//! OpenAI Chat Completions egress transport. Maps unified ChatRequest → the
//! `/v1/chat/completions` body, parses the response, and normalizes usage into
//! NON-OVERLAPPING TokenUsage (OpenAI's `prompt_tokens` INCLUDES cached tokens,
//! so cached are subtracted out of input — preserving the spine's exact cost
//! math). Idempotency-key is sent as the `Idempotency-Key` header (no double-
//! billing). Streaming added in Task 10. Tool/structured-output FULL fidelity is
//! P1.3; here tool defs/calls pass through their natural OpenAI shape.

use std::collections::HashMap;

use async_trait::async_trait;
use gateway_spine::TokenUsage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::{ContentPart, ImageSource, Message, Role};
use crate::provider::{Credentials, DeltaStream, Provider, ProviderCapabilities, ProviderError};
use crate::req::ChatRequest;
use crate::resp::{ChatResponse, FinishReason};
use crate::toolcall::{ToolCall, ToolChoice, ToolDef};

const DEFAULT_BASE_URL: &str = "https://api.openai.com";

pub struct OpenAi {
    http: reqwest::Client,
}

impl Default for OpenAi {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAi {
    pub fn new() -> Self {
        OpenAi {
            http: reqwest::Client::new(),
        }
    }

    fn base_url<'a>(&self, creds: &'a Credentials) -> &'a str {
        creds.base_url.as_deref().unwrap_or(DEFAULT_BASE_URL)
    }
}

// ---- wire structs (private) ----

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireReqToolDef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<i64>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<WireStreamOptions>,
}

#[derive(Serialize)]
struct WireStreamOptions {
    include_usage: bool,
}

#[derive(Serialize)]
struct WireMessage {
    role: &'static str,
    /// String for text-only, array of typed parts for multimodal, or omitted for an
    /// assistant turn carrying only tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<WireReqCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize)]
struct WireReqToolDef {
    #[serde(rename = "type")]
    kind: &'static str,
    function: WireReqToolFn,
}

#[derive(Serialize)]
struct WireReqToolFn {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: serde_json::Value,
}

#[derive(Serialize)]
struct WireReqCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: WireReqCallFn,
}

#[derive(Serialize)]
struct WireReqCallFn {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct WireResponse {
    id: Option<String>,
    model: String,
    choices: Vec<WireChoice>,
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct WireChoice {
    message: WireRespMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct WireRespMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<WireToolCall>,
}

#[derive(Deserialize)]
struct WireToolCall {
    id: String,
    function: WireFunction,
}

#[derive(Deserialize)]
struct WireFunction {
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Deserialize)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: i64,
    #[serde(default)]
    completion_tokens: i64,
    #[serde(default)]
    total_tokens: Option<i64>,
    #[serde(default)]
    prompt_tokens_details: Option<WirePromptDetails>,
    #[serde(default)]
    completion_tokens_details: Option<WireCompletionDetails>,
    #[serde(flatten)]
    unknown_fields: HashMap<String, Value>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct WirePromptDetails {
    #[serde(default)]
    cached_tokens: i64,
    #[serde(default)]
    audio_tokens: i64,
    #[serde(flatten)]
    unknown_fields: HashMap<String, Value>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct WireCompletionDetails {
    #[serde(default)]
    reasoning_tokens: i64,
    #[serde(default)]
    audio_tokens: i64,
    #[serde(default)]
    accepted_prediction_tokens: i64,
    #[serde(default)]
    rejected_prediction_tokens: i64,
    #[serde(flatten)]
    unknown_fields: HashMap<String, Value>,
}

// ---- streaming wire structs ----

#[derive(Deserialize)]
struct WireStreamChunk {
    #[serde(default)]
    choices: Vec<WireStreamChoice>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct WireStreamChoice {
    #[serde(default)]
    delta: WireDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct WireDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<WireToolCallDelta>,
}

#[derive(Deserialize)]
struct WireToolCallDelta {
    index: i64,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<WireFunctionDelta>,
}

#[derive(Deserialize)]
struct WireFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

// ---- mapping ----

fn role_str(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// Build the OpenAI `content` value: a bare string for text-only messages, an array
/// of typed parts when an image is present, or `None` for an assistant turn that
/// carried only tool calls.
fn content_json(m: &Message) -> Option<serde_json::Value> {
    let has_image = m
        .content
        .iter()
        .any(|p| matches!(p, ContentPart::Image { .. }));
    if !has_image {
        let text = m.text_content();
        if text.is_empty() && !m.tool_calls.is_empty() {
            return None;
        }
        return Some(serde_json::Value::String(text));
    }
    let parts: Vec<serde_json::Value> = m
        .content
        .iter()
        .map(|p| match p {
            ContentPart::Text { text } => serde_json::json!({"type": "text", "text": text}),
            ContentPart::Image { source } => {
                let url = match source {
                    ImageSource::Url { url } => url.clone(),
                    ImageSource::Base64 { media_type, data } => {
                        format!("data:{media_type};base64,{data}")
                    }
                };
                serde_json::json!({"type": "image_url", "image_url": {"url": url}})
            }
        })
        .collect();
    Some(serde_json::Value::Array(parts))
}

fn map_messages(messages: &[Message]) -> Vec<WireMessage> {
    messages
        .iter()
        .map(|m| WireMessage {
            role: role_str(m.role),
            content: content_json(m),
            tool_calls: m
                .tool_calls
                .iter()
                .map(|t| WireReqCall {
                    id: t.id.clone(),
                    kind: "function",
                    function: WireReqCallFn {
                        name: t.name.clone(),
                        arguments: t.arguments.clone(),
                    },
                })
                .collect(),
            tool_call_id: m.tool_call_id.clone(),
        })
        .collect()
}

fn map_tools(tools: &[ToolDef]) -> Vec<WireReqToolDef> {
    tools
        .iter()
        .map(|t| WireReqToolDef {
            kind: "function",
            function: WireReqToolFn {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
            },
        })
        .collect()
}

fn map_tool_choice(tc: Option<&ToolChoice>) -> Option<serde_json::Value> {
    tc.map(|c| match c {
        ToolChoice::Auto => serde_json::Value::String("auto".into()),
        ToolChoice::None => serde_json::Value::String("none".into()),
        ToolChoice::Required => serde_json::Value::String("required".into()),
        ToolChoice::Function { name } => {
            serde_json::json!({"type": "function", "function": {"name": name}})
        }
    })
}

fn map_finish(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("stop") => FinishReason::Stop,
        Some("length") => FinishReason::Length,
        Some("tool_calls") => FinishReason::ToolCalls,
        Some("content_filter") => FinishReason::ContentFilter,
        _ => FinishReason::Unknown,
    }
}

fn usage_unknown_field_is_cost_critical(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    key.contains("token") || key.contains("cache") || key.contains("usage")
}

fn openai_usage_drift_reason(u: &WireUsage) -> Option<String> {
    let total = u
        .total_tokens
        .unwrap_or_else(|| u.prompt_tokens + u.completion_tokens);
    if total > 0 && u.prompt_tokens == 0 && u.completion_tokens == 0 {
        return Some(format!(
            "usage total_tokens={total} but prompt_tokens=0 and completion_tokens=0"
        ));
    }

    if let Some(key) = u
        .unknown_fields
        .keys()
        .find(|key| usage_unknown_field_is_cost_critical(key))
    {
        return Some(format!("unknown cost-critical usage field `{key}`"));
    }

    if let Some(details) = &u.prompt_tokens_details
        && let Some(key) = details
            .unknown_fields
            .keys()
            .find(|key| usage_unknown_field_is_cost_critical(key))
    {
        return Some(format!(
            "unknown cost-critical prompt_tokens_details field `{key}`"
        ));
    }

    if let Some(details) = &u.completion_tokens_details
        && let Some(key) = details
            .unknown_fields
            .keys()
            .find(|key| usage_unknown_field_is_cost_critical(key))
    {
        return Some(format!(
            "unknown cost-critical completion_tokens_details field `{key}`"
        ));
    }

    None
}

/// OpenAI `prompt_tokens` INCLUDES cached; split into non-overlapping buckets.
fn map_usage(u: Option<WireUsage>) -> Result<TokenUsage, ProviderError> {
    let Some(u) = u else {
        return Ok(TokenUsage::default());
    };
    if let Some(reason) = openai_usage_drift_reason(&u) {
        return Err(ProviderError::SchemaDrift {
            provider: "openai",
            reason,
        });
    }
    let cached = u
        .prompt_tokens_details
        .map(|d| d.cached_tokens)
        .unwrap_or(0);
    Ok(TokenUsage {
        input_tokens: (u.prompt_tokens - cached).max(0),
        output_tokens: u.completion_tokens,
        cache_read_tokens: cached,
        cache_write_tokens: 0,
    })
}

fn map_response(w: WireResponse) -> Result<ChatResponse, ProviderError> {
    let choice = w.choices.into_iter().next();
    let (content, tool_calls, finish) = match choice {
        Some(c) => {
            let content = c
                .message
                .content
                .filter(|s| !s.is_empty())
                .map(|s| vec![ContentPart::text(s)])
                .unwrap_or_default();
            let tool_calls: Vec<ToolCall> = c
                .message
                .tool_calls
                .into_iter()
                .map(|t| ToolCall {
                    id: t.id,
                    name: t.function.name,
                    arguments: t.function.arguments,
                })
                .collect();
            (content, tool_calls, map_finish(c.finish_reason.as_deref()))
        }
        None => (Vec::new(), Vec::new(), FinishReason::Unknown),
    };
    Ok(ChatResponse {
        model: w.model,
        content,
        tool_calls,
        finish_reason: finish,
        usage: map_usage(w.usage)?,
        provider_response_id: w.id,
    })
}

/// Parse one OpenAI stream `data:` payload into a unified delta. Returns `None`
/// for chunks that carry nothing useful. Tool-call argument fragments are
/// relayed per-index; AGGREGATION is P1.3.
fn parse_stream_chunk(payload: &str) -> Result<Option<crate::stream::StreamDelta>, ProviderError> {
    use crate::stream::{StreamDelta, ToolCallDelta};
    let chunk: WireStreamChunk =
        serde_json::from_str(payload).map_err(|e| ProviderError::Decode(e.to_string()))?;
    let mut delta = StreamDelta::default();
    if let Some(choice) = chunk.choices.into_iter().next() {
        if let Some(c) = choice.delta.content
            && !c.is_empty()
        {
            delta.content_delta = Some(c);
        }
        for tc in choice.delta.tool_calls {
            let (name, args) = match tc.function {
                Some(f) => (f.name, f.arguments),
                None => (None, None),
            };
            delta.tool_call_deltas.push(ToolCallDelta {
                index: tc.index,
                id: tc.id,
                name,
                arguments_delta: args,
            });
        }
        if let Some(f) = choice.finish_reason {
            delta.finish_reason = Some(map_finish(Some(f.as_str())));
        }
    }
    if let Some(u) = chunk.usage {
        delta.usage = Some(map_usage(Some(u))?);
    }
    if delta.is_empty() {
        Ok(None)
    } else {
        Ok(Some(delta))
    }
}

#[async_trait]
impl Provider for OpenAi {
    fn id(&self) -> &str {
        "openai"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_vision: true,
            supports_idempotency: true,
        }
    }

    async fn chat(
        &self,
        req: &ChatRequest,
        creds: &Credentials,
        idempotency_key: &str,
    ) -> Result<ChatResponse, ProviderError> {
        let url = format!("{}/v1/chat/completions", self.base_url(creds));
        let wire = WireRequest {
            model: &req.model,
            messages: map_messages(&req.messages),
            tools: map_tools(&req.tools),
            tool_choice: map_tool_choice(req.tool_choice.as_ref()),
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            stream: false,
            stream_options: None,
        };
        let mut rb = self
            .http
            .post(url)
            .bearer_auth(&creds.api_key)
            .header("Idempotency-Key", idempotency_key)
            .json(&wire);
        for (k, v) in &creds.extra_headers {
            rb = rb.header(k, v);
        }
        let resp = rb
            .send()
            .await
            .map_err(|e| ProviderError::Transport(e.to_string()))?;
        let status = resp.status();
        if status == reqwest::StatusCode::UNAUTHORIZED {
            return Err(ProviderError::Auth);
        }
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<i64>().ok());
            return Err(ProviderError::RateLimited {
                retry_after_secs: retry,
            });
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Upstream {
                status: status.as_u16(),
                body,
            });
        }
        let wire: WireResponse = resp
            .json()
            .await
            .map_err(|e| ProviderError::Decode(e.to_string()))?;
        map_response(wire)
    }

    async fn stream(
        &self,
        req: &ChatRequest,
        creds: &Credentials,
        idempotency_key: &str,
    ) -> Result<DeltaStream, ProviderError> {
        use crate::sse::{SseDecoder, SseEvent};
        use futures::StreamExt;

        let url = format!("{}/v1/chat/completions", self.base_url(creds));
        let wire = WireRequest {
            model: &req.model,
            messages: map_messages(&req.messages),
            tools: map_tools(&req.tools),
            tool_choice: map_tool_choice(req.tool_choice.as_ref()),
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            stream: true,
            stream_options: Some(WireStreamOptions {
                include_usage: true,
            }),
        };
        let mut rb = self
            .http
            .post(url)
            .bearer_auth(&creds.api_key)
            .header("Idempotency-Key", idempotency_key)
            .json(&wire);
        for (k, v) in &creds.extra_headers {
            rb = rb.header(k, v);
        }
        let resp = rb
            .send()
            .await
            .map_err(|e| ProviderError::Transport(e.to_string()))?;
        let status = resp.status();
        if status == reqwest::StatusCode::UNAUTHORIZED {
            return Err(ProviderError::Auth);
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Upstream {
                status: status.as_u16(),
                body,
            });
        }

        let byte_stream = resp.bytes_stream();
        let out = futures::stream::unfold(
            (
                byte_stream,
                SseDecoder::new(),
                Vec::<Result<crate::stream::StreamDelta, ProviderError>>::new(),
                false,
            ),
            |(mut bytes, mut decoder, mut pending, mut done)| async move {
                loop {
                    if let Some(item) = pending.pop() {
                        return Some((item, (bytes, decoder, pending, done)));
                    }
                    if done {
                        return None;
                    }
                    match bytes.next().await {
                        Some(Ok(chunk)) => {
                            let mut produced = Vec::new();
                            for ev in decoder.push(chunk) {
                                match ev {
                                    SseEvent::Done => done = true,
                                    SseEvent::Data(payload) => match parse_stream_chunk(&payload) {
                                        Ok(Some(d)) => produced.push(Ok(d)),
                                        Ok(None) => {}
                                        Err(e) => produced.push(Err(e)),
                                    },
                                }
                            }
                            // emit in order: push reversed so pop() yields FIFO
                            produced.reverse();
                            pending = produced;
                        }
                        Some(Err(e)) => {
                            done = true;
                            pending = vec![Err(ProviderError::Transport(e.to_string()))];
                        }
                        None => {
                            // Upstream ended (possibly aborted). Surface a terminal
                            // Unknown-finish so usage/finish is never silently lost.
                            done = true;
                        }
                    }
                }
            },
        );
        Ok(Box::pin(out))
    }
}
