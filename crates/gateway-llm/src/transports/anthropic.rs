//! Anthropic Messages (`/v1/messages`) egress transport. System messages are
//! hoisted into the top-level `system` field; the rest map to user/assistant
//! turns. Auth is `x-api-key` + `anthropic-version`; `anthropic-beta` and other
//! overrides ride via `Credentials.extra_headers` (Claude Code requires header
//! forwarding). Usage is already non-overlapping (input excludes cache). `max_
//! tokens` is REQUIRED by Anthropic — we default it when unset. Streaming in
//! Task 12; full tool/structured-output fidelity is P1.3.

use std::collections::HashMap;

use async_trait::async_trait;
use gateway_spine::TokenUsage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::{ContentPart, Message, Role};
use crate::provider::{Credentials, DeltaStream, Provider, ProviderCapabilities, ProviderError};
use crate::req::ChatRequest;
use crate::resp::{ChatResponse, FinishReason};
use crate::toolcall::ToolCall;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: i64 = 4096;

pub struct Anthropic {
    http: reqwest::Client,
}

impl Default for Anthropic {
    fn default() -> Self {
        Self::new()
    }
}

impl Anthropic {
    pub fn new() -> Self {
        Anthropic {
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
    max_tokens: i64,
    messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
}

#[derive(Serialize)]
struct WireMessage {
    role: &'static str,
    content: String,
}

#[derive(Deserialize)]
struct WireResponse {
    id: Option<String>,
    model: String,
    stop_reason: Option<String>,
    #[serde(default)]
    content: Vec<WireContentBlock>,
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct WireUsage {
    #[serde(default)]
    input_tokens: i64,
    #[serde(default)]
    output_tokens: i64,
    #[serde(default)]
    cache_read_input_tokens: i64,
    #[serde(default)]
    cache_creation_input_tokens: i64,
    #[serde(flatten)]
    unknown_fields: HashMap<String, Value>,
}

// ---- streaming wire structs ----

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireStreamEvent {
    MessageStart {
        message: WireStreamStartMessage,
    },
    ContentBlockDelta {
        delta: WireStreamDelta,
    },
    MessageDelta {
        delta: WireStreamMessageDelta,
        #[serde(default)]
        usage: Option<WireDeltaUsage>,
    },
    MessageStop,
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct WireStreamStartMessage {
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireStreamDelta {
    TextDelta {
        text: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct WireStreamMessageDelta {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct WireDeltaUsage {
    #[serde(default)]
    output_tokens: i64,
}

// ---- mapping ----

/// Hoist system turns into the top-level `system` string; map the rest.
fn split_messages(messages: &[Message]) -> (Option<String>, Vec<WireMessage>) {
    let mut system = String::new();
    let mut out = Vec::new();
    for m in messages {
        match m.role {
            Role::System => {
                if !system.is_empty() {
                    system.push('\n');
                }
                system.push_str(&m.text_content());
            }
            Role::User | Role::Tool => {
                out.push(WireMessage {
                    role: "user",
                    content: m.text_content(),
                });
            }
            Role::Assistant => {
                out.push(WireMessage {
                    role: "assistant",
                    content: m.text_content(),
                });
            }
        }
    }
    (
        if system.is_empty() {
            None
        } else {
            Some(system)
        },
        out,
    )
}

fn map_finish(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("end_turn") | Some("stop_sequence") => FinishReason::Stop,
        Some("max_tokens") => FinishReason::Length,
        Some("tool_use") => FinishReason::ToolCalls,
        _ => FinishReason::Unknown,
    }
}

fn usage_unknown_field_is_cost_critical(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    key.contains("token") || key.contains("cache") || key.contains("usage")
}

fn map_usage(u: Option<WireUsage>) -> Result<TokenUsage, ProviderError> {
    let Some(u) = u else {
        return Ok(TokenUsage::default());
    };
    if let Some(key) = u
        .unknown_fields
        .keys()
        .find(|key| usage_unknown_field_is_cost_critical(key))
    {
        return Err(ProviderError::SchemaDrift {
            provider: "anthropic",
            reason: format!("unknown cost-critical usage field `{key}`"),
        });
    }
    Ok(TokenUsage {
        input_tokens: u.input_tokens,
        output_tokens: u.output_tokens,
        cache_read_tokens: u.cache_read_input_tokens,
        cache_write_tokens: u.cache_creation_input_tokens,
    })
}

fn map_response(w: WireResponse) -> Result<ChatResponse, ProviderError> {
    let mut content = Vec::new();
    let mut tool_calls = Vec::new();
    for block in w.content {
        match block {
            WireContentBlock::Text { text } => content.push(ContentPart::text(text)),
            WireContentBlock::ToolUse { id, name, input } => tool_calls.push(ToolCall {
                id,
                name,
                arguments: input.to_string(),
            }),
            WireContentBlock::Other => {}
        }
    }
    Ok(ChatResponse {
        model: w.model,
        content,
        tool_calls,
        finish_reason: map_finish(w.stop_reason.as_deref()),
        usage: map_usage(w.usage)?,
        provider_response_id: w.id,
    })
}

/// Fold one Anthropic stream event into (optional emitted delta, usage accumulator
/// mutation). Returns the unified delta to relay, if any.
fn fold_stream_event(
    payload: &str,
    acc_usage: &mut TokenUsage,
) -> Result<Option<crate::stream::StreamDelta>, ProviderError> {
    use crate::stream::StreamDelta;
    let ev: WireStreamEvent =
        serde_json::from_str(payload).map_err(|e| ProviderError::Decode(e.to_string()))?;
    match ev {
        WireStreamEvent::MessageStart { message } => {
            if let Some(u) = message.usage {
                let mapped = map_usage(Some(u))?;
                acc_usage.input_tokens = mapped.input_tokens;
                acc_usage.cache_read_tokens = mapped.cache_read_tokens;
                acc_usage.cache_write_tokens = mapped.cache_write_tokens;
            }
            Ok(None)
        }
        WireStreamEvent::ContentBlockDelta { delta } => match delta {
            WireStreamDelta::TextDelta { text } => Ok(Some(StreamDelta::text(text))),
            // Tool-call argument fragments: relay carried in P1.3's aggregation;
            // here we surface them as a content-less tool delta on index 0.
            WireStreamDelta::InputJsonDelta { partial_json } => {
                use crate::stream::ToolCallDelta;
                Ok(Some(StreamDelta {
                    tool_call_deltas: vec![ToolCallDelta {
                        index: 0,
                        id: None,
                        name: None,
                        arguments_delta: Some(partial_json),
                    }],
                    ..Default::default()
                }))
            }
            WireStreamDelta::Other => Ok(None),
        },
        WireStreamEvent::MessageDelta { delta, usage } => {
            if let Some(u) = usage {
                acc_usage.output_tokens = u.output_tokens;
            }
            // Terminal delta: carry finish + the accumulated usage so it is never lost.
            Ok(Some(StreamDelta {
                finish_reason: Some(map_finish(delta.stop_reason.as_deref())),
                usage: Some(*acc_usage),
                ..Default::default()
            }))
        }
        WireStreamEvent::MessageStop | WireStreamEvent::Other => Ok(None),
    }
}

#[async_trait]
impl Provider for Anthropic {
    fn id(&self) -> &str {
        "anthropic"
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
        let url = format!("{}/v1/messages", self.base_url(creds));
        let (system, messages) = split_messages(&req.messages);
        let wire = WireRequest {
            model: &req.model,
            max_tokens: req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            messages,
            system,
            temperature: req.temperature,
            stream: false,
        };
        let mut rb = self
            .http
            .post(url)
            .header("x-api-key", &creds.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
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

        let url = format!("{}/v1/messages", self.base_url(creds));
        let (system, messages) = split_messages(&req.messages);
        let wire = WireRequest {
            model: &req.model,
            max_tokens: req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            messages,
            system,
            temperature: req.temperature,
            stream: true,
        };
        let mut rb = self
            .http
            .post(url)
            .header("x-api-key", &creds.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
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
        type State = (
            std::pin::Pin<Box<dyn futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>,
            SseDecoder,
            Vec<Result<crate::stream::StreamDelta, ProviderError>>,
            TokenUsage,
            bool,
        );
        let init: State = (
            Box::pin(byte_stream),
            SseDecoder::new(),
            Vec::new(),
            TokenUsage::default(),
            false,
        );
        let out = futures::stream::unfold(
            init,
            |(mut bytes, mut decoder, mut pending, mut acc, mut done)| async move {
                loop {
                    if let Some(item) = pending.pop() {
                        return Some((item, (bytes, decoder, pending, acc, done)));
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
                                    SseEvent::Data(payload) => {
                                        match fold_stream_event(&payload, &mut acc) {
                                            Ok(Some(d)) => produced.push(Ok(d)),
                                            Ok(None) => {}
                                            Err(e) => produced.push(Err(e)),
                                        }
                                    }
                                }
                            }
                            produced.reverse();
                            pending = produced;
                        }
                        Some(Err(e)) => {
                            done = true;
                            pending = vec![Err(ProviderError::Transport(e.to_string()))];
                        }
                        None => done = true,
                    }
                }
            },
        );
        Ok(Box::pin(out))
    }
}
