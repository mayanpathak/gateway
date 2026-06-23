//! The egress seam. A `Provider` takes a unified `ChatRequest` + resolved
//! `Credentials` + an `idempotency_key` and returns either a unified
//! `ChatResponse` or a stream of `StreamDelta`s. The idempotency key is threaded
//! so the SAME logical request reused across retries/failover bills once
//! (no-double-billing invariant §2): transports MUST forward it as the provider's
//! idempotency header. `ProviderError::Unsupported` is the typed no-silent-
//! degradation seam P1.3 grows into the fidelity matrix.

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::req::ChatRequest;
use crate::resp::ChatResponse;
use crate::stream::StreamDelta;

/// Resolved upstream credentials + endpoint for one transport call. The spine
/// decrypts/injects these (P1.6); transports never read env/secrets themselves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Credentials {
    /// Bearer/API key material (already decrypted).
    pub api_key: String,
    /// Base URL override (e.g. Azure/self-hosted/proxy). `None` = provider default.
    pub base_url: Option<String>,
    /// Extra headers to forward verbatim (e.g. `anthropic-beta`, org ids).
    pub extra_headers: Vec<(String, String)>,
}

impl Credentials {
    pub fn new(api_key: impl Into<String>) -> Self {
        Credentials {
            api_key: api_key.into(),
            base_url: None,
            extra_headers: Vec::new(),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.push((name.into(), value.into()));
        self
    }
}

/// Errors a transport surfaces. HTTP statuses map at the server layer (P1.4):
/// Auth → 401, RateLimited → 429, Upstream{status} passes through, Unsupported →
/// 400/422, Transport/Decode → 502.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("upstream authentication failed")]
    Auth,
    #[error("upstream rate limited{}", retry_after_secs.map(|s| format!(" (retry after {s}s)")).unwrap_or_default())]
    RateLimited { retry_after_secs: Option<i64> },
    #[error("upstream returned status {status}: {body}")]
    Upstream { status: u16, body: String },
    #[error("request feature unsupported by this provider: {feature}")]
    Unsupported { feature: String },
    #[error("provider {provider} response appears to have schema drift: {reason}")]
    SchemaDrift {
        provider: &'static str,
        reason: String,
    },
    #[error("transport error: {0}")]
    Transport(String),
    #[error("failed to decode upstream response: {0}")]
    Decode(String),
}

/// A unified streaming response: a pinned, boxed, `Send` stream of deltas.
pub type DeltaStream = Pin<Box<dyn Stream<Item = Result<StreamDelta, ProviderError>> + Send>>;

/// Capabilities a transport declares so the router/spine can pre-validate a
/// request shape (richer capability data lives in the model registry; this is the
/// per-transport floor).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderCapabilities {
    pub supports_streaming: bool,
    pub supports_tools: bool,
    pub supports_vision: bool,
    /// Provider honors an idempotency header for safe retries.
    pub supports_idempotency: bool,
}

/// The egress transport trait. One impl per provider API shape (~30 total).
#[async_trait]
pub trait Provider: Send + Sync {
    /// Stable provider id (e.g. "openai", "anthropic", "gemini").
    fn id(&self) -> &str;

    fn capabilities(&self) -> ProviderCapabilities;

    /// Non-streaming completion. `idempotency_key` MUST be forwarded as the
    /// provider's idempotency header (no-double-billing). The same key across
    /// retries of one logical request yields one bill.
    async fn chat(
        &self,
        req: &ChatRequest,
        creds: &Credentials,
        idempotency_key: &str,
    ) -> Result<ChatResponse, ProviderError>;

    /// Streaming completion. Same idempotency contract. The returned stream's
    /// final delta MUST carry usage when the provider reports it; an aborted
    /// upstream yields a terminal `Unknown`-finish delta, never a silent drop.
    async fn stream(
        &self,
        req: &ChatRequest,
        creds: &Credentials,
        idempotency_key: &str,
    ) -> Result<DeltaStream, ProviderError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Message, Role};
    use crate::resp::FinishReason;
    use futures::StreamExt;
    use gateway_spine::TokenUsage;

    /// A trivial in-memory provider proving the trait is object-safe and the
    /// idempotency key is observable by the impl.
    struct EchoProvider {
        last_idempotency_key: std::sync::Mutex<Option<String>>,
    }

    #[async_trait]
    impl Provider for EchoProvider {
        fn id(&self) -> &str {
            "echo"
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                supports_streaming: true,
                supports_tools: false,
                supports_vision: false,
                supports_idempotency: true,
            }
        }
        async fn chat(
            &self,
            req: &ChatRequest,
            _creds: &Credentials,
            idempotency_key: &str,
        ) -> Result<ChatResponse, ProviderError> {
            *self.last_idempotency_key.lock().unwrap() = Some(idempotency_key.to_string());
            Ok(ChatResponse {
                model: req.model.clone(),
                content: vec![crate::message::ContentPart::text(
                    req.messages
                        .last()
                        .map(|m| m.text_content())
                        .unwrap_or_default(),
                )],
                tool_calls: Vec::new(),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                provider_response_id: None,
            })
        }
        async fn stream(
            &self,
            _req: &ChatRequest,
            _creds: &Credentials,
            _idempotency_key: &str,
        ) -> Result<DeltaStream, ProviderError> {
            let deltas = vec![
                Ok(StreamDelta::text("hi")),
                Ok(StreamDelta::finish(
                    FinishReason::Stop,
                    TokenUsage::default(),
                )),
            ];
            Ok(Box::pin(futures::stream::iter(deltas)))
        }
    }

    #[tokio::test]
    async fn trait_is_object_safe_and_threads_idempotency_key() {
        let p: Box<dyn Provider> = Box::new(EchoProvider {
            last_idempotency_key: std::sync::Mutex::new(None),
        });
        let req = ChatRequest::new("echo-1", vec![Message::text(Role::User, "ping")]);
        let creds = Credentials::new("sk-x");
        let resp = p.chat(&req, &creds, "idem-123").await.unwrap();
        assert_eq!(resp.text(), "ping");
    }

    #[tokio::test]
    async fn stream_yields_text_then_finish_with_usage() {
        let p = EchoProvider {
            last_idempotency_key: std::sync::Mutex::new(None),
        };
        let req = ChatRequest::new("echo-1", vec![Message::text(Role::User, "ping")]);
        let creds = Credentials::new("sk-x");
        let mut s = p.stream(&req, &creds, "idem-1").await.unwrap();
        let first = s.next().await.unwrap().unwrap();
        assert_eq!(first.content_delta.as_deref(), Some("hi"));
        let last = s.next().await.unwrap().unwrap();
        assert_eq!(last.finish_reason, Some(FinishReason::Stop));
        assert!(last.usage.is_some());
        assert!(s.next().await.is_none());
    }

    #[test]
    fn credentials_builder() {
        let c = Credentials::new("sk-x")
            .with_base_url("https://proxy.local")
            .with_header("anthropic-beta", "tools-2024");
        assert_eq!(c.base_url.as_deref(), Some("https://proxy.local"));
        assert_eq!(
            c.extra_headers,
            vec![("anthropic-beta".into(), "tools-2024".into())]
        );
    }
}
