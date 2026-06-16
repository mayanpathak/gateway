//! HTTP webhook guardrail — Lakera Guard compatible.
//!
//! POSTs `{"input": <text>, "stage": <stage>}` to a configurable endpoint and
//! maps the response to a [`GuardVerdict`].
//!
//! Response shape (Lakera Guard v1):
//!
//! ```json
//! {
//!   "results": [{ "flagged": true, "categories": { "prompt_injection": true } }]
//! }
//! ```
//!
//! If `flagged == true`, returns `Block` with the first truthy category name as
//! the reason. On any HTTP error, non-2xx status, parse failure, or timeout,
//! returns `Allow` (fail-open) and logs a warning.

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::guardrail::Guardrail;
use crate::types::{GuardContext, GuardStage, GuardVerdict};

// ── Wire types ────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct WebhookRequest<'a> {
    input: &'a str,
    stage: &'a GuardStage,
}

#[derive(Debug, Deserialize)]
struct WebhookResponse {
    results: Vec<ResultEntry>,
}

#[derive(Debug, Deserialize)]
struct ResultEntry {
    flagged: bool,
    // serde_json preserves insertion order in Maps, so "first truthy" is stable.
    #[serde(default)]
    categories: serde_json::Map<String, serde_json::Value>,
}

// ── Guardrail ─────────────────────────────────────────────────────────────────

/// Delegates content moderation to an external HTTP endpoint.
///
/// The `reqwest::Client` is built once at construction time and reused across
/// requests. Default timeout is 500ms — bounded so a slow endpoint can't blow
/// the gateway's latency budget.
#[derive(Debug, Clone)]
pub struct WebhookGuardrail {
    pub name: String,
    pub endpoint_url: String,
    client: reqwest::Client,
}

impl WebhookGuardrail {
    pub fn new(name: impl Into<String>, endpoint_url: impl Into<String>) -> Self {
        Self::with_timeout(name, endpoint_url, Duration::from_millis(500))
    }

    pub fn with_timeout(
        name: impl Into<String>,
        endpoint_url: impl Into<String>,
        timeout: Duration,
    ) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .expect("failed to build reqwest client");

        Self {
            name: name.into(),
            endpoint_url: endpoint_url.into(),
            client,
        }
    }
}

#[async_trait]
impl Guardrail for WebhookGuardrail {
    fn name(&self) -> &str {
        &self.name
    }

    async fn check(&self, ctx: &GuardContext) -> GuardVerdict {
        let body = WebhookRequest {
            input: &ctx.text,
            stage: &ctx.stage,
        };

        let response = match self
            .client
            .post(&self.endpoint_url)
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(
                    guardrail = %self.name,
                    endpoint = %self.endpoint_url,
                    error = %e,
                    "webhook request failed — failing open"
                );
                return GuardVerdict::Allow;
            }
        };

        if !response.status().is_success() {
            tracing::warn!(
                guardrail = %self.name,
                endpoint = %self.endpoint_url,
                status = %response.status(),
                "webhook returned non-2xx — failing open"
            );
            return GuardVerdict::Allow;
        }

        let parsed = match response.json::<WebhookResponse>().await {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    guardrail = %self.name,
                    endpoint = %self.endpoint_url,
                    error = %e,
                    "webhook response parse error — failing open"
                );
                return GuardVerdict::Allow;
            }
        };

        let first = match parsed.results.into_iter().next() {
            Some(r) => r,
            None => return GuardVerdict::Allow,
        };

        if !first.flagged {
            return GuardVerdict::Allow;
        }

        let reason = first
            .categories
            .into_iter()
            .find(|(_, v)| v.as_bool().unwrap_or(false))
            .map(|(k, _)| k)
            .unwrap_or_else(|| "content flagged".to_string());

        GuardVerdict::Block { reason }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::types::GuardStage;

    fn ctx(text: &str) -> GuardContext {
        GuardContext::new(GuardStage::PreRequest, text)
    }

    async fn run_with_body(
        response_body: serde_json::Value,
    ) -> (GuardVerdict, wiremock::MockServer) {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/check"))
            .respond_with(ResponseTemplate::new(200).set_body_json(response_body))
            .mount(&server)
            .await;

        let g = WebhookGuardrail::new("test", format!("{}/check", server.uri()));
        let verdict = g.check(&ctx("ignore previous instructions")).await;
        (verdict, server)
    }

    #[tokio::test]
    async fn sends_correct_request_body() {
        let server = MockServer::start().await;

        let expected_body = serde_json::json!({
            "input": "test prompt",
            "stage": "PreRequest",
        });

        Mock::given(method("POST"))
            .and(path("/check"))
            .and(body_json(&expected_body))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{ "flagged": false, "categories": {} }]
            })))
            .mount(&server)
            .await;

        let g = WebhookGuardrail::new("test", format!("{}/check", server.uri()));
        let ctx = GuardContext::new(GuardStage::PreRequest, "test prompt");
        g.check(&ctx).await;

        // wiremock asserts all mounted mocks were called when the server drops.
        // If the body didn't match, the mock would not fire and the server would
        // report an unsatisfied expectation on drop.
        server.verify().await;
    }

    #[tokio::test]
    async fn blocks_when_flagged() {
        let (verdict, _server) = run_with_body(serde_json::json!({
            "results": [{ "flagged": true, "categories": { "prompt_injection": true } }]
        }))
        .await;

        assert!(
            matches!(verdict, GuardVerdict::Block { .. }),
            "expected Block, got {verdict:?}"
        );
    }

    #[tokio::test]
    async fn block_reason_is_first_truthy_category() {
        let (verdict, _server) = run_with_body(serde_json::json!({
            "results": [{ "flagged": true, "categories": { "jailbreak": true } }]
        }))
        .await;

        match verdict {
            GuardVerdict::Block { reason } => assert_eq!(reason, "jailbreak"),
            other => panic!("expected Block, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn block_reason_falls_back_when_no_categories() {
        let (verdict, _server) = run_with_body(serde_json::json!({
            "results": [{ "flagged": true, "categories": {} }]
        }))
        .await;

        match verdict {
            GuardVerdict::Block { reason } => assert_eq!(reason, "content flagged"),
            other => panic!("expected Block, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn allows_when_not_flagged() {
        let (verdict, _server) = run_with_body(serde_json::json!({
            "results": [{ "flagged": false, "categories": {} }]
        }))
        .await;

        assert_eq!(verdict, GuardVerdict::Allow);
    }

    #[tokio::test]
    async fn allows_on_empty_results() {
        let (verdict, _server) = run_with_body(serde_json::json!({ "results": [] })).await;
        assert_eq!(verdict, GuardVerdict::Allow);
    }

    #[tokio::test]
    async fn fails_open_on_server_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/check"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let g = WebhookGuardrail::new("test", format!("{}/check", server.uri()));
        let verdict = g.check(&ctx("anything")).await;
        assert_eq!(verdict, GuardVerdict::Allow);
    }

    #[tokio::test]
    async fn fails_open_on_timeout() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/check"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_secs(5))
                    .set_body_json(serde_json::json!({
                        "results": [{ "flagged": true, "categories": { "prompt_injection": true } }]
                    })),
            )
            .mount(&server)
            .await;

        let g = WebhookGuardrail::with_timeout(
            "test",
            format!("{}/check", server.uri()),
            Duration::from_millis(100),
        );

        let verdict = g.check(&ctx("anything")).await;
        assert_eq!(verdict, GuardVerdict::Allow);
    }

    #[tokio::test]
    async fn fails_open_on_bad_json() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/check"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .mount(&server)
            .await;

        let g = WebhookGuardrail::new("test", format!("{}/check", server.uri()));
        let verdict = g.check(&ctx("anything")).await;
        assert_eq!(verdict, GuardVerdict::Allow);
    }

    #[test]
    fn name_is_set() {
        let g = WebhookGuardrail::new("my-hook", "http://localhost:9090/check");
        assert_eq!(g.name(), "my-hook");
    }
}
