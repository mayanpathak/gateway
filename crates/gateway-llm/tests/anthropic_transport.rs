//! Anthropic /v1/messages egress, non-streaming, mocked. Asserts request mapping
//! (system hoisted out of messages, x-api-key + anthropic-version + idempotency),
//! response mapping (stop_reason → finish_reason), and usage extraction. Anthropic
//! already reports NON-overlapping buckets (input excludes cache), so the mapping
//! is direct.

use gateway_llm::message::{Message, Role};
use gateway_llm::provider::{Credentials, Provider, ProviderError};
use gateway_llm::req::ChatRequest;
use gateway_llm::resp::FinishReason;
use gateway_llm::transports::anthropic::Anthropic;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn anthropic_messages_maps_request_response_and_usage() {
    let server = MockServer::start().await;
    let body = std::fs::read_to_string("tests/fixtures/anthropic_messages.json").unwrap();

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "sk-ant"))
        .and(header("anthropic-version", "2023-06-01"))
        .and(header("idempotency-key", "idem-a"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .expect(1)
        .mount(&server)
        .await;

    let provider = Anthropic::new();
    let creds = Credentials::new("sk-ant").with_base_url(server.uri());
    let req = ChatRequest::new(
        "claude-3-5-sonnet-20241022",
        vec![
            Message::text(Role::System, "Be terse."),
            Message::text(Role::User, "Hi"),
        ],
    );

    let resp = provider.chat(&req, &creds, "idem-a").await.unwrap();

    assert_eq!(resp.text(), "Hi from Claude");
    assert_eq!(resp.finish_reason, FinishReason::Stop);
    assert_eq!(resp.usage.input_tokens, 800);
    assert_eq!(resp.usage.output_tokens, 500);
    assert_eq!(resp.usage.cache_read_tokens, 200);
    assert_eq!(resp.usage.cache_write_tokens, 50);
    assert_eq!(resp.provider_response_id.as_deref(), Some("msg_01ABC"));
}

#[tokio::test]
async fn anthropic_stream_accumulates_usage_and_text() {
    use futures::StreamExt;
    let server = MockServer::start().await;
    let sse = std::fs::read_to_string("tests/fixtures/anthropic_stream.sse").unwrap();

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("idempotency-key", "idem-as"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;

    let provider = Anthropic::new();
    let creds = Credentials::new("sk-ant").with_base_url(server.uri());
    let mut req = ChatRequest::new(
        "claude-3-5-sonnet-20241022",
        vec![Message::text(Role::User, "Hi")],
    );
    req.stream = true;

    let mut stream = provider.stream(&req, &creds, "idem-as").await.unwrap();

    let mut text = String::new();
    let mut finish = None;
    let mut usage = None;
    while let Some(item) = stream.next().await {
        let d = item.unwrap();
        if let Some(c) = d.content_delta {
            text.push_str(&c);
        }
        if let Some(f) = d.finish_reason {
            finish = Some(f);
        }
        if let Some(u) = d.usage {
            usage = Some(u);
        }
    }

    assert_eq!(text, "Hello");
    assert_eq!(finish, Some(FinishReason::Stop));
    let u = usage.expect("usage must be emitted on the terminal delta");
    assert_eq!(u.input_tokens, 800);
    assert_eq!(u.cache_read_tokens, 200);
    assert_eq!(u.output_tokens, 2);
}

#[tokio::test]
async fn anthropic_unknown_cache_usage_field_fails_closed() {
    let server = MockServer::start().await;
    let body = r#"{
      "id": "msg_drift",
      "type": "message",
      "model": "claude-3-5-sonnet-20241022",
      "stop_reason": "end_turn",
      "content": [
        { "type": "text", "text": "Hi" }
      ],
      "usage": {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_input_tokens": 10,
        "cache_v2_input_tokens": 30
      }
    }"#;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&server)
        .await;

    let provider = Anthropic::new();
    let creds = Credentials::new("sk-ant").with_base_url(server.uri());
    let req = ChatRequest::new(
        "claude-3-5-sonnet-20241022",
        vec![Message::text(Role::User, "Hi")],
    );

    let err = provider
        .chat(&req, &creds, "idem-ant-drift")
        .await
        .unwrap_err();
    match err {
        ProviderError::SchemaDrift { provider, reason } => {
            assert_eq!(provider, "anthropic");
            assert!(reason.contains("cache_v2_input_tokens"));
        }
        other => panic!("expected SchemaDrift, got {other:?}"),
    }
}

#[tokio::test]
async fn anthropic_unknown_non_usage_field_remains_permissive() {
    let server = MockServer::start().await;
    let body = r#"{
      "id": "msg_extra",
      "type": "message",
      "model": "claude-3-5-sonnet-20241022",
      "stop_reason": "end_turn",
      "content": [
        { "type": "text", "text": "Hi" }
      ],
      "usage": {
        "input_tokens": 100,
        "output_tokens": 20,
        "billing_mode": "standard"
      }
    }"#;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&server)
        .await;

    let provider = Anthropic::new();
    let creds = Credentials::new("sk-ant").with_base_url(server.uri());
    let req = ChatRequest::new(
        "claude-3-5-sonnet-20241022",
        vec![Message::text(Role::User, "Hi")],
    );

    let resp = provider.chat(&req, &creds, "idem-ant-extra").await.unwrap();
    assert_eq!(resp.usage.input_tokens, 100);
    assert_eq!(resp.usage.output_tokens, 20);
}
