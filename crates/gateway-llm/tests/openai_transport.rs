//! OpenAI Chat Completions egress, non-streaming, against a mocked upstream.
//! Asserts: (a) request mapping (model/messages/idempotency header), (b) response
//! mapping (content/finish_reason), (c) usage extraction normalized into
//! non-overlapping TokenUsage (cached split out of prompt_tokens).

use gateway_llm::message::{Message, Role};
use gateway_llm::provider::{Credentials, Provider, ProviderError};
use gateway_llm::req::ChatRequest;
use gateway_llm::resp::FinishReason;
use gateway_llm::transports::openai::OpenAi;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn openai_chat_maps_request_response_and_usage() {
    let server = MockServer::start().await;
    let body = std::fs::read_to_string("tests/fixtures/openai_chat.json").unwrap();

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer sk-test"))
        .and(header("idempotency-key", "idem-xyz"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAi::new();
    let creds = Credentials::new("sk-test").with_base_url(server.uri());
    let req = ChatRequest::new("gpt-4o", vec![Message::text(Role::User, "Hi")]);

    let resp = provider.chat(&req, &creds, "idem-xyz").await.unwrap();

    assert_eq!(resp.model, "gpt-4o");
    assert_eq!(resp.text(), "Hello there!");
    assert_eq!(resp.finish_reason, FinishReason::Stop);
    // prompt_tokens(1000) includes cached(200) → input=800, cache_read=200.
    assert_eq!(resp.usage.input_tokens, 800);
    assert_eq!(resp.usage.cache_read_tokens, 200);
    assert_eq!(resp.usage.output_tokens, 500);
    assert_eq!(
        resp.provider_response_id.as_deref(),
        Some("chatcmpl-abc123")
    );
}

#[tokio::test]
async fn openai_stream_yields_text_deltas_then_usage() {
    use futures::StreamExt;
    let server = MockServer::start().await;
    let sse = std::fs::read_to_string("tests/fixtures/openai_stream.sse").unwrap();

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("idempotency-key", "idem-s"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;

    let provider = OpenAi::new();
    let creds = Credentials::new("sk-test").with_base_url(server.uri());
    let mut req = ChatRequest::new("gpt-4o", vec![Message::text(Role::User, "Hi")]);
    req.stream = true;

    let mut stream = provider.stream(&req, &creds, "idem-s").await.unwrap();

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
    assert_eq!(usage.unwrap().output_tokens, 2);
}

#[tokio::test]
async fn openai_usage_total_without_breakdown_fails_closed_as_schema_drift() {
    let server = MockServer::start().await;
    let body = r#"{
      "id": "chatcmpl-drift",
      "model": "gpt-4o",
      "choices": [
        {
          "message": { "role": "assistant", "content": "Hello" },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "total_tokens": 1200,
        "input_token_count": 1000,
        "output_token_count": 200
      }
    }"#;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&server)
        .await;

    let provider = OpenAi::new();
    let creds = Credentials::new("sk-test").with_base_url(server.uri());
    let req = ChatRequest::new("gpt-4o", vec![Message::text(Role::User, "Hi")]);

    let err = provider.chat(&req, &creds, "idem-drift").await.unwrap_err();
    match err {
        ProviderError::SchemaDrift { provider, reason } => {
            assert_eq!(provider, "openai");
            assert!(reason.contains("total_tokens=1200"));
        }
        other => panic!("expected SchemaDrift, got {other:?}"),
    }
}

#[tokio::test]
async fn openai_unknown_cache_usage_field_fails_closed() {
    let server = MockServer::start().await;
    let body = r#"{
      "id": "chatcmpl-cache-drift",
      "model": "gpt-4o",
      "choices": [
        {
          "message": { "role": "assistant", "content": "Hello" },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "cache_details": { "cached_tokens": 800 }
      }
    }"#;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&server)
        .await;

    let provider = OpenAi::new();
    let creds = Credentials::new("sk-test").with_base_url(server.uri());
    let req = ChatRequest::new("gpt-4o", vec![Message::text(Role::User, "Hi")]);

    let err = provider
        .chat(&req, &creds, "idem-cache-drift")
        .await
        .unwrap_err();
    match err {
        ProviderError::SchemaDrift { provider, reason } => {
            assert_eq!(provider, "openai");
            assert!(reason.contains("cache_details"));
        }
        other => panic!("expected SchemaDrift, got {other:?}"),
    }
}

#[tokio::test]
async fn openai_unknown_non_usage_field_remains_permissive() {
    let server = MockServer::start().await;
    let body = r#"{
      "id": "chatcmpl-extra",
      "model": "gpt-4o",
      "choices": [
        {
          "message": { "role": "assistant", "content": "Hello" },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "service_tier": "default"
      }
    }"#;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&server)
        .await;

    let provider = OpenAi::new();
    let creds = Credentials::new("sk-test").with_base_url(server.uri());
    let req = ChatRequest::new("gpt-4o", vec![Message::text(Role::User, "Hi")]);

    let resp = provider.chat(&req, &creds, "idem-extra").await.unwrap();
    assert_eq!(resp.usage.input_tokens, 10);
    assert_eq!(resp.usage.output_tokens, 2);
}
