//! The HTTP-facing error. Every lifecycle failure becomes a `GatewayError`,
//! which carries the HTTP status and an OpenAI-shaped JSON error body so SDK
//! clients parse it. `SpineError` and `ProviderError` map in here — this is the
//! single place the governance taxonomy meets HTTP status codes (design §6).

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use gateway_llm::ProviderError;
use gateway_spine::SpineError;

#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    #[error("missing or malformed Authorization header")]
    MissingAuth,
    #[error("invalid api key")]
    InvalidKey,
    #[error("{0}")]
    Spine(#[from] SpineError),
    #[error("{0}")]
    Provider(#[from] ProviderError),
    #[error("invalid request: {0}")]
    BadRequest(String),
    #[error("feature not supported: {0}")]
    Unsupported(String),
    /// A guardrail blocked the request or response content (design §2/§P4).
    /// Mapped to 403 Forbidden so the caller sees a policy denial, never a
    /// provider/upstream error.
    #[error("blocked by guardrail: {0}")]
    GuardBlocked(String),
}

impl GatewayError {
    /// The HTTP status this error maps to (design §6).
    pub fn status(&self) -> StatusCode {
        match self {
            GatewayError::MissingAuth | GatewayError::InvalidKey => StatusCode::UNAUTHORIZED,
            GatewayError::Spine(e) => match e {
                SpineError::BudgetExceeded { .. } | SpineError::RateLimited { .. } => {
                    StatusCode::TOO_MANY_REQUESTS
                }
                SpineError::KeyRevoked { .. } | SpineError::KeyExpired { .. } => {
                    StatusCode::UNAUTHORIZED
                }
                SpineError::ModelNotAllowed { .. } => StatusCode::FORBIDDEN,
                SpineError::UnknownModel { .. } => StatusCode::BAD_REQUEST,
                SpineError::NoSuchKey { .. } => StatusCode::UNAUTHORIZED,
                SpineError::NoSuchReservation => StatusCode::INTERNAL_SERVER_ERROR,
            },
            GatewayError::Provider(e) => match e {
                ProviderError::Auth => StatusCode::BAD_GATEWAY,
                ProviderError::RateLimited { .. } => StatusCode::TOO_MANY_REQUESTS,
                // A 4xx from upstream is the client's fault (bad model, bad request) —
                // pass it through so callers see 404/400, not a misleading 502.
                ProviderError::Upstream { status, .. } if (400..500).contains(status) => {
                    StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY)
                }
                ProviderError::Upstream { .. } => StatusCode::BAD_GATEWAY,
                ProviderError::Unsupported { .. } => StatusCode::NOT_IMPLEMENTED,
                ProviderError::SchemaDrift { .. }
                | ProviderError::Transport(_)
                | ProviderError::Decode(_) => StatusCode::BAD_GATEWAY,
            },
            GatewayError::BadRequest(_) => StatusCode::BAD_REQUEST,
            GatewayError::Unsupported(_) => StatusCode::NOT_IMPLEMENTED,
            GatewayError::GuardBlocked(_) => StatusCode::FORBIDDEN,
        }
    }

    /// OpenAI-shaped error "type" string clients switch on.
    pub fn error_type(&self) -> &'static str {
        match self {
            GatewayError::MissingAuth | GatewayError::InvalidKey => "authentication_error",
            GatewayError::Spine(SpineError::BudgetExceeded { .. }) => "insufficient_quota",
            GatewayError::Spine(SpineError::RateLimited { .. }) => "rate_limit_error",
            GatewayError::Spine(SpineError::KeyRevoked { .. } | SpineError::KeyExpired { .. }) => {
                "authentication_error"
            }
            GatewayError::Spine(SpineError::ModelNotAllowed { .. })
            | GatewayError::GuardBlocked(_) => "permission_error",
            GatewayError::Spine(SpineError::UnknownModel { .. }) | GatewayError::BadRequest(_) => {
                "invalid_request_error"
            }
            GatewayError::Provider(_) => "upstream_error",
            _ => "api_error",
        }
    }
}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({
            "error": {
                "message": self.to_string(),
                "type": self.error_type(),
            }
        });
        (self.status(), Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gateway_spine::RateDimension;

    #[test]
    fn budget_exceeded_is_429_insufficient_quota() {
        let e = GatewayError::Spine(SpineError::budget_exceeded(
            "k",
            gateway_spine::Usd::from_micros(2),
            gateway_spine::Usd::from_micros(1),
        ));
        assert_eq!(e.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(e.error_type(), "insufficient_quota");
    }

    #[test]
    fn rate_limited_is_429() {
        let e = GatewayError::Spine(SpineError::RateLimited {
            key_id: "k".into(),
            dimension: RateDimension::Requests,
        });
        assert_eq!(e.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[test]
    fn revoked_and_missing_auth_are_401() {
        assert_eq!(GatewayError::MissingAuth.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(GatewayError::InvalidKey.status(), StatusCode::UNAUTHORIZED);
        let e = GatewayError::Spine(SpineError::KeyRevoked { key_id: "k".into() });
        assert_eq!(e.status(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn model_not_allowed_is_403_unknown_is_400() {
        let na = GatewayError::Spine(SpineError::ModelNotAllowed {
            key_id: "k".into(),
            model: "x".into(),
        });
        assert_eq!(na.status(), StatusCode::FORBIDDEN);
        let unk = GatewayError::Spine(SpineError::UnknownModel { model: "x".into() });
        assert_eq!(unk.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn provider_unsupported_is_501() {
        let e = GatewayError::Provider(ProviderError::Unsupported {
            feature: "audio".into(),
        });
        assert_eq!(e.status(), StatusCode::NOT_IMPLEMENTED);
    }
}
