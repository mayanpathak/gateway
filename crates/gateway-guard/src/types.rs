//! Core types: stages, context, verdicts, enforcement modes, and errors.

use serde::{Deserialize, Serialize};

/// The pipeline stage at which a guardrail is evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuardStage {
    /// Before the request is forwarded to the LLM provider.
    PreRequest,
    /// After the response is received from the LLM provider.
    PostResponse,
    /// Before a tool/function call is dispatched.
    PreToolCall,
    /// After a tool/function call result is received.
    PostToolResult,
}

/// The context passed to every guardrail during evaluation.
#[derive(Debug, Clone)]
pub struct GuardContext {
    /// Which pipeline stage triggered this evaluation.
    pub stage: GuardStage,
    /// The text content to inspect (prompt, completion, tool input/output, etc.).
    pub text: String,
    /// The virtual key id that originated the request, if any.
    pub key_id: Option<String>,
    /// The model being used, if known.
    pub model: Option<String>,
    /// Arbitrary tags for routing/policy decisions.
    pub tags: Vec<String>,
}

impl GuardContext {
    /// Construct a minimal context for testing or quick checks.
    pub fn new(stage: GuardStage, text: impl Into<String>) -> Self {
        Self {
            stage,
            text: text.into(),
            key_id: None,
            model: None,
            tags: Vec::new(),
        }
    }
}

/// The decision returned by a [`crate::Guardrail`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum GuardVerdict {
    /// Request is clean — proceed normally.
    Allow,
    /// Request must be blocked; surface `reason` to the caller.
    Block { reason: String },
    /// Sensitive content was found; the guardrail provides a sanitised replacement
    /// in `redacted_text`. The chain applies the replacement for subsequent
    /// guardrails.
    Mask { redacted_text: String },
    /// Content is noteworthy but not severe enough to block; annotate and continue.
    Flag { reason: String },
}

/// How a [`crate::chain::GuardEntry`] responds when its guardrail fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnforcementMode {
    /// Verdict is enforced: `Block` short-circuits, `Mask` is applied.
    Enforce,
    /// Record what the guardrail *would* have done, but never block or mutate.
    ObserveOnly,
    /// Simulate the full chain — record hypothetical outcomes without acting.
    DryRun,
}

/// Errors produced during guardrail construction.
#[derive(Debug, thiserror::Error)]
pub enum GuardError {
    #[error("regex compilation failed: {0}")]
    RegexCompile(String),
    #[error("JSON schema validation error: {0}")]
    SchemaValidation(String),
    #[error("invalid guardrail config: {0}")]
    InvalidConfig(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_verdict_serde_allow() {
        let v = GuardVerdict::Allow;
        let json = serde_json::to_string(&v).unwrap();
        let back: GuardVerdict = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn guard_verdict_serde_block() {
        let v = GuardVerdict::Block {
            reason: "contains secret".into(),
        };
        let json = serde_json::to_string(&v).unwrap();
        let back: GuardVerdict = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn guard_verdict_serde_mask() {
        let v = GuardVerdict::Mask {
            redacted_text: "[REDACTED]".into(),
        };
        let json = serde_json::to_string(&v).unwrap();
        let back: GuardVerdict = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn guard_verdict_serde_flag() {
        let v = GuardVerdict::Flag {
            reason: "unusual pattern".into(),
        };
        let json = serde_json::to_string(&v).unwrap();
        let back: GuardVerdict = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn guard_context_new() {
        let ctx = GuardContext::new(GuardStage::PreRequest, "hello world");
        assert_eq!(ctx.stage, GuardStage::PreRequest);
        assert_eq!(ctx.text, "hello world");
        assert!(ctx.key_id.is_none());
        assert!(ctx.tags.is_empty());
    }
}
