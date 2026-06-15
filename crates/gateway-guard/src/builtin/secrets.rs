//! Secrets-detection guardrail.
//!
//! Blocks requests that contain recognisable secret or token patterns:
//! - `sk-...` — OpenAI API keys
//! - `AKIA...` — AWS access key IDs
//! - `ghp_...` — GitHub personal access tokens
//! - `xoxb-` / `xoxp-` — Slack bot / user tokens
//! - `glpat-` — GitLab personal access tokens
//! - `hf_...` — Hugging Face access tokens

use async_trait::async_trait;
use regex::Regex;

use crate::guardrail::Guardrail;
use crate::types::{GuardContext, GuardVerdict};

/// Pattern plus human-readable label for each secret type.
struct SecretPattern {
    label: &'static str,
    re: Regex,
}

/// Guardrail that blocks any text containing known provider secret formats.
#[derive(Debug)]
pub struct SecretsGuardrail {
    patterns: Vec<SecretPatternWrap>,
}

struct SecretPatternWrap(SecretPattern);

impl std::fmt::Debug for SecretPatternWrap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecretPattern")
            .field("label", &self.0.label)
            .finish_non_exhaustive()
    }
}

impl SecretsGuardrail {
    /// Create a new `SecretsGuardrail` pre-loaded with all built-in patterns.
    pub fn new() -> Self {
        let patterns = vec![
            SecretPatternWrap(SecretPattern {
                label: "OpenAI API key",
                // sk- followed by at least 20 alphanumeric/punctuation chars.
                re: Regex::new(r"sk-[A-Za-z0-9_\-]{20,}").unwrap(),
            }),
            SecretPatternWrap(SecretPattern {
                label: "AWS access key ID",
                // AKIA followed by 16 uppercase alphanumeric chars.
                re: Regex::new(r"AKIA[A-Z0-9]{16}").unwrap(),
            }),
            SecretPatternWrap(SecretPattern {
                label: "GitHub personal access token",
                // ghp_ prefix with 36+ chars.
                re: Regex::new(r"ghp_[A-Za-z0-9]{36,}").unwrap(),
            }),
            SecretPatternWrap(SecretPattern {
                label: "Slack bot token",
                re: Regex::new(r"xoxb-[0-9]+-[0-9]+-[A-Za-z0-9]+").unwrap(),
            }),
            SecretPatternWrap(SecretPattern {
                label: "Slack user token",
                re: Regex::new(r"xoxp-[0-9]+-[0-9]+-[0-9]+-[A-Za-z0-9]+").unwrap(),
            }),
            SecretPatternWrap(SecretPattern {
                label: "GitLab personal access token",
                re: Regex::new(r"glpat-[A-Za-z0-9_\-]{20,}").unwrap(),
            }),
            SecretPatternWrap(SecretPattern {
                label: "Hugging Face access token",
                // Matches common Hugging Face access token formats observed in the wild.
                re: Regex::new(r"hf_[A-Za-z0-9]{34,40}").unwrap(),
            }),
        ];
        Self { patterns }
    }
}

impl Default for SecretsGuardrail {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Guardrail for SecretsGuardrail {
    fn name(&self) -> &str {
        "secrets"
    }

    async fn check(&self, ctx: &GuardContext) -> GuardVerdict {
        for p in &self.patterns {
            if p.0.re.is_match(&ctx.text) {
                return GuardVerdict::Block {
                    reason: format!("detected {} in content", p.0.label),
                };
            }
        }
        GuardVerdict::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GuardStage;

    fn ctx(text: &str) -> GuardContext {
        GuardContext::new(GuardStage::PreRequest, text)
    }

    #[tokio::test]
    async fn blocks_openai_key() {
        let g = SecretsGuardrail::new();
        let verdict = g.check(&ctx("My key is sk-abc123XYZabcXYZabcXYZabc")).await;
        assert!(
            matches!(verdict, GuardVerdict::Block { .. }),
            "expected Block, got {verdict:?}"
        );
    }

    #[tokio::test]
    async fn blocks_aws_key() {
        let g = SecretsGuardrail::new();
        let verdict = g.check(&ctx("AWS key: AKIAIOSFODNN7EXAMPLE")).await;
        assert!(
            matches!(verdict, GuardVerdict::Block { .. }),
            "expected Block, got {verdict:?}"
        );
    }

    #[tokio::test]
    async fn blocks_github_pat() {
        let g = SecretsGuardrail::new();
        let token = format!("ghp_{}", "a".repeat(36));
        let verdict = g.check(&ctx(&format!("token={token}"))).await;
        assert!(
            matches!(verdict, GuardVerdict::Block { .. }),
            "expected Block for GitHub PAT, got {verdict:?}"
        );
    }

    #[tokio::test]
    async fn blocks_slack_bot_token() {
        let g = SecretsGuardrail::new();
        let verdict = g.check(&ctx("xoxb-12345-12345-abcXYZabc123")).await;
        assert!(
            matches!(verdict, GuardVerdict::Block { .. }),
            "expected Block for Slack token"
        );
    }

    #[tokio::test]
    async fn blocks_gitlab_pat() {
        let g = SecretsGuardrail::new();
        let token = format!("glpat-{}", "z".repeat(20));
        let verdict = g.check(&ctx(&token)).await;
        assert!(
            matches!(verdict, GuardVerdict::Block { .. }),
            "expected Block for GitLab PAT"
        );
    }

    #[tokio::test]
    async fn blocks_huggingface_token_34_chars() {
        let g = SecretsGuardrail::new();
        let token = format!("hf_{}", "a".repeat(34));
        let verdict = g.check(&ctx(&format!("HF_TOKEN={token}"))).await;
        assert!(
            matches!(verdict, GuardVerdict::Block { .. }),
            "expected Block for Hugging Face token (34 chars), got {verdict:?}"
        );
    }

    #[tokio::test]
    async fn blocks_huggingface_token_40_chars() {
        let g = SecretsGuardrail::new();
        let token = format!("hf_{}", "b".repeat(40));
        let verdict = g.check(&ctx(&format!("HF_TOKEN={token}"))).await;
        assert!(
            matches!(verdict, GuardVerdict::Block { .. }),
            "expected Block for Hugging Face token (40 chars), got {verdict:?}"
        );
    }

    #[tokio::test]
    async fn allows_huggingface_token_too_short() {
        let g = SecretsGuardrail::new();
        // 10 chars after hf_ — well below the minimum, should not be flagged.
        let verdict = g.check(&ctx("hf_abc1234567")).await;
        assert_eq!(
            verdict,
            GuardVerdict::Allow,
            "short hf_ string should not be flagged"
        );
    }

    #[tokio::test]
    async fn huggingface_token_has_correct_label() {
        let g = SecretsGuardrail::new();
        let token = format!("hf_{}", "a".repeat(34));
        let verdict = g.check(&ctx(&token)).await;

        match verdict {
            GuardVerdict::Block { reason } => {
                assert!(reason.contains("Hugging Face"), "got reason: {reason}");
            }
            other => panic!("expected Block, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn allows_clean_text() {
        let g = SecretsGuardrail::new();
        let verdict = g
            .check(&ctx("The quick brown fox jumped over the lazy dog."))
            .await;
        assert_eq!(verdict, GuardVerdict::Allow);
    }
}
