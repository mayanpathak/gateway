//! PII detection and redaction guardrail.
//!
//! Uses regex-based detection to find and redact common PII patterns:
//! - Email addresses
//! - Phone numbers (US/international)
//! - Credit-card-shaped 16-digit groups
//! - SSN-shaped strings (`XXX-XX-XXXX`)
//! - IBAN-shaped strings (2-letter country code + 2 check digits + BBAN,
//!   per ISO 13616; shape-matched only - mod-97 checksum not verified)
//! - Common API-key shapes (`Bearer <token>`)

use async_trait::async_trait;
use regex::Regex;

use crate::guardrail::Guardrail;
use crate::types::{GuardContext, GuardVerdict};

/// Regex patterns used for PII detection.
struct PiiPatterns {
    email: Regex,
    phone: Regex,
    credit_card: Regex,
    ssn: Regex,
    api_key: Regex,
    iban: Regex,
}

impl PiiPatterns {
    fn build() -> Self {
        Self {
            // Standard email pattern.
            email: Regex::new(r"(?i)[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}").unwrap(),
            // US/international phone: +1-234-567-8901, (234) 567-8901, 234-567-8901, etc.
            phone: Regex::new(r"(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}").unwrap(),
            // 16-digit groups (credit card): NNNN NNNN NNNN NNNN or NNNN-NNNN-NNNN-NNNN.
            credit_card: Regex::new(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b").unwrap(),
            // SSN: XXX-XX-XXXX
            ssn: Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap(),
            // Generic API key (Bearer token or long alphanum string ≥ 20 chars).
            api_key: Regex::new(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*").unwrap(),
            // IBAN: 2-letter ISO 3166-1 alpha-2 country code, 2 decimal check
            // digits, then 11–30 alphanumeric BBAN characters (total 15–34
            // characters, per ISO 13616).  Matched in canonical upper-case form
            // without spaces.  The mod-97 checksum is intentionally not
            // verified — this is a shape match for redaction, consistent with
            // how every other pattern in this file works.
            iban: Regex::new(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b").unwrap(),
        }
    }
}

/// Guardrail that detects and redacts PII from the context text.
///
/// When any PII pattern is found the guardrail returns [`GuardVerdict::Mask`]
/// with all matches replaced by their respective placeholder tokens. If no PII
/// is detected it returns [`GuardVerdict::Allow`].
#[derive(Debug)]
pub struct PiiGuardrail {
    patterns: std::sync::Arc<PiiPatterns>,
}

impl std::fmt::Debug for PiiPatterns {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PiiPatterns").finish_non_exhaustive()
    }
}

impl PiiGuardrail {
    /// Create a new `PiiGuardrail` with the built-in patterns.
    pub fn new() -> Self {
        Self {
            patterns: std::sync::Arc::new(PiiPatterns::build()),
        }
    }

    /// Apply all redactions and return the sanitised text.
    fn redact(&self, text: &str) -> String {
        let p = &self.patterns;
        // Order matters: apply more-specific patterns first so they don't
        // interfere with each other (credit-card before phone, since CC is 16
        // digits and phone is 10).
        let t = p.credit_card.replace_all(text, "[CREDIT_CARD]");
        let t = p.iban.replace_all(&t, "[IBAN]");
        let t = p.ssn.replace_all(&t, "[SSN]");
        let t = p.phone.replace_all(&t, "[PHONE]");
        let t = p.email.replace_all(&t, "[EMAIL]");
        let t = p.api_key.replace_all(&t, "[API_KEY]");
        t.into_owned()
    }

    /// Return `true` if any PII pattern matches `text`.
    fn has_pii(&self, text: &str) -> bool {
        let p = &self.patterns;
        p.email.is_match(text)
            || p.credit_card.is_match(text)
            || p.ssn.is_match(text)
            || p.phone.is_match(text)
            || p.api_key.is_match(text)
            || p.iban.is_match(text)
    }
}

impl Default for PiiGuardrail {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Guardrail for PiiGuardrail {
    fn name(&self) -> &str {
        "pii"
    }

    async fn check(&self, ctx: &GuardContext) -> GuardVerdict {
        if self.has_pii(&ctx.text) {
            GuardVerdict::Mask {
                redacted_text: self.redact(&ctx.text),
            }
        } else {
            GuardVerdict::Allow
        }
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
    async fn masks_email() {
        let g = PiiGuardrail::new();
        let verdict = g
            .check(&ctx("Contact us at alice@example.com for help"))
            .await;
        match verdict {
            GuardVerdict::Mask { redacted_text } => {
                assert!(
                    !redacted_text.contains("alice@example.com"),
                    "email should be redacted, got: {redacted_text}"
                );
                assert!(redacted_text.contains("[EMAIL]"));
            }
            other => panic!("expected Mask, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn masks_credit_card() {
        let g = PiiGuardrail::new();
        let verdict = g.check(&ctx("Card: 4111 1111 1111 1111 expire soon")).await;
        match verdict {
            GuardVerdict::Mask { redacted_text } => {
                assert!(
                    !redacted_text.contains("4111"),
                    "CC should be redacted, got: {redacted_text}"
                );
                assert!(redacted_text.contains("[CREDIT_CARD]"));
            }
            other => panic!("expected Mask, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn masks_ssn() {
        let g = PiiGuardrail::new();
        let verdict = g.check(&ctx("SSN: 123-45-6789")).await;
        match verdict {
            GuardVerdict::Mask { redacted_text } => {
                assert!(!redacted_text.contains("123-45-6789"));
                assert!(redacted_text.contains("[SSN]"));
            }
            other => panic!("expected Mask, got {other:?}"),
        }
    }

    // Canonical example from the IBAN Wikipedia article / ISO 13616 test vectors.
    #[tokio::test]
    async fn masks_iban() {
        let g = PiiGuardrail::new();
        let verdict = g.check(&ctx("IBAN: GB82WEST12345698765432")).await;
        match verdict {
            GuardVerdict::Mask { redacted_text } => {
                assert!(
                    !redacted_text.contains("GB82WEST12345698765432"),
                    "IBAN should be redacted, got: {redacted_text}"
                );
                assert!(redacted_text.contains("[IBAN]"));
            }
            other => panic!("expected Mask, got {other:?}"),
        }
    }

    // A German IBAN to verify the pattern isn't GB-specific.
    #[tokio::test]
    async fn masks_german_iban() {
        let g = PiiGuardrail::new();
        let verdict = g
            .check(&ctx("Please transfer to DE89370400440532013000"))
            .await;
        match verdict {
            GuardVerdict::Mask { redacted_text } => {
                assert!(!redacted_text.contains("DE89370400440532013000"));
                assert!(redacted_text.contains("[IBAN]"));
            }
            other => panic!("expected Mask, got {other:?}"),
        }
    }

    // Negative test: a short uppercase string below the IBAN minimum BBAN
    // length (11 chars) must not be flagged.
    #[tokio::test]
    async fn does_not_flag_short_uppercase_string() {
        let g = PiiGuardrail::new();
        let verdict = g.check(&ctx("Reference code: GB82ABCD")).await;
        assert_eq!(
            verdict,
            GuardVerdict::Allow,
            "short uppercase string below IBAN minimum length should not be flagged"
        );
    }

    #[tokio::test]
    async fn allows_clean_text() {
        let g = PiiGuardrail::new();
        let verdict = g.check(&ctx("The weather today is sunny and warm.")).await;
        assert_eq!(verdict, GuardVerdict::Allow);
    }
}
