//! Build a [`GuardChain`] from deserialized config data.
//!
//! This module deliberately uses a small borrowed view type instead of depending
//! on `gateway-config`, keeping crate dependencies one-way and avoiding a cycle.

use std::sync::Arc;

use serde_json::Value;

use crate::WebhookGuardrail;
use crate::builtin::{
    JsonSchemaGuardrail, KeywordBanlistGuardrail, PiiGuardrail, RegexDenylistGuardrail,
    SecretsGuardrail,
};
use crate::chain::GuardChain;
use crate::types::{EnforcementMode, GuardError};

pub struct GuardrailRuleView<'a> {
    pub guardrail_type: &'a str,
    pub mode: &'a str,
    pub keywords: &'a [String],
    pub pattern: Option<&'a str>,
    pub label: Option<&'a str>,
    pub schema: Option<&'a Value>,
    pub url: Option<&'a str>,

    /// P2 stub: parsed and passed through, but not routed per stage yet.
    pub stages: &'a [String],
}

pub fn chain_from_rules(rules: &[GuardrailRuleView<'_>]) -> Result<GuardChain, GuardError> {
    let mut chain = GuardChain::new();

    for rule in rules {
        let mode = parse_mode(rule.mode);

        match rule.guardrail_type {
            "secrets" => {
                chain = chain.push(mode, Arc::new(SecretsGuardrail::new()));
            }
            "pii" => {
                chain = chain.push(mode, Arc::new(PiiGuardrail::new()));
            }
            "keyword" => {
                if rule.keywords.is_empty() {
                    return Err(GuardError::InvalidConfig(
                        "keyword guardrail requires at least one keyword".into(),
                    ));
                }

                chain = chain.push(
                    mode,
                    Arc::new(KeywordBanlistGuardrail::new(
                        rule.keywords.iter().map(String::as_str),
                    )),
                );
            }
            "regex_deny" => {
                let pat = rule.pattern.ok_or_else(|| {
                    GuardError::RegexCompile(
                        "regex_deny rule is missing the 'pattern' field".into(),
                    )
                })?;

                chain = chain.push(mode, Arc::new(RegexDenylistGuardrail::new([pat])?));
            }
            "json_schema" => {
                let schema = rule.schema.ok_or_else(|| {
                    GuardError::SchemaValidation(
                        "json_schema rule is missing the 'schema' field".into(),
                    )
                })?;

                let label = rule.label.unwrap_or("json-schema");

                chain = chain.push(
                    mode,
                    Arc::new(JsonSchemaGuardrail::new(label, schema.clone())?),
                );
            }
            "webhook" => {
                let url = rule.url.ok_or_else(|| {
                    GuardError::SchemaValidation("webhook rule is missing the 'url' field".into())
                })?;

                let label = rule.label.unwrap_or("webhook");

                chain = chain.push(mode, Arc::new(WebhookGuardrail::new(label, url)));
            }
            other => {
                return Err(GuardError::InvalidConfig(format!(
                    "unknown guardrail type: '{other}'"
                )));
            }
        }
    }

    Ok(chain)
}

pub fn default_chain() -> GuardChain {
    GuardChain::new()
        .push(EnforcementMode::Enforce, Arc::new(SecretsGuardrail::new()))
        .push(EnforcementMode::Enforce, Arc::new(PiiGuardrail::new()))
}

fn parse_mode(mode: &str) -> EnforcementMode {
    match mode {
        "observe_only" => EnforcementMode::ObserveOnly,
        "dry_run" => EnforcementMode::DryRun,
        _ => EnforcementMode::Enforce,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GuardContext, GuardStage, GuardVerdict};

    fn ctx(text: &str) -> GuardContext {
        GuardContext::new(GuardStage::PreRequest, text)
    }

    #[tokio::test]
    async fn empty_rules_returns_empty_chain_that_allows_everything() {
        let chain = chain_from_rules(&[]).unwrap();

        let result = chain
            .run(&ctx("sk-ant-api03-FAKEFAKEFAKEFAKEFAKE is my key"))
            .await;

        assert_eq!(result.final_verdict, GuardVerdict::Allow);
    }

    #[tokio::test]
    async fn default_chain_blocks_secrets() {
        let chain = default_chain();

        let result = chain.run(&ctx("sk-ant-api03-FAKEFAKEFAKEFAKEFAKE")).await;

        assert!(matches!(result.final_verdict, GuardVerdict::Block { .. }));
    }

    #[tokio::test]
    async fn secrets_enforce_blocks() {
        let rules = [GuardrailRuleView {
            guardrail_type: "secrets",
            mode: "enforce",
            keywords: &[],
            pattern: None,
            label: None,
            schema: None,
            url: None,
            stages: &[],
        }];

        let chain = chain_from_rules(&rules).unwrap();

        let result = chain.run(&ctx("sk-ant-api03-FAKEFAKEFAKEFAKEFAKE")).await;

        assert!(matches!(result.final_verdict, GuardVerdict::Block { .. }));
    }

    #[tokio::test]
    async fn secrets_observe_only_does_not_block_but_records() {
        let rules = [GuardrailRuleView {
            guardrail_type: "secrets",
            mode: "observe_only",
            keywords: &[],
            pattern: None,
            label: None,
            schema: None,
            url: None,
            stages: &[],
        }];

        let chain = chain_from_rules(&rules).unwrap();

        let result = chain.run(&ctx("sk-ant-api03-FAKEFAKEFAKEFAKEFAKE")).await;

        assert_eq!(result.final_verdict, GuardVerdict::Allow);

        assert!(matches!(
            result.per_guardrail[0].2,
            GuardVerdict::Block { .. }
        ));
    }

    #[tokio::test]
    async fn pii_enforce_masks_email() {
        let rules = [GuardrailRuleView {
            guardrail_type: "pii",
            mode: "enforce",
            keywords: &[],
            pattern: None,
            label: None,
            schema: None,
            url: None,
            stages: &[],
        }];

        let chain = chain_from_rules(&rules).unwrap();

        let result = chain.run(&ctx("email me at bob@example.com")).await;

        assert!(matches!(result.final_verdict, GuardVerdict::Mask { .. }));
    }

    #[tokio::test]
    async fn keyword_enforce_blocks_on_match() {
        let kws = vec!["forbidden".to_string()];

        let rules = [GuardrailRuleView {
            guardrail_type: "keyword",
            mode: "enforce",
            keywords: &kws,
            pattern: None,
            label: None,
            schema: None,
            url: None,
            stages: &[],
        }];

        let chain = chain_from_rules(&rules).unwrap();

        let result = chain.run(&ctx("this is forbidden text")).await;

        assert!(matches!(result.final_verdict, GuardVerdict::Block { .. }));
    }

    #[test]
    fn keyword_rule_with_no_keywords_returns_err() {
        let rules = [GuardrailRuleView {
            guardrail_type: "keyword",
            mode: "enforce",
            keywords: &[],
            pattern: None,
            label: None,
            schema: None,
            url: None,
            stages: &[],
        }];

        assert!(chain_from_rules(&rules).is_err());
    }

    #[tokio::test]
    async fn regex_deny_enforce_blocks_on_match() {
        let pat = r"\bpassword\b".to_string();

        let rules = [GuardrailRuleView {
            guardrail_type: "regex_deny",
            mode: "enforce",
            keywords: &[],
            pattern: Some(&pat),
            label: Some("password"),
            schema: None,
            url: None,
            stages: &[],
        }];

        let chain = chain_from_rules(&rules).unwrap();

        let result = chain.run(&ctx("reset my password please")).await;

        assert!(matches!(result.final_verdict, GuardVerdict::Block { .. }));
    }

    #[test]
    fn regex_deny_missing_pattern_returns_err() {
        let rules = [GuardrailRuleView {
            guardrail_type: "regex_deny",
            mode: "enforce",
            keywords: &[],
            pattern: None,
            label: None,
            schema: None,
            url: None,
            stages: &[],
        }];

        assert!(chain_from_rules(&rules).is_err());
    }

    #[test]
    fn unknown_type_returns_err() {
        let rules = [GuardrailRuleView {
            guardrail_type: "notreal",
            mode: "enforce",
            keywords: &[],
            pattern: None,
            label: None,
            schema: None,
            url: None,
            stages: &[],
        }];

        assert!(chain_from_rules(&rules).is_err());
    }
}
