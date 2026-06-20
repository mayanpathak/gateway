//! Load pipeline: read → interpolate `${ENV}` → parse → schema-validate →
//! semantic-validate. `validate` runs the whole pipeline WITHOUT applying (the
//! `--dry-run` path). Referential checks the schema can't express live in
//! `validate_semantics` (a route's provider must exist; key ids unique).

use std::collections::HashSet;

use serde_json::Value;

use crate::error::ConfigError;
use crate::interpolate::interpolate;
use crate::model::{Config, GuardrailType};
use crate::schema::validate_structure;

/// Parse + validate a config string (already env-interpolated). Returns the typed
/// `Config` on success — this is the `validate` / `--dry-run` entry point.
pub fn validate(raw_json: &str) -> Result<Config, ConfigError> {
    let value: Value = serde_json::from_str(raw_json).map_err(|e| ConfigError::Parse {
        detail: e.to_string(),
    })?;

    validate_structure(&value)?;

    let config: Config = serde_json::from_value(value).map_err(|e| ConfigError::Parse {
        detail: e.to_string(),
    })?;

    validate_semantics(&config)?;
    Ok(config)
}

/// Full load: interpolate `${ENV}` first, then `validate`.
pub fn load(
    raw_with_env_refs: &str,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<Config, ConfigError> {
    let interpolated = interpolate(raw_with_env_refs, lookup)?;
    validate(&interpolated)
}

/// Cross-row referential integrity the JSON Schema can't express.
pub fn validate_semantics(config: &Config) -> Result<(), ConfigError> {
    // Unique provider ids.
    let mut provider_ids = HashSet::new();
    for p in &config.providers {
        if !provider_ids.insert(&p.id) {
            return Err(ConfigError::Validation {
                detail: format!("duplicate provider id: {}", p.id),
            });
        }
    }

    // Unique key ids.
    let mut key_ids = HashSet::new();
    for k in &config.keys {
        if !key_ids.insert(&k.id) {
            return Err(ConfigError::Validation {
                detail: format!("duplicate key id: {}", k.id),
            });
        }
    }

    // Every route references a declared provider.
    for r in &config.routes {
        if !provider_ids.contains(&r.provider) {
            return Err(ConfigError::Validation {
                detail: format!("route {} references unknown provider {}", r.id, r.provider),
            });
        }
    }

    // Config version check. `Config.version` is an i64, not an Option.
    if config.version > 1 {
        return Err(ConfigError::Validation {
            detail: format!(
                "unsupported config version {}; this gateway supports up to version 1",
                config.version
            ),
        });
    }

    // Guardrail policy and rule validation.
    let mut guardrail_ids = HashSet::new();

    for g in &config.guardrails {
        if !guardrail_ids.insert(&g.id) {
            return Err(ConfigError::Validation {
                detail: format!("duplicate guardrail id: {}", g.id),
            });
        }

        for target in &g.apply_to {
            if target != "*" && !key_ids.contains(target) {
                return Err(ConfigError::Validation {
                    detail: format!(
                        "guardrail {} apply_to references unknown key {}",
                        g.id, target
                    ),
                });
            }
        }

        for rule in &g.rules {
            match rule.guardrail_type {
                GuardrailType::Secrets | GuardrailType::Pii => {}

                GuardrailType::Keyword => {
                    if rule.keywords.is_empty() {
                        return Err(ConfigError::Validation {
                            detail: format!(
                                "guardrail '{}': keyword rule requires at least one keyword",
                                g.id
                            ),
                        });
                    }
                }

                GuardrailType::RegexDeny => {
                    let pat = rule
                        .pattern
                        .as_deref()
                        .ok_or_else(|| ConfigError::Validation {
                            detail: format!(
                                "guardrail '{}': regex_deny rule missing 'pattern'",
                                g.id
                            ),
                        })?;

                    const MAX_PATTERN_BYTES: usize = 4_096;

                    if pat.len() > MAX_PATTERN_BYTES {
                        return Err(ConfigError::Validation {
                            detail: format!(
                                "guardrail '{}': regex_deny pattern exceeds {} byte limit ({} bytes)",
                                g.id,
                                MAX_PATTERN_BYTES,
                                pat.len()
                            ),
                        });
                    }

                    regex::Regex::new(pat).map_err(|e| ConfigError::Validation {
                        detail: format!("guardrail '{}': invalid regex '{}': {}", g.id, pat, e),
                    })?;
                }

                GuardrailType::JsonSchema => {
                    let schema = rule
                        .schema
                        .as_ref()
                        .ok_or_else(|| ConfigError::Validation {
                            detail: format!(
                                "guardrail '{}': json_schema rule missing 'schema'",
                                g.id
                            ),
                        })?;

                    jsonschema::validator_for(schema).map_err(|e| ConfigError::Validation {
                        detail: format!("guardrail '{}': invalid json schema: {}", g.id, e),
                    })?;
                }

                GuardrailType::Webhook => {
                    if rule.url.as_deref().unwrap_or_default().trim().is_empty() {
                        return Err(ConfigError::Validation {
                            detail: format!("guardrail '{}': webhook rule missing 'url'", g.id),
                        });
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::interpolate::map_lookup;

    #[test]
    fn validate_accepts_a_good_config() {
        let raw = r#"{
            "providers": [{ "id": "openai", "kind": "openai" }],
            "keys": [{ "id": "k1", "max_budget_usd": 10.0 }],
            "routes": [{ "id": "r1", "model": "gpt-4o", "provider": "openai" }]
        }"#;

        let c = validate(raw).unwrap();
        assert_eq!(c.routes.len(), 1);
    }

    #[test]
    fn route_to_unknown_provider_is_rejected() {
        let raw = r#"{
            "providers": [{ "id": "openai", "kind": "openai" }],
            "routes": [{ "id": "r1", "model": "gpt-4o", "provider": "ghost" }]
        }"#;

        assert!(matches!(validate(raw), Err(ConfigError::Validation { .. })));
    }

    #[test]
    fn duplicate_key_ids_are_rejected() {
        let raw = r#"{ "keys": [{ "id": "k1" }, { "id": "k1" }] }"#;

        assert!(matches!(validate(raw), Err(ConfigError::Validation { .. })));
    }

    #[test]
    fn load_interpolates_then_validates() {
        let m: HashMap<String, String> = [("OPENAI_API_KEY".to_string(), "sk-live".to_string())]
            .into_iter()
            .collect();

        let raw = r#"{ "providers": [{ "id": "openai", "kind": "openai", "api_key": "${OPENAI_API_KEY}" }] }"#;

        let c = load(raw, &map_lookup(&m)).unwrap();

        assert_eq!(c.providers[0].api_key.as_deref(), Some("sk-live"));
    }

    #[test]
    fn load_fails_closed_on_missing_env() {
        let m: HashMap<String, String> = HashMap::new();

        let raw =
            r#"{ "providers": [{ "id": "openai", "kind": "openai", "api_key": "${MISSING}" }] }"#;

        assert!(matches!(
            load(raw, &map_lookup(&m)),
            Err(ConfigError::Interpolation { .. })
        ));
    }

    #[test]
    fn guardrail_apply_to_unknown_key_is_rejected() {
        let raw = r#"{
            "keys": [{ "id": "k1" }],
            "guardrails": [{
                "id": "g1",
                "apply_to": ["ghost"],
                "rules": [{ "type": "secrets" }]
            }]
        }"#;

        assert!(matches!(validate(raw), Err(ConfigError::Validation { .. })));
    }

    #[test]
    fn guardrail_wildcard_apply_to_is_accepted() {
        let raw = r#"{
            "guardrails": [{
                "id": "g1",
                "apply_to": ["*"],
                "rules": [{ "type": "secrets" }]
            }]
        }"#;

        validate(raw).unwrap();
    }

    #[test]
    fn keyword_without_keywords_is_rejected() {
        let raw = r#"{
            "guardrails": [{
                "id": "g1",
                "rules": [{ "type": "keyword" }]
            }]
        }"#;

        assert!(matches!(validate(raw), Err(ConfigError::Validation { .. })));
    }

    #[test]
    fn regex_deny_without_pattern_is_rejected() {
        let raw = r#"{
            "guardrails": [{
                "id": "g1",
                "rules": [{ "type": "regex_deny" }]
            }]
        }"#;

        assert!(matches!(validate(raw), Err(ConfigError::Validation { .. })));
    }

    #[test]
    fn regex_deny_bad_pattern_is_rejected() {
        let raw = r#"{
            "guardrails": [{
                "id": "g1",
                "rules": [{ "type": "regex_deny", "pattern": "(unclosed" }]
            }]
        }"#;

        assert!(matches!(validate(raw), Err(ConfigError::Validation { .. })));
    }

    #[test]
    fn regex_deny_oversized_pattern_is_rejected() {
        let huge_pattern = "a".repeat(5_000);

        let raw = format!(
            r#"{{"guardrails":[{{"id":"g1","rules":[{{"type":"regex_deny","pattern":"{huge_pattern}"}}]}}]}}"#
        );

        assert!(matches!(
            validate(&raw),
            Err(ConfigError::Validation { .. })
        ));
    }

    #[test]
    fn json_schema_without_schema_is_rejected() {
        let raw = r#"{
            "guardrails": [{
                "id": "g1",
                "rules": [{ "type": "json_schema" }]
            }]
        }"#;

        assert!(matches!(validate(raw), Err(ConfigError::Validation { .. })));
    }

    #[test]
    fn webhook_without_url_is_rejected() {
        let raw = r#"{
            "guardrails": [{
                "id": "g1",
                "rules": [{ "type": "webhook" }]
            }]
        }"#;

        assert!(matches!(validate(raw), Err(ConfigError::Validation { .. })));
    }
}
