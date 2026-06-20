//! The JSON Schema that validates a config BEFORE it touches running state — the
//! one gate for UI = API = CLI = Git. Structural rules live here (required ids,
//! types, non-negative budgets); cross-row referential checks (a route's
//! provider exists) live in `load::validate_semantics`.

use serde_json::{Value, json};

use crate::error::ConfigError;

/// The config JSON Schema (draft 2020-12).
pub fn config_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "version": { "type": "integer", "minimum": 1 },
            "providers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "kind"],
                    "properties": {
                        "id": { "type": "string", "minLength": 1 },
                        "kind": { "type": "string", "minLength": 1 },
                        "base_url": { "type": "string" },
                        "api_key": { "type": "string" }
                    }
                }
            },
            "keys": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {
                        "id": { "type": "string", "minLength": 1 },
                        "max_budget_usd": { "type": "number", "minimum": 0 },
                        "rpm": { "type": "integer", "minimum": 0 },
                        "tpm": { "type": "integer", "minimum": 0 },
                        "max_parallel": { "type": "integer", "minimum": 0 },
                        "model_allowlist": {
                            "type": "array",
                            "items": { "type": "string" }
                        }
                    }
                }
            },
            "routes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "model", "provider"],
                    "properties": {
                        "id": { "type": "string", "minLength": 1 },
                        "model": { "type": "string", "minLength": 1 },
                        "provider": { "type": "string", "minLength": 1 }
                    }
                }
            },
            "guardrails": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {
                        "id": { "type": "string", "minLength": 1 },
                        "apply_to": {
                            "type": "array",
                            "items": { "type": "string", "minLength": 1 }
                        },
                        "rules": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["type"],
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "secrets",
                                            "pii",
                                            "keyword",
                                            "regex_deny",
                                            "json_schema",
                                            "webhook"
                                        ]
                                    },
                                    "mode": {
                                        "type": "string",
                                        "enum": [
                                            "enforce",
                                            "observe_only",
                                            "dry_run"
                                        ]
                                    },
                                    "stages": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "enum": [
                                                "pre_request",
                                                "post_response",
                                                "pre_tool_call",
                                                "post_tool_result"
                                            ]
                                        }
                                    },
                                    "keywords": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "minLength": 1
                                        }
                                    },
                                    "pattern": { "type": "string" },
                                    "label": { "type": "string" },
                                    "schema": { "type": "object" },
                                    "url": { "type": "string", "minLength": 1 }
                                },
                                "additionalProperties": false
                            }
                        }
                    },
                    "additionalProperties": false
                }
            }
        }
    })
}

/// Validate a raw JSON value against the schema. Returns the first error message.
pub fn validate_structure(value: &Value) -> Result<(), ConfigError> {
    let schema = config_schema();
    let compiled = jsonschema::validator_for(&schema).map_err(|e| ConfigError::Validation {
        detail: format!("schema compile: {e}"),
    })?;

    if let Some(err) = compiled.iter_errors(value).next() {
        return Err(ConfigError::Validation {
            detail: err.to_string(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_config_passes() {
        let v = json!({
            "version": 1,
            "providers": [{ "id": "openai", "kind": "openai" }],
            "keys": [{ "id": "k1", "max_budget_usd": 10.0 }]
        });

        validate_structure(&v).unwrap();
    }

    #[test]
    fn provider_missing_id_fails() {
        let v = json!({
            "providers": [{ "kind": "openai" }]
        });

        assert!(matches!(
            validate_structure(&v),
            Err(ConfigError::Validation { .. })
        ));
    }

    #[test]
    fn negative_budget_fails() {
        let v = json!({
            "keys": [{ "id": "k1", "max_budget_usd": -5.0 }]
        });

        assert!(matches!(
            validate_structure(&v),
            Err(ConfigError::Validation { .. })
        ));
    }

    #[test]
    fn guardrails_schema_accepts_valid_policy() {
        let v = json!({
            "guardrails": [{
                "id": "global",
                "apply_to": ["*"],
                "rules": [{
                    "type": "regex_deny",
                    "mode": "enforce",
                    "stages": ["pre_request"],
                    "pattern": "\\bpassword\\b",
                    "label": "password"
                }]
            }]
        });

        validate_structure(&v).unwrap();
    }

    #[test]
    fn guardrails_schema_rejects_unknown_type() {
        let v = json!({
            "guardrails": [{
                "id": "global",
                "rules": [{
                    "type": "not_real"
                }]
            }]
        });

        assert!(matches!(
            validate_structure(&v),
            Err(ConfigError::Validation { .. })
        ));
    }
}
