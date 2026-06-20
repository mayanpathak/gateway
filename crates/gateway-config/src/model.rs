use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_version")]
    pub version: i64,
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
    #[serde(default)]
    pub keys: Vec<KeyConfig>,
    #[serde(default)]
    pub routes: Vec<RouteConfig>,
    #[serde(default)]
    pub guardrails: Vec<GuardrailConfig>,
    #[serde(default)]
    pub registry_overrides: Vec<RegistryOverride>,
}

fn default_version() -> i64 {
    1
}

impl Default for Config {
    fn default() -> Self {
        Self {
            version: 1,
            providers: Vec::new(),
            keys: Vec::new(),
            routes: Vec::new(),
            guardrails: Vec::new(),
            registry_overrides: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub id: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// `${OPENAI_API_KEY}`-style ref resolved at load; never the literal secret.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyConfig {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_budget_usd: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rpm: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tpm: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_parallel: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_allowlist: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteConfig {
    pub id: String,
    pub model: String,
    pub provider: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardrailConfig {
    pub id: String,

    /// P2 stub: parsed and validated, but not routed per key yet.
    ///
    /// Use `["*"]` or omit for a global policy. Key-specific entries are
    /// accepted so config files are forward-compatible, but the binary currently
    /// applies only the first global policy at startup and warns about skipped
    /// key-scoped policies.
    #[serde(default)]
    pub apply_to: Vec<String>,

    #[serde(default)]
    pub rules: Vec<GuardrailRuleConfig>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardrailRuleConfig {
    #[serde(rename = "type")]
    pub guardrail_type: GuardrailType,

    #[serde(default)]
    pub mode: GuardrailMode,

    /// P2 stub: parsed and validated, but not routed per stage yet.
    ///
    /// The current runtime installs one chain and runs it at both pre-request
    /// and post-response. If this is non-empty, startup logs a warning.
    #[serde(default)]
    pub stages: Vec<GuardrailStage>,

    #[serde(default)]
    pub keywords: Vec<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardrailType {
    Secrets,
    Pii,
    Keyword,
    RegexDeny,
    JsonSchema,
    Webhook,
}

// #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardrailMode {
    #[default]
    Enforce,
    ObserveOnly,
    DryRun,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardrailStage {
    PreRequest,
    PostResponse,
    PreToolCall,
    PostToolResult,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegistryOverride {
    pub model_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_per_mtok: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_per_mtok: Option<i64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_config_roundtrips_json() {
        let c = Config::default();
        let json = serde_json::to_string(&c).unwrap();
        let back: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn missing_sections_default_to_empty() {
        // A minimal file with only a provider parses; the rest default.
        let json =
            r#"{"providers":[{"id":"openai","kind":"openai","api_key":"${OPENAI_API_KEY}"}]}"#;
        let c: Config = serde_json::from_str(json).unwrap();
        assert_eq!(c.version, 1);
        assert_eq!(c.providers.len(), 1);
        assert!(c.keys.is_empty());
        assert_eq!(c.providers[0].api_key.as_deref(), Some("${OPENAI_API_KEY}"));
    }

    #[test]
    fn guardrail_policy_defaults_parse() {
        let json = r#"{
            "guardrails": [{
                "id": "global",
                "rules": [{ "type": "secrets" }]
            }]
        }"#;

        let c: Config = serde_json::from_str(json).unwrap();

        assert_eq!(c.guardrails.len(), 1);
        assert_eq!(c.guardrails[0].apply_to.len(), 0);
        assert_eq!(c.guardrails[0].rules[0].mode, GuardrailMode::Enforce);
    }

    #[test]
    fn guardrail_rule_payloads_parse() {
        let json = r#"{
            "guardrails": [{
                "id": "global",
                "apply_to": ["*"],
                "rules": [{
                    "type": "regex_deny",
                    "mode": "observe_only",
                    "stages": ["pre_request"],
                    "pattern": "\\bpassword\\b",
                    "label": "password"
                }]
            }]
        }"#;

        let c: Config = serde_json::from_str(json).unwrap();
        let rule = &c.guardrails[0].rules[0];

        assert_eq!(rule.guardrail_type, GuardrailType::RegexDeny);
        assert_eq!(rule.mode, GuardrailMode::ObserveOnly);
        assert_eq!(rule.stages, vec![GuardrailStage::PreRequest]);
        assert_eq!(rule.pattern.as_deref(), Some("\\bpassword\\b"));
    }
}
