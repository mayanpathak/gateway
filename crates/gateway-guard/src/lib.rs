//! # gateway-guard
//!
//! Guardrails framework for the Oximy Gateway.
//!
//! Provides a composable, async guardrail pipeline with:
//! - Built-in deterministic guardrails: PII redaction, secrets detection,
//!   keyword banning, regex denylist, and JSON schema validation.
//! - [`chain::GuardChain`]: ordered evaluation with `Enforce`, `ObserveOnly`,
//!   and `DryRun` enforcement modes.
//! - [`webhook::WebhookGuardrail`]: seam for external HTTP content-moderation
//!   endpoints (Lakera Guard, Azure Content Safety, custom).
//!
//! # Quick start
//!
//! ```rust
//! use std::sync::Arc;
//! use gateway_guard::{
//!     GuardChain, GuardContext, GuardStage, EnforcementMode,
//!     builtin::{PiiGuardrail, SecretsGuardrail},
//! };
//!
//! # #[tokio::main]
//! # async fn main() {
//! let chain = GuardChain::new()
//!     .push(EnforcementMode::Enforce, Arc::new(PiiGuardrail::new()))
//!     .push(EnforcementMode::Enforce, Arc::new(SecretsGuardrail::new()));
//!
//! let ctx = GuardContext::new(GuardStage::PreRequest, "Send to user@example.com");
//! let verdict = chain.run(&ctx).await;
//! println!("{:?}", verdict.final_verdict);
//! # }
//! ```
//!
//! Part of [Oximy Gateway](https://github.com/oximyhq/gateway).

#![forbid(unsafe_code)]
#![deny(clippy::collapsible_if)]

pub mod builder_from_config;
pub mod builtin;
pub mod chain;
pub mod guardrail;
pub mod types;
pub mod webhook;

// Re-export the most commonly used items at crate root.
pub use builder_from_config::{GuardrailRuleView, chain_from_rules, default_chain};
pub use chain::{ChainVerdict, GuardChain, GuardEntry};
pub use guardrail::Guardrail;
pub use types::{EnforcementMode, GuardContext, GuardError, GuardStage, GuardVerdict};
pub use webhook::WebhookGuardrail;

// Expose spine types so callers don't need to depend on gateway-spine directly
// for the common audit / error types used alongside guard operations.
pub use gateway_spine::{AuditEvent, AuditSink, SpineError};
