//! # gateway-config
//!
//! The single declarative projection of gateway state. UI = API = CLI = Git, all
//! through one diff/apply engine. Kills the yaml-vs-DB split brain.
//!
//! ## What this crate provides
//!
//! - **`Config` model** — schema-validated serde structs: providers, virtual keys,
//!   routes, guardrail attachments, registry overrides.
//! - **JSON Schema** — structural validation (required ids, non-negative budgets);
//!   `validate_semantics` handles cross-row referential checks.
//! - **Env-var interpolation** — `${NAME}` refs resolved at load; missing vars are
//!   a hard error (fail-closed).
//! - **`load`/`validate`/`--dry-run`** — the full pipeline without applying.
//! - **decK-style `diff`/`apply`/`dump`** — ordered typed change set; one engine
//!   for UI, API, CLI, and Git.
//! - **File-watch hot reload** — bad configs are rejected; the last good config
//!   keeps serving.
//! - **`MasterKey`** — XChaCha20-Poly1305 AEAD sealing for provider secrets.
//! - **`ConfigStore` trait** + **`SqliteConfigStore`** — durable config-plane
//!   persistence (separate from the full spine `Store` which is a later milestone).
//!
//! Part of [Oximy Gateway](https://github.com/oximyhq/gateway) — the unified,
//! Apache-2.0 LLM + MCP gateway. See `docs/2026-06-10-oximy-gateway-design.md`.

#![forbid(unsafe_code)]
#![deny(clippy::collapsible_if)]

pub mod apply;
pub mod crypto;
pub mod diff;
pub mod dump;
pub mod error;
pub mod interpolate;
pub mod load;
pub mod model;
pub mod schema;
pub mod store;
pub mod watch;

// ── Flat re-exports for ergonomic use ────────────────────────────────────────

pub use apply::apply;
pub use crypto::MasterKey;
pub use diff::{Change, Diff, diff};
pub use dump::dump;
pub use error::ConfigError;
pub use interpolate::{interpolate, map_lookup};
pub use load::{load, validate, validate_semantics};
pub use model::{
    Config, GuardrailConfig, GuardrailMode, GuardrailRuleConfig, GuardrailStage, GuardrailType,
    KeyConfig, ProviderConfig, RegistryOverride, RouteConfig,
};
pub use schema::{config_schema, validate_structure};
pub use store::{ConfigStore, MemConfigStore, SqliteConfigStore, StoredKey, StoredProvider};
pub use watch::{load_file, watch};
