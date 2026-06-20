//! # Oximy Gateway
//!
//! The unified, fastest, open-source LLM + MCP gateway. Single static binary,
//! embedded dashboard, agent-first control plane (CLI + admin-MCP + config-as-code).
//!
//! `oximy-gateway up` boots the gateway and opens the dashboard.
//!
//! See `docs/2026-06-10-oximy-gateway-design.md`.

#![forbid(unsafe_code)]

mod cli;
mod firstboot;
mod state_file;

use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;

use cli::{Cli, Command, KeysCommand, UpArgs};

/// Bundled model catalog in models.dev API snapshot format (5000+ models, 142 providers).
/// Embedded at compile time — no disk dependency for the base catalog.
const BUNDLED_MODELS_DEV: &str = include_str!("models-dev.json");

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Version => {
            println!("oximy-gateway {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Command::Up(args) => match run_up(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("oximy-gateway up failed: {e:#}");
                ExitCode::from(70)
            }
        },
        Command::Keys(args) => match run_keys(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("oximy-gateway keys failed: {e:#}");
                ExitCode::from(70)
            }
        },
    }
}

/// Boot the gateway: resolve state dir → first-boot seed → register providers →
/// build AppState → start HTTP server → open browser.
fn run_up(args: UpArgs) -> anyhow::Result<()> {
    tokio::runtime::Runtime::new()?.block_on(run_up_async(args))
}

async fn run_up_async(args: UpArgs) -> anyhow::Result<()> {
    use gateway_cache::build_registry_from_models_dev;
    use gateway_control::cache_handle::memory_cache_handle;
    // 8A: removed `use gateway_control::guard::default_chain;`
    use gateway_control::keystore::{MutableKeyStore, PersistHook};
    use gateway_control::providers::{Deployment, ProviderRegistry};
    use gateway_control::state::AppState;
    use gateway_llm::Credentials;
    use gateway_llm::transports::openai::OpenAi;
    use gateway_spine::{MemoryAudit, SystemClock, Usd, VirtualKey};

    // ── 1. Resolve + create the state directory ───────────────────────────────
    let data_dir = cli::resolve_data_dir(args.dir.as_deref())?;
    std::fs::create_dir_all(&data_dir)?;
    let state_path = cli::state_path(&data_dir);
    tracing::info!(dir = %data_dir.display(), "data directory");

    // ── 2. Load or initialize the key store from the JSON state file ──────────
    let sf = Arc::new(state_file::StateFile::load_or_create(&state_path)?);

    // First boot: seed admin key, persist, print once.
    let clock = SystemClock;
    if let Some(minted) = firstboot::ensure_admin_key(sf.as_ref(), &clock)? {
        sf.save(&state_path)?;
        print_minted_key(&minted);
    } else if args.print_key {
        eprintln!(
            "An admin key already exists for this data dir ({}).\n\
             The secret is never recoverable; rotate it via the dashboard\n\
             or `oximy-gateway keys` CLI if lost.",
            data_dir.display()
        );
    }

    // ── 2b. Config file (light) ───────────────────────────────────────────────
    let config_path = data_dir.join("oximy-gateway.json");
    let config_keys = sf.load_keys();
    let config = load_or_seed_config(&config_path, &config_keys)?;

    // ── 3. Build the mutable, live key store with a file-persistence hook ─────
    // The hook is called after every `insert` / `revoke` to write the state file.
    struct FileHook {
        sf: Arc<state_file::StateFile>,
        path: std::path::PathBuf,
    }
    impl PersistHook for FileHook {
        fn persist(&self, keys: &[VirtualKey]) -> anyhow::Result<()> {
            // Re-populate the state file from the live key list.
            for k in keys {
                crate::firstboot::KeyStore::insert_key(self.sf.as_ref(), k)?;
            }
            self.sf.save(&self.path)
        }
    }

    let hook = Arc::new(FileHook {
        sf: Arc::clone(&sf),
        path: state_path.clone(),
    });
    let ks = Arc::new(MutableKeyStore::with_hook(hook));
    // Seed from the state file without calling the persist hook.
    ks.seed(sf.load_keys());

    // ── 4. Register LLM providers from env vars ───────────────────────────────
    // Interior mutability: `insert` takes `&self`, so the registry need not be
    // `mut` and can be shared (the admin `POST /v1/admin/providers` route mutates
    // it at runtime through the same seam).
    let providers = ProviderRegistry::new();

    // OpenAI native
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY")
        && !api_key.is_empty()
    {
        let mut creds = Credentials::new(api_key);
        if let Ok(base) = std::env::var("OPENAI_BASE_URL")
            && !base.is_empty()
        {
            creds = creds.with_base_url(base);
        }
        providers.insert(
            "openai",
            Deployment {
                provider: Arc::new(OpenAi::new()),
                credentials: Arc::new(creds),
            },
        );
        tracing::info!("provider registered: openai");
    }

    // OpenRouter (OpenAI-compatible)
    if let Ok(api_key) = std::env::var("OPENROUTER_API_KEY")
        && !api_key.is_empty()
    {
        providers.insert(
            "openrouter",
            Deployment {
                provider: Arc::new(OpenAi::new()),
                credentials: Arc::new(
                    Credentials::new(api_key).with_base_url("https://openrouter.ai/api"),
                ),
            },
        );
        tracing::info!("provider registered: openrouter (OpenAI-compatible)");
    }

    // Anthropic native
    if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY")
        && !api_key.is_empty()
    {
        use gateway_llm::transports::anthropic::Anthropic;
        providers.insert(
            "anthropic",
            Deployment {
                provider: Arc::new(Anthropic::new()),
                credentials: Arc::new(Credentials::new(api_key)),
            },
        );
        tracing::info!("provider registered: anthropic");
    }

    // Gemini native
    if let Ok(api_key) = std::env::var("GEMINI_API_KEY")
        && !api_key.is_empty()
    {
        use gateway_llm::transports::gemini::Gemini;
        // The models.dev catalog keys Google models under the `google` provider id,
        // so register the native transport there; keep `gemini` as an alias for any
        // route/override that uses the friendlier name.
        let gemini = Arc::new(Gemini::new());
        let creds = Arc::new(Credentials::new(api_key));
        providers.insert(
            "google",
            Deployment {
                provider: gemini.clone(),
                credentials: creds.clone(),
            },
        );
        providers.insert(
            "gemini",
            Deployment {
                provider: gemini,
                credentials: creds,
            },
        );
        tracing::info!("provider registered: google (gemini, native)");
    }

    // ── 4b. OpenAI-compatible provider presets ────────────────────────────────
    register_compat_provider(
        &providers,
        "GROQ_API_KEY",
        "groq",
        "https://api.groq.com/openai",
    );
    register_compat_provider(
        &providers,
        "TOGETHER_API_KEY",
        "together",
        "https://api.together.xyz",
    );
    register_compat_provider(
        &providers,
        "FIREWORKS_API_KEY",
        "fireworks",
        "https://api.fireworks.ai/inference",
    );
    register_compat_provider(
        &providers,
        "DEEPSEEK_API_KEY",
        "deepseek",
        "https://api.deepseek.com",
    );
    register_compat_provider(&providers, "XAI_API_KEY", "xai", "https://api.x.ai");
    register_compat_provider(
        &providers,
        "MISTRAL_API_KEY",
        "mistral",
        "https://api.mistral.ai",
    );
    register_compat_provider(
        &providers,
        "PERPLEXITY_API_KEY",
        "perplexity",
        "https://api.perplexity.ai",
    );
    register_compat_provider(
        &providers,
        "CEREBRAS_API_KEY",
        "cerebras",
        "https://api.cerebras.ai",
    );

    // ── 4c. Re-register providers persisted by the admin API at runtime ───────
    // These were added via `POST /v1/admin/providers` in a previous run and are
    // stored in the JSON state file. Each is an OpenAI-compatible deployment.
    for sp in sf.load_providers() {
        providers.insert(
            sp.id.clone(),
            Deployment::openai_compat(sp.api_key.clone(), sp.base_url.clone()),
        );
        tracing::info!(provider_id = %sp.id, base_url = %sp.base_url, "provider re-registered from state file");
    }

    if providers.is_empty() {
        eprintln!(
            "  Warning: no provider API keys found. The server will start but\n\
             \x20          chat requests will fail until at least one key is set.\n\
             \x20          Supported env vars: OPENAI_API_KEY, ANTHROPIC_API_KEY,\n\
             \x20          GEMINI_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY,\n\
             \x20          TOGETHER_API_KEY, FIREWORKS_API_KEY, DEEPSEEK_API_KEY,\n\
             \x20          XAI_API_KEY, MISTRAL_API_KEY, PERPLEXITY_API_KEY, CEREBRAS_API_KEY."
        );
    }

    // ── 5. Open (or create) the durable SQLite store ─────────────────────────
    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| format!("sqlite:{}", data_dir.join("gateway.db").display()));
    let durable_store = Arc::new(
        gateway_store::Store::connect(&db_url)
            .await
            .map_err(|e| anyhow::anyhow!("failed to open gateway store at {db_url}: {e}"))?,
    );
    tracing::info!(db = %db_url, "gateway store opened");

    // One-time migration: import any keys already in the JSON state file into
    // the SQLite store so they survive the transition without re-keying.
    for key in sf.load_keys() {
        let sk = gateway_store::StoredKey {
            id: key.id.clone(),
            name: key.id.clone(),
            token_hash: key.token_hash.clone(),
            token_prefix: key.token_prefix.clone(),
            budget_micros: key.max_budget.map(|u| u.micros()),
            spent_micros: 0,
            rpm: key.limits.rpm,
            tpm: key.limits.tpm,
            max_parallel: key.limits.max_parallel,
            model_allowlist: key.model_allowlist.clone(),
            expires_at_ms: None,
            revoked: key.revoked,
            parent_id: key.parent_id.clone(),
            created_at_ms: 0,
        };
        // upsert is idempotent: safe to call on every startup
        if let Err(e) = durable_store.upsert_key(&sk).await {
            tracing::warn!(key_id = %key.id, err = %e, "failed to migrate key to store");
        }
    }

    // ── 5b. Spin up the telemetry writer ─────────────────────────────────────
    use gateway_telemetry::{
        DEFAULT_CHANNEL_CAPACITY, GatewayMetrics, MemorySpendStore, spawn as spawn_telemetry,
    };
    let metrics = Arc::new(GatewayMetrics::new());
    let spend_store = Arc::new(MemorySpendStore::new());
    let (telem_sink, _telem_writer) = spawn_telemetry(
        Arc::clone(&spend_store),
        Arc::clone(&metrics),
        DEFAULT_CHANNEL_CAPACITY,
    );

    // ── 6. Build the model registry from the bundled models.dev catalog ──────
    // User-override file: <data_dir>/models.json (flat-array format, merged over
    // the bundled catalog). Overrides win by id.
    let user_overrides_path = data_dir.join("models.json");
    let user_overrides_json: Option<String> = if user_overrides_path.exists() {
        match std::fs::read_to_string(&user_overrides_path) {
            Ok(s) => {
                tracing::info!(path = %user_overrides_path.display(), "loading user model overrides");
                Some(s)
            }
            Err(e) => {
                tracing::warn!(path = %user_overrides_path.display(), err = %e, "failed to read user model overrides; using bundled catalog only");
                None
            }
        }
    } else {
        None
    };

    let model_registry =
        build_registry_from_models_dev(BUNDLED_MODELS_DEV, user_overrides_json.as_deref())
            .map_err(|e| anyhow::anyhow!("failed to build model registry: {e}"))?;

    tracing::info!(
        models = model_registry.len(),
        "loaded models from models.dev catalog"
    );

    // ── 6a. Build AppState (with L1 cache pre-wired) ─────────────────────────
    // 8B: build guard chain from config before constructing AppState
    let mut state_inner = AppState::with_parts_and_telemetry(
        ks,
        Arc::new(SystemClock),
        providers,
        Arc::new(build_guard_chain_from_config(config.as_ref())?),
        Arc::new(MemoryAudit::new()),
        telem_sink,
        metrics,
        Arc::clone(&spend_store) as Arc<dyn gateway_telemetry::SpendStore>,
        Arc::clone(&durable_store),
    );

    {
        let mut reg = state_inner.registry.write().unwrap();
        for entry in model_registry.all_entries() {
            reg.insert(entry);
        }
    }

    // ── 6b. Wire the L1 in-memory cache into AppState ────────────────────────
    state_inner.cache = Some(memory_cache_handle(SystemClock, 3600));

    // ── 6b2. Wire the runtime-provider persistence hook ──────────────────────
    // When the admin `POST /v1/admin/providers` route adds a provider, this hook
    // writes it into the JSON state file so it survives a restart (re-registered
    // in §4c above on the next boot).
    struct ProviderFileHook {
        sf: Arc<state_file::StateFile>,
        path: std::path::PathBuf,
    }
    impl gateway_control::providers::ProviderPersist for ProviderFileHook {
        fn persist(
            &self,
            provider: &gateway_control::providers::RuntimeProvider,
        ) -> anyhow::Result<()> {
            self.sf.insert_provider(state_file::StoredProvider {
                id: provider.id.clone(),
                base_url: provider.base_url.clone(),
                api_key: provider.api_key.clone(),
            });
            self.sf.save(&self.path)
        }
        fn remove(&self, id: &str) -> anyhow::Result<()> {
            self.sf.remove_provider(id);
            self.sf.save(&self.path)
        }
    }
    state_inner.provider_persist = Some(Arc::new(ProviderFileHook {
        sf: Arc::clone(&sf),
        path: state_path.clone(),
    }));

    let state = Arc::new(state_inner);

    // Set budget for all keys.
    for key in sf.load_keys() {
        state.ledger.set_budget(&key.id, key.max_budget, Usd::ZERO);
    }

    // ── 6c. Route overrides from OXIMY_ROUTES (JSON: model → Route) ──────────
    if let Ok(raw) = std::env::var("OXIMY_ROUTES")
        && !raw.trim().is_empty()
    {
        match serde_json::from_str::<std::collections::HashMap<String, gateway_route::Route>>(&raw)
        {
            Ok(routes) => {
                for (model, route) in routes {
                    tracing::info!(model = %model, targets = route.targets.len(), "route override installed");
                    state.set_route(model, route);
                }
            }
            Err(e) => {
                eprintln!("  Warning: OXIMY_ROUTES is not valid JSON ({e}); ignoring.");
            }
        }
    }

    // ── 6d. Route overrides from config file ─────────────────────────────────
    if let Some(cfg) = &config {
        apply_config(cfg, &state);
    }

    // ── 6e. Upstream MCP servers from OXIMY_MCP_SERVERS ──────────────────────
    if let Ok(raw) = std::env::var("OXIMY_MCP_SERVERS")
        && !raw.trim().is_empty()
    {
        register_mcp_servers(&state, &raw).await;
    }

    // ── 6e2. Re-seed per-key MCP tool ACLs from persisted keys ───────────────
    // The federation ACL is in-memory; rebuild it from each key's persisted
    // tool_allowlist so a restart can't silently re-open a restricted key.
    for key in sf.load_keys() {
        if let Some(allow) = &key.tool_allowlist {
            let set: std::collections::HashSet<String> = allow.iter().cloned().collect();
            state.federation.acl_mut().await.set(&key.id, Some(set));
        }
    }

    // ── 6f. Spawn background reservation sweep ───────────────────────────────
    // Sweeps stale (crashed/orphaned) reservations every 60s so they don't
    // block future budget reservations indefinitely.
    {
        let sweep_store = Arc::clone(&durable_store);
        tokio::spawn(async move {
            let ttl_ms: i64 = 5 * 60 * 1_000; // 5 minutes
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as i64;
                match sweep_store.sweep_stale_reservations(ttl_ms, now_ms).await {
                    Ok(n) if n > 0 => tracing::info!(swept = n, "swept stale reservations"),
                    Ok(_) => {}
                    Err(e) => tracing::warn!(err = %e, "reservation sweep failed"),
                }
            }
        });
    }

    // ── 7. Build the app router: API + dashboard (mounted last) ───────────────
    let api = gateway_control::router(state);
    let app = api.merge(gateway_dash::dash_router());

    // ── 8. Bind + serve with graceful shutdown ────────────────────────────────
    let addr: std::net::SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    let host_display = display_host(&args.host);
    let base = format!("http://{host_display}:{}", bound.port());

    eprintln!(
        "\n  Oximy Gateway is running.\n\
         \n\
         \x20 Dashboard:  {base}/\n\
         \x20 API base:   {base}/v1\n\
         \x20 Health:     {base}/health\n\
         \x20 Models:     {base}/v1/models (auth required)\n"
    );
    tracing::info!(
        url = %format!("{base}/"),
        assets = gateway_dash::asset_count(),
        "gateway ready"
    );

    // Best-effort open the browser (ignore failure on headless servers).
    if !args.no_open && gateway_dash::index_present() {
        let _ = open::that(format!("{base}/"));
    }

    // Graceful shutdown on SIGINT / SIGTERM.
    let shutdown = async {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            let mut sigint = signal(SignalKind::interrupt()).expect("SIGINT handler");
            let mut sigterm = signal(SignalKind::terminate()).expect("SIGTERM handler");
            tokio::select! {
                _ = sigint.recv() => tracing::info!("SIGINT received — shutting down"),
                _ = sigterm.recv() => tracing::info!("SIGTERM received — shutting down"),
            }
        }
        #[cfg(not(unix))]
        {
            tokio::signal::ctrl_c().await.ok();
            tracing::info!("Ctrl-C received — shutting down");
        }
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;

    tracing::info!("gateway stopped");
    Ok(())
}

/// Register an OpenAI-compatible provider when the given env key is set.
fn register_compat_provider(
    providers: &gateway_control::providers::ProviderRegistry,
    env_key: &str,
    provider_id: &'static str,
    base_url: &'static str,
) {
    use gateway_control::providers::Deployment;

    if let Ok(api_key) = std::env::var(env_key)
        && !api_key.is_empty()
    {
        providers.insert(provider_id, Deployment::openai_compat(api_key, base_url));
        tracing::info!(
            provider_id,
            base_url,
            "provider registered (OpenAI-compatible)"
        );
    }
}

/// Light config file: load `oximy-gateway.json` from the data dir. On first boot
/// (no file exists) write a commented example. Config is additive — env still wins
/// for provider keys; config can add routes/model overrides/guardrails.
fn load_or_seed_config(
    config_path: &std::path::Path,
    keys: &[gateway_spine::VirtualKey],
) -> anyhow::Result<Option<FileConfig>> {
    if config_path.exists() {
        let text = std::fs::read_to_string(config_path)
            .map_err(|e| anyhow::anyhow!("reading config {}: {e}", config_path.display()))?;

        let raw_value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("parsing config {}: {e}", config_path.display()))?;

        validate_config_guardrails(config_path, &raw_value, keys)?;

        let cfg: FileConfig = serde_json::from_value(raw_value)
            .map_err(|e| anyhow::anyhow!("parsing config {}: {e}", config_path.display()))?;

        tracing::info!(path = %config_path.display(), "config file loaded");
        Ok(Some(cfg))
    } else {
        let example = r#"{
  "_comment": "Oximy Gateway config - edit and restart to apply. All fields are optional.",
  "routes": {},
  "model_overrides": [],
  "guardrails": [{
    "id": "global",
    "apply_to": ["*"],
    "rules": [
      { "type": "secrets", "mode": "enforce" },
      { "type": "pii", "mode": "enforce" }
    ]
  }]
}
"#;
        if let Err(e) = std::fs::write(config_path, example) {
            tracing::warn!(path = %config_path.display(), err = %e, "could not write example config");
        } else {
            tracing::info!(path = %config_path.display(), "wrote example config");
        }
        Ok(None)
    }
}

fn validate_config_guardrails(
    config_path: &std::path::Path,
    raw_config: &serde_json::Value,
    keys: &[gateway_spine::VirtualKey],
) -> anyhow::Result<()> {
    let guardrails = raw_config
        .get("guardrails")
        .cloned()
        .unwrap_or_else(|| serde_json::json!([]));

    let keys = keys
        .iter()
        .map(|key| serde_json::json!({ "id": key.id }))
        .collect::<Vec<_>>();

    let validation_doc = serde_json::json!({
        "version": 1,
        "keys": keys,
        "guardrails": guardrails,
    });

    let validation_raw = serde_json::to_string(&validation_doc)?;
    gateway_config::validate(&validation_raw).map_err(|e| {
        anyhow::anyhow!(
            "invalid guardrails config in {}: {e}",
            config_path.display()
        )
    })?;

    Ok(())
}

/// Minimal config file schema — only what we act on here. Other fields (providers,
/// keys) are ignored; they are managed by env vars / the `keys` CLI.
#[derive(serde::Deserialize, Default)]
struct FileConfig {
    #[serde(default)]
    routes: std::collections::HashMap<String, FileRoute>,
    #[serde(default)]
    model_overrides: Vec<serde_json::Value>,
    #[serde(default)]
    guardrails: Vec<gateway_config::GuardrailConfig>,
}

#[derive(serde::Deserialize)]
struct FileRoute {
    targets: Vec<FileRouteTarget>,
    #[serde(default = "default_strategy")]
    strategy: String,
}

fn default_strategy() -> String {
    "failover".into()
}

#[derive(serde::Deserialize)]
struct FileRouteTarget {
    provider_id: String,
    model: String,
}

fn build_guard_chain_from_config(
    config: Option<&FileConfig>,
) -> anyhow::Result<gateway_guard::GuardChain> {
    use gateway_guard::builder_from_config::{GuardrailRuleView, chain_from_rules, default_chain};

    let cfg = match config {
        Some(c) if !c.guardrails.is_empty() => c,
        _ => {
            tracing::debug!("no guardrails config present; using built-in default chain");
            return Ok(default_chain());
        }
    };

    let key_scoped_count = cfg
        .guardrails
        .iter()
        .filter(|g| !g.apply_to.is_empty() && !g.apply_to.iter().any(|t| t == "*"))
        .count();

    if key_scoped_count > 0 {
        tracing::warn!(
            count = key_scoped_count,
            "oximy-gateway.json contains key-scoped guardrail policies, but per-key \
             routing is not yet implemented; these policies will be skipped"
        );
    }

    let policy = match cfg
        .guardrails
        .iter()
        .find(|g| g.apply_to.is_empty() || g.apply_to.iter().any(|t| t == "*"))
    {
        Some(p) => p,
        None => {
            tracing::info!("guardrails config has no global policy; using built-in default chain");
            return Ok(default_chain());
        }
    };

    let staged_count = policy.rules.iter().filter(|r| !r.stages.is_empty()).count();
    if staged_count > 0 {
        tracing::warn!(
            policy_id = %policy.id,
            count = staged_count,
            "guardrail rules configure stages, but per-stage routing is not yet \
             implemented; configured rules will run wherever the installed chain runs"
        );
    }

    if policy.rules.is_empty() {
        tracing::info!(
            policy_id = %policy.id,
            "global guardrail policy has no rules; using built-in default chain"
        );
        return Ok(default_chain());
    }

    let stage_strings = policy
        .rules
        .iter()
        .map(|rule| {
            rule.stages
                .iter()
                .map(|stage| guardrail_stage_as_str(*stage).to_string())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let views = policy
        .rules
        .iter()
        .zip(stage_strings.iter())
        .map(|(rule, stages)| GuardrailRuleView {
            guardrail_type: guardrail_type_as_str(rule.guardrail_type),
            mode: guardrail_mode_as_str(rule.mode),
            keywords: &rule.keywords,
            pattern: rule.pattern.as_deref(),
            label: rule.label.as_deref(),
            schema: rule.schema.as_ref(),
            url: rule.url.as_deref(),
            stages,
        })
        .collect::<Vec<_>>();

    chain_from_rules(&views).map_err(|e| {
        anyhow::anyhow!(
            "failed to build guard chain from config policy '{}': {e}",
            policy.id
        )
    })
}

fn guardrail_type_as_str(t: gateway_config::GuardrailType) -> &'static str {
    match t {
        gateway_config::GuardrailType::Secrets => "secrets",
        gateway_config::GuardrailType::Pii => "pii",
        gateway_config::GuardrailType::Keyword => "keyword",
        gateway_config::GuardrailType::RegexDeny => "regex_deny",
        gateway_config::GuardrailType::JsonSchema => "json_schema",
        gateway_config::GuardrailType::Webhook => "webhook",
    }
}

fn guardrail_mode_as_str(mode: gateway_config::GuardrailMode) -> &'static str {
    match mode {
        gateway_config::GuardrailMode::Enforce => "enforce",
        gateway_config::GuardrailMode::ObserveOnly => "observe_only",
        gateway_config::GuardrailMode::DryRun => "dry_run",
    }
}

fn guardrail_stage_as_str(stage: gateway_config::GuardrailStage) -> &'static str {
    match stage {
        gateway_config::GuardrailStage::PreRequest => "pre_request",
        gateway_config::GuardrailStage::PostResponse => "post_response",
        gateway_config::GuardrailStage::PreToolCall => "pre_tool_call",
        gateway_config::GuardrailStage::PostToolResult => "post_tool_result",
    }
}

/// Apply config-file routes and model overrides to an already-built AppState.
fn apply_config<C: gateway_spine::Clock + 'static>(
    cfg: &FileConfig,
    state: &gateway_control::state::AppState<C>,
) {
    use gateway_cache::build_registry_from_models_dev;
    // 8E: guardrails are installed before AppState construction by
    // build_guard_chain_from_config(). Hot reload requires a future swappable
    // guard-chain holder and is intentionally out of scope for this PR.

    for (model_id, file_route) in &cfg.routes {
        // Skip comment keys (keys starting with "_")
        if model_id.starts_with('_') {
            continue;
        }
        let targets: Vec<gateway_route::RouteTarget> = file_route
            .targets
            .iter()
            .map(|t| gateway_route::RouteTarget::new(&t.provider_id, &t.model))
            .collect();
        if targets.is_empty() {
            continue;
        }
        let strategy = match file_route.strategy.as_str() {
            "weighted" => gateway_route::Strategy::Weighted,
            "latency_aware" | "latency-aware" => gateway_route::Strategy::LatencyAware,
            _ => gateway_route::Strategy::Failover,
        };
        let route = gateway_route::Route::new(targets, strategy);
        tracing::info!(model = %model_id, strategy = %file_route.strategy, "config route installed");
        state.set_route(model_id.clone(), route);
    }

    if !cfg.model_overrides.is_empty() {
        let overrides_json = match serde_json::to_string(&cfg.model_overrides) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(err = %e, "could not serialize config model_overrides; skipping");
                return;
            }
        };
        // Re-build from the bundled catalog + config overrides and merge into state registry.
        match build_registry_from_models_dev(BUNDLED_MODELS_DEV, Some(&overrides_json)) {
            Ok(reg) => {
                let mut state_reg = state.registry.write().unwrap();
                for entry in reg.all_entries() {
                    state_reg.insert(entry);
                }
                tracing::info!(
                    count = cfg.model_overrides.len(),
                    "config model overrides applied"
                );
            }
            Err(e) => {
                tracing::warn!(err = %e, "config model_overrides parse error; skipping");
            }
        }
    }
}

/// Parse `OXIMY_MCP_SERVERS` (JSON array) and register + refresh each upstream
/// MCP server on the federation.
async fn register_mcp_servers<C: gateway_spine::Clock + 'static>(
    state: &gateway_control::state::AppState<C>,
    raw: &str,
) {
    use gateway_mcp::{HttpTransport, McpServer, StdioTransport};

    #[derive(serde::Deserialize)]
    struct McpServerCfg {
        name: String,
        url: Option<String>,
        command: Option<String>,
        #[serde(default)]
        args: Vec<String>,
        /// Custom headers applied to every outbound request for url servers.
        #[serde(default)]
        headers: std::collections::HashMap<String, String>,
        /// Convenience: if set (and no explicit Authorization header given),
        /// adds `Authorization: Bearer <token>`.
        #[serde(default)]
        token: Option<String>,
    }

    let cfgs: Vec<McpServerCfg> = match serde_json::from_str(raw) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  Warning: OXIMY_MCP_SERVERS is not valid JSON ({e}); ignoring.");
            return;
        }
    };

    for cfg in cfgs {
        let server = if let Some(url) = cfg.url {
            // Collect custom headers, then inject a Bearer token if `token` is
            // set and no Authorization header was provided explicitly.
            let mut headers: Vec<(String, String)> = cfg
                .headers
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            let has_auth = headers
                .iter()
                .any(|(k, _)| k.eq_ignore_ascii_case("authorization"));
            if let Some(token) = cfg.token.as_ref()
                && !has_auth
            {
                headers.push(("Authorization".to_string(), format!("Bearer {token}")));
            }
            McpServer::new(
                cfg.name.clone(),
                Arc::new(HttpTransport::with_headers(cfg.name.clone(), url, headers)),
            )
        } else if let Some(command) = cfg.command {
            let mut cmd = tokio::process::Command::new(&command);
            cmd.args(&cfg.args);
            match StdioTransport::spawn(cfg.name.clone(), &mut cmd).await {
                Ok(t) => McpServer::new(cfg.name.clone(), Arc::new(t)),
                Err(e) => {
                    eprintln!("  Warning: MCP server '{}' failed to spawn: {e}", cfg.name);
                    continue;
                }
            }
        } else {
            eprintln!(
                "  Warning: MCP server '{}' has neither 'url' nor 'command'; skipping.",
                cfg.name
            );
            continue;
        };

        state.federation.register_server(server).await;
        match state.federation.refresh_server(&cfg.name).await {
            Ok(names) => {
                tracing::info!(server = %cfg.name, tools = names.len(), "MCP server registered");
            }
            Err(e) => {
                eprintln!(
                    "  Warning: MCP server '{}' tool refresh failed: {e}",
                    cfg.name
                );
            }
        }
    }
}

// ── `keys` subcommand implementation ─────────────────────────────────────────

fn run_keys(args: cli::KeysArgs) -> anyhow::Result<()> {
    tokio::runtime::Runtime::new()?.block_on(run_keys_async(args))
}

async fn run_keys_async(args: cli::KeysArgs) -> anyhow::Result<()> {
    use gateway_spine::{Clock, RateLimits, SystemClock, Usd, VirtualKey};

    let data_dir = cli::resolve_data_dir(args.dir.as_deref())?;
    std::fs::create_dir_all(&data_dir)?;
    let state_path = cli::state_path(&data_dir);
    let sf = state_file::StateFile::load_or_create(&state_path)?;

    match args.subcommand {
        KeysCommand::Create {
            name,
            budget_usd,
            models,
        } => {
            let clock = SystemClock;
            let ts = clock.now_ms();

            // Generate a secret.
            let secret = firstboot::generate_secret();
            let key_id = format!(
                "key_{}",
                name.as_deref()
                    .map(|n| n.replace(' ', "_"))
                    .unwrap_or_else(|| format!("user_{ts}"))
            );
            let token_prefix: String = secret.chars().take(12).collect();

            let max_budget = budget_usd.map(Usd::from_dollars_f64);
            let model_allowlist = models.map(|m| {
                m.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
            });

            let key = VirtualKey {
                id: key_id.clone(),
                token_hash: VirtualKey::hash_secret(&secret),
                token_prefix: token_prefix.clone(),
                max_budget,
                limits: RateLimits::default(),
                model_allowlist,
                tool_allowlist: None,
                expires_at: None,
                revoked: false,
                parent_id: None,
            };

            crate::firstboot::KeyStore::insert_key(&sf, &key)?;
            sf.save(&state_path)?;

            println!("\n  Key created successfully!");
            println!("  ID:     {key_id}");
            println!("  Prefix: {token_prefix}");
            if let Some(b) = budget_usd {
                println!("  Budget: ${b:.2}");
            } else {
                println!("  Budget: unlimited");
            }
            println!(
                "\n  Secret (shown ONCE — store it now):\n\n    {secret}\n\n  \
                 Use as: Authorization: Bearer {secret}\n"
            );
        }

        KeysCommand::List => {
            let keys = sf.load_keys();
            if keys.is_empty() {
                println!("No keys found in {}", data_dir.display());
                return Ok(());
            }
            println!(
                "\n  {:<35} {:<14} {:<12} {:<10} MODELS",
                "ID", "PREFIX", "BUDGET", "REVOKED"
            );
            println!("  {}", "-".repeat(85));
            for k in keys {
                let budget = match k.max_budget {
                    Some(b) => format!("${:.4}", b.as_dollars_f64()),
                    None => "unlimited".into(),
                };
                let models = match &k.model_allowlist {
                    Some(list) => list.join(","),
                    None => "all".into(),
                };
                println!(
                    "  {:<35} {:<14} {:<12} {:<10} {}",
                    k.id,
                    k.token_prefix,
                    budget,
                    if k.revoked { "yes" } else { "no" },
                    models
                );
            }
            println!();
        }

        KeysCommand::Revoke { id } => {
            sf.revoke_key(&id)?;
            sf.save(&state_path)?;
            println!("Key '{id}' revoked and persisted.");
        }
    }

    Ok(())
}

/// Print the freshly minted admin secret exactly once.
fn print_minted_key(minted: &firstboot::MintedKey) {
    eprintln!(
        "\n  \u{250c}\u{2500} First boot \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\n\
          \u{2502}  A default admin key was created. It is shown ONCE:\n\
          \u{2502}\n\
          \u{2502}     {secret}\n\
          \u{2502}\n\
          \u{2502}  Use it as your Bearer token for the API and dashboard.\n\
          \u{2502}  Store it now \u{2014} it cannot be recovered.\n\
          \u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\n",
        secret = minted.secret
    );
}

/// For display, show 127.0.0.1 even when bound to 0.0.0.0.
fn display_host(host: &str) -> &str {
    if host == "0.0.0.0" { "127.0.0.1" } else { host }
}
