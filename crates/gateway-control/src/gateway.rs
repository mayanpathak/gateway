//! The per-request lifecycle, free of HTTP. `Gateway::run` executes the design
//! §6 order exactly: authenticate (caller passes the resolved key) → allowlist
//! → rate-limit acquire → budget reserve → guard pre-hook → provider egress
//! (with the idempotency key) → commit ACTUAL cost from provider usage →
//! release the parallel slot → return the response + computed `usage.cost`.
//!
//! Failure handling is fail-closed and leak-free: a denial before egress
//! releases nothing (nothing was acquired past the failing step); a denial or
//! error AFTER reserve releases the reservation and the parallel slot so a
//! failed request never strands budget or a concurrency slot. The idempotency
//! key is minted once here and is the SAME value the provider sees on any retry
//! (retries are P1.5; the seam is correct now).

use std::sync::Arc;

use futures::stream::{Stream, StreamExt};
use gateway_guard::{GuardContext, GuardStage, GuardVerdict};
use gateway_llm::{ChatRequest, ChatResponse, ContentPart, ProviderError, Role, StreamDelta};
use gateway_route::{Route, RouteError, Router};
use gateway_spine::{AuditEvent, Usd, VirtualKey};
use gateway_store::StoreError;

use crate::error::GatewayError;
use crate::route_exec::RegistryExecutor;
use crate::state::AppState;

/// A completed non-streaming call: the provider response plus the authoritative
/// cost committed to the ledger (`usage.cost`).
#[derive(Debug)]
pub struct Completed {
    pub response: ChatResponse,
    pub cost: Usd,
    pub idempotency_key: String,
    /// `provider_id/model` of the target that actually served the response
    /// (surfaced as the `x-served-by` header).
    pub served_by: String,
    /// Whether the router fell back to a non-primary target (surfaced as the
    /// `x-fallback-fired` header).
    pub fallback_fired: bool,
}

/// The output of a streaming run: the wrapped delta stream (commit-on-terminal)
/// plus the minted idempotency key for response headers. The stream yields the
/// SAME `StreamDelta`s the provider produced; the cost commit is a side effect
/// that fires on the terminal (usage-carrying) delta.
pub struct CompletedStream {
    pub stream: std::pin::Pin<Box<dyn Stream<Item = Result<StreamDelta, ProviderError>> + Send>>,
    pub idempotency_key: String,
}

/// A conservative per-request token estimate used for the pre-call budget
/// reservation and TPM check. True-up happens at commit from real usage.
fn estimate_tokens(req: &ChatRequest) -> i64 {
    // ~4 chars/token over all text parts, plus the max_tokens ceiling for output.
    let input_chars: usize = req
        .messages
        .iter()
        .flat_map(|m| m.content.iter())
        .filter_map(|p| match p {
            gateway_llm::ContentPart::Text { text } => Some(text.len()),
            _ => None,
        })
        .sum();
    let input_est = (input_chars / 4).max(1) as i64;
    let output_est = req.max_tokens.unwrap_or(1024);
    input_est + output_est
}

impl<C: gateway_spine::Clock> AppState<C> {
    /// Estimate the worst-case USD for a request, for the fail-closed reserve.
    /// Uses the model's price if known; an unknown model is a hard error here
    /// (cost-correctness: we never reserve against a guessed price).
    fn estimate_cost(&self, model: &str, est_tokens: i64) -> Result<Usd, GatewayError> {
        let reg = self.registry.read().unwrap();
        let entry = reg.get(model).ok_or_else(|| {
            GatewayError::Spine(gateway_spine::SpineError::UnknownModel {
                model: model.to_string(),
            })
        })?;
        // Treat the whole estimate as output tokens (the most expensive bucket).
        let usage = gateway_spine::TokenUsage {
            output_tokens: est_tokens,
            ..Default::default()
        };
        Ok(entry.price.cost(&usage))
    }

    /// The route used to dispatch `model`. Returns a configured multi-target
    /// route override if one is installed for this model id; otherwise the
    /// default single target = (`provider_id`, `model`) — behaviour identical to
    /// the pre-routing direct dispatch.
    fn route_for(&self, model: &str, provider_id: &str) -> Route {
        self.routes
            .read()
            .unwrap()
            .get(model)
            .cloned()
            .unwrap_or_else(|| Route::single(provider_id, model))
    }
}

/// Map a `gateway_route::RouteError` to the HTTP-facing `GatewayError`.
/// A terminal provider error keeps its original status mapping; exhaustion /
/// no-targets become a 502 (we tried and could not get an upstream response).
fn route_error_to_gateway(e: RouteError) -> GatewayError {
    match e {
        RouteError::TerminalError(pe) => GatewayError::Provider(pe),
        RouteError::NoTargets | RouteError::AllTargetsExhausted { .. } => {
            GatewayError::Provider(ProviderError::Upstream {
                status: 502,
                body: e.to_string(),
            })
        }
        RouteError::Internal(msg) => GatewayError::BadRequest(msg),
    }
}

/// Concatenate all user-authored prompt text in a request — the surface the
/// `PreRequest` guard inspects. System/assistant/tool turns are excluded so the
/// guard reasons about the caller's own content.
fn user_prompt_text(req: &ChatRequest) -> String {
    req.messages
        .iter()
        .filter(|m| m.role == Role::User)
        .map(|m| m.text_content())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Replace every user message's text content with `masked` (single combined
/// blob) while preserving non-text parts (images) and non-user turns. Used when
/// the `PreRequest` guard returns a `Mask` verdict.
fn apply_user_mask(req: &mut ChatRequest, masked: &str) {
    let mut applied = false;
    for m in req.messages.iter_mut() {
        if m.role != Role::User {
            continue;
        }
        // Keep any image parts; replace the text with the masked blob exactly once.
        let images: Vec<ContentPart> = m
            .content
            .iter()
            .filter(|p| matches!(p, ContentPart::Image { .. }))
            .cloned()
            .collect();
        m.content.clear();
        if !applied {
            m.content.push(ContentPart::text(masked.to_string()));
            applied = true;
        }
        m.content.extend(images);
    }
}

/// Replace a response's text content with `masked`, preserving any non-text
/// (image) parts and the tool calls. Used when the `PostResponse` guard masks.
fn mask_response_text(resp: &mut ChatResponse, masked: &str) {
    let images: Vec<ContentPart> = resp
        .content
        .iter()
        .filter(|p| matches!(p, ContentPart::Image { .. }))
        .cloned()
        .collect();
    resp.content.clear();
    resp.content.push(ContentPart::text(masked.to_string()));
    resp.content.extend(images);
}

/// Map a StoreError from a reserve call to a GatewayError, also releasing the
/// parallel slot. Returns `None` for NotFound (treated as unlimited key).
fn map_reserve_error(
    e: StoreError,
    key_id: &str,
    budget_from_key: Option<Usd>,
) -> Result<Option<String>, GatewayError> {
    match e {
        StoreError::NotFound => {
            // Key not in store: treated as unlimited (no durable budget tracking).
            Ok(None)
        }
        StoreError::BudgetExceeded {
            budget_micros,
            spent_micros,
            reserved_micros,
            requested_micros,
        } => {
            let _ = budget_from_key; // used for context if needed
            Err(GatewayError::Spine(
                gateway_spine::SpineError::BudgetExceeded {
                    key_id: key_id.to_string(),
                    would_spend_micros: spent_micros + reserved_micros + requested_micros,
                    budget_micros,
                },
            ))
        }
        e => Err(GatewayError::BadRequest(format!("store error: {e}"))),
    }
}

/// The lifecycle. Generic over the clock so tests inject `MockClock`.
pub struct Gateway;

impl Gateway {
    /// Run one non-streaming chat call end-to-end. `key` is the already
    /// authenticated `VirtualKey` (the handler resolves it via `auth`).
    pub async fn run<C: gateway_spine::Clock + 'static>(
        state: &AppState<C>,
        key: &VirtualKey,
        req: &ChatRequest,
    ) -> Result<Completed, GatewayError> {
        // Owned request: the PreRequest guard may rewrite the prompt (Mask), so
        // the value we hand to egress can differ from the caller's input.
        let mut req = req.clone();
        let req = &mut req;
        let model = req.model.clone();
        let model = model.as_str();

        // 1. model allowlist
        if !key.allows_model(model) {
            Self::audit(
                state,
                key,
                "request.denied",
                model,
                "denied",
                "model_not_allowed",
            );
            return Err(GatewayError::Spine(
                gateway_spine::SpineError::ModelNotAllowed {
                    key_id: key.id.clone(),
                    model: model.to_string(),
                },
            ));
        }

        // 2. resolve the egress provider for this model (unknown model → 400
        //    here, before any acquisition) and confirm it has a deployment. The
        //    Router resolves the per-target deployment at egress; this is the
        //    fast-fail validation for the request's own model.
        let provider_id = {
            let reg = state.registry.read().unwrap();
            reg.get(model).map(|e| e.provider.clone()).ok_or_else(|| {
                GatewayError::Spine(gateway_spine::SpineError::UnknownModel {
                    model: model.to_string(),
                })
            })?
        };
        if !state.providers.contains(&provider_id) {
            return Err(GatewayError::BadRequest(format!(
                "no egress configured for provider {provider_id}"
            )));
        }

        let est_tokens = estimate_tokens(req);

        // 3. rate-limit acquire (RPM/TPM/parallel). On failure nothing else has
        //    been acquired, so just propagate.
        state
            .limiter
            .acquire(&key.id, &key.limits, est_tokens)
            .map_err(|e| {
                Self::audit(
                    state,
                    key,
                    "request.denied",
                    model,
                    "denied",
                    "rate_limited",
                );
                GatewayError::Spine(e)
            })?;

        // 4. budget reserve (fail-closed, BEFORE egress) via the durable store.
        //    On failure, release the parallel slot we just acquired, then propagate.
        //    `None` reservation = key not in store → unlimited key, no tracking.
        let est_cost = match state.estimate_cost(model, est_tokens) {
            Ok(c) => c,
            Err(e) => {
                state.limiter.release_parallel(&key.id);
                return Err(e);
            }
        };
        let reservation: Option<String> = match state.store.reserve(&key.id, est_cost).await {
            Ok(res_id) => Some(res_id),
            Err(e) => {
                state.limiter.release_parallel(&key.id);
                match map_reserve_error(e, &key.id, key.max_budget) {
                    Ok(None) => {
                        // unlimited key — also update the in-memory ledger for tests
                        None
                    }
                    Ok(Some(_)) => unreachable!(),
                    Err(ge) => {
                        Self::audit(
                            state,
                            key,
                            "request.denied",
                            model,
                            "denied",
                            "budget_exceeded",
                        );
                        return Err(ge);
                    }
                }
            }
        };

        // Also update the in-memory ledger so tests that read state.ledger still work.
        let ledger_reservation = state.ledger.reserve(&key.id, est_cost).ok();

        // 5. PreRequest guard over the user prompt. A Block releases the
        //    reservation + parallel slot before any egress (403); a Mask rewrites
        //    the outgoing prompt in place. Runs BEFORE egress so a secret-laden
        //    prompt is never forwarded upstream.
        let pre_ctx = GuardContext {
            stage: GuardStage::PreRequest,
            text: user_prompt_text(req),
            key_id: Some(key.id.clone()),
            model: Some(model.to_string()),
            tags: vec![],
        };
        match state.guard.run(&pre_ctx).await.final_verdict {
            GuardVerdict::Block { reason } => {
                if let Some(res_id) = &reservation {
                    let _ = state.store.release(res_id).await;
                }
                if let Some(lr) = ledger_reservation {
                    let _ = state.ledger.release(lr);
                }
                state.limiter.release_parallel(&key.id);
                Self::audit(state, key, "request.denied", model, "denied", &reason);
                return Err(GatewayError::GuardBlocked(reason));
            }
            GuardVerdict::Mask { redacted_text } => {
                apply_user_mask(req, &redacted_text);
                Self::audit(state, key, "guard.masked", model, "masked", "pre_request");
            }
            GuardVerdict::Allow | GuardVerdict::Flag { .. } => {}
        }

        // 6. egress — mint ONE idempotency key (no-double-billing) and dispatch
        //    through the Router.
        let idempotency_key = uuid::Uuid::new_v4().to_string();
        let route = state.route_for(model, &provider_id);
        let executor = RegistryExecutor::new(&state.providers, idempotency_key.clone());
        let router = Router::new(route, state.clock.clone());
        let (response, meta) = match router.call(&executor, req).await {
            Ok(ok) => ok,
            Err(e) => {
                if let Some(res_id) = &reservation {
                    let _ = state.store.release(res_id).await;
                }
                if let Some(lr) = ledger_reservation {
                    let _ = state.ledger.release(lr);
                }
                state.limiter.release_parallel(&key.id);
                let ge = route_error_to_gateway(e);
                Self::audit(state, key, "request.error", model, "error", &ge.to_string());
                return Err(ge);
            }
        };
        let mut response = response;
        let served_by = format!("{}/{}", meta.provider_id, meta.model);
        let fallback_fired = meta.used_fallback || meta.hedge_won;

        // 7. commit ACTUAL cost from provider usage (true-up). Cost is computed
        //    from the registry for the model that ACTUALLY served the request.
        let actual_cost = {
            let reg = state.registry.read().unwrap();
            reg.cost(&meta.model, &response.usage)
                .or_else(|| reg.cost(model, &response.usage))
                .unwrap_or(Usd::ZERO)
        };

        // Commit to the durable store (async).
        if let Some(res_id) = &reservation
            && let Err(e) = state.store.commit(res_id, actual_cost).await
        {
            tracing::warn!(err = %e, "store commit failed (spend may be under-counted)");
        }
        // Also commit in the in-memory ledger so tests can assert on it.
        if let Some(lr) = ledger_reservation {
            let _ = state.ledger.commit(lr, actual_cost);
        }
        state.limiter.release_parallel(&key.id);

        // 8. PostResponse guard over the assistant output.
        let post_ctx = GuardContext {
            stage: GuardStage::PostResponse,
            text: response.text(),
            key_id: Some(key.id.clone()),
            model: Some(model.to_string()),
            tags: vec![],
        };
        match state.guard.run(&post_ctx).await.final_verdict {
            GuardVerdict::Block { reason } => {
                Self::audit(state, key, "response.blocked", model, "denied", &reason);
                return Err(GatewayError::GuardBlocked(reason));
            }
            GuardVerdict::Mask { redacted_text } => {
                mask_response_text(&mut response, &redacted_text);
                Self::audit(state, key, "guard.masked", model, "masked", "post_response");
            }
            GuardVerdict::Allow | GuardVerdict::Flag { .. } => {}
        }

        Self::audit(
            state,
            key,
            "request.complete",
            model,
            "ok",
            &format!("{} µUSD", actual_cost.micros()),
        );

        Ok(Completed {
            response,
            cost: actual_cost,
            idempotency_key,
            served_by,
            fallback_fired,
        })
    }

    /// Streaming variant. Same admission order as `run`; egress returns a
    /// delta stream that we wrap to commit the actual cost from the terminal
    /// delta's usage and release the parallel slot when the stream ends. If the
    /// stream is dropped before completion (client abort), the wrapper commits
    /// whatever usage arrived and releases the slot — usage is never lost and a
    /// reservation is never stranded.
    pub async fn run_stream<C: gateway_spine::Clock + 'static>(
        state: Arc<AppState<C>>,
        key: &VirtualKey,
        req: &ChatRequest,
    ) -> Result<CompletedStream, GatewayError> {
        // Owned request so the PreRequest guard can rewrite the prompt (Mask).
        let mut req = req.clone();
        let req = &mut req;
        let model = req.model.clone();

        if !key.allows_model(&model) {
            Self::audit(
                &state,
                key,
                "request.denied",
                &model,
                "denied",
                "model_not_allowed",
            );
            return Err(GatewayError::Spine(
                gateway_spine::SpineError::ModelNotAllowed {
                    key_id: key.id.clone(),
                    model: model.clone(),
                },
            ));
        }

        let provider_id = {
            let reg = state.registry.read().unwrap();
            reg.get(&model).map(|e| e.provider.clone()).ok_or_else(|| {
                GatewayError::Spine(gateway_spine::SpineError::UnknownModel {
                    model: model.clone(),
                })
            })?
        };
        let deployment = state.providers.get(&provider_id).ok_or_else(|| {
            GatewayError::BadRequest(format!("no egress configured for provider {provider_id}"))
        })?;

        let est_tokens = estimate_tokens(req);

        state
            .limiter
            .acquire(&key.id, &key.limits, est_tokens)
            .map_err(GatewayError::Spine)?;

        let est_cost = match state.estimate_cost(&model, est_tokens) {
            Ok(c) => c,
            Err(e) => {
                state.limiter.release_parallel(&key.id);
                return Err(e);
            }
        };

        // Durable reserve via store.
        let reservation: Option<String> = match state.store.reserve(&key.id, est_cost).await {
            Ok(res_id) => Some(res_id),
            Err(e) => {
                state.limiter.release_parallel(&key.id);
                match map_reserve_error(e, &key.id, key.max_budget) {
                    Ok(None) => None,
                    Ok(Some(_)) => unreachable!(),
                    Err(ge) => return Err(ge),
                }
            }
        };

        // In-memory ledger reserve so tests can assert on ledger state.
        let ledger_reservation = state.ledger.reserve(&key.id, est_cost).ok();

        // PreRequest guard over the prompt.
        let pre_ctx = GuardContext {
            stage: GuardStage::PreRequest,
            text: user_prompt_text(req),
            key_id: Some(key.id.clone()),
            model: Some(model.clone()),
            tags: vec![],
        };
        match state.guard.run(&pre_ctx).await.final_verdict {
            GuardVerdict::Block { reason } => {
                if let Some(res_id) = &reservation {
                    let _ = state.store.release(res_id).await;
                }
                if let Some(lr) = ledger_reservation {
                    let _ = state.ledger.release(lr);
                }
                state.limiter.release_parallel(&key.id);
                Self::audit(&state, key, "request.denied", &model, "denied", &reason);
                return Err(GatewayError::GuardBlocked(reason));
            }
            GuardVerdict::Mask { redacted_text } => {
                apply_user_mask(req, &redacted_text);
            }
            GuardVerdict::Allow | GuardVerdict::Flag { .. } => {}
        }

        let idempotency_key = uuid::Uuid::new_v4().to_string();
        let inner = match deployment
            .provider
            .stream(req, &deployment.credentials, &idempotency_key)
            .await
        {
            Ok(s) => s,
            Err(e) => {
                if let Some(res_id) = &reservation {
                    let _ = state.store.release(res_id).await;
                }
                if let Some(lr) = ledger_reservation {
                    let _ = state.ledger.release(lr);
                }
                state.limiter.release_parallel(&key.id);
                return Err(GatewayError::Provider(e));
            }
        };

        let wrapped = Self::wrap_stream_for_commit(
            state,
            key.id.clone(),
            model,
            reservation,
            ledger_reservation,
            inner,
        );
        Ok(CompletedStream {
            stream: Box::pin(wrapped),
            idempotency_key,
        })
    }

    /// Wrap a provider delta stream so the terminal usage commits cost + releases
    /// the parallel slot exactly once, whether the stream completes or is dropped.
    fn wrap_stream_for_commit<C: gateway_spine::Clock + 'static>(
        state: Arc<AppState<C>>,
        key_id: String,
        model: String,
        reservation: Option<String>,
        ledger_reservation: Option<gateway_spine::ReservationId>,
        inner: std::pin::Pin<Box<dyn Stream<Item = Result<StreamDelta, ProviderError>> + Send>>,
    ) -> impl Stream<Item = Result<StreamDelta, ProviderError>> + Send {
        // State carried across the stream: the latest usage seen + a guard that
        // commits on Drop so an aborted stream still trues-up.
        struct CommitGuard<C: gateway_spine::Clock> {
            state: Arc<AppState<C>>,
            key_id: String,
            model: String,
            /// `Some(...)` while guard is active; `take()` on first drop.
            /// Inner `Option<String>` = the store reservation id (None = unlimited).
            reservation: Option<Option<String>>,
            /// In-memory ledger reservation for tests.
            ledger_reservation: Option<Option<gateway_spine::ReservationId>>,
            last_usage: Option<gateway_spine::TokenUsage>,
            /// Tokio runtime handle for spawning the async commit from Drop.
            rt: tokio::runtime::Handle,
        }
        impl<C: gateway_spine::Clock> Drop for CommitGuard<C> {
            fn drop(&mut self) {
                if let Some(maybe_res) = self.reservation.take() {
                    let cost = {
                        let reg = self.state.registry.read().unwrap();
                        self.last_usage
                            .and_then(|u| reg.cost(&self.model, &u))
                            .unwrap_or(Usd::ZERO)
                    };

                    // Commit to durable store (fire and forget).
                    if let Some(res_id) = maybe_res {
                        let store = self.state.store.clone();
                        let res_clone = res_id.clone();
                        let cost_clone = cost;
                        self.rt.spawn(async move {
                            if let Err(e) = store.commit(&res_clone, cost_clone).await {
                                tracing::warn!(err = %e, "stream commit to store failed");
                            }
                        });
                    }

                    // Commit in in-memory ledger for tests.
                    if let Some(Some(lr)) = self.ledger_reservation.take() {
                        let _ = self.state.ledger.commit(lr, cost);
                    }

                    self.state.limiter.release_parallel(&self.key_id);
                    self.state.audit.record(AuditEvent {
                        ts_ms: self.state.clock.now_ms(),
                        actor: self.key_id.clone(),
                        action: "request.complete".into(),
                        target: self.model.clone(),
                        outcome: "ok".into(),
                        detail: Some(format!("{} µUSD (stream)", cost.micros())),
                    });
                }
            }
        }

        let rt = tokio::runtime::Handle::current();
        let guard = CommitGuard {
            state,
            key_id,
            model,
            reservation: Some(reservation),
            ledger_reservation: Some(ledger_reservation),
            last_usage: None,
            rt,
        };

        futures::stream::unfold((inner, guard), |(mut inner, mut guard)| async move {
            match inner.next().await {
                Some(Ok(delta)) => {
                    if let Some(u) = delta.usage {
                        guard.last_usage = Some(u);
                    }
                    Some((Ok(delta), (inner, guard)))
                }
                Some(Err(e)) => Some((Err(e), (inner, guard))),
                None => None, // `guard` drops here → commit fires
            }
        })
    }

    fn audit<C: gateway_spine::Clock>(
        state: &AppState<C>,
        key: &VirtualKey,
        action: &str,
        target: &str,
        outcome: &str,
        detail: &str,
    ) {
        state.audit.record(AuditEvent {
            ts_ms: state.clock.now_ms(),
            actor: key.id.clone(),
            action: action.to_string(),
            target: target.to_string(),
            outcome: outcome.to_string(),
            detail: Some(detail.to_string()),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guard::empty_chain;
    use crate::keystore::StaticKeyStore;
    use crate::providers::{Deployment, ProviderRegistry};
    use async_trait::async_trait;
    use gateway_llm::{
        ChatResponse, ContentPart, Credentials, DeltaStream, FinishReason, Message, Provider,
        ProviderCapabilities, ProviderError, Role,
    };
    use gateway_spine::{
        MemoryAudit, MockClock, ModelEntry, ModelPrice, RateLimits, TokenUsage, Usd, VirtualKey,
    };
    use gateway_store::StoredKey;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A provider that records how many times it was called + the last
    /// idempotency key, and returns a fixed usage.
    struct MockProvider {
        calls: AtomicUsize,
        last_idem: std::sync::Mutex<Option<String>>,
        last_prompt: std::sync::Mutex<Option<String>>,
        usage: TokenUsage,
    }

    impl MockProvider {
        fn new(usage: TokenUsage) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                last_idem: std::sync::Mutex::new(None),
                last_prompt: std::sync::Mutex::new(None),
                usage,
            }
        }
    }

    #[async_trait]
    impl Provider for MockProvider {
        fn id(&self) -> &str {
            "mock"
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                supports_streaming: true,
                supports_tools: true,
                supports_vision: false,
                supports_idempotency: true,
            }
        }
        async fn chat(
            &self,
            req: &ChatRequest,
            _creds: &Credentials,
            idempotency_key: &str,
        ) -> Result<ChatResponse, ProviderError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            *self.last_idem.lock().unwrap() = Some(idempotency_key.to_string());
            *self.last_prompt.lock().unwrap() = Some(
                req.messages
                    .iter()
                    .map(|m| m.text_content())
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
            Ok(ChatResponse {
                model: req.model.clone(),
                content: vec![ContentPart::text("hello")],
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: self.usage,
                provider_response_id: Some("resp_1".into()),
            })
        }
        async fn stream(
            &self,
            _req: &ChatRequest,
            _creds: &Credentials,
            _idempotency_key: &str,
        ) -> Result<DeltaStream, ProviderError> {
            unreachable!("non-streaming test")
        }
    }

    fn gpt4o() -> ModelEntry {
        ModelEntry {
            id: "gpt-4o".into(),
            provider: "openai".into(),
            price: ModelPrice {
                input_per_mtok: 2_500_000,
                output_per_mtok: 10_000_000,
                cache_read_per_mtok: 1_250_000,
                cache_write_per_mtok: 0,
            },
            context_window: Some(128_000),
            max_output_tokens: Some(16_384),
            supports_tools: true,
            supports_vision: true,
            supports_streaming: true,
        }
    }

    fn key(budget: Option<Usd>, allow: Option<Vec<String>>, limits: RateLimits) -> VirtualKey {
        VirtualKey {
            id: "key_1".into(),
            token_hash: VirtualKey::hash_secret("sk-test"),
            token_prefix: "sk-test".into(),
            max_budget: budget,
            limits,
            model_allowlist: allow,
            tool_allowlist: None,
            expires_at: None,
            revoked: false,
            parent_id: None,
        }
    }

    fn make_stored_key(id: &str, budget: Option<Usd>) -> StoredKey {
        StoredKey {
            id: id.to_string(),
            name: format!("key-{id}"),
            token_hash: format!("hash-{id}"),
            token_prefix: "sk-test".to_string(),
            budget_micros: budget.map(|u| u.micros()),
            spent_micros: 0,
            rpm: None,
            tpm: None,
            max_parallel: None,
            model_allowlist: None,
            expires_at_ms: None,
            revoked: false,
            parent_id: None,
            created_at_ms: 0,
        }
    }

    /// Build an AppState wired to a shared MockProvider so tests can inspect it.
    async fn state_with(provider: Arc<MockProvider>, budget: Option<Usd>) -> AppState<MockClock> {
        let mut ks = StaticKeyStore::new();
        ks.insert(key(budget, None, RateLimits::default()));
        let providers = ProviderRegistry::new();
        providers.insert(
            "openai",
            Deployment {
                provider: provider.clone(),
                credentials: Arc::new(Credentials::new("sk-up")),
            },
        );
        let store = Arc::new(
            gateway_store::Store::connect("sqlite::memory:")
                .await
                .unwrap(),
        );
        // Seed the key in the store with budget so reserve works.
        store
            .upsert_key(&make_stored_key("key_1", budget))
            .await
            .unwrap();

        let state = AppState::with_parts(
            Arc::new(ks),
            Arc::new(MockClock::new(1_000)),
            providers,
            Arc::new(empty_chain()),
            Arc::new(MemoryAudit::new()),
            store,
        );
        state.registry.write().unwrap().insert(gpt4o());
        state.ledger.set_budget("key_1", budget, Usd::ZERO);
        state
    }

    /// Like `state_with` but installs the production `default_chain` guard
    /// (secrets-block + PII-mask) so guard integration can be exercised.
    async fn state_with_default_guard(
        provider: Arc<MockProvider>,
        budget: Option<Usd>,
    ) -> AppState<MockClock> {
        let mut ks = StaticKeyStore::new();
        ks.insert(key(budget, None, RateLimits::default()));
        let providers = ProviderRegistry::new();
        providers.insert(
            "openai",
            Deployment {
                provider: provider.clone(),
                credentials: Arc::new(Credentials::new("sk-up")),
            },
        );
        let store = Arc::new(
            gateway_store::Store::connect("sqlite::memory:")
                .await
                .unwrap(),
        );
        store
            .upsert_key(&make_stored_key("key_1", budget))
            .await
            .unwrap();

        let state = AppState::with_parts(
            Arc::new(ks),
            Arc::new(MockClock::new(1_000)),
            providers,
            Arc::new(crate::guard::default_chain()),
            Arc::new(MemoryAudit::new()),
            store,
        );
        state.registry.write().unwrap().insert(gpt4o());
        state.ledger.set_budget("key_1", budget, Usd::ZERO);
        state
    }

    /// Build an AppState wired to a shared MockProvider and a guard chain
    /// assembled from the given config-style rule views.
    async fn state_with_rules(
        provider: Arc<MockProvider>,
        budget: Option<Usd>,
        rules: &[gateway_guard::builder_from_config::GuardrailRuleView<'_>],
    ) -> AppState<MockClock> {
        let chain = gateway_guard::builder_from_config::chain_from_rules(rules).unwrap();
        let mut ks = StaticKeyStore::new();
        ks.insert(key(budget, None, RateLimits::default()));
        let providers = ProviderRegistry::new();
        providers.insert(
            "openai",
            Deployment {
                provider: provider.clone(),
                credentials: Arc::new(Credentials::new("sk-up")),
            },
        );
        let store = Arc::new(
            gateway_store::Store::connect("sqlite::memory:")
                .await
                .unwrap(),
        );
        store
            .upsert_key(&make_stored_key("key_1", budget))
            .await
            .unwrap();

        let state = AppState::with_parts(
            Arc::new(ks),
            Arc::new(MockClock::new(1_000)),
            providers,
            Arc::new(chain),
            Arc::new(MemoryAudit::new()),
            store,
        );
        state.registry.write().unwrap().insert(gpt4o());
        state.ledger.set_budget("key_1", budget, Usd::ZERO);
        state
    }

    fn chat_req() -> ChatRequest {
        ChatRequest::new("gpt-4o", vec![Message::text(Role::User, "hi there")])
    }

    #[tokio::test]
    async fn guard_blocks_secret_prompt_before_egress() {
        let provider = Arc::new(MockProvider::new(TokenUsage::default()));
        let state =
            state_with_default_guard(provider.clone(), Some(Usd::from_dollars_f64(1.0))).await;
        let k = key(
            Some(Usd::from_dollars_f64(1.0)),
            None,
            RateLimits::default(),
        );

        // A live OpenAI-shaped key in the prompt must be blocked pre-egress.
        let req = ChatRequest::new(
            "gpt-4o",
            vec![Message::text(
                Role::User,
                "please use my key sk-ant-api03-FAKEFAKEFAKEFAKEFAKE to call the api",
            )],
        );
        let err = Gateway::run(&state, &k, &req).await.unwrap_err();
        assert_eq!(err.status(), axum::http::StatusCode::FORBIDDEN);
        assert!(matches!(err, GatewayError::GuardBlocked(_)));
        // The provider must NOT have been called, and no spend/reservation leaked.
        assert_eq!(
            provider.calls.load(Ordering::SeqCst),
            0,
            "secret prompt must never reach the provider"
        );
        assert_eq!(state.ledger.reserved("key_1"), Usd::ZERO);
        assert_eq!(state.ledger.spent("key_1"), Usd::ZERO);
    }

    #[tokio::test]
    async fn guard_masks_pii_prompt_then_calls_provider() {
        let usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        };
        let provider = Arc::new(MockProvider::new(usage));
        let state =
            state_with_default_guard(provider.clone(), Some(Usd::from_dollars_f64(1.0))).await;
        let k = key(
            Some(Usd::from_dollars_f64(1.0)),
            None,
            RateLimits::default(),
        );

        let req = ChatRequest::new(
            "gpt-4o",
            vec![Message::text(
                Role::User,
                "email me at alice@example.com about the order",
            )],
        );
        // PII is masked, not blocked — the call still succeeds.
        let done = Gateway::run(&state, &k, &req).await.unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(done.response.text(), "hello");
        // The provider saw the masked prompt (no raw email).
        let seen = provider.last_prompt.lock().unwrap().clone().unwrap();
        assert!(
            !seen.contains("alice@example.com"),
            "provider must see redacted prompt, got: {seen}"
        );
        assert!(seen.contains("[EMAIL]"));
    }

    #[tokio::test]
    async fn guard_allows_clean_prompt() {
        let usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        };
        let provider = Arc::new(MockProvider::new(usage));
        let state =
            state_with_default_guard(provider.clone(), Some(Usd::from_dollars_f64(1.0))).await;
        let k = key(
            Some(Usd::from_dollars_f64(1.0)),
            None,
            RateLimits::default(),
        );
        let done = Gateway::run(&state, &k, &chat_req()).await.unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(done.response.text(), "hello");
    }

    #[tokio::test]
    async fn happy_path_commits_actual_cost() {
        // 1000 in + 500 out → $0.0025 + $0.005 = $0.0075 = 7_500 µUSD
        let usage = TokenUsage {
            input_tokens: 1000,
            output_tokens: 500,
            ..Default::default()
        };
        let provider = Arc::new(MockProvider::new(usage));
        let state = state_with(provider.clone(), Some(Usd::from_dollars_f64(1.0))).await;
        let k = key(
            Some(Usd::from_dollars_f64(1.0)),
            None,
            RateLimits::default(),
        );

        let done = Gateway::run(&state, &k, &chat_req()).await.unwrap();

        assert_eq!(done.cost, Usd::from_micros(7_500));
        assert_eq!(state.ledger.spent("key_1"), Usd::from_micros(7_500));
        assert_eq!(
            state.ledger.reserved("key_1"),
            Usd::ZERO,
            "reservation trued up"
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        // idempotency key was minted + passed
        assert!(provider.last_idem.lock().unwrap().is_some());
    }

    #[tokio::test]
    async fn budget_exceeded_never_calls_provider() {
        let usage = TokenUsage {
            output_tokens: 500,
            ..Default::default()
        };
        let provider = Arc::new(MockProvider::new(usage));
        // tiny budget that the worst-case estimate blows
        let state = state_with(provider.clone(), Some(Usd::from_micros(1))).await;
        let k = key(Some(Usd::from_micros(1)), None, RateLimits::default());

        let err = Gateway::run(&state, &k, &chat_req()).await.unwrap_err();
        assert_eq!(err.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            provider.calls.load(Ordering::SeqCst),
            0,
            "fail-closed: no egress"
        );
        // no stranded reservation or parallel slot
        assert_eq!(state.ledger.reserved("key_1"), Usd::ZERO);
    }

    #[tokio::test]
    async fn disallowed_model_is_403_no_egress() {
        let provider = Arc::new(MockProvider::new(TokenUsage::default()));
        let state = state_with(provider.clone(), Some(Usd::from_dollars_f64(1.0))).await;
        let k = key(
            Some(Usd::from_dollars_f64(1.0)),
            Some(vec!["claude-3-5-sonnet".into()]),
            RateLimits::default(),
        );
        let err = Gateway::run(&state, &k, &chat_req()).await.unwrap_err();
        assert_eq!(err.status(), axum::http::StatusCode::FORBIDDEN);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn unknown_model_is_400() {
        let provider = Arc::new(MockProvider::new(TokenUsage::default()));
        let state = state_with(provider.clone(), Some(Usd::from_dollars_f64(1.0))).await;
        let k = key(
            Some(Usd::from_dollars_f64(1.0)),
            None,
            RateLimits::default(),
        );
        let req = ChatRequest::new("mystery", vec![Message::text(Role::User, "hi")]);
        let err = Gateway::run(&state, &k, &req).await.unwrap_err();
        assert_eq!(err.status(), axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn provider_error_releases_reservation() {
        struct Failing;
        #[async_trait]
        impl Provider for Failing {
            fn id(&self) -> &str {
                "fail"
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities {
                    supports_streaming: false,
                    supports_tools: false,
                    supports_vision: false,
                    supports_idempotency: false,
                }
            }
            async fn chat(
                &self,
                _req: &ChatRequest,
                _creds: &Credentials,
                _idempotency_key: &str,
            ) -> Result<ChatResponse, ProviderError> {
                Err(ProviderError::Upstream {
                    status: 500,
                    body: "boom".into(),
                })
            }
            async fn stream(
                &self,
                _req: &ChatRequest,
                _creds: &Credentials,
                _idempotency_key: &str,
            ) -> Result<DeltaStream, ProviderError> {
                unreachable!()
            }
        }

        let mut ks = StaticKeyStore::new();
        ks.insert(key(
            Some(Usd::from_dollars_f64(1.0)),
            None,
            RateLimits::default(),
        ));
        let providers = ProviderRegistry::new();
        providers.insert(
            "openai",
            Deployment {
                provider: Arc::new(Failing),
                credentials: Arc::new(Credentials::new("x")),
            },
        );
        let store = Arc::new(
            gateway_store::Store::connect("sqlite::memory:")
                .await
                .unwrap(),
        );
        store
            .upsert_key(&make_stored_key("key_1", Some(Usd::from_dollars_f64(1.0))))
            .await
            .unwrap();

        let state = AppState::with_parts(
            Arc::new(ks),
            Arc::new(MockClock::new(0)),
            providers,
            Arc::new(empty_chain()),
            Arc::new(MemoryAudit::new()),
            store,
        );
        state.registry.write().unwrap().insert(gpt4o());
        state
            .ledger
            .set_budget("key_1", Some(Usd::from_dollars_f64(1.0)), Usd::ZERO);

        let k = key(
            Some(Usd::from_dollars_f64(1.0)),
            None,
            RateLimits::default(),
        );
        let err = Gateway::run(&state, &k, &chat_req()).await.unwrap_err();
        assert_eq!(err.status(), axum::http::StatusCode::BAD_GATEWAY);
        assert_eq!(
            state.ledger.reserved("key_1"),
            Usd::ZERO,
            "released on egress error"
        );
        assert_eq!(state.ledger.spent("key_1"), Usd::ZERO, "nothing billed");
    }

    #[tokio::test]
    async fn streaming_commits_from_terminal_delta_usage() {
        use gateway_llm::FinishReason;

        struct StreamProvider {
            usage: TokenUsage,
        }
        #[async_trait]
        impl Provider for StreamProvider {
            fn id(&self) -> &str {
                "stream"
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities {
                    supports_streaming: true,
                    supports_tools: false,
                    supports_vision: false,
                    supports_idempotency: true,
                }
            }
            async fn chat(
                &self,
                _req: &ChatRequest,
                _creds: &Credentials,
                _idempotency_key: &str,
            ) -> Result<ChatResponse, ProviderError> {
                unreachable!()
            }
            async fn stream(
                &self,
                _req: &ChatRequest,
                _creds: &Credentials,
                _idempotency_key: &str,
            ) -> Result<DeltaStream, ProviderError> {
                let deltas = vec![
                    Ok(StreamDelta::text("hel")),
                    Ok(StreamDelta::text("lo")),
                    Ok(StreamDelta::finish(FinishReason::Stop, self.usage)),
                ];
                Ok(Box::pin(futures::stream::iter(deltas)))
            }
        }

        let mut ks = StaticKeyStore::new();
        ks.insert(key(
            Some(Usd::from_dollars_f64(1.0)),
            None,
            RateLimits::default(),
        ));
        let providers = ProviderRegistry::new();
        let usage = TokenUsage {
            input_tokens: 1000,
            output_tokens: 500,
            ..Default::default()
        };
        providers.insert(
            "openai",
            Deployment {
                provider: Arc::new(StreamProvider { usage }),
                credentials: Arc::new(Credentials::new("x")),
            },
        );
        let store = Arc::new(
            gateway_store::Store::connect("sqlite::memory:")
                .await
                .unwrap(),
        );
        store
            .upsert_key(&make_stored_key("key_1", Some(Usd::from_dollars_f64(1.0))))
            .await
            .unwrap();

        let state = Arc::new(AppState::with_parts(
            Arc::new(ks),
            Arc::new(MockClock::new(0)),
            providers,
            Arc::new(empty_chain()),
            Arc::new(MemoryAudit::new()),
            store,
        ));
        state.registry.write().unwrap().insert(gpt4o());
        state
            .ledger
            .set_budget("key_1", Some(Usd::from_dollars_f64(1.0)), Usd::ZERO);

        let k = key(
            Some(Usd::from_dollars_f64(1.0)),
            None,
            RateLimits::default(),
        );
        let completed = Gateway::run_stream(state.clone(), &k, &chat_req())
            .await
            .unwrap();

        // drain the whole stream
        let mut s = completed.stream;
        let mut chunks = 0;
        while let Some(item) = s.next().await {
            item.unwrap();
            chunks += 1;
        }
        drop(s); // ensure the commit guard drops
        // Give tokio a moment to process the spawned commit task.
        tokio::task::yield_now().await;
        assert_eq!(chunks, 3);
        assert_eq!(state.ledger.spent("key_1"), Usd::from_micros(7_500));
        assert_eq!(state.ledger.reserved("key_1"), Usd::ZERO);
    }

    #[tokio::test]
    async fn aborted_stream_still_commits_partial_usage() {
        use gateway_llm::FinishReason;

        struct AbortProvider {
            usage: TokenUsage,
        }
        #[async_trait]
        impl Provider for AbortProvider {
            fn id(&self) -> &str {
                "abort"
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities {
                    supports_streaming: true,
                    supports_tools: false,
                    supports_vision: false,
                    supports_idempotency: true,
                }
            }
            async fn chat(
                &self,
                _req: &ChatRequest,
                _creds: &Credentials,
                _idempotency_key: &str,
            ) -> Result<ChatResponse, ProviderError> {
                unreachable!()
            }
            async fn stream(
                &self,
                _req: &ChatRequest,
                _creds: &Credentials,
                _idempotency_key: &str,
            ) -> Result<DeltaStream, ProviderError> {
                // usage arrives on the FIRST delta, then more content would follow
                let deltas = vec![
                    Ok(StreamDelta::finish(FinishReason::Stop, self.usage)),
                    Ok(StreamDelta::text("more")),
                ];
                Ok(Box::pin(futures::stream::iter(deltas)))
            }
        }

        let mut ks = StaticKeyStore::new();
        ks.insert(key(
            Some(Usd::from_dollars_f64(1.0)),
            None,
            RateLimits::default(),
        ));
        let providers = ProviderRegistry::new();
        let usage = TokenUsage {
            input_tokens: 1000,
            output_tokens: 500,
            ..Default::default()
        };
        providers.insert(
            "openai",
            Deployment {
                provider: Arc::new(AbortProvider { usage }),
                credentials: Arc::new(Credentials::new("x")),
            },
        );
        let store = Arc::new(
            gateway_store::Store::connect("sqlite::memory:")
                .await
                .unwrap(),
        );
        store
            .upsert_key(&make_stored_key("key_1", Some(Usd::from_dollars_f64(1.0))))
            .await
            .unwrap();

        let state = Arc::new(AppState::with_parts(
            Arc::new(ks),
            Arc::new(MockClock::new(0)),
            providers,
            Arc::new(empty_chain()),
            Arc::new(MemoryAudit::new()),
            store,
        ));
        state.registry.write().unwrap().insert(gpt4o());
        state
            .ledger
            .set_budget("key_1", Some(Usd::from_dollars_f64(1.0)), Usd::ZERO);

        let k = key(
            Some(Usd::from_dollars_f64(1.0)),
            None,
            RateLimits::default(),
        );
        let completed = Gateway::run_stream(state.clone(), &k, &chat_req())
            .await
            .unwrap();

        // read ONLY the first delta (which carried usage), then DROP the stream.
        let mut s = completed.stream;
        let first = s.next().await.unwrap();
        first.unwrap();
        drop(s); // abort → CommitGuard drops → commit fires from last_usage
        // Give tokio a moment to process the spawned commit task.
        tokio::task::yield_now().await;

        assert_eq!(
            state.ledger.spent("key_1"),
            Usd::from_micros(7_500),
            "usage not lost on abort"
        );
        assert_eq!(
            state.ledger.reserved("key_1"),
            Usd::ZERO,
            "no stranded reservation"
        );
    }

    // ── Routing integration ───────────────────────────────────────────────────

    /// A provider that fails its first `chat` with a retryable error, then
    /// succeeds — used to prove the router fails over to the backup target.
    struct FlakyProvider {
        id: &'static str,
        calls: AtomicUsize,
        fail_first: bool,
    }
    #[async_trait]
    impl Provider for FlakyProvider {
        fn id(&self) -> &str {
            self.id
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                supports_streaming: false,
                supports_tools: false,
                supports_vision: false,
                supports_idempotency: true,
            }
        }
        async fn chat(
            &self,
            req: &ChatRequest,
            _creds: &Credentials,
            _idempotency_key: &str,
        ) -> Result<ChatResponse, ProviderError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail_first && n == 0 {
                return Err(ProviderError::Upstream {
                    status: 503,
                    body: "service unavailable".into(),
                });
            }
            Ok(ChatResponse {
                model: req.model.clone(),
                content: vec![ContentPart::text("ok")],
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: TokenUsage {
                    input_tokens: 100,
                    output_tokens: 50,
                    ..Default::default()
                },
                provider_response_id: None,
            })
        }
        async fn stream(
            &self,
            _req: &ChatRequest,
            _creds: &Credentials,
            _idempotency_key: &str,
        ) -> Result<DeltaStream, ProviderError> {
            unreachable!()
        }
    }

    #[tokio::test]
    async fn failover_route_uses_backup_on_retryable_error() {
        let primary = Arc::new(FlakyProvider {
            id: "primary",
            calls: AtomicUsize::new(0),
            fail_first: true,
        });
        let backup = Arc::new(FlakyProvider {
            id: "backup",
            calls: AtomicUsize::new(0),
            fail_first: false,
        });

        let mut ks = StaticKeyStore::new();
        ks.insert(key(
            Some(Usd::from_dollars_f64(1.0)),
            None,
            RateLimits::default(),
        ));
        let providers = ProviderRegistry::new();
        providers.insert(
            "primary",
            Deployment {
                provider: primary.clone(),
                credentials: Arc::new(Credentials::new("p")),
            },
        );
        providers.insert(
            "backup",
            Deployment {
                provider: backup.clone(),
                credentials: Arc::new(Credentials::new("b")),
            },
        );
        let store = Arc::new(
            gateway_store::Store::connect("sqlite::memory:")
                .await
                .unwrap(),
        );
        store
            .upsert_key(&make_stored_key("key_1", Some(Usd::from_dollars_f64(1.0))))
            .await
            .unwrap();

        let state = AppState::with_parts(
            Arc::new(ks),
            Arc::new(MockClock::new(0)),
            providers,
            Arc::new(empty_chain()),
            Arc::new(MemoryAudit::new()),
            store,
        );
        // The request model is "gpt-4o" served by the "primary" provider.
        let mut entry = gpt4o();
        entry.provider = "primary".into();
        state.registry.write().unwrap().insert(entry);
        state
            .ledger
            .set_budget("key_1", Some(Usd::from_dollars_f64(1.0)), Usd::ZERO);

        // Install a failover route: primary → backup, no backoff in tests.
        let mut route = Route::new(
            vec![
                gateway_route::RouteTarget::new("primary", "gpt-4o"),
                gateway_route::RouteTarget::new("backup", "gpt-4o"),
            ],
            gateway_route::Strategy::Failover,
        );
        route.failure_threshold = 1;
        route.base_backoff_ms = 0;
        route.max_attempts = 4;
        state.set_route("gpt-4o", route);

        let k = key(
            Some(Usd::from_dollars_f64(1.0)),
            None,
            RateLimits::default(),
        );
        let done = Gateway::run(&state, &k, &chat_req()).await.unwrap();

        // Primary failed once; backup served the response.
        assert_eq!(primary.calls.load(Ordering::SeqCst), 1);
        assert_eq!(backup.calls.load(Ordering::SeqCst), 1);
        assert_eq!(done.served_by, "backup/gpt-4o");
        assert!(done.fallback_fired, "fallback should have fired");
        // Cost trued up (100 in + 50 out on gpt-4o pricing) and committed once.
        assert_eq!(done.cost, Usd::from_micros(750));
        assert_eq!(state.ledger.spent("key_1"), Usd::from_micros(750));
        assert_eq!(state.ledger.reserved("key_1"), Usd::ZERO);
    }

    #[tokio::test]
    async fn single_deployment_model_routes_unchanged() {
        // With no route override, a model with one deployment behaves exactly as
        // the direct-dispatch path: served_by = its provider, no fallback.
        let usage = TokenUsage {
            input_tokens: 1000,
            output_tokens: 500,
            ..Default::default()
        };
        let provider = Arc::new(MockProvider::new(usage));
        let state = state_with(provider.clone(), Some(Usd::from_dollars_f64(1.0))).await;
        let k = key(
            Some(Usd::from_dollars_f64(1.0)),
            None,
            RateLimits::default(),
        );
        let done = Gateway::run(&state, &k, &chat_req()).await.unwrap();
        assert_eq!(done.served_by, "openai/gpt-4o");
        assert!(!done.fallback_fired);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(done.cost, Usd::from_micros(7_500));
    }

    // ── Config-chain guard integration ────────────────────────────────────────

    #[tokio::test]
    async fn config_keyword_chain_blocks_matching_prompt() {
        let kws = vec!["forbidden-word".to_string()];
        let rules = [gateway_guard::builder_from_config::GuardrailRuleView {
            guardrail_type: "keyword",
            mode: "enforce",
            keywords: &kws,
            pattern: None,
            label: None,
            schema: None,
            url: None,
            stages: &[],
        }];
        let provider = Arc::new(MockProvider::new(TokenUsage::default()));
        let state =
            state_with_rules(provider.clone(), Some(Usd::from_dollars_f64(1.0)), &rules).await;
        let k = key(
            Some(Usd::from_dollars_f64(1.0)),
            None,
            RateLimits::default(),
        );
        let req = ChatRequest::new(
            "gpt-4o",
            vec![Message::text(
                Role::User,
                "please share the forbidden-word plan",
            )],
        );

        let err = Gateway::run(&state, &k, &req).await.unwrap_err();
        assert_eq!(err.status(), axum::http::StatusCode::FORBIDDEN);
        assert!(matches!(err, GatewayError::GuardBlocked(_)));
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        assert_eq!(state.ledger.reserved("key_1"), Usd::ZERO);
    }

    #[tokio::test]
    async fn config_observe_only_does_not_block_request() {
        let kws = vec!["forbidden-word".to_string()];
        let rules = [gateway_guard::builder_from_config::GuardrailRuleView {
            guardrail_type: "keyword",
            mode: "observe_only",
            keywords: &kws,
            pattern: None,
            label: None,
            schema: None,
            url: None,
            stages: &[],
        }];
        let usage = TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            ..Default::default()
        };
        let provider = Arc::new(MockProvider::new(usage));
        let state =
            state_with_rules(provider.clone(), Some(Usd::from_dollars_f64(1.0)), &rules).await;
        let k = key(
            Some(Usd::from_dollars_f64(1.0)),
            None,
            RateLimits::default(),
        );
        let req = ChatRequest::new(
            "gpt-4o",
            vec![Message::text(
                Role::User,
                "the forbidden-word should pass through in observe_only",
            )],
        );

        Gateway::run(&state, &k, &req).await.unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn config_regex_deny_blocks_pattern_match() {
        let pat = r"\bpassword\b".to_string();
        let rules = [gateway_guard::builder_from_config::GuardrailRuleView {
            guardrail_type: "regex_deny",
            mode: "enforce",
            keywords: &[],
            pattern: Some(&pat),
            label: Some("password"),
            schema: None,
            url: None,
            stages: &[],
        }];
        let provider = Arc::new(MockProvider::new(TokenUsage::default()));
        let state =
            state_with_rules(provider.clone(), Some(Usd::from_dollars_f64(1.0)), &rules).await;
        let k = key(
            Some(Usd::from_dollars_f64(1.0)),
            None,
            RateLimits::default(),
        );
        let req = ChatRequest::new(
            "gpt-4o",
            vec![Message::text(Role::User, "reset my password now")],
        );

        let err = Gateway::run(&state, &k, &req).await.unwrap_err();
        assert_eq!(err.status(), axum::http::StatusCode::FORBIDDEN);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn empty_config_chain_allows_clean_request() {
        let rules: &[gateway_guard::builder_from_config::GuardrailRuleView<'_>] = &[];
        let usage = TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            ..Default::default()
        };
        let provider = Arc::new(MockProvider::new(usage));
        let state =
            state_with_rules(provider.clone(), Some(Usd::from_dollars_f64(1.0)), rules).await;
        let k = key(
            Some(Usd::from_dollars_f64(1.0)),
            None,
            RateLimits::default(),
        );

        Gateway::run(&state, &k, &chat_req()).await.unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    }
}
