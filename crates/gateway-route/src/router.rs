//! The `Router` struct: applies strategy + retries + cooldown + hedging.

use std::sync::Arc;
use std::time::Duration;

use rand::Rng;
use tokio::time::timeout;

use gateway_llm::{ChatRequest, ChatResponse, ProviderError};
use gateway_spine::Clock;

use crate::error::RouteError;
use crate::executor::TargetExecutor;
use crate::route::{Route, RouteTarget, Strategy};
use crate::strategy::TargetStateVec;

/// Metadata about how a request was served. Attached to every successful
/// `Router::call` result.
#[derive(Debug, Clone, PartialEq)]
pub struct RouterMeta {
    /// Index of the target (in `Route::targets`) that provided the response.
    pub target_index: usize,
    /// Provider id of the winning target.
    pub provider_id: String,
    /// Model id of the winning target.
    pub model: String,
    /// Whether the router had to fall back to a non-primary target.
    pub used_fallback: bool,
    /// Whether hedging fired and the backup target won.
    pub hedge_won: bool,
    /// Total number of provider call attempts made.
    pub attempt_count: u32,
}

/// A successful call result: the response plus routing metadata.
pub type RouterResult = (ChatResponse, RouterMeta);

/// Classify a `ProviderError` into retryable vs terminal.
fn is_retryable(err: &ProviderError) -> bool {
    match err {
        ProviderError::RateLimited { .. } => true,
        ProviderError::Transport(_) => true,
        ProviderError::Upstream { status, .. } => *status >= 500,
        // Auth, Unsupported, SchemaDrift, Decode, and 4xx are terminal.
        ProviderError::Auth => false,
        ProviderError::Unsupported { .. } => false,
        ProviderError::SchemaDrift { .. } => false,
        ProviderError::Decode(_) => false,
    }
}

/// Compute the backoff delay for `attempt` (0-based).
/// Returns the clamped jittered exponential delay in milliseconds.
fn backoff_ms(attempt: u32, base_ms: u64, max_ms: u64) -> u64 {
    // 2^attempt * base, capped at max, plus ±25% jitter.
    let exp = (base_ms as f64) * (2_u64.saturating_pow(attempt) as f64);
    let capped = exp.min(max_ms as f64) as u64;
    let jitter_range = capped / 4;
    if jitter_range == 0 {
        return capped;
    }
    let jitter = rand::thread_rng().gen_range(0..=jitter_range);
    capped + jitter
}

/// How long to honor a `RateLimited { retry_after_secs }` hint (capped to
/// avoid abusive upstreams holding us too long).
const MAX_RATE_LIMIT_SLEEP_SECS: u64 = 60;

/// The routing engine. Constructed once per route config; `call` is invoked
/// per request. The `TargetStateVec` is shared across concurrent calls so
/// EWMA latency and cooldown state accumulate across the life of the router.
pub struct Router<C: Clock> {
    route: Route,
    clock: Arc<C>,
    state: TargetStateVec,
}

impl<C: Clock + 'static> Router<C> {
    /// Create a new router. `clock` is injected for testability.
    pub fn new(route: Route, clock: Arc<C>) -> Self {
        let n = route.targets.len();
        Router {
            route,
            clock,
            state: TargetStateVec::new(n),
        }
    }

    /// Execute the route. Returns the first successful response plus metadata,
    /// or a `RouteError` if all attempts failed.
    pub async fn call(
        &self,
        executor: &dyn TargetExecutor,
        request: &ChatRequest,
    ) -> Result<RouterResult, RouteError> {
        if self.route.targets.is_empty() {
            return Err(RouteError::NoTargets);
        }

        match self.route.strategy {
            Strategy::Single => self.call_single(executor, request).await,
            Strategy::Failover => self.call_failover(executor, request).await,
            Strategy::Weighted => self.call_weighted(executor, request).await,
            Strategy::LatencyAware => self.call_latency_aware(executor, request).await,
        }
    }

    // ── Strategy implementations ────────────────────────────────────────────

    /// Single: use the first target; retry on retryable errors (no failover).
    async fn call_single(
        &self,
        executor: &dyn TargetExecutor,
        request: &ChatRequest,
    ) -> Result<RouterResult, RouteError> {
        let target = &self.route.targets[0];
        let mut attempts = 0u32;

        loop {
            attempts += 1;
            let start = self.clock.now_ms();
            let result = executor.execute(target, request).await;
            let elapsed = (self.clock.now_ms() - start).max(0) as f64;

            match result {
                Ok(resp) => {
                    self.state.record_success(0, elapsed);
                    let meta = RouterMeta {
                        target_index: 0,
                        provider_id: target.provider_id.clone(),
                        model: target.model.clone(),
                        used_fallback: false,
                        hedge_won: false,
                        attempt_count: attempts,
                    };
                    return Ok((resp, meta));
                }
                Err(err) => {
                    self.state
                        .record_failure(0, self.route.failure_threshold, self.clock.as_ref());

                    if !is_retryable(&err) {
                        return Err(RouteError::TerminalError(err));
                    }
                    if attempts >= self.route.max_attempts {
                        return Err(RouteError::AllTargetsExhausted { attempts });
                    }
                    self.sleep_for_error(&err, attempts).await;
                }
            }
        }
    }

    /// Failover: try targets in order; advance to the next on retryable error.
    async fn call_failover(
        &self,
        executor: &dyn TargetExecutor,
        request: &ChatRequest,
    ) -> Result<RouterResult, RouteError> {
        let mut attempts = 0u32;
        let mut used_fallback = false;

        // Check if hedging is configured and there's a second target.
        if self.route.hedge_after_ms.is_some() && self.route.targets.len() >= 2 {
            return self.call_with_hedge(executor, request).await;
        }

        for (idx, target) in self.route.targets.iter().enumerate() {
            // Skip targets in cooldown (try to recover first).
            if !self
                .state
                .maybe_recover(idx, self.route.cooldown_ms, self.clock.as_ref())
            {
                tracing::debug!(
                    provider_id = %target.provider_id,
                    "skipping target {} (in cooldown)",
                    idx
                );
                continue;
            }

            if idx > 0 {
                used_fallback = true;
            }

            // Retry loop within this target.
            let mut target_attempts = 0u32;
            loop {
                if attempts >= self.route.max_attempts {
                    return Err(RouteError::AllTargetsExhausted { attempts });
                }
                attempts += 1;
                target_attempts += 1;

                let start = self.clock.now_ms();
                let result = executor.execute(target, request).await;
                let elapsed = (self.clock.now_ms() - start).max(0) as f64;

                match result {
                    Ok(resp) => {
                        self.state.record_success(idx, elapsed);
                        let meta = RouterMeta {
                            target_index: idx,
                            provider_id: target.provider_id.clone(),
                            model: target.model.clone(),
                            used_fallback,
                            hedge_won: false,
                            attempt_count: attempts,
                        };
                        return Ok((resp, meta));
                    }
                    Err(err) => {
                        self.state.record_failure(
                            idx,
                            self.route.failure_threshold,
                            self.clock.as_ref(),
                        );

                        if !is_retryable(&err) {
                            return Err(RouteError::TerminalError(err));
                        }

                        // After failure_threshold retries on this target, move on.
                        if target_attempts >= self.route.failure_threshold {
                            break; // advance to next target
                        }

                        self.sleep_for_error(&err, target_attempts).await;
                    }
                }
            }
        }

        Err(RouteError::AllTargetsExhausted { attempts })
    }

    /// Weighted: sample a target proportionally by weight, retry on retryable errors.
    async fn call_weighted(
        &self,
        executor: &dyn TargetExecutor,
        request: &ChatRequest,
    ) -> Result<RouterResult, RouteError> {
        let mut attempts = 0u32;

        loop {
            if attempts >= self.route.max_attempts {
                return Err(RouteError::AllTargetsExhausted { attempts });
            }

            // Sample a healthy target by weight.
            let Some((idx, target)) = self.pick_weighted() else {
                return Err(RouteError::AllTargetsExhausted { attempts });
            };

            attempts += 1;
            let start = self.clock.now_ms();
            let result = executor.execute(target, request).await;
            let elapsed = (self.clock.now_ms() - start).max(0) as f64;

            match result {
                Ok(resp) => {
                    self.state.record_success(idx, elapsed);
                    let meta = RouterMeta {
                        target_index: idx,
                        provider_id: target.provider_id.clone(),
                        model: target.model.clone(),
                        used_fallback: idx != 0,
                        hedge_won: false,
                        attempt_count: attempts,
                    };
                    return Ok((resp, meta));
                }
                Err(err) => {
                    self.state.record_failure(
                        idx,
                        self.route.failure_threshold,
                        self.clock.as_ref(),
                    );

                    if !is_retryable(&err) {
                        return Err(RouteError::TerminalError(err));
                    }

                    self.sleep_for_error(&err, attempts).await;
                }
            }
        }
    }

    /// Latency-aware: always pick the target with the lowest EWMA; fallback
    /// to failover order for targets with no latency data yet.
    async fn call_latency_aware(
        &self,
        executor: &dyn TargetExecutor,
        request: &ChatRequest,
    ) -> Result<RouterResult, RouteError> {
        let mut attempts = 0u32;
        let mut used_fallback = false;
        // Targets we have NOT yet tried in this call (to avoid infinite loops
        // when all targets keep failing).
        let mut remaining: Vec<usize> = (0..self.route.targets.len()).collect();

        loop {
            if remaining.is_empty() || attempts >= self.route.max_attempts {
                return Err(RouteError::AllTargetsExhausted { attempts });
            }

            // Pick the healthy target with the lowest EWMA from `remaining`.
            let Some(pos_in_remaining) = self.pick_lowest_latency(&remaining) else {
                return Err(RouteError::AllTargetsExhausted { attempts });
            };
            let idx = remaining[pos_in_remaining];
            let target = &self.route.targets[idx];

            if idx != 0 {
                used_fallback = true;
            }

            attempts += 1;
            let start = self.clock.now_ms();
            let result = executor.execute(target, request).await;
            let elapsed = (self.clock.now_ms() - start).max(0) as f64;

            match result {
                Ok(resp) => {
                    self.state.record_success(idx, elapsed);
                    let meta = RouterMeta {
                        target_index: idx,
                        provider_id: target.provider_id.clone(),
                        model: target.model.clone(),
                        used_fallback,
                        hedge_won: false,
                        attempt_count: attempts,
                    };
                    return Ok((resp, meta));
                }
                Err(err) => {
                    self.state.record_failure(
                        idx,
                        self.route.failure_threshold,
                        self.clock.as_ref(),
                    );

                    if !is_retryable(&err) {
                        return Err(RouteError::TerminalError(err));
                    }

                    // Remove this target from remaining after it has reached
                    // failure_threshold consecutive failures (i.e., in cooldown).
                    if self
                        .state
                        .is_in_cooldown(idx, self.route.cooldown_ms, self.clock.as_ref())
                    {
                        remaining.remove(pos_in_remaining);
                    }

                    self.sleep_for_error(&err, attempts).await;
                }
            }
        }
    }

    /// LLM-aware hedging: fire primary target; if it hasn't responded within
    /// `hedge_after_ms`, fire the secondary target; take whichever finishes
    /// first. Only fires for non-streaming (fallback only before first token).
    ///
    /// Falls back to normal failover if there's only one target.
    async fn call_with_hedge(
        &self,
        executor: &dyn TargetExecutor,
        request: &ChatRequest,
    ) -> Result<RouterResult, RouteError> {
        let hedge_delay = match self.route.hedge_after_ms {
            Some(d) => d,
            None => return self.call_failover_no_hedge(executor, request).await,
        };

        if self.route.targets.len() < 2 {
            return self.call_failover_no_hedge(executor, request).await;
        }

        let primary_target = self.route.targets[0].clone();
        let secondary_target = self.route.targets[1].clone();

        let primary_start = self.clock.now_ms();

        // Spawn primary call.
        // We use tokio::select! with a timeout for the hedge delay, then race.
        let primary_fut = async { executor.execute(&primary_target, request).await };

        let hedge_delay_dur = Duration::from_millis(hedge_delay);

        // Race: primary completes within hedge_delay → no hedge needed.
        let primary_result = timeout(hedge_delay_dur, primary_fut).await;

        match primary_result {
            Ok(Ok(resp)) => {
                // Primary succeeded before hedge fired.
                let elapsed = (self.clock.now_ms() - primary_start).max(0) as f64;
                self.state.record_success(0, elapsed);
                let meta = RouterMeta {
                    target_index: 0,
                    provider_id: primary_target.provider_id.clone(),
                    model: primary_target.model.clone(),
                    used_fallback: false,
                    hedge_won: false,
                    attempt_count: 1,
                };
                return Ok((resp, meta));
            }
            Ok(Err(err)) => {
                // Primary failed (within hedge window). Check if terminal.
                self.state
                    .record_failure(0, self.route.failure_threshold, self.clock.as_ref());
                if !is_retryable(&err) {
                    return Err(RouteError::TerminalError(err));
                }
                // Retryable: fall through to secondary.
            }
            Err(_timeout) => {
                // Primary is still running but too slow — fire the hedge.
                // We need to race them. Use tokio::select with new futures.
                // Re-execute primary (the previous future was dropped) and race
                // against secondary.
                // Note: idempotency key is reused so no double-billing.
            }
        }

        // Hedge or retry: race primary vs secondary.
        // For simplicity and correctness we spawn both and take the first.
        let p_target = primary_target.clone();
        let s_target = secondary_target.clone();

        // We need both executor calls to run concurrently. Since TargetExecutor
        // is not Clone, we use tokio::select on the two futures directly.
        let primary_retry = executor.execute(&p_target, request);
        let secondary_call = executor.execute(&s_target, request);

        tokio::select! {
            res = primary_retry => {
                match res {
                    Ok(resp) => {
                        let elapsed = (self.clock.now_ms() - primary_start).max(0) as f64;
                        self.state.record_success(0, elapsed);
                        let meta = RouterMeta {
                            target_index: 0,
                            provider_id: p_target.provider_id.clone(),
                            model: p_target.model.clone(),
                            used_fallback: false,
                            hedge_won: false,
                            attempt_count: 2,
                        };
                        Ok((resp, meta))
                    }
                    Err(err) => {
                        self.state.record_failure(0, self.route.failure_threshold, self.clock.as_ref());
                        // Try secondary directly.
                        let sec_start = self.clock.now_ms();
                        match executor.execute(&s_target, request).await {
                            Ok(resp) => {
                                let elapsed = (self.clock.now_ms() - sec_start).max(0) as f64;
                                self.state.record_success(1, elapsed);
                                let meta = RouterMeta {
                                    target_index: 1,
                                    provider_id: s_target.provider_id.clone(),
                                    model: s_target.model.clone(),
                                    used_fallback: true,
                                    hedge_won: true,
                                    attempt_count: 3,
                                };
                                Ok((resp, meta))
                            }
                            Err(sec_err) => {
                                self.state.record_failure(1, self.route.failure_threshold, self.clock.as_ref());
                                if is_retryable(&err) {
                                    Err(RouteError::AllTargetsExhausted { attempts: 3 })
                                } else {
                                    Err(RouteError::TerminalError(sec_err))
                                }
                            }
                        }
                    }
                }
            }
            res = secondary_call => {
                match res {
                    Ok(resp) => {
                        let elapsed = (self.clock.now_ms() - primary_start).max(0) as f64;
                        self.state.record_success(1, elapsed);
                        let meta = RouterMeta {
                            target_index: 1,
                            provider_id: s_target.provider_id.clone(),
                            model: s_target.model.clone(),
                            used_fallback: true,
                            hedge_won: true,
                            attempt_count: 2,
                        };
                        Ok((resp, meta))
                    }
                    Err(err) => {
                        self.state.record_failure(1, self.route.failure_threshold, self.clock.as_ref());
                        if !is_retryable(&err) {
                            return Err(RouteError::TerminalError(err));
                        }
                        Err(RouteError::AllTargetsExhausted { attempts: 2 })
                    }
                }
            }
        }
    }

    /// Failover without hedge (called when hedge is not configured).
    async fn call_failover_no_hedge(
        &self,
        executor: &dyn TargetExecutor,
        request: &ChatRequest,
    ) -> Result<RouterResult, RouteError> {
        let mut attempts = 0u32;
        let mut used_fallback = false;

        for (idx, target) in self.route.targets.iter().enumerate() {
            if !self
                .state
                .maybe_recover(idx, self.route.cooldown_ms, self.clock.as_ref())
            {
                continue;
            }

            if idx > 0 {
                used_fallback = true;
            }

            let mut target_attempts = 0u32;
            loop {
                if attempts >= self.route.max_attempts {
                    return Err(RouteError::AllTargetsExhausted { attempts });
                }
                attempts += 1;
                target_attempts += 1;

                let start = self.clock.now_ms();
                let result = executor.execute(target, request).await;
                let elapsed = (self.clock.now_ms() - start).max(0) as f64;

                match result {
                    Ok(resp) => {
                        self.state.record_success(idx, elapsed);
                        let meta = RouterMeta {
                            target_index: idx,
                            provider_id: target.provider_id.clone(),
                            model: target.model.clone(),
                            used_fallback,
                            hedge_won: false,
                            attempt_count: attempts,
                        };
                        return Ok((resp, meta));
                    }
                    Err(err) => {
                        self.state.record_failure(
                            idx,
                            self.route.failure_threshold,
                            self.clock.as_ref(),
                        );

                        if !is_retryable(&err) {
                            return Err(RouteError::TerminalError(err));
                        }

                        if target_attempts >= self.route.failure_threshold {
                            break;
                        }

                        self.sleep_for_error(&err, target_attempts).await;
                    }
                }
            }
        }

        Err(RouteError::AllTargetsExhausted { attempts })
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    /// Sleep according to the error type: honour `retry_after_secs` for
    /// RateLimited; use exponential backoff for other retryable errors.
    async fn sleep_for_error(&self, err: &ProviderError, attempt: u32) {
        let delay_ms = match err {
            ProviderError::RateLimited {
                retry_after_secs: Some(secs),
            } => {
                let capped = (*secs as u64).min(MAX_RATE_LIMIT_SLEEP_SECS);
                capped * 1000
            }
            _ => backoff_ms(
                attempt.saturating_sub(1),
                self.route.base_backoff_ms,
                self.route.max_backoff_ms,
            ),
        };

        if delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        }
    }

    /// Weighted random selection: pick one healthy target proportional to
    /// its weight. Returns `(index, &RouteTarget)` or `None` if all targets
    /// are in cooldown / have zero weight.
    fn pick_weighted(&self) -> Option<(usize, &RouteTarget)> {
        let weights: Vec<f64> = self
            .route
            .targets
            .iter()
            .enumerate()
            .map(|(i, t)| {
                if self
                    .state
                    .maybe_recover(i, self.route.cooldown_ms, self.clock.as_ref())
                    && !self
                        .state
                        .is_in_cooldown(i, self.route.cooldown_ms, self.clock.as_ref())
                {
                    t.effective_weight()
                } else {
                    0.0
                }
            })
            .collect();

        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return None;
        }

        let mut sample = rand::thread_rng().r#gen::<f64>() * total;
        for (i, w) in weights.iter().enumerate() {
            sample -= w;
            if sample <= 0.0 {
                return Some((i, &self.route.targets[i]));
            }
        }
        // Floating-point edge case: return the last non-zero target.
        weights
            .iter()
            .enumerate()
            .rev()
            .find(|&(_, &w)| w > 0.0)
            .map(|(i, _)| (i, &self.route.targets[i]))
    }

    /// Among the `remaining` indices, pick the one with the lowest EWMA
    /// latency that is not in cooldown. Targets with no EWMA data get an
    /// optimistic estimate of 0 ms (explored first so they can build history).
    /// Returns the position within `remaining`.
    fn pick_lowest_latency(&self, remaining: &[usize]) -> Option<usize> {
        let mut best_pos: Option<usize> = None;
        let mut best_latency = f64::MAX;

        for (pos, &idx) in remaining.iter().enumerate() {
            if !self
                .state
                .maybe_recover(idx, self.route.cooldown_ms, self.clock.as_ref())
            {
                continue; // skip targets still in cooldown
            }
            // Use EWMA if available; new targets get an optimistic 0 so they
            // are tried first and can build their latency history.
            let lat = self.state.ewma_ms(idx).unwrap_or(0.0);
            if lat < best_latency {
                best_latency = lat;
                best_pos = Some(pos);
            }
        }

        best_pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU32, Ordering};

    use async_trait::async_trait;
    use gateway_llm::{ChatRequest, ChatResponse, FinishReason, Message, Role};
    use gateway_spine::{MockClock, TokenUsage};

    use crate::route::{Route, RouteTarget, Strategy};

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_response(model: &str) -> ChatResponse {
        use gateway_llm::message::ContentPart;
        ChatResponse {
            model: model.to_string(),
            content: vec![ContentPart::text("ok")],
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
            usage: TokenUsage::default(),
            provider_response_id: None,
        }
    }

    fn make_request() -> ChatRequest {
        ChatRequest::new("gpt-4o", vec![Message::text(Role::User, "hi")])
    }

    // ── Mock executor ─────────────────────────────────────────────────────────

    /// A mock executor that maps provider_id → a canned sequence of results.
    struct MockExecutor {
        results: Mutex<HashMap<String, Vec<Result<ChatResponse, ProviderError>>>>,
        call_count: AtomicU32,
    }

    impl MockExecutor {
        fn new() -> Self {
            MockExecutor {
                results: Mutex::new(HashMap::new()),
                call_count: AtomicU32::new(0),
            }
        }

        fn add_results(
            &self,
            provider_id: &str,
            results: Vec<Result<ChatResponse, ProviderError>>,
        ) {
            self.results
                .lock()
                .unwrap()
                .entry(provider_id.to_string())
                .or_default()
                .extend(results);
        }

        fn call_count(&self) -> u32 {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl TargetExecutor for MockExecutor {
        async fn execute(
            &self,
            target: &RouteTarget,
            _request: &ChatRequest,
        ) -> Result<ChatResponse, ProviderError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let mut map = self.results.lock().unwrap();
            let queue = map.entry(target.provider_id.clone()).or_default();
            if queue.is_empty() {
                // Default: success
                return Ok(make_response(&target.model));
            }
            queue.remove(0)
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// Single strategy: succeeds on first call.
    #[tokio::test]
    async fn single_success() {
        let clock = Arc::new(MockClock::new(0));
        let route = Route::single("openai", "gpt-4o");
        let router = Router::new(route, clock);
        let executor = MockExecutor::new();
        let req = make_request();
        let (resp, meta) = router.call(&executor, &req).await.unwrap();
        assert_eq!(resp.model, "gpt-4o");
        assert_eq!(meta.attempt_count, 1);
        assert!(!meta.used_fallback);
        assert!(!meta.hedge_won);
    }

    /// No targets → NoTargets error.
    #[tokio::test]
    async fn no_targets_returns_error() {
        let clock = Arc::new(MockClock::new(0));
        let route = Route::new(vec![], Strategy::Failover);
        let router = Router::new(route, clock);
        let executor = MockExecutor::new();
        let req = make_request();
        let err = router.call(&executor, &req).await.unwrap_err();
        assert!(matches!(err, RouteError::NoTargets));
    }

    /// Failover: first target fails with retryable, second succeeds.
    #[tokio::test]
    async fn failover_on_retryable_error() {
        let clock = Arc::new(MockClock::new(0));
        let mut route = Route::new(
            vec![
                RouteTarget::new("primary", "gpt-4o"),
                RouteTarget::new("backup", "gpt-4o-mini"),
            ],
            Strategy::Failover,
        );
        route.max_attempts = 5;
        route.failure_threshold = 1;
        route.base_backoff_ms = 0; // no sleep in tests

        let router = Router::new(route, clock);
        let executor = MockExecutor::new();

        // Primary fails with transport error (retryable).
        executor.add_results(
            "primary",
            vec![Err(ProviderError::Transport("timeout".into()))],
        );

        let req = make_request();
        let (resp, meta) = router.call(&executor, &req).await.unwrap();

        assert_eq!(resp.model, "gpt-4o-mini");
        assert!(meta.used_fallback);
        assert_eq!(meta.target_index, 1);
    }

    /// Terminal error stops immediately without trying other targets.
    #[tokio::test]
    async fn terminal_error_stops_immediately() {
        let clock = Arc::new(MockClock::new(0));
        let mut route = Route::new(
            vec![
                RouteTarget::new("primary", "gpt-4o"),
                RouteTarget::new("backup", "gpt-4o-mini"),
            ],
            Strategy::Failover,
        );
        route.failure_threshold = 1;
        route.base_backoff_ms = 0;

        let router = Router::new(route, clock);
        let executor = MockExecutor::new();
        executor.add_results("primary", vec![Err(ProviderError::Auth)]);

        let req = make_request();
        let err = router.call(&executor, &req).await.unwrap_err();
        assert!(matches!(
            err,
            RouteError::TerminalError(ProviderError::Auth)
        ));
        // Backup should NOT have been called.
        assert_eq!(executor.call_count(), 1);
    }

    /// Cooldown: after failure_threshold failures the target goes in cooldown;
    /// auto-recovers after the window elapses.
    #[tokio::test]
    async fn cooldown_opens_then_recovers() {
        use crate::strategy::TargetState;
        let clock = MockClock::new(0);

        let mut state = TargetState::new();
        // Threshold = 3: first 2 failures don't trip it.
        assert!(!state.record_failure(3, &clock));
        assert!(!state.record_failure(3, &clock));
        // Third failure trips cooldown.
        assert!(state.record_failure(3, &clock));
        assert!(state.is_in_cooldown(30_000, &clock));

        // Before window: still in cooldown.
        clock.advance(29_999);
        assert!(state.is_in_cooldown(30_000, &clock));

        // After window: auto-recovers.
        clock.advance(1);
        assert!(!state.is_in_cooldown(30_000, &clock));
        assert!(state.maybe_recover(30_000, &clock));
    }

    /// Weighted distribution: with weights [1, 9], target 1 should win ~90%.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn weighted_distribution_roughly_correct() {
        let clock = Arc::new(MockClock::new(0));
        let route = Route::new(
            vec![
                RouteTarget::new("low", "model-a").with_weight(1.0),
                RouteTarget::new("high", "model-b").with_weight(9.0),
            ],
            Strategy::Weighted,
        );

        let router = Router::new(route, clock);
        let executor = MockExecutor::new();
        let req = make_request();

        let mut counts = [0u32; 2];
        let n = 200;
        for _ in 0..n {
            let (_, meta) = router.call(&executor, &req).await.unwrap();
            counts[meta.target_index] += 1;
        }

        // With weights 1:9 we expect ~20 for target 0 and ~180 for target 1.
        // Allow wide tolerance (±20%) for a random test.
        let frac_high = counts[1] as f64 / n as f64;
        assert!(
            frac_high > 0.70 && frac_high < 0.98,
            "expected ~90% for high-weight target, got {:.1}%",
            frac_high * 100.0
        );
    }

    /// Failover with all targets in cooldown returns AllTargetsExhausted.
    #[tokio::test]
    async fn failover_all_in_cooldown_exhausted() {
        let clock = Arc::new(MockClock::new(0));
        let mut route = Route::new(
            vec![RouteTarget::new("a", "m1"), RouteTarget::new("b", "m2")],
            Strategy::Failover,
        );
        route.failure_threshold = 1;
        route.cooldown_ms = 60_000;
        route.max_attempts = 10;
        route.base_backoff_ms = 0;

        let router = Router::new(route, Arc::clone(&clock));
        let executor = MockExecutor::new();

        // Pre-seed both into cooldown by failing them.
        executor.add_results(
            "a",
            vec![
                Err(ProviderError::Transport("fail".into())),
                Err(ProviderError::Transport("fail".into())),
            ],
        );
        executor.add_results(
            "b",
            vec![
                Err(ProviderError::Transport("fail".into())),
                Err(ProviderError::Transport("fail".into())),
            ],
        );

        let req = make_request();
        // First call will try both and fail.
        let _ = router.call(&executor, &req).await;

        // Now both are in cooldown — subsequent call should get Exhausted quickly.
        let err = router.call(&executor, &req).await.unwrap_err();
        assert!(matches!(err, RouteError::AllTargetsExhausted { .. }));
    }

    /// Hedging: slow primary + fast secondary → secondary wins.
    #[tokio::test]
    async fn hedging_fast_secondary_wins() {
        let clock = Arc::new(MockClock::new(0));
        let mut route = Route::new(
            vec![
                RouteTarget::new("slow", "model-slow"),
                RouteTarget::new("fast", "model-fast"),
            ],
            Strategy::Failover,
        );
        // Hedge fires after 10ms; slow primary takes 200ms.
        route.hedge_after_ms = Some(10);
        route.max_attempts = 3;
        route.base_backoff_ms = 0;
        route.failure_threshold = 3;

        let router = Router::new(route, clock);

        // slow = takes 200ms; fast = instant.
        struct HedgeExecutor;
        #[async_trait]
        impl TargetExecutor for HedgeExecutor {
            async fn execute(
                &self,
                target: &RouteTarget,
                _request: &ChatRequest,
            ) -> Result<ChatResponse, ProviderError> {
                if target.provider_id == "slow" {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
                Ok(make_response(&target.model))
            }
        }

        let req = make_request();
        let (resp, meta) = router.call(&HedgeExecutor, &req).await.unwrap();

        // The fast secondary should have won.
        assert_eq!(resp.model, "model-fast");
        assert!(meta.hedge_won, "expected hedge_won to be true");
        assert!(meta.used_fallback, "expected used_fallback to be true");
    }

    /// Latency-aware: after seeding EWMA, picks the faster target.
    #[tokio::test]
    async fn latency_aware_picks_lowest_ewma() {
        // We can't inject clock into the EWMA measurement directly since the
        // router uses clock.now_ms() for wall-clock elapsed. Instead we use the
        // real executor with fixed latencies and check target selection
        // indirectly via many calls.
        let clock = Arc::new(MockClock::new(0));
        let route = Route::new(
            vec![
                RouteTarget::new("slow", "model-slow"),
                RouteTarget::new("fast", "model-fast"),
            ],
            Strategy::LatencyAware,
        );

        let router = Router::new(route, Arc::clone(&clock));

        // Seed EWMA by manually advancing the clock between sub-calls.
        // We reach into `state` indirectly: call the router with a timed mock
        // that advances the clock. However, the router calls clock.now_ms()
        // before and after executor.execute, so we can control latency by
        // advancing the clock *inside* the executor.

        struct ClockAdvancingExecutor {
            clock: Arc<MockClock>,
        }
        #[async_trait]
        impl TargetExecutor for ClockAdvancingExecutor {
            async fn execute(
                &self,
                target: &RouteTarget,
                _request: &ChatRequest,
            ) -> Result<ChatResponse, ProviderError> {
                if target.provider_id == "slow" {
                    self.clock.advance(100); // simulate 100ms latency
                } else {
                    self.clock.advance(5); // simulate 5ms latency
                }
                Ok(make_response(&target.model))
            }
        }

        let executor = ClockAdvancingExecutor {
            clock: Arc::clone(&clock),
        };
        let req = make_request();

        // Warm up latency estimates: alternate to force both targets to get data.
        // First call: no data, picks "slow" (first in list); second call: slow
        // has data, fast has none — so we need to seed fast too.
        // Actually LatencyAware picks first available target until all have data;
        // let's run several calls and check the final distribution.
        let mut slow_count = 0u32;
        let mut fast_count = 0u32;
        for _ in 0..20 {
            let (_, meta) = router.call(&executor, &req).await.unwrap();
            if meta.provider_id == "slow" {
                slow_count += 1;
            } else {
                fast_count += 1;
            }
        }

        // After warming up, fast (5ms EWMA) should be picked more than slow (100ms).
        // We allow the first few calls to "slow" (no data → picks first).
        assert!(
            fast_count > slow_count,
            "expected fast to be picked more; fast={fast_count} slow={slow_count}"
        );
    }

    /// Rate-limited retry: router honours retry_after_secs hint.
    /// Uses very small values so the test doesn't sleep long.
    #[tokio::test]
    async fn rate_limited_retry_succeeds() {
        let clock = Arc::new(MockClock::new(0));
        let mut route = Route::single("openai", "gpt-4o");
        route.max_attempts = 3;
        route.base_backoff_ms = 0;

        let router = Router::new(route, clock);
        let executor = MockExecutor::new();
        executor.add_results(
            "openai",
            vec![
                Err(ProviderError::RateLimited {
                    retry_after_secs: Some(0), // instant retry in tests
                }),
                Ok(make_response("gpt-4o")),
            ],
        );

        let req = make_request();
        let (resp, meta) = router.call(&executor, &req).await.unwrap();
        assert_eq!(resp.model, "gpt-4o");
        assert_eq!(meta.attempt_count, 2);
    }

    /// Upstream 4xx is terminal (non-retryable).
    #[tokio::test]
    async fn upstream_4xx_is_terminal() {
        let clock = Arc::new(MockClock::new(0));
        let route = Route::single("openai", "gpt-4o");
        let router = Router::new(route, clock);
        let executor = MockExecutor::new();
        executor.add_results(
            "openai",
            vec![Err(ProviderError::Upstream {
                status: 400,
                body: "bad request".into(),
            })],
        );

        let req = make_request();
        let err = router.call(&executor, &req).await.unwrap_err();
        assert!(matches!(
            err,
            RouteError::TerminalError(ProviderError::Upstream { status: 400, .. })
        ));
        assert_eq!(executor.call_count(), 1);
    }

    /// Upstream 5xx is retryable.
    #[tokio::test]
    async fn upstream_5xx_is_retryable() {
        let clock = Arc::new(MockClock::new(0));
        let mut route = Route::single("openai", "gpt-4o");
        route.max_attempts = 3;
        route.base_backoff_ms = 0;

        let router = Router::new(route, clock);
        let executor = MockExecutor::new();
        executor.add_results(
            "openai",
            vec![
                Err(ProviderError::Upstream {
                    status: 500,
                    body: "internal server error".into(),
                }),
                Ok(make_response("gpt-4o")),
            ],
        );

        let req = make_request();
        let (_, meta) = router.call(&executor, &req).await.unwrap();
        assert_eq!(meta.attempt_count, 2);
    }
}
