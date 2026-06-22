//! # gateway-telemetry
//!
//! Async, off-hot-path request-log + spend store; OTel GenAI/MCP semconv span
//! emit; authenticated Prometheus; optional Oximy OTEL/ClickHouse export adapter
//! (default-off). Owns the invariants: telemetry never blocks/fails a request,
//! `/metrics` is auth-by-default, content capture is fail-safe metadata-only,
//! and the standalone posture is preserved (no external export unless opted in).
//!
//! Part of [Oximy Gateway](https://github.com/oximyhq/gateway). See
//! `docs/2026-06-10-oximy-gateway-design.md` (§3, §9) and `docs/plans/`.

#![forbid(unsafe_code)]

pub mod durable_store;
pub mod export;
pub mod headers;
pub mod metrics;
pub mod otel;
pub mod policy;
pub mod prom;
pub mod row;
pub mod sink;
pub mod store;

pub use durable_store::{DurableStoreError, SqliteSpendStore};
pub use export::{ExportConfig, Exporter, NoopExporter, build_exporter};
pub use headers::{
    CACHE_HEADER, COST_HEADER, FALLBACK_HEADER, OVERHEAD_HEADER, SERVED_BY_HEADER, cost_usd_string,
    overhead_ms_string,
};
pub use metrics::GatewayMetrics;
pub use otel::{AttrValue, SpanAttr, span_attrs, span_name};
pub use policy::{CapturePolicy, GlobalCapture, RequestCapturePref};
pub use prom::{METRICS_CONTENT_TYPE, MetricsEndpoint, MetricsResponse};
pub use row::{CacheStatus, CaptureMode, RequestKind, RequestLogRow};
pub use sink::{
    DEFAULT_CHANNEL_CAPACITY, TelemetrySink, TelemetryWriter, spawn, spawn_with_exporter,
};
pub use store::{GroupBy, MemorySpendStore, SpendBucket, SpendStore, TimeRange};
