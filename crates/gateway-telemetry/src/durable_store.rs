//! Durable, crash-safe `SpendStore` backed by SQLite.
//!
//! `MemorySpendStore` is correct but process-local: request-log rows disappear
//! on restart. This module adds a durable implementation behind the existing
//! `SpendStore` trait without changing callers.
//!
//! `SpendStore` is synchronous by design, so this implementation uses
//! `rusqlite` instead of `sqlx`. Telemetry remains fail-open: write/query
//! errors are logged and do not propagate into request handling.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;

use gateway_spine::{TokenUsage, Usd};
use rusqlite::{Connection, params};
use sha2::{Digest, Sha256};

use crate::row::{CacheStatus, CaptureMode, RequestKind, RequestLogRow};
use crate::store::{GroupBy, SpendBucket, SpendStore, TimeRange};

const SCHEMA_SQL: &str = include_str!("durable_store_schema.sql");

#[derive(Debug, thiserror::Error)]
pub enum DurableStoreError {
    #[error("failed to open database at {path}: {source}")]
    Open {
        path: String,
        #[source]
        source: rusqlite::Error,
    },
    #[error("failed to initialize durable spend store schema: {0}")]
    Schema(#[from] rusqlite::Error),
}

pub struct SqliteSpendStore {
    conn: Mutex<Connection>,
}

impl SqliteSpendStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, DurableStoreError> {
        let path = path.as_ref();
        let conn = Connection::open(path).map_err(|source| DurableStoreError::Open {
            path: path.display().to_string(),
            source,
        })?;
        Self::init(conn)
    }

    fn init(conn: Connection) -> Result<Self, DurableStoreError> {
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.busy_timeout(std::time::Duration::from_millis(5_000))?;
        conn.execute_batch(SCHEMA_SQL)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn row_uid(row: &RequestLogRow) -> String {
        let encoded =
            serde_json::to_vec(row).expect("serializing RequestLogRow for row_uid cannot fail");
        let mut h = Sha256::new();
        h.update(encoded);
        hex::encode(h.finalize())
    }
}

fn kind_to_str(kind: RequestKind) -> &'static str {
    match kind {
        RequestKind::Llm => "llm",
        RequestKind::McpTool => "mcp_tool",
    }
}

fn kind_from_str(value: &str) -> RequestKind {
    match value {
        "mcp_tool" => RequestKind::McpTool,
        _ => RequestKind::Llm,
    }
}

fn cache_status_to_str(status: CacheStatus) -> &'static str {
    match status {
        CacheStatus::Hit => "hit",
        CacheStatus::Miss => "miss",
        CacheStatus::Bypass => "bypass",
    }
}

fn cache_status_from_str(value: &str) -> CacheStatus {
    match value {
        "hit" => CacheStatus::Hit,
        "bypass" => CacheStatus::Bypass,
        _ => CacheStatus::Miss,
    }
}

fn capture_mode_to_str(mode: CaptureMode) -> &'static str {
    match mode {
        CaptureMode::Full => "full",
        CaptureMode::Metadata => "metadata",
    }
}

fn capture_mode_from_str(value: &str) -> CaptureMode {
    match value {
        "full" => CaptureMode::Full,
        _ => CaptureMode::Metadata,
    }
}

impl SpendStore for SqliteSpendStore {
    fn append(&self, row: RequestLogRow) {
        let row_uid = Self::row_uid(&row);
        let tags_json = serde_json::to_string(&row.tags).unwrap_or_else(|_| "[]".to_string());
        let conn = self.conn.lock().unwrap_or_else(|p| p.into_inner());

        let result = conn.execute(
            "INSERT OR IGNORE INTO request_log (
                row_uid, ts_ms, kind, key_id, team_id, user_id, tags, model, provider,
                input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                cost_micros, latency_ms, ttft_ms, status, served_by, fallback_fired,
                cache_status, capture_mode, request_text, response_text
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13,
                ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23
            )",
            params![
                row_uid,
                row.ts_ms,
                kind_to_str(row.kind),
                row.key_id.as_str(),
                row.team_id.as_deref(),
                row.user_id.as_deref(),
                tags_json,
                row.model.as_str(),
                row.provider.as_str(),
                row.usage.input_tokens,
                row.usage.output_tokens,
                row.usage.cache_read_tokens,
                row.usage.cache_write_tokens,
                row.cost.micros(),
                row.latency_ms,
                row.ttft_ms,
                row.status as i64,
                row.served_by.as_str(),
                row.fallback_fired as i64,
                cache_status_to_str(row.cache_status),
                capture_mode_to_str(row.capture_mode),
                row.request_text.as_deref(),
                row.response_text.as_deref(),
            ],
        );

        if let Err(error) = result {
            tracing::error!(
                %error,
                key_id = %row.key_id,
                model = %row.model,
                "durable spend-store write failed; telemetry is fail-open"
            );
        }
    }

    fn query(
        &self,
        group_by: GroupBy,
        range: TimeRange,
        tag_filter: Option<&str>,
    ) -> Vec<SpendBucket> {
        let conn = self.conn.lock().unwrap_or_else(|p| p.into_inner());
        let mut stmt = match conn.prepare(
            "SELECT key_id, team_id, user_id, tags, model,
                    input_tokens, output_tokens, cost_micros
             FROM request_log
             WHERE (?1 IS NULL OR ts_ms >= ?1)
               AND (?2 IS NULL OR ts_ms < ?2)",
        ) {
            Ok(stmt) => stmt,
            Err(error) => {
                tracing::error!(%error, "durable spend-store query prepare failed");
                return Vec::new();
            }
        };

        let rows = match stmt.query_map(params![range.since_ms, range.until_ms], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, i64>(6)?,
                row.get::<_, i64>(7)?,
            ))
        }) {
            Ok(rows) => rows,
            Err(error) => {
                tracing::error!(%error, "durable spend-store query failed");
                return Vec::new();
            }
        };

        let mut acc: BTreeMap<String, SpendBucket> = BTreeMap::new();
        for row in rows {
            let (
                key_id,
                team_id,
                user_id,
                tags_json,
                model,
                input_tokens,
                output_tokens,
                cost_micros,
            ) = match row {
                Ok(row) => row,
                Err(error) => {
                    tracing::error!(%error, "durable spend-store row decode failed");
                    continue;
                }
            };

            let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
            if let Some(tag) = tag_filter
                && !tags.iter().any(|existing| existing == tag)
            {
                continue;
            }

            let group_keys = match group_by {
                GroupBy::Key => vec![key_id],
                GroupBy::Team => vec![team_id.unwrap_or_else(|| "(none)".to_string())],
                GroupBy::User => vec![user_id.unwrap_or_else(|| "(none)".to_string())],
                GroupBy::Model => vec![model],
                GroupBy::Tag => {
                    if tags.is_empty() {
                        vec!["(untagged)".to_string()]
                    } else {
                        tags
                    }
                }
            };

            for key in group_keys {
                let bucket = acc.entry(key.clone()).or_insert_with(|| SpendBucket {
                    group: key,
                    requests: 0,
                    input_tokens: 0,
                    output_tokens: 0,
                    cost: Usd::ZERO,
                });
                bucket.requests += 1;
                bucket.input_tokens += input_tokens;
                bucket.output_tokens += output_tokens;
                bucket.cost += Usd::from_micros(cost_micros);
            }
        }

        acc.into_values().collect()
    }

    fn recent(&self, range: TimeRange, limit: usize) -> Vec<RequestLogRow> {
        let conn = self.conn.lock().unwrap_or_else(|p| p.into_inner());
        let mut stmt = match conn.prepare(
            "SELECT ts_ms, kind, key_id, team_id, user_id, tags, model, provider,
                    input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                    cost_micros, latency_ms, ttft_ms, status, served_by, fallback_fired,
                    cache_status, capture_mode, request_text, response_text
             FROM request_log
             WHERE (?1 IS NULL OR ts_ms >= ?1)
               AND (?2 IS NULL OR ts_ms < ?2)
             ORDER BY ts_ms DESC, seq DESC
             LIMIT ?3",
        ) {
            Ok(stmt) => stmt,
            Err(error) => {
                tracing::error!(%error, "durable spend-store recent prepare failed");
                return Vec::new();
            }
        };

        let rows =
            match stmt.query_map(params![range.since_ms, range.until_ms, limit as i64], |r| {
                let tags_json: String = r.get(5)?;
                Ok(RequestLogRow {
                    ts_ms: r.get(0)?,
                    kind: kind_from_str(&r.get::<_, String>(1)?),
                    key_id: r.get(2)?,
                    team_id: r.get(3)?,
                    user_id: r.get(4)?,
                    tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                    model: r.get(6)?,
                    provider: r.get(7)?,
                    usage: TokenUsage {
                        input_tokens: r.get(8)?,
                        output_tokens: r.get(9)?,
                        cache_read_tokens: r.get(10)?,
                        cache_write_tokens: r.get(11)?,
                    },
                    cost: Usd::from_micros(r.get(12)?),
                    latency_ms: r.get(13)?,
                    ttft_ms: r.get(14)?,
                    status: r.get::<_, i64>(15)? as u16,
                    served_by: r.get(16)?,
                    fallback_fired: r.get::<_, i64>(17)? != 0,
                    cache_status: cache_status_from_str(&r.get::<_, String>(18)?),
                    capture_mode: capture_mode_from_str(&r.get::<_, String>(19)?),
                    request_text: r.get(20)?,
                    response_text: r.get(21)?,
                })
            }) {
                Ok(rows) => rows,
                Err(error) => {
                    tracing::error!(%error, "durable spend-store recent query failed");
                    return Vec::new();
                }
            };

        let mut out = Vec::new();
        for row in rows {
            match row {
                Ok(row) => out.push(row),
                Err(error) => {
                    tracing::error!(%error, "durable spend-store recent row decode failed");
                }
            }
        }
        out
    }

    fn row_count(&self) -> usize {
        let conn = self.conn.lock().unwrap_or_else(|p| p.into_inner());
        conn.query_row("SELECT COUNT(*) FROM request_log", [], |row| {
            row.get::<_, i64>(0)
        })
        .map(|count| count as usize)
        .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::MemorySpendStore;

    fn row(
        ts: i64,
        key: &str,
        team: &str,
        model: &str,
        cost_micros: i64,
        tags: &[&str],
    ) -> RequestLogRow {
        RequestLogRow {
            ts_ms: ts,
            kind: RequestKind::Llm,
            key_id: key.to_string(),
            team_id: Some(team.to_string()),
            user_id: Some("user_1".to_string()),
            tags: tags.iter().map(|tag| tag.to_string()).collect(),
            model: model.to_string(),
            provider: "openai".to_string(),
            usage: TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                cache_read_tokens: 0,
                cache_write_tokens: 0,
            },
            cost: Usd::from_micros(cost_micros),
            latency_ms: 120,
            ttft_ms: Some(40),
            status: 200,
            served_by: "openai/gpt-4o".to_string(),
            fallback_fired: false,
            cache_status: CacheStatus::Miss,
            capture_mode: CaptureMode::Metadata,
            request_text: None,
            response_text: None,
        }
    }

    #[test]
    fn append_and_row_count() {
        let store = SqliteSpendStore::open(":memory:").unwrap();
        assert_eq!(store.row_count(), 0);
        store.append(row(1, "k1", "t1", "gpt-4o", 100, &[]));
        store.append(row(2, "k1", "t1", "gpt-4o", 200, &[]));
        assert_eq!(store.row_count(), 2);
    }

    #[test]
    fn durability_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("telemetry.db");

        {
            let store = SqliteSpendStore::open(&path).unwrap();
            store.append(row(1, "k1", "t1", "gpt-4o", 7_500, &["prod"]));
            store.append(row(2, "k1", "t1", "gpt-4o", 2_500, &["prod"]));
            assert_eq!(store.row_count(), 2);
        }

        let reopened = SqliteSpendStore::open(&path).unwrap();
        assert_eq!(reopened.row_count(), 2);
        let recent = reopened.recent(TimeRange::default(), 10);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].ts_ms, 2);
        assert_eq!(recent[1].ts_ms, 1);
    }

    #[test]
    fn duplicate_append_is_idempotent() {
        let store = SqliteSpendStore::open(":memory:").unwrap();
        let request_row = row(1, "k1", "t1", "gpt-4o", 100, &[]);
        store.append(request_row.clone());
        store.append(request_row);
        assert_eq!(store.row_count(), 1);
    }

    #[test]
    fn distinct_rows_are_not_collapsed() {
        let store = SqliteSpendStore::open(":memory:").unwrap();
        store.append(row(1, "k1", "t1", "gpt-4o", 100, &[]));
        store.append(row(1, "k1", "t1", "gpt-4o", 200, &[]));
        assert_eq!(store.row_count(), 2);
    }

    #[test]
    fn query_group_by_key_matches_memory_store() {
        let memory = MemorySpendStore::new();
        let durable = SqliteSpendStore::open(":memory:").unwrap();

        let rows = vec![
            row(1, "k1", "t1", "gpt-4o", 100, &[]),
            row(2, "k1", "t1", "gpt-4o", 200, &[]),
            row(3, "k2", "t2", "claude", 50, &[]),
        ];

        for request_row in rows {
            memory.append(request_row.clone());
            durable.append(request_row);
        }

        let mut memory_buckets = memory.query(GroupBy::Key, TimeRange::default(), None);
        let mut durable_buckets = durable.query(GroupBy::Key, TimeRange::default(), None);
        memory_buckets.sort_by(|a, b| a.group.cmp(&b.group));
        durable_buckets.sort_by(|a, b| a.group.cmp(&b.group));

        assert_eq!(memory_buckets, durable_buckets);
    }

    #[test]
    fn query_group_by_tag_explodes_multi_tag_rows() {
        let store = SqliteSpendStore::open(":memory:").unwrap();
        store.append(row(1, "k1", "t1", "gpt-4o", 100, &["prod", "agent"]));
        store.append(row(2, "k1", "t1", "gpt-4o", 100, &["prod"]));

        let mut buckets = store.query(GroupBy::Tag, TimeRange::default(), None);
        buckets.sort_by(|a, b| a.group.cmp(&b.group));

        assert_eq!(buckets.len(), 2);
        assert_eq!(buckets[0].group, "agent");
        assert_eq!(buckets[0].cost, Usd::from_micros(100));
        assert_eq!(buckets[1].group, "prod");
        assert_eq!(buckets[1].cost, Usd::from_micros(200));
    }

    #[test]
    fn query_time_range_is_half_open() {
        let store = SqliteSpendStore::open(":memory:").unwrap();
        store.append(row(10, "k", "t", "m", 1, &[]));
        store.append(row(20, "k", "t", "m", 1, &[]));
        store.append(row(30, "k", "t", "m", 1, &[]));

        let range = TimeRange {
            since_ms: Some(20),
            until_ms: Some(30),
        };
        let buckets = store.query(GroupBy::Key, range, None);
        assert_eq!(buckets[0].requests, 1);
    }

    #[test]
    fn query_tag_filter_restricts_rows() {
        let store = SqliteSpendStore::open(":memory:").unwrap();
        store.append(row(1, "k1", "t1", "gpt-4o", 100, &["prod"]));
        store.append(row(2, "k2", "t1", "gpt-4o", 100, &["dev"]));

        let buckets = store.query(GroupBy::Key, TimeRange::default(), Some("prod"));
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].group, "k1");
    }

    #[test]
    fn recent_returns_newest_first_and_respects_limit() {
        let store = SqliteSpendStore::open(":memory:").unwrap();
        store.append(row(1, "k", "t", "m", 1, &[]));
        store.append(row(2, "k", "t", "m", 1, &[]));
        store.append(row(3, "k", "t", "m", 1, &[]));

        let recent = store.recent(TimeRange::default(), 2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].ts_ms, 3);
        assert_eq!(recent[1].ts_ms, 2);
    }

    #[test]
    fn recent_uses_insert_order_for_same_timestamp() {
        let store = SqliteSpendStore::open(":memory:").unwrap();
        store.append(row(1, "k", "t", "first", 1, &[]));
        store.append(row(1, "k", "t", "second", 1, &[]));

        let recent = store.recent(TimeRange::default(), 2);
        assert_eq!(recent[0].model, "second");
        assert_eq!(recent[1].model, "first");
    }

    #[test]
    fn full_capture_mode_text_roundtrips() {
        let store = SqliteSpendStore::open(":memory:").unwrap();
        let mut request_row = row(1, "k1", "t1", "gpt-4o", 100, &[]);
        request_row.capture_mode = CaptureMode::Full;
        request_row.request_text = Some("hello".to_string());
        request_row.response_text = Some("hi".to_string());

        store.append(request_row);

        let back = store.recent(TimeRange::default(), 1);
        assert_eq!(back[0].capture_mode, CaptureMode::Full);
        assert_eq!(back[0].request_text.as_deref(), Some("hello"));
        assert_eq!(back[0].response_text.as_deref(), Some("hi"));
    }
}
