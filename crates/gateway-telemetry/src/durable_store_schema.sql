-- Durable request-log table for SqliteSpendStore.
-- Append-only by convention: this crate inserts rows with INSERT OR IGNORE,
-- keyed by row_uid, and never updates or deletes request-log rows.

CREATE TABLE IF NOT EXISTS request_log (
    seq                INTEGER PRIMARY KEY AUTOINCREMENT,
    row_uid            TEXT NOT NULL UNIQUE,
    ts_ms              INTEGER NOT NULL,
    kind               TEXT NOT NULL,
    key_id             TEXT NOT NULL,
    team_id            TEXT,
    user_id            TEXT,
    tags               TEXT NOT NULL,
    model              TEXT NOT NULL,
    provider           TEXT NOT NULL,
    input_tokens       INTEGER NOT NULL,
    output_tokens      INTEGER NOT NULL,
    cache_read_tokens  INTEGER NOT NULL,
    cache_write_tokens INTEGER NOT NULL,
    cost_micros        INTEGER NOT NULL,
    latency_ms         INTEGER NOT NULL,
    ttft_ms            INTEGER,
    status             INTEGER NOT NULL,
    served_by          TEXT NOT NULL,
    fallback_fired     INTEGER NOT NULL,
    cache_status       TEXT NOT NULL,
    capture_mode       TEXT NOT NULL,
    request_text       TEXT,
    response_text      TEXT
);

CREATE INDEX IF NOT EXISTS idx_request_log_ts ON request_log(ts_ms);
CREATE INDEX IF NOT EXISTS idx_request_log_key_id ON request_log(key_id);
CREATE INDEX IF NOT EXISTS idx_request_log_model ON request_log(model);

