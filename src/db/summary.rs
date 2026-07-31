use super::models::SummaryProcess;
use chrono::Utc;
use serde_json::Value;
use sqlx::SqlitePool;

pub async fn get_summary(
    pool: &SqlitePool,
    meeting_id: &str,
) -> Result<Option<SummaryProcess>, sqlx::Error> {
    sqlx::query_as::<_, SummaryProcess>(
        r#"
        SELECT meeting_id, status, created_at, updated_at, error, result,
               start_time, end_time, chunk_count, processing_time, metadata,
               result_backup, result_backup_timestamp
        FROM summary_processes WHERE meeting_id = ?
        "#,
    )
    .bind(meeting_id)
    .fetch_optional(pool)
    .await
}

async fn run_create_or_reset(
    executor: impl sqlx::Executor<'_, Database = sqlx::Sqlite>,
    meeting_id: &str,
) -> Result<(), sqlx::Error> {
    let now = Utc::now().to_rfc3339();
    sqlx::query(
        r#"
        INSERT INTO summary_processes
            (meeting_id, status, created_at, updated_at, start_time, result, error)
        VALUES (?, 'pending', ?, ?, ?, NULL, NULL)
        ON CONFLICT(meeting_id) DO UPDATE SET
            status = 'pending',
            updated_at = excluded.updated_at,
            start_time = excluded.start_time,
            result = NULL,
            result_backup = result,
            result_backup_timestamp = excluded.updated_at,
            error = NULL
        "#,
    )
    .bind(meeting_id)
    .bind(&now)
    .bind(&now)
    .bind(&now)
    .execute(executor)
    .await?;
    Ok(())
}

pub async fn create_or_reset_process(
    pool: &SqlitePool,
    meeting_id: &str,
) -> Result<(), sqlx::Error> {
    run_create_or_reset(pool, meeting_id).await
}

async fn run_mark_completed(
    executor: impl sqlx::Executor<'_, Database = sqlx::Sqlite>,
    meeting_id: &str,
    result: &Value,
    chunk_count: i64,
    processing_time: f64,
) -> Result<(), sqlx::Error> {
    let now = Utc::now().to_rfc3339();
    let result_str = serde_json::to_string(result)
        .map_err(|e| sqlx::Error::Protocol(format!("serialize: {e}")))?;
    sqlx::query(
        r#"
        UPDATE summary_processes
        SET status = 'completed', result = ?, updated_at = ?, end_time = ?,
            chunk_count = ?, processing_time = ?, error = NULL,
            result_backup = NULL, result_backup_timestamp = NULL
        WHERE meeting_id = ?
        "#,
    )
    .bind(result_str)
    .bind(&now)
    .bind(&now)
    .bind(chunk_count)
    .bind(processing_time)
    .bind(meeting_id)
    .execute(executor)
    .await?;
    Ok(())
}

pub async fn mark_completed(
    pool: &SqlitePool,
    meeting_id: &str,
    result: &Value,
    chunk_count: i64,
    processing_time: f64,
) -> Result<(), sqlx::Error> {
    run_mark_completed(pool, meeting_id, result, chunk_count, processing_time).await
}

pub async fn mark_failed(
    pool: &SqlitePool,
    meeting_id: &str,
    error: &str,
) -> Result<(), sqlx::Error> {
    let now = Utc::now().to_rfc3339();
    sqlx::query(
        r#"
        UPDATE summary_processes
        SET status = 'failed', error = ?, updated_at = ?, end_time = ?,
            result = COALESCE(result_backup, result),
            result_backup = NULL, result_backup_timestamp = NULL
        WHERE meeting_id = ?
        "#,
    )
    .bind(error)
    .bind(&now)
    .bind(&now)
    .bind(meeting_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Associate a free-form summary JSON with a meeting (create process row if needed).
pub async fn store_summary_for_meeting(
    pool: &SqlitePool,
    meeting_id: &str,
    summary: &Value,
) -> Result<(), sqlx::Error> {
    store_summary_for_meeting_with_stats(pool, meeting_id, summary, 1, 0.0).await
}

/// Atomic store: reset process row, write completed summary, and touch the
/// meeting inside a single transaction so a stale result is never observable.
pub async fn store_summary_for_meeting_with_stats(
    pool: &SqlitePool,
    meeting_id: &str,
    summary: &Value,
    chunk_count: i64,
    processing_time: f64,
) -> Result<(), sqlx::Error> {
    let mut tx = pool.begin().await?;
    run_create_or_reset(&mut *tx, meeting_id).await?;
    run_mark_completed(&mut *tx, meeting_id, summary, chunk_count, processing_time).await?;
    // Touch meeting so list ordering / GUI picks up the write.
    let now = Utc::now().to_rfc3339();
    sqlx::query("UPDATE meetings SET updated_at = ? WHERE id = ?")
        .bind(now)
        .bind(meeting_id)
        .execute(&mut *tx)
        .await?;
    tx.commit().await
}

/// Load stored summary plain text for a meeting (Meeticulous plain + legacy JSON).
pub async fn load_summary_plain_text(
    pool: &SqlitePool,
    meeting_id: &str,
) -> Result<Option<(String, String)>, sqlx::Error> {
    let row = get_summary(pool, meeting_id).await?;
    let Some(sp) = row else {
        return Ok(None);
    };
    let Some(result) = sp.result else {
        return Ok(None);
    };
    let status = sp.status;
    let text = parse_stored_summary_body(&result);
    if text.trim().is_empty() {
        return Ok(None);
    }
    Ok(Some((text, status)))
}

/// Extract display body from summary_processes.result JSON/text.
pub fn parse_stored_summary_body(result: &str) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(result) {
        if let Some(t) = v.get("text").and_then(|x| x.as_str()) {
            return t.to_string();
        }
        if let Some(s) = v.get("summary").and_then(|x| x.as_str()) {
            // Legacy structured JSON
            let mut out = s.to_string();
            for (key, label) in [
                ("key_points", "Key points"),
                ("action_items", "Action items"),
                ("decisions", "Decisions"),
            ] {
                if let Some(arr) = v.get(key).and_then(|a| a.as_array()) {
                    if !arr.is_empty() {
                        out.push_str(&format!("\n\n## {label}\n"));
                        for item in arr {
                            if let Some(s) = item.as_str() {
                                out.push_str(&format!("- {s}\n"));
                            }
                        }
                    }
                }
            }
            return out;
        }
        // Unknown JSON object — pretty-print as last resort
        return serde_json::to_string_pretty(&v).unwrap_or_else(|_| result.to_string());
    }
    result.to_string()
}
