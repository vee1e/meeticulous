use super::models::Meeting;
use chrono::Utc;
use sqlx::SqlitePool;
use uuid::Uuid;

/// List all meetings newest-first (Meetily order).
pub async fn list_meetings(pool: &SqlitePool) -> Result<Vec<Meeting>, sqlx::Error> {
    sqlx::query_as::<_, Meeting>(
        "SELECT id, title, created_at, updated_at, folder_path FROM meetings ORDER BY created_at DESC",
    )
    .fetch_all(pool)
    .await
}

/// Fetch one meeting by id.
pub async fn get_meeting(
    pool: &SqlitePool,
    meeting_id: &str,
) -> Result<Option<Meeting>, sqlx::Error> {
    sqlx::query_as::<_, Meeting>(
        "SELECT id, title, created_at, updated_at, folder_path FROM meetings WHERE id = ?",
    )
    .bind(meeting_id)
    .fetch_optional(pool)
    .await
}

/// Create a new meeting. Returns the meeting id (`meeting-<uuid>`).
pub async fn create_meeting(
    pool: &SqlitePool,
    title: &str,
    folder_path: Option<&str>,
) -> Result<String, sqlx::Error> {
    let meeting_id = format!("meeting-{}", Uuid::new_v4());
    let now = Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO meetings (id, title, created_at, updated_at, folder_path) VALUES (?, ?, ?, ?, ?)",
    )
    .bind(&meeting_id)
    .bind(title)
    .bind(&now)
    .bind(&now)
    .bind(folder_path)
    .execute(pool)
    .await?;
    Ok(meeting_id)
}

/// Update meeting title.
pub async fn update_meeting_title(
    pool: &SqlitePool,
    meeting_id: &str,
    title: &str,
) -> Result<bool, sqlx::Error> {
    let now = Utc::now().to_rfc3339();
    let res = sqlx::query("UPDATE meetings SET title = ?, updated_at = ? WHERE id = ?")
        .bind(title)
        .bind(now)
        .bind(meeting_id)
        .execute(pool)
        .await?;
    Ok(res.rows_affected() > 0)
}

/// Touch updated_at.
pub async fn touch_meeting(pool: &SqlitePool, meeting_id: &str) -> Result<(), sqlx::Error> {
    let now = Utc::now().to_rfc3339();
    sqlx::query("UPDATE meetings SET updated_at = ? WHERE id = ?")
        .bind(now)
        .bind(meeting_id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Delete meeting and cascaded rows (explicit deletes for safety).
pub async fn delete_meeting(pool: &SqlitePool, meeting_id: &str) -> Result<bool, sqlx::Error> {
    let mut tx = pool.begin().await?;
    sqlx::query("DELETE FROM transcript_chunks WHERE meeting_id = ?")
        .bind(meeting_id)
        .execute(&mut *tx)
        .await?;
    sqlx::query("DELETE FROM summary_processes WHERE meeting_id = ?")
        .bind(meeting_id)
        .execute(&mut *tx)
        .await?;
    sqlx::query("DELETE FROM meeting_notes WHERE meeting_id = ?")
        .bind(meeting_id)
        .execute(&mut *tx)
        .await?;
    sqlx::query("DELETE FROM transcripts WHERE meeting_id = ?")
        .bind(meeting_id)
        .execute(&mut *tx)
        .await?;
    let res = sqlx::query("DELETE FROM meetings WHERE id = ?")
        .bind(meeting_id)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    Ok(res.rows_affected() > 0)
}
