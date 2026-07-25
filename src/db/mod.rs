//! SQLite access against Meetily's `meeting_minutes.sqlite` schema.

mod meetings;
mod models;
mod settings;
mod summary;
mod transcripts;

pub use meetings::*;
pub use models::*;
pub use settings::*;
pub use summary::*;
pub use transcripts::*;

use crate::paths::MeetilyPaths;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
use sqlx::SqlitePool;
use std::str::FromStr;

/// Open (or create) the Meetily database at the given path and apply schema.
pub async fn open_database(db_path: &std::path::Path) -> anyhow::Result<SqlitePool> {
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let url = format!("sqlite:{}?mode=rwc", db_path.display());
    let options = SqliteConnectOptions::from_str(&url)?
        .create_if_missing(true)
        .journal_mode(SqliteJournalMode::Wal)
        .foreign_keys(true);

    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect_with(options)
        .await?;

    apply_schema(&pool).await?;
    Ok(pool)
}

/// Open using resolved Meetily paths.
pub async fn open_meetily_database(paths: &MeetilyPaths) -> anyhow::Result<SqlitePool> {
    // Prefer existing sqlite; if only legacy .db exists, copy like Meetily does.
    if !paths.db_path.exists() && paths.legacy_db_path.exists() {
        std::fs::copy(&paths.legacy_db_path, &paths.db_path)?;
    }
    open_database(&paths.db_path).await
}

/// Idempotent schema matching Meetily production tables (CREATE IF NOT EXISTS).
/// Does not run sqlx migrations against a live Meetily DB (those are already applied);
/// for empty/test DBs this creates the full table set the TUI needs.
pub async fn apply_schema(pool: &SqlitePool) -> anyhow::Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS meetings (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            folder_path TEXT
        );

        CREATE TABLE IF NOT EXISTS transcripts (
            id TEXT PRIMARY KEY,
            meeting_id TEXT NOT NULL,
            transcript TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            summary TEXT,
            action_items TEXT,
            key_points TEXT,
            audio_start_time REAL,
            audio_end_time REAL,
            duration REAL,
            speaker TEXT,
            FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS summary_processes (
            meeting_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            error TEXT,
            result TEXT,
            start_time TEXT,
            end_time TEXT,
            chunk_count INTEGER DEFAULT 0,
            processing_time REAL DEFAULT 0.0,
            metadata TEXT,
            result_backup TEXT,
            result_backup_timestamp TEXT,
            FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS transcript_chunks (
            meeting_id TEXT PRIMARY KEY,
            meeting_name TEXT,
            transcript_text TEXT NOT NULL,
            model TEXT NOT NULL,
            model_name TEXT NOT NULL,
            chunk_size INTEGER,
            overlap INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS settings (
            id TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            whisperModel TEXT NOT NULL,
            groqApiKey TEXT,
            openaiApiKey TEXT,
            anthropicApiKey TEXT,
            ollamaApiKey TEXT,
            openRouterApiKey TEXT,
            ollamaEndpoint TEXT,
            customOpenAIConfig TEXT,
            geminiApiKey TEXT
        );

        CREATE TABLE IF NOT EXISTS transcript_settings (
            id TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            whisperApiKey TEXT,
            deepgramApiKey TEXT,
            elevenLabsApiKey TEXT,
            groqApiKey TEXT,
            openaiApiKey TEXT
        );

        CREATE TABLE IF NOT EXISTS meeting_notes (
            meeting_id TEXT PRIMARY KEY NOT NULL,
            notes_markdown TEXT,
            notes_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
        );
        "#,
    )
    .execute(pool)
    .await?;

    // Best-effort column adds for older empty test DBs created with partial schema.
    let _ = sqlx::query("ALTER TABLE meetings ADD COLUMN folder_path TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE transcripts ADD COLUMN audio_start_time REAL")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE transcripts ADD COLUMN audio_end_time REAL")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE transcripts ADD COLUMN duration REAL")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE transcripts ADD COLUMN speaker TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE settings ADD COLUMN openRouterApiKey TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE settings ADD COLUMN ollamaEndpoint TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE settings ADD COLUMN customOpenAIConfig TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE settings ADD COLUMN geminiApiKey TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE summary_processes ADD COLUMN result_backup TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query(
        "ALTER TABLE summary_processes ADD COLUMN result_backup_timestamp TEXT",
    )
    .execute(pool)
    .await;

    Ok(())
}

/// Graceful WAL checkpoint + pool close.
pub async fn cleanup(pool: &SqlitePool) -> anyhow::Result<()> {
    let _ = sqlx::query("PRAGMA wal_checkpoint(TRUNCATE)")
        .execute(pool)
        .await;
    pool.close().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paths::MeetilyPaths;

    #[tokio::test]
    async fn roundtrip_meeting_and_transcript() {
        let tmp = tempfile::tempdir().unwrap();
        let app = tmp.path().join("com.meetily.ai");
        let rec = tmp.path().join("Movies").join("meetily-recordings");
        let paths = MeetilyPaths::with_dirs(app, rec);
        paths.ensure_dirs().unwrap();

        let pool = open_database(&paths.db_path).await.unwrap();
        assert!(paths.db_path.file_name().unwrap() == "meeting_minutes.sqlite");

        let meeting_id = create_meeting(&pool, "Test Meeting", None)
            .await
            .unwrap();
        assert!(meeting_id.starts_with("meeting-"));

        append_transcript_segment(
            &pool,
            &meeting_id,
            "Hello from meeticulous",
            "2026-01-01T00:00:00Z",
            Some(0.0),
            Some(1.5),
            Some(1.5),
            None,
        )
        .await
        .unwrap();

        let list = list_meetings(&pool).await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].title, "Test Meeting");
        assert_eq!(list[0].id, meeting_id);

        let segs = load_transcripts(&pool, &meeting_id).await.unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].transcript, "Hello from meeticulous");

        let text = load_transcript_text(&pool, &meeting_id).await.unwrap();
        assert!(text.contains("Hello from meeticulous"));
        assert!(text.contains("[00:00]"), "expected MM:SS timestamp, got: {text}");

        cleanup(&pool).await.unwrap();
    }

    #[test]
    fn media_timestamp_format() {
        assert_eq!(format_media_timestamp(0.0), "00:00");
        assert_eq!(format_media_timestamp(65.2), "01:05");
        assert_eq!(format_media_timestamp(3599.0), "59:59");
        assert_eq!(format_media_timestamp(3600.0), "01:00:00");
        assert_eq!(format_media_timestamp(3661.0), "01:01:01");
    }
}
