use super::models::TranscriptSegment;
use super::touch_meeting_in_tx;
use sqlx::SqlitePool;
use uuid::Uuid;

/// Load all transcript segments for a meeting, ordered by audio timing then timestamp.
pub async fn load_transcripts(
    pool: &SqlitePool,
    meeting_id: &str,
) -> Result<Vec<TranscriptSegment>, sqlx::Error> {
    sqlx::query_as::<_, TranscriptSegment>(
        r#"
        SELECT id, meeting_id, transcript, timestamp, summary, action_items, key_points,
               audio_start_time, audio_end_time, duration, speaker
        FROM transcripts
        WHERE meeting_id = ?
        ORDER BY COALESCE(audio_start_time, 0) ASC, timestamp ASC
        "#,
    )
    .bind(meeting_id)
    .fetch_all(pool)
    .await
}

/// Format seconds as media timestamp:
/// - under 1 hour → `MM:SS` (e.g. `00:42`)
/// - 1 hour or more → `HH:MM:SS` (e.g. `01:05:03`)
pub fn format_media_timestamp(secs: f64) -> String {
    let total = if secs.is_finite() && secs > 0.0 {
        secs.floor() as u64
    } else {
        0
    };
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h:02}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

/// Format one segment line with timestamp (and optional speaker).
pub fn format_segment_line(seg: &TranscriptSegment) -> String {
    let ts = format_media_timestamp(seg.audio_start_time.unwrap_or(0.0));
    match &seg.speaker {
        Some(sp) if !sp.is_empty() => format!("[{ts}] [{sp}] {}", seg.transcript),
        _ => format!("[{ts}] {}", seg.transcript),
    }
}

/// Concatenated plain-text transcript for display / summary input.
/// Display lines include media timestamps from `audio_start_time`.
pub async fn load_transcript_text(
    pool: &SqlitePool,
    meeting_id: &str,
) -> Result<String, sqlx::Error> {
    let segs = load_transcripts(pool, meeting_id).await?;
    Ok(segs
        .into_iter()
        .map(|s| format_segment_line(&s))
        .collect::<Vec<_>>()
        .join("\n"))
}

/// Transcript body for LLM summary: plain text without timestamps (cleaner for models).
pub async fn load_transcript_text_plain(
    pool: &SqlitePool,
    meeting_id: &str,
) -> Result<String, sqlx::Error> {
    let segs = load_transcripts(pool, meeting_id).await?;
    Ok(segs
        .into_iter()
        .map(|s| {
            if let Some(sp) = s.speaker.filter(|x| !x.is_empty()) {
                format!("[{sp}] {}", s.transcript)
            } else {
                s.transcript
            }
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

/// Append one transcript segment to an existing meeting.
#[allow(clippy::too_many_arguments)]
pub async fn append_transcript_segment(
    pool: &SqlitePool,
    meeting_id: &str,
    text: &str,
    timestamp: &str,
    audio_start_time: Option<f64>,
    audio_end_time: Option<f64>,
    duration: Option<f64>,
    speaker: Option<&str>,
) -> Result<String, sqlx::Error> {
    let id = format!("transcript-{}", Uuid::new_v4());
    let mut tx = pool.begin().await?;
    sqlx::query(
        r#"
        INSERT INTO transcripts
            (id, meeting_id, transcript, timestamp, audio_start_time, audio_end_time, duration, speaker)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(&id)
    .bind(meeting_id)
    .bind(text)
    .bind(timestamp)
    .bind(audio_start_time)
    .bind(audio_end_time)
    .bind(duration)
    .bind(speaker)
    .execute(&mut *tx)
    .await?;
    touch_meeting_in_tx(&mut tx, meeting_id).await?;
    tx.commit().await?;
    Ok(id)
}

/// Import an entire meeting's transcript segments in ONE transaction: a row per
/// line plus the `transcript_chunks` upsert. Caller owns meeting lifecycle.
pub async fn import_meeting_with_segments(
    pool: &SqlitePool,
    meeting_id: &str,
    meeting_name: &str,
    lines: &[&str],
    model: &str,
    model_name: &str,
) -> Result<(), sqlx::Error> {
    let mut tx = pool.begin().await?;
    for (i, line) in lines.iter().enumerate() {
        let id = format!("transcript-{}", Uuid::new_v4());
        let timestamp = chrono::Local::now().format("%H:%M:%S").to_string();
        let start = i as f64 * 2.0;
        sqlx::query(
            r#"
            INSERT INTO transcripts
                (id, meeting_id, transcript, timestamp, audio_start_time, audio_end_time, duration, speaker)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
            "#,
        )
        .bind(&id)
        .bind(meeting_id)
        .bind(*line)
        .bind(&timestamp)
        .bind(start)
        .bind(start + 2.0)
        .bind(2.0)
        .execute(&mut *tx)
        .await?;
    }
    let transcript_text = lines.join("\n");
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        r#"
        INSERT INTO transcript_chunks
            (meeting_id, meeting_name, transcript_text, model, model_name, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(meeting_id) DO UPDATE SET
            meeting_name = excluded.meeting_name,
            transcript_text = excluded.transcript_text,
            model = excluded.model,
            model_name = excluded.model_name,
            created_at = excluded.created_at
        "#,
    )
    .bind(meeting_id)
    .bind(meeting_name)
    .bind(&transcript_text)
    .bind(model)
    .bind(model_name)
    .bind(now)
    .execute(&mut *tx)
    .await?;
    tx.commit().await
}

/// Upsert transcript_chunks blob used by summary pipeline (Meetily shape).
pub async fn upsert_transcript_chunk(
    pool: &SqlitePool,
    meeting_id: &str,
    meeting_name: &str,
    transcript_text: &str,
    model: &str,
    model_name: &str,
) -> Result<(), sqlx::Error> {
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        r#"
        INSERT INTO transcript_chunks
            (meeting_id, meeting_name, transcript_text, model, model_name, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(meeting_id) DO UPDATE SET
            meeting_name = excluded.meeting_name,
            transcript_text = excluded.transcript_text,
            model = excluded.model,
            model_name = excluded.model_name,
            created_at = excluded.created_at
        "#,
    )
    .bind(meeting_id)
    .bind(meeting_name)
    .bind(transcript_text)
    .bind(model)
    .bind(model_name)
    .bind(now)
    .execute(pool)
    .await?;
    Ok(())
}
