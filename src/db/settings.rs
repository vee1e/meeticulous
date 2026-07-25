use super::models::{Setting, TranscriptSetting};
use sqlx::SqlitePool;

pub async fn get_model_config(pool: &SqlitePool) -> Result<Option<Setting>, sqlx::Error> {
    // SELECT * can fail if column order differs; use explicit columns matching Setting.
    sqlx::query_as::<_, Setting>(
        r#"
        SELECT id, provider, model, whisperModel,
               groqApiKey, openaiApiKey, anthropicApiKey, ollamaApiKey,
               openRouterApiKey, ollamaEndpoint, customOpenAIConfig, geminiApiKey
        FROM settings LIMIT 1
        "#,
    )
    .fetch_optional(pool)
    .await
}

pub async fn save_model_config(
    pool: &SqlitePool,
    provider: &str,
    model: &str,
    whisper_model: &str,
    ollama_endpoint: Option<&str>,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        INSERT INTO settings (id, provider, model, whisperModel, ollamaEndpoint)
        VALUES ('1', ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            provider = excluded.provider,
            model = excluded.model,
            whisperModel = excluded.whisperModel,
            ollamaEndpoint = excluded.ollamaEndpoint
        "#,
    )
    .bind(provider)
    .bind(model)
    .bind(whisper_model)
    .bind(ollama_endpoint)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn get_transcript_config(
    pool: &SqlitePool,
) -> Result<Option<TranscriptSetting>, sqlx::Error> {
    sqlx::query_as::<_, TranscriptSetting>(
        r#"
        SELECT id, provider, model, whisperApiKey, deepgramApiKey,
               elevenLabsApiKey, groqApiKey, openaiApiKey
        FROM transcript_settings LIMIT 1
        "#,
    )
    .fetch_optional(pool)
    .await
}

pub async fn save_transcript_config(
    pool: &SqlitePool,
    provider: &str,
    model: &str,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        INSERT INTO transcript_settings (id, provider, model)
        VALUES ('1', ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            provider = excluded.provider,
            model = excluded.model
        "#,
    )
    .bind(provider)
    .bind(model)
    .execute(pool)
    .await?;
    Ok(())
}

/// Resolve API key for a summary provider from Meetily settings columns.
pub async fn get_api_key_for_provider(
    pool: &SqlitePool,
    provider: &str,
) -> Result<Option<String>, sqlx::Error> {
    let col = match provider {
        "openai" => "openaiApiKey",
        "claude" => "anthropicApiKey",
        "ollama" => "ollamaApiKey",
        "groq" => "groqApiKey",
        "openrouter" => "openRouterApiKey",
        "gemini" => "geminiApiKey",
        "builtin-ai" => return Ok(None),
        _ => return Ok(None),
    };
    let q = format!("SELECT {} FROM settings WHERE id = '1' LIMIT 1", col);
    let key: Option<Option<String>> = sqlx::query_scalar(&q).fetch_optional(pool).await?;
    Ok(key.flatten())
}
