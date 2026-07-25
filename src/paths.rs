//! Meetily-compatible path resolution.
//!
//! meeticulous **must** use the exact same durable paths as the Meetily GUI
//! (`com.meetily.ai` app identifier). Never create a separate app-support tree
//! under a meeticulous-specific identity for models / transcripts / DB / recordings.

use std::path::{Path, PathBuf};

/// Tauri / Meetily application identifier — drives Application Support folder name.
pub const MEETILY_APP_IDENTIFIER: &str = "com.meetily.ai";

/// SQLite database filename used by Meetily production.
pub const MEETING_MINUTES_DB_FILENAME: &str = "meeting_minutes.sqlite";

/// Legacy backend DB filename (auto-migrated by Meetily if present).
pub const MEETING_MINUTES_LEGACY_DB_FILENAME: &str = "meeting_minutes.db";

/// Models subdirectory under app data.
pub const MODELS_DIRNAME: &str = "models";

/// Parakeet models live under `models/parakeet/`.
pub const PARAKEET_MODELS_DIRNAME: &str = "parakeet";

/// Built-in summary models under `models/summary/`.
pub const SUMMARY_MODELS_DIRNAME: &str = "summary";

/// Default recordings folder name under Movies (macOS).
pub const RECORDINGS_DIRNAME: &str = "meetily-recordings";

/// Recording preferences JSON filename (Tauri store).
pub const RECORDING_PREFERENCES_FILENAME: &str = "recording_preferences.json";

/// General preferences JSON filename.
pub const PREFERENCES_FILENAME: &str = "preferences.json";

/// Override root for tests. When set, all path helpers resolve under this dir
/// instead of the real `~/Library/Application Support/com.meetily.ai`.
static TEST_APP_DATA_ROOT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Install a test-only app-data root. Call once per process from tests.
pub fn set_test_app_data_root(path: PathBuf) {
    let _ = TEST_APP_DATA_ROOT.set(path);
}

/// Clear is not supported (OnceLock); tests should use unique temp roots via
/// explicit `MeetilyPaths::with_app_data_dir` instead when isolation is needed.
///
/// Resolved Meetily data locations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeetilyPaths {
    pub app_data_dir: PathBuf,
    pub db_path: PathBuf,
    pub legacy_db_path: PathBuf,
    pub models_dir: PathBuf,
    pub parakeet_models_dir: PathBuf,
    pub summary_models_dir: PathBuf,
    pub recordings_dir: PathBuf,
    pub recording_preferences_path: PathBuf,
    pub preferences_path: PathBuf,
}

impl MeetilyPaths {
    /// Production paths matching Meetily GUI on macOS.
    pub fn resolve() -> anyhow::Result<Self> {
        if let Some(test_root) = TEST_APP_DATA_ROOT.get() {
            return Ok(Self::with_app_data_dir(
                test_root.clone(),
                default_recordings_dir(),
            ));
        }
        Ok(Self::with_app_data_dir(
            meetily_app_data_dir()?,
            default_recordings_dir(),
        ))
    }

    /// Build paths from an explicit app-data directory (tests / overrides).
    /// Recordings still default to `~/Movies/meetily-recordings` unless overridden
    /// via `with_dirs`.
    pub fn with_app_data_dir(app_data_dir: PathBuf, recordings_dir: PathBuf) -> Self {
        let models_dir = app_data_dir.join(MODELS_DIRNAME);
        Self {
            db_path: app_data_dir.join(MEETING_MINUTES_DB_FILENAME),
            legacy_db_path: app_data_dir.join(MEETING_MINUTES_LEGACY_DB_FILENAME),
            parakeet_models_dir: models_dir.join(PARAKEET_MODELS_DIRNAME),
            summary_models_dir: models_dir.join(SUMMARY_MODELS_DIRNAME),
            models_dir,
            recording_preferences_path: app_data_dir.join(RECORDING_PREFERENCES_FILENAME),
            preferences_path: app_data_dir.join(PREFERENCES_FILENAME),
            app_data_dir,
            recordings_dir,
        }
    }

    /// Fully custom dirs (unit tests).
    pub fn with_dirs(app_data_dir: PathBuf, recordings_dir: PathBuf) -> Self {
        Self::with_app_data_dir(app_data_dir, recordings_dir)
    }

    /// Ensure app-data and models directories exist (does not touch live user data
    /// structure beyond create-if-missing for standard subdirs).
    pub fn ensure_dirs(&self) -> anyhow::Result<()> {
        std::fs::create_dir_all(&self.app_data_dir)?;
        std::fs::create_dir_all(&self.models_dir)?;
        std::fs::create_dir_all(&self.parakeet_models_dir)?;
        std::fs::create_dir_all(&self.summary_models_dir)?;
        std::fs::create_dir_all(&self.recordings_dir)?;
        Ok(())
    }
}

/// `~/Library/Application Support/com.meetily.ai` on macOS.
///
/// Uses the same resolution as Tauri's `app_data_dir()` for identifier
/// `com.meetily.ai`: macOS Application Support + reverse-DNS folder name.
pub fn meetily_app_data_dir() -> anyhow::Result<PathBuf> {
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("could not resolve home directory"))?;
    Ok(home
        .join("Library")
        .join("Application Support")
        .join(MEETILY_APP_IDENTIFIER))
}

/// Default recordings root: `~/Movies/meetily-recordings` on macOS.
///
/// Matches Meetily's `get_default_recordings_folder()` (uses `dirs::video_dir()`).
pub fn default_recordings_dir() -> PathBuf {
    if let Some(movies) = dirs::video_dir() {
        movies.join(RECORDINGS_DIRNAME)
    } else if let Some(docs) = dirs::document_dir() {
        docs.join(RECORDINGS_DIRNAME)
    } else {
        PathBuf::from(".").join(RECORDINGS_DIRNAME)
    }
}

/// Convenience: sqlite path under app data.
pub fn meeting_minutes_sqlite_path() -> anyhow::Result<PathBuf> {
    Ok(meetily_app_data_dir()?.join(MEETING_MINUTES_DB_FILENAME))
}

/// Convenience: models dir.
pub fn models_dir() -> anyhow::Result<PathBuf> {
    Ok(meetily_app_data_dir()?.join(MODELS_DIRNAME))
}

/// Format paths for `--paths` CLI output.
pub fn format_paths_report(paths: &MeetilyPaths) -> String {
    format!(
        "meeticulous data paths (shared with Meetily GUI)\n\
         app_data:     {}\n\
         database:     {}\n\
         models:       {}\n\
         parakeet:     {}\n\
         summary:      {}\n\
         recordings:   {}\n",
        paths.app_data_dir.display(),
        paths.db_path.display(),
        paths.models_dir.display(),
        paths.parakeet_models_dir.display(),
        paths.summary_models_dir.display(),
        paths.recordings_dir.display(),
    )
}

/// Assert a path ends with the expected Meetily relative components.
pub fn path_ends_with(path: &Path, components: &[&str]) -> bool {
    let parts: Vec<_> = path
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();
    if parts.len() < components.len() {
        return false;
    }
    parts[parts.len() - components.len()..] == *components
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_data_dir_is_com_meetily_ai() {
        let dir = meetily_app_data_dir().expect("home");
        assert!(
            path_ends_with(&dir, &["Library", "Application Support", "com.meetily.ai"]),
            "unexpected app data dir: {}",
            dir.display()
        );
        // Must not invent a meeticulous-specific tree for durable data.
        assert!(!dir.to_string_lossy().contains("meeticulous"));
    }

    #[test]
    fn db_and_models_under_app_data() {
        let paths = MeetilyPaths::resolve().expect("resolve");
        assert_eq!(
            paths.db_path,
            paths.app_data_dir.join("meeting_minutes.sqlite")
        );
        assert_eq!(paths.models_dir, paths.app_data_dir.join("models"));
        assert_eq!(
            paths.parakeet_models_dir,
            paths.app_data_dir.join("models").join("parakeet")
        );
        assert!(path_ends_with(
            &paths.app_data_dir,
            &["Library", "Application Support", "com.meetily.ai"]
        ));
        assert!(path_ends_with(
            &paths.db_path,
            &[
                "Library",
                "Application Support",
                "com.meetily.ai",
                "meeting_minutes.sqlite"
            ]
        ));
        assert!(path_ends_with(
            &paths.models_dir,
            &["Library", "Application Support", "com.meetily.ai", "models"]
        ));
    }

    #[test]
    fn default_recordings_is_movies_meetily_recordings() {
        let rec = default_recordings_dir();
        assert!(
            path_ends_with(&rec, &["Movies", "meetily-recordings"])
                || path_ends_with(&rec, &["Documents", "meetily-recordings"]),
            "unexpected recordings dir: {}",
            rec.display()
        );
        let paths = MeetilyPaths::resolve().expect("resolve");
        assert_eq!(paths.recordings_dir, rec);
    }

    #[test]
    fn with_dirs_isolates_test_roots() {
        let tmp = tempfile::tempdir().unwrap();
        let app = tmp.path().join("com.meetily.ai");
        let rec = tmp.path().join("Movies").join("meetily-recordings");
        let paths = MeetilyPaths::with_dirs(app.clone(), rec.clone());
        assert_eq!(paths.app_data_dir, app);
        assert_eq!(paths.db_path, app.join("meeting_minutes.sqlite"));
        assert_eq!(paths.models_dir, app.join("models"));
        assert_eq!(paths.recordings_dir, rec);
    }
}
