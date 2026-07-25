//! Transcription model discovery under Meetily's shared `models/` tree.
//!
//! Whisper: `~/Library/Application Support/com.meetily.ai/models/ggml-*.bin`
//! Parakeet: `.../models/parakeet/<model-name>/` with ONNX assets

use crate::paths::MeetilyPaths;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Catalog entry aligned with Meetily `WHISPER_MODEL_CATALOG`.
pub const WHISPER_MODEL_CATALOG: &[(&str, &str, u32, &str, &str)] = &[
    ("tiny", "ggml-tiny.bin", 74, "Decent", "Very Fast"),
    ("base", "ggml-base.bin", 142, "Good", "Fast"),
    ("small", "ggml-small.bin", 466, "Good", "Medium"),
    ("medium", "ggml-medium.bin", 1463, "High", "Slow"),
    ("large-v3-turbo", "ggml-large-v3-turbo.bin", 1549, "High", "Medium"),
    ("large-v3", "ggml-large-v3.bin", 2951, "High", "Slow"),
    ("tiny-q5_1", "ggml-tiny-q5_1.bin", 31, "Decent", "Very Fast"),
    ("base-q5_1", "ggml-base-q5_1.bin", 57, "Good", "Fast"),
    ("small-q5_1", "ggml-small-q5_1.bin", 181, "Good", "Fast"),
    ("medium-q5_0", "ggml-medium-q5_0.bin", 514, "High", "Medium"),
    ("large-v3-turbo-q5_0", "ggml-large-v3-turbo-q5_0.bin", 547, "High", "Medium"),
    ("large-v3-q5_0", "ggml-large-v3-q5_0.bin", 1031, "High", "Slow"),
];

pub const DEFAULT_WHISPER_MODEL: &str = "large-v3-turbo";
pub const DEFAULT_PARAKEET_MODEL: &str = "parakeet-tdt-0.6b-v3-int8";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TranscriptionProvider {
    #[serde(rename = "localWhisper")]
    LocalWhisper,
    #[serde(rename = "parakeet")]
    Parakeet,
}

impl TranscriptionProvider {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::LocalWhisper => "localWhisper",
            Self::Parakeet => "parakeet",
        }
    }

    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s {
            "localWhisper" | "whisper" | "local_whisper" => Some(Self::LocalWhisper),
            "parakeet" => Some(Self::Parakeet),
            _ => None,
        }
    }
}

impl std::fmt::Display for TranscriptionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DiscoveredModel {
    pub provider: TranscriptionProvider,
    pub name: String,
    pub path: PathBuf,
    pub size_mb: u64,
    pub available: bool,
    pub accuracy: String,
    pub speed: String,
}

/// Discover Whisper + Parakeet models under a Meetily-layout models directory.
pub fn discover_models(models_dir: &Path) -> Vec<DiscoveredModel> {
    let mut out = Vec::new();
    out.extend(discover_whisper_models(models_dir));
    out.extend(discover_parakeet_models(&models_dir.join("parakeet")));
    out
}

/// Discover using resolved Meetily paths.
pub fn discover_models_for_paths(paths: &MeetilyPaths) -> Vec<DiscoveredModel> {
    discover_models(&paths.models_dir)
}

pub fn discover_whisper_models(models_dir: &Path) -> Vec<DiscoveredModel> {
    let mut models = Vec::new();
    for &(name, filename, expected_mb, accuracy, speed) in WHISPER_MODEL_CATALOG {
        let path = models_dir.join(filename);
        let (available, size_mb) = if path.is_file() {
            let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            let mb = bytes / (1024 * 1024);
            // Available if the file is a substantial ggml blob (>1 MiB). Catalog `expected_mb`
            // is used for display; incomplete downloads under 1 MiB stay unavailable.
            let _ = expected_mb;
            let ok = bytes > 1_048_576;
            (ok, if mb == 0 && bytes > 0 { 1 } else { mb })
        } else {
            (false, 0)
        };
        models.push(DiscoveredModel {
            provider: TranscriptionProvider::LocalWhisper,
            name: name.to_string(),
            path,
            size_mb,
            available,
            accuracy: accuracy.to_string(),
            speed: speed.to_string(),
        });
    }
    models
}

/// Parakeet models are directories under `models/parakeet/` that contain ONNX files
/// (encoder / decoder / vocab) — same layout Meetily's ParakeetEngine expects.
pub fn discover_parakeet_models(parakeet_dir: &Path) -> Vec<DiscoveredModel> {
    let mut models = Vec::new();
    if !parakeet_dir.is_dir() {
        return models;
    }
    let entries = match std::fs::read_dir(parakeet_dir) {
        Ok(e) => e,
        Err(_) => return models,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = match path.file_name().and_then(|s| s.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        let available = is_parakeet_model_dir(&path);
        let size_mb = dir_size_mb(&path);
        models.push(DiscoveredModel {
            provider: TranscriptionProvider::Parakeet,
            name,
            path,
            size_mb,
            available,
            accuracy: "High".to_string(),
            speed: "Very Fast".to_string(),
        });
    }
    models.sort_by(|a, b| a.name.cmp(&b.name));
    models
}

fn is_parakeet_model_dir(dir: &Path) -> bool {
    let has_onnx = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten().any(|e| {
                e.path()
                    .extension()
                    .and_then(|x| x.to_str())
                    .map(|x| x.eq_ignore_ascii_case("onnx"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);
    let has_vocab = dir.join("vocab.txt").is_file();
    has_onnx && has_vocab
}

fn dir_size_mb(dir: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            if let Ok(meta) = e.metadata() {
                if meta.is_file() {
                    total += meta.len();
                }
            }
        }
    }
    total / (1024 * 1024)
}

/// Resolve path for a selected provider/model name under models root.
pub fn resolve_model_path(
    models_dir: &Path,
    provider: TranscriptionProvider,
    model_name: &str,
) -> Option<PathBuf> {
    match provider {
        TranscriptionProvider::LocalWhisper => {
            for &(name, filename, ..) in WHISPER_MODEL_CATALOG {
                if name == model_name {
                    let p = models_dir.join(filename);
                    return if p.exists() { Some(p) } else { Some(p) };
                }
            }
            let p = models_dir.join(format!("ggml-{model_name}.bin"));
            Some(p)
        }
        TranscriptionProvider::Parakeet => {
            let p = models_dir.join("parakeet").join(model_name);
            Some(p)
        }
    }
}

/// Persist selection into Meetily's `transcript_settings` table shape via caller DB helpers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelSelection {
    pub provider: TranscriptionProvider,
    pub model: String,
}

impl Default for ModelSelection {
    fn default() -> Self {
        Self {
            provider: TranscriptionProvider::Parakeet,
            model: DEFAULT_PARAKEET_MODEL.to_string(),
        }
    }
}

/// Load selection from JSON prefs file under app data (fallback when DB empty).
/// Meetily primarily uses SQLite; we also accept a simple `meeticulous-model.json`
/// only for TUI-local cache — but preferred store is transcript_settings in sqlite.
pub fn load_selection_from_app_data(app_data: &Path) -> Option<ModelSelection> {
    let p = app_data.join("meeticulous-model-selection.json");
    let data = std::fs::read_to_string(p).ok()?;
    serde_json::from_str(&data).ok()
}

pub fn save_selection_to_app_data(app_data: &Path, sel: &ModelSelection) -> anyhow::Result<()> {
    let p = app_data.join("meeticulous-model-selection.json");
    std::fs::write(p, serde_json::to_string_pretty(sel)?)?;
    Ok(())
}

/// Name kept for call sites that also write `transcript_settings` via `db::save_transcript_config`.
pub fn save_transcript_config_bridge() {
    // Selection persistence is handled by `save_selection_to_app_data` + DB helpers.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paths::MeetilyPaths;

    #[test]
    fn discover_whisper_and_parakeet_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let models = tmp.path().join("models");
        std::fs::create_dir_all(models.join("parakeet").join("parakeet-tdt-0.6b-v3-int8")).unwrap();
        // fake whisper
        let whisper = models.join("ggml-tiny.bin");
        std::fs::write(&whisper, vec![0u8; 3 * 1024 * 1024]).unwrap();
        // fake parakeet onnx + vocab
        let pk = models.join("parakeet").join("parakeet-tdt-0.6b-v3-int8");
        std::fs::write(pk.join("encoder-model.int8.onnx"), b"onnx").unwrap();
        std::fs::write(pk.join("vocab.txt"), b"a\nb\n").unwrap();

        let found = discover_models(&models);
        let tiny = found
            .iter()
            .find(|m| m.name == "tiny" && m.provider == TranscriptionProvider::LocalWhisper)
            .expect("tiny");
        assert!(tiny.available);
        let pk_m = found
            .iter()
            .find(|m| m.name == "parakeet-tdt-0.6b-v3-int8")
            .expect("parakeet");
        assert!(pk_m.available);
        assert_eq!(pk_m.provider, TranscriptionProvider::Parakeet);
    }

    #[test]
    fn selection_persists_under_app_data_root() {
        let tmp = tempfile::tempdir().unwrap();
        let app = tmp.path().join("com.meetily.ai");
        let rec = tmp.path().join("Movies").join("meetily-recordings");
        let paths = MeetilyPaths::with_dirs(app.clone(), rec);
        paths.ensure_dirs().unwrap();

        let sel = ModelSelection {
            provider: TranscriptionProvider::LocalWhisper,
            model: "base".to_string(),
        };
        save_selection_to_app_data(&paths.app_data_dir, &sel).unwrap();
        let loaded = load_selection_from_app_data(&paths.app_data_dir).unwrap();
        assert_eq!(loaded, sel);
        // Stored under shared Meetily root naming, not a meeticulous app-support tree.
        assert!(paths.app_data_dir.ends_with("com.meetily.ai"));
    }

    #[test]
    fn resolve_paths_match_meetily_conventions() {
        let models = PathBuf::from("/tmp/com.meetily.ai/models");
        let w = resolve_model_path(&models, TranscriptionProvider::LocalWhisper, "tiny").unwrap();
        assert!(w.ends_with("ggml-tiny.bin"));
        let p = resolve_model_path(
            &models,
            TranscriptionProvider::Parakeet,
            "parakeet-tdt-0.6b-v3-int8",
        )
        .unwrap();
        assert!(p.ends_with("parakeet/parakeet-tdt-0.6b-v3-int8") || p.components().any(|c| c.as_os_str() == "parakeet"));
    }
}
