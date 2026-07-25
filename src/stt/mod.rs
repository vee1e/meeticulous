//! Live speech-to-text for recording sessions.
//!
//! Uses Meetily's shared Parakeet ONNX models under
//! `~/Library/Application Support/com.meetily.ai/models/parakeet/`.

mod parakeet_model;

pub use parakeet_model::{ParakeetError, ParakeetModel, TimestampedResult};

use crate::models::{resolve_model_path, ModelSelection, TranscriptionProvider};
use crate::paths::MeetilyPaths;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Rolling diagnostics shown in the TUI while recording.
#[derive(Debug, Clone)]
pub struct RecordingDiagnostics {
    pub mic_device: String,
    pub mic_ok: bool,
    pub mic_error: Option<String>,
    pub system_ok: bool,
    pub system_device: String,
    pub system_error: Option<String>,
    pub sample_rate: u32,
    pub channels: u16,
    pub buffer_samples: usize,
    pub buffer_secs: f32,
    pub rms: f32,
    pub peak: f32,
    pub level_db: f32,
    pub stt_engine: String,
    pub stt_model_path: String,
    pub stt_status: String,
    pub last_stt_ms: Option<u64>,
    pub chunks_processed: u64,
    pub chunks_skipped_silence: u64,
    pub segments_emitted: u64,
    pub processed_audio_secs: f32,
    pub log: VecDeque<String>,
}

impl Default for RecordingDiagnostics {
    fn default() -> Self {
        Self {
            mic_device: "(none)".into(),
            mic_ok: false,
            mic_error: None,
            system_ok: false,
            system_device: "(none)".into(),
            system_error: None,
            sample_rate: 0,
            channels: 0,
            buffer_samples: 0,
            buffer_secs: 0.0,
            rms: 0.0,
            peak: 0.0,
            level_db: -120.0,
            stt_engine: "none".into(),
            stt_model_path: String::new(),
            stt_status: "idle".into(),
            last_stt_ms: None,
            chunks_processed: 0,
            chunks_skipped_silence: 0,
            segments_emitted: 0,
            processed_audio_secs: 0.0,
            log: VecDeque::with_capacity(32),
        }
    }
}

impl RecordingDiagnostics {
    pub fn push_log(&mut self, line: impl AsRef<str>) {
        let ts = chrono::Local::now().format("%H:%M:%S");
        self.log.push_back(format!("[{ts}] {}", line.as_ref()));
        while self.log.len() > 24 {
            self.log.pop_front();
        }
    }

    /// Multi-line status for the TUI.
    pub fn format_verbose(&self) -> String {
        let mic = if self.mic_ok {
            format!("OK · {}", self.mic_device)
        } else {
            format!(
                "FAIL · {}",
                self.mic_error.as_deref().unwrap_or("no input device")
            )
        };
        let sys = if self.system_ok {
            format!("OK · {}", self.system_device)
        } else {
            format!(
                "FAIL · {}",
                self.system_error
                    .as_deref()
                    .unwrap_or("not started")
            )
        };
        let level_bar = level_meter(self.rms);
        let mut s = format!(
            "system: {sys}\n\
             mic: {mic}\n\
             capture: {sr} Hz · {ch} ch · buffer {buf_s:.1}s ({buf_n} samples)\n\
             level: {bar}  rms={rms:.4} peak={peak:.4} ({db:.0} dBFS)\n\
             stt: {engine} · {status}\n\
             model: {model}\n\
             progress: processed={proc:.1}s  chunks={chunks}  silence_skip={skip}  segments={segs}\n\
             last_stt: {last}\n\
             note: system = Core Audio process tap (Zoom/Meet/etc). Needs Audio Capture permission.\n\
             if system FAIL: System Settings → Privacy & Security → Audio Capture → enable Terminal/iTerm, quit+reopen",
            sr = self.sample_rate,
            ch = self.channels,
            buf_s = self.buffer_secs,
            buf_n = self.buffer_samples,
            bar = level_bar,
            rms = self.rms,
            peak = self.peak,
            db = self.level_db,
            engine = self.stt_engine,
            status = self.stt_status,
            model = if self.stt_model_path.is_empty() {
                "(none)"
            } else {
                &self.stt_model_path
            },
            proc = self.processed_audio_secs,
            chunks = self.chunks_processed,
            skip = self.chunks_skipped_silence,
            segs = self.segments_emitted,
            last = self
                .last_stt_ms
                .map(|ms| format!("{ms} ms"))
                .unwrap_or_else(|| "—".into()),
        );
        if !self.log.is_empty() {
            s.push_str("\n--- log ---\n");
            for line in &self.log {
                s.push_str(line);
                s.push('\n');
            }
        }
        s
    }
}

fn level_meter(rms: f32) -> String {
    let n = ((rms * 40.0).clamp(0.0, 20.0)) as usize;
    let mut bar = String::from("[");
    for i in 0..20 {
        bar.push(if i < n { '#' } else { '.' });
    }
    bar.push(']');
    bar
}

/// Pending transcript line produced by the STT worker (TUI/DB consumer).
#[derive(Debug, Clone)]
pub struct SttSegment {
    pub text: String,
    pub audio_start: f64,
    pub audio_end: f64,
}

/// Shared STT output queue + diagnostics.
pub type DiagHandle = Arc<Mutex<RecordingDiagnostics>>;
pub type SegmentQueue = Arc<Mutex<Vec<SttSegment>>>;

/// Resolve model directory for the current selection.
pub fn resolve_stt_model_dir(paths: &MeetilyPaths, selection: &ModelSelection) -> Option<PathBuf> {
    let p = resolve_model_path(&paths.models_dir, selection.provider, &selection.model)?;
    match selection.provider {
        TranscriptionProvider::Parakeet => {
            if p.is_dir() {
                Some(p)
            } else {
                None
            }
        }
        TranscriptionProvider::LocalWhisper => {
            // Whisper ggml file path — live STT path is Parakeet-first for now.
            if p.is_file() {
                Some(p)
            } else {
                None
            }
        }
    }
}

/// Downmix interleaved multi-channel frames to mono.
pub fn downmix_interleaved(interleaved: &[f32], channels: u16) -> Vec<f32> {
    let ch = channels.max(1) as usize;
    if ch == 1 {
        return interleaved.to_vec();
    }
    let frames = interleaved.len() / ch;
    let mut mono = Vec::with_capacity(frames);
    for i in 0..frames {
        let mut sum = 0.0f32;
        for c in 0..ch {
            sum += interleaved[i * ch + c];
        }
        mono.push(sum / ch as f32);
    }
    mono
}

/// Linear resample mono audio to target rate.
pub fn resample_linear(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if input.is_empty() || from_rate == 0 {
        return Vec::new();
    }
    if from_rate == to_rate {
        return input.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 / ratio;
        let i0 = src.floor() as usize;
        let i1 = (i0 + 1).min(input.len() - 1);
        let t = (src - i0 as f64) as f32;
        let s = input[i0] * (1.0 - t) + input[i1] * t;
        out.push(s);
    }
    out
}

pub fn compute_rms_peak(samples: &[f32]) -> (f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let mut sum = 0.0f32;
    let mut peak = 0.0f32;
    for &s in samples {
        let a = s.abs();
        sum += s * s;
        if a > peak {
            peak = a;
        }
    }
    let rms = (sum / samples.len() as f32).sqrt();
    (rms, peak)
}

pub fn rms_to_db(rms: f32) -> f32 {
    if rms <= 1e-9 {
        -120.0
    } else {
        20.0 * rms.log10()
    }
}

/// Spawn background STT worker. Returns join handle (or None if spawn failed).
pub fn spawn_stt_worker(
    samples: Arc<Mutex<Vec<f32>>>,
    sample_rate: u32,
    channels: u16,
    stop: Arc<std::sync::atomic::AtomicBool>,
    diag: DiagHandle,
    out_segments: SegmentQueue,
    paths: MeetilyPaths,
    selection: ModelSelection,
    session_start: Instant,
) -> Option<std::thread::JoinHandle<()>> {
    // Tests / CI: skip heavy ONNX + mic-level loops unless explicitly enabled.
    if std::env::var_os("MEETICULOUS_DISABLE_STT").is_some() {
        if let Ok(mut d) = diag.lock() {
            d.stt_status = "disabled (MEETICULOUS_DISABLE_STT)".into();
            d.push_log("STT worker not started (MEETICULOUS_DISABLE_STT)");
        }
        return None;
    }
    std::thread::Builder::new()
        .name("meeticulous-stt".into())
        .spawn(move || {
            stt_worker_loop(
                samples,
                sample_rate,
                channels,
                stop,
                diag,
                out_segments,
                paths,
                selection,
                session_start,
            );
        })
        .ok()
}

fn stt_worker_loop(
    samples: Arc<Mutex<Vec<f32>>>,
    sample_rate: u32,
    channels: u16,
    stop: Arc<std::sync::atomic::AtomicBool>,
    diag: DiagHandle,
    out_segments: SegmentQueue,
    paths: MeetilyPaths,
    selection: ModelSelection,
    session_start: Instant,
) {
    {
        let mut d = diag.lock().unwrap();
        d.stt_engine = format!("{} / {}", selection.provider, selection.model);
        d.stt_status = "loading model…".into();
        d.push_log(format!(
            "STT worker start provider={} model={}",
            selection.provider, selection.model
        ));
    }

    let model_dir = match selection.provider {
        TranscriptionProvider::Parakeet => resolve_stt_model_dir(&paths, &selection),
        TranscriptionProvider::LocalWhisper => {
            // Prefer parakeet if whisper selected but only parakeet present
            let whisper = resolve_stt_model_dir(&paths, &selection);
            let mut parakeet_sel = selection.clone();
            parakeet_sel.provider = TranscriptionProvider::Parakeet;
            parakeet_sel.model = crate::models::DEFAULT_PARAKEET_MODEL.to_string();
            let pk = resolve_stt_model_dir(&paths, &parakeet_sel);
            if whisper.is_none() && pk.is_some() {
                let mut d = diag.lock().unwrap();
                d.push_log("Whisper ggml not found; falling back to Parakeet for live STT");
                d.stt_engine = format!("parakeet / {}", parakeet_sel.model);
                pk
            } else {
                whisper
            }
        }
    };

    let Some(model_path) = model_dir else {
        let mut d = diag.lock().unwrap();
        d.stt_status = "NO MODEL — install Parakeet under models/parakeet/".into();
        d.push_log(format!(
            "ERROR: no model for {} / {} under {}",
            selection.provider,
            selection.model,
            paths.models_dir.display()
        ));
        // Keep updating levels even without STT
        levels_only_loop(&samples, sample_rate, channels, &stop, &diag);
        return;
    };

    {
        let mut d = diag.lock().unwrap();
        d.stt_model_path = model_path.display().to_string();
        d.push_log(format!("loading ONNX from {}", model_path.display()));
    }

    let quantized = model_path
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.contains("int8"))
        .unwrap_or(true);

    let mut model = match ParakeetModel::new(&model_path, quantized) {
        Ok(m) => m,
        Err(e) => {
            // try opposite quantization flag
            match ParakeetModel::new(&model_path, !quantized) {
                Ok(m) => m,
                Err(e2) => {
                    let mut d = diag.lock().unwrap();
                    d.stt_status = format!("model load failed: {e2}");
                    d.push_log(format!("ERROR load model: {e} / retry: {e2}"));
                    levels_only_loop(&samples, sample_rate, channels, &stop, &diag);
                    return;
                }
            }
        }
    };

    {
        let mut d = diag.lock().unwrap();
        d.stt_status = "ready · waiting for speech".into();
        d.push_log("Parakeet model loaded OK");
    }

    let mut cursor: usize = 0; // index into mono sample history (device rate, mono)
    let mut mono_history: Vec<f32> = Vec::new();
    // Snapshot length of capture buffer last time we pulled (detect ring drain)
    let mut last_buf_len: usize = 0;
    let mut absolute_mono_offset: u64 = 0; // total mono samples ever (for timestamps)

    let chunk_secs = 3.0f32;
    let hop_secs = 2.5f32;
    let min_rms = 0.008f32;
    let ch = channels.max(1) as usize;

    while !stop.load(std::sync::atomic::Ordering::SeqCst) {
        // Pull capture buffer snapshot. If len shrank, ring was drained — resync.
        let buf_snapshot = samples.lock().unwrap().clone();
        if buf_snapshot.len() < last_buf_len {
            // Buffer drained from the front; keep mono_history, start consuming from end.
            last_buf_len = 0;
            {
                let mut d = diag.lock().unwrap();
                d.push_log("capture buffer rotated — resyncing audio cursor");
            }
        }
        if buf_snapshot.len() > last_buf_len {
            let new_interleaved = &buf_snapshot[last_buf_len..];
            last_buf_len = buf_snapshot.len();
            let usable = new_interleaved.len() - (new_interleaved.len() % ch);
            if usable > 0 {
                let mono = downmix_interleaved(&new_interleaved[..usable], channels);
                mono_history.extend_from_slice(&mono);
            }
        }

        // Level from last 0.25s
        let window = ((sample_rate as f32) * 0.25) as usize;
        let recent = if mono_history.len() > window {
            &mono_history[mono_history.len() - window..]
        } else {
            mono_history.as_slice()
        };
        let (rms, peak) = compute_rms_peak(recent);
        {
            let mut d = diag.lock().unwrap();
            d.buffer_samples = mono_history.len();
            d.buffer_secs = mono_history.len() as f32 / sample_rate.max(1) as f32;
            d.rms = rms;
            d.peak = peak;
            d.level_db = rms_to_db(rms);
            d.sample_rate = sample_rate;
            d.channels = channels;
        }

        let chunk_samples = ((sample_rate as f32) * chunk_secs) as usize;
        let hop_samples = ((sample_rate as f32) * hop_secs) as usize;

        if mono_history.len().saturating_sub(cursor) >= chunk_samples {
            let end = cursor + chunk_samples;
            let chunk = mono_history[cursor..end].to_vec();
            let (c_rms, _) = compute_rms_peak(&chunk);
            let audio_start =
                (absolute_mono_offset + cursor as u64) as f64 / sample_rate as f64;
            let audio_end =
                (absolute_mono_offset + end as u64) as f64 / sample_rate as f64;

            if c_rms < min_rms {
                let mut d = diag.lock().unwrap();
                d.chunks_skipped_silence += 1;
                d.stt_status = format!("silence skip (rms={c_rms:.4}) · listening…");
                d.processed_audio_secs = audio_end as f32;
                if d.chunks_skipped_silence % 5 == 1 {
                    d.push_log(format!(
                        "skip silence chunk @{audio_start:.1}s rms={c_rms:.4} (need ≥{min_rms})"
                    ));
                }
                cursor += hop_samples;
            } else {
                {
                    let mut d = diag.lock().unwrap();
                    d.stt_status = format!("transcribing {chunk_secs:.0}s @{audio_start:.1}s…");
                    d.push_log(format!(
                        "STT chunk @{audio_start:.1}-{audio_end:.1}s rms={c_rms:.4} samples={}",
                        chunk.len()
                    ));
                }

                let t0 = Instant::now();
                let resampled = resample_linear(&chunk, sample_rate, TARGET_SAMPLE_RATE);
                let result = model.transcribe_samples(resampled);
                let ms = t0.elapsed().as_millis() as u64;

                match result {
                    Ok(ts) => {
                        let text = ts.text.trim().to_string();
                        let mut d = diag.lock().unwrap();
                        d.chunks_processed += 1;
                        d.last_stt_ms = Some(ms);
                        d.processed_audio_secs = audio_end as f32;
                        if text.is_empty() {
                            d.stt_status = format!("empty text ({ms}ms) · listening…");
                            d.push_log(format!(
                                "STT empty @{audio_start:.1}s in {ms}ms (model heard silence/noise)"
                            ));
                        } else {
                            d.segments_emitted += 1;
                            d.stt_status = format!("ok · last \"{}\"", truncate(&text, 40));
                            d.push_log(format!("STT +{ms}ms: {text}"));
                            drop(d);
                            if let Ok(mut q) = out_segments.lock() {
                                q.push(SttSegment {
                                    text,
                                    audio_start: session_start.elapsed().as_secs_f64()
                                        - (audio_end - audio_start)
                                        + audio_start.min(session_start.elapsed().as_secs_f64()),
                                    // Prefer buffer-relative times
                                    audio_end,
                                });
                                // Fix start/end to buffer-relative which matches session if capture started with session
                                if let Some(last) = q.last_mut() {
                                    last.audio_start = audio_start;
                                    last.audio_end = audio_end;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let mut d = diag.lock().unwrap();
                        d.chunks_processed += 1;
                        d.last_stt_ms = Some(ms);
                        d.stt_status = format!("STT error: {e}");
                        d.push_log(format!("ERROR STT @{audio_start:.1}s: {e}"));
                    }
                }
                cursor += hop_samples;
            }

            // Cap mono history to ~2 minutes to bound memory; adjust cursor
            const MAX_MONO: usize = 48_000 * 120;
            if mono_history.len() > MAX_MONO {
                let drop_n = mono_history.len() - MAX_MONO;
                mono_history.drain(0..drop_n);
                cursor = cursor.saturating_sub(drop_n);
                absolute_mono_offset += drop_n as u64;
            }
        }

        std::thread::sleep(Duration::from_millis(200));
    }

    let mut d = diag.lock().unwrap();
    d.stt_status = "stopped".into();
    d.push_log("STT worker stopped");
}

fn levels_only_loop(
    samples: &Arc<Mutex<Vec<f32>>>,
    sample_rate: u32,
    channels: u16,
    stop: &Arc<std::sync::atomic::AtomicBool>,
    diag: &DiagHandle,
) {
    while !stop.load(std::sync::atomic::Ordering::SeqCst) {
        let buf = samples.lock().unwrap().clone();
        let mono = downmix_interleaved(&buf, channels);
        let window = mono.len().saturating_sub(((sample_rate as f32) * 0.25) as usize);
        let recent = &mono[window..];
        let (rms, peak) = compute_rms_peak(recent);
        {
            let mut d = diag.lock().unwrap();
            d.buffer_samples = mono.len();
            d.buffer_secs = mono.len() as f32 / sample_rate.max(1) as f32;
            d.rms = rms;
            d.peak = peak;
            d.level_db = rms_to_db(rms);
            d.sample_rate = sample_rate;
            d.channels = channels;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max).collect();
        format!("{t}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_stereo() {
        let interleaved = vec![1.0, -1.0, 0.5, 0.5];
        let mono = downmix_interleaved(&interleaved, 2);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.0).abs() < 1e-6);
        assert!((mono[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn resample_doubles_length() {
        let input = vec![0.0, 1.0, 0.0, -1.0];
        let out = resample_linear(&input, 8_000, 16_000);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn diagnostics_verbose_includes_mic_and_stt() {
        let mut d = RecordingDiagnostics::default();
        d.mic_ok = true;
        d.mic_device = "Built-in".into();
        d.stt_status = "ready".into();
        d.push_log("hello");
        let v = d.format_verbose();
        assert!(v.contains("mic:"));
        assert!(v.contains("stt:"));
        assert!(v.contains("hello"));
    }
}
