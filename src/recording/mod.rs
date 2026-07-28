//! Local recording session: capture mic audio, live STT, append transcript to Meetily DB.

use crate::db::{append_transcript_segment, create_meeting, upsert_transcript_chunk, Meeting};
use crate::models::ModelSelection;
use crate::paths::MeetilyPaths;
use crate::stt::{spawn_stt_worker, DiagHandle, RecordingDiagnostics, SegmentQueue, SttSegment};
use chrono::Utc;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{WavSpec, WavWriter};
use sqlx::SqlitePool;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Live transcript line shown in the TUI.
#[derive(Debug, Clone)]
pub struct LiveSegment {
    pub text: String,
    pub timestamp: String,
    pub audio_start: f64,
    pub audio_end: f64,
}

/// WAV file shared by audio-capture callbacks for the lifetime of a recording.
pub(crate) type LiveWavWriter = Arc<Mutex<Option<WavWriter<BufWriter<File>>>>>;

/// Handle controlling a background recording + STT session.
pub struct RecordingHandle {
    pub meeting_id: String,
    pub title: String,
    pub folder_path: PathBuf,
    pub wav_path: PathBuf,
    stop_flag: Arc<AtomicBool>,
    wav_writer: LiveWavWriter,
    started: Instant,
    #[allow(dead_code)]
    started_at: chrono::DateTime<Utc>,
    live: Arc<Mutex<Vec<LiveSegment>>>,
    pub diagnostics: DiagHandle,
    pending_stt: SegmentQueue,
    /// Optional mic stream kept alive for the session lifetime.
    _mic_stream: Option<cpal::Stream>,
    /// macOS system-audio process tap (Meetily Core Audio path).
    #[cfg(target_os = "macos")]
    _system_audio: Option<crate::system_audio::SystemAudioSession>,
    stt_join: Option<std::thread::JoinHandle<()>>,
}

impl RecordingHandle {
    pub fn elapsed_secs(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    pub fn live_segments(&self) -> Vec<LiveSegment> {
        self.live.lock().map(|g| g.clone()).unwrap_or_default()
    }

    pub fn diagnostics_snapshot(&self) -> RecordingDiagnostics {
        self.diagnostics
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    pub fn verbose_status(&self) -> String {
        self.diagnostics
            .lock()
            .map(|g| g.format_verbose())
            .unwrap_or_else(|_| "diagnostics unavailable".into())
    }
}

/// Start a meeting + mic capture + live STT session.
pub async fn start_recording(
    pool: &SqlitePool,
    paths: &MeetilyPaths,
    title: Option<&str>,
    selection: &ModelSelection,
) -> anyhow::Result<RecordingHandle> {
    let started_at = Utc::now();
    let stamp = started_at.format("%Y-%m-%d_%H-%M-%S");
    let title = title
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("Meeting {stamp}"));

    let folder_name = format!("Meeting {stamp}_{}", started_at.format("%Y-%m-%d_%H-%M"));
    let folder_path = paths.recordings_dir.join(&folder_name);
    std::fs::create_dir_all(&folder_path)?;
    let wav_path = folder_path.join("recording.wav");

    let meeting_id = create_meeting(pool, &title, Some(&folder_path.to_string_lossy())).await?;

    upsert_transcript_chunk(
        pool,
        &meeting_id,
        &title,
        "",
        selection.provider.as_str(),
        &selection.model,
    )
    .await?;

    let stop_flag = Arc::new(AtomicBool::new(false));
    // This feed is drained by the STT worker so live transcription never has to
    // scan the recording buffer or depend on its bounded length changing.
    let stt_samples = Arc::new(Mutex::new(Vec::<f32>::new()));
    let wav_writer: LiveWavWriter = Arc::new(Mutex::new(None));
    let live = Arc::new(Mutex::new(Vec::new()));
    let diagnostics: DiagHandle = Arc::new(Mutex::new(RecordingDiagnostics::default()));
    let pending_stt: SegmentQueue = Arc::new(Mutex::new(Vec::new()));

    {
        let mut d = diagnostics.lock().unwrap();
        d.push_log(format!("session start meeting_id={meeting_id}"));
        d.push_log(format!("folder={}", folder_path.display()));
        d.push_log(format!(
            "requested STT {} / {}",
            selection.provider, selection.model
        ));
    }

    // --- System audio first (Zoom/Meet/etc), then mic as secondary mix ---
    // WAV is written live (unbounded) as samples arrive — never a rolling
    // in-memory window. That old 3-minute drain was the source of truncated files.
    let mut sample_rate: u32 = 16_000;
    let mut channels: u16 = 1;

    #[cfg(target_os = "macos")]
    let system_audio = {
        match crate::system_audio::start_system_audio_capture(
            &wav_path,
            wav_writer.clone(),
            stt_samples.clone(),
            stop_flag.clone(),
        ) {
            Ok(sess) => {
                sample_rate = sess.sample_rate.max(1);
                channels = 1; // process tap is mono
                let mut d = diagnostics.lock().unwrap();
                d.system_ok = true;
                d.system_device = sess.device_name.clone();
                d.sample_rate = sample_rate;
                d.channels = channels;
                d.push_log(format!(
                    "SYSTEM AUDIO OK: {} @ {} Hz (Core Audio process tap)",
                    sess.device_name, sample_rate
                ));
                d.push_log("capturing what your Mac plays (Zoom/Meet/browser) — not just the mic");
                d.push_log(format!(
                    "live WAV open at {} (full session, no duration cap)",
                    wav_path.display()
                ));
                Some(sess)
            }
            Err(e) => {
                let mut d = diagnostics.lock().unwrap();
                d.system_ok = false;
                d.system_error = Some(e.to_string());
                d.push_log(format!("SYSTEM AUDIO FAIL: {e}"));
                d.push_log(
                    "Grant Audio Capture: System Settings → Privacy & Security → Audio Capture",
                );
                d.push_log(
                    "Enable for Terminal / iTerm / your shell host, fully quit & reopen, then retry",
                );
                None
            }
        }
    };
    #[cfg(not(target_os = "macos"))]
    let system_audio = ();

    // Mic: only feed the STT buffer when system audio is unavailable.
    // When system tap is live, it already captures meeting playback; mixing both
    // into one stream desyncs the timeline. Mic is still opened for diagnostics.
    #[cfg(target_os = "macos")]
    let system_up = system_audio.is_some();
    #[cfg(not(target_os = "macos"))]
    let system_up = false;

    let mic_wav_writer = if system_up {
        // Black-hole: open mic for status only, don't append into the meeting WAV.
        Arc::new(Mutex::new(None))
    } else {
        // Mic-only path: open the live WAV before the capture stream starts.
        *wav_writer
            .lock()
            .map_err(|_| anyhow::anyhow!("WAV writer lock poisoned"))? =
            Some(create_wav_writer(&wav_path, sample_rate, 1)?);
        if let Ok(mut d) = diagnostics.lock() {
            d.push_log(format!(
                "live WAV open at {} @ {} Hz (mic-only, no duration cap)",
                wav_path.display(),
                sample_rate
            ));
        }
        wav_writer.clone()
    };
    let mic_stt_feed = if system_up {
        Arc::new(Mutex::new(Vec::<f32>::new()))
    } else {
        stt_samples.clone()
    };

    let mic_stream = match start_cpal_capture(
        mic_wav_writer,
        mic_stt_feed,
        stop_flag.clone(),
        diagnostics.clone(),
        sample_rate,
    ) {
        Ok((stream, mic_sr, mic_ch)) => {
            let _ = (mic_sr, mic_ch);
            if system_up {
                let mut d = diagnostics.lock().unwrap();
                d.push_log("mic open for presence; STT uses SYSTEM AUDIO only (meeting playback)");
            }
            stream
        }
        Err(e) => {
            let mut d = diagnostics.lock().unwrap();
            d.mic_ok = false;
            d.mic_error = Some(e.to_string());
            d.push_log(format!("MIC ERROR: {e}"));
            None
        }
    };

    {
        let mut d = diagnostics.lock().unwrap();
        if !d.system_ok && !d.mic_ok {
            d.push_log("NO AUDIO SOURCES — grant mic + Audio Capture permissions");
        } else if d.system_ok {
            d.push_log("primary source: SYSTEM AUDIO (meeting playback)");
        } else {
            d.push_log("primary source: MIC only (system tap unavailable)");
        }
    }

    let started = Instant::now();
    let stt_join = spawn_stt_worker(
        stt_samples,
        sample_rate,
        channels,
        stop_flag.clone(),
        diagnostics.clone(),
        pending_stt.clone(),
        paths.clone(),
        selection.clone(),
        started,
    );

    Ok(RecordingHandle {
        meeting_id,
        title,
        folder_path,
        wav_path,
        stop_flag,
        wav_writer,
        started,
        started_at,
        live,
        diagnostics,
        pending_stt,
        _mic_stream: mic_stream,
        #[cfg(target_os = "macos")]
        _system_audio: system_audio,
        stt_join,
    })
}

/// Open default mic. When `target_rate` differs from the device rate, samples are
/// linearly resampled to `target_rate` mono so they mix cleanly with system audio.
fn start_cpal_capture(
    wav_writer: LiveWavWriter,
    stt_samples: Arc<Mutex<Vec<f32>>>,
    stop_flag: Arc<AtomicBool>,
    diag: DiagHandle,
    target_rate: u32,
) -> anyhow::Result<(Option<cpal::Stream>, u32, u16)> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("no default input device (check mic permission)"))?;
    let device_name = device.name().unwrap_or_else(|_| "(unnamed input)".into());
    let config = device
        .default_input_config()
        .map_err(|e| anyhow::anyhow!("input config: {e}"))?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels();
    let stream_config: cpal::StreamConfig = config.clone().into();

    {
        let mut d = diag.lock().unwrap();
        d.mic_ok = true;
        d.mic_device = device_name.clone();
        d.push_log(format!(
            "mic opened: {device_name} @ {sample_rate} Hz, {channels} ch, format={:?} (mix→{target_rate} Hz mono)",
            config.sample_format()
        ));
    }

    let capture = CaptureContext {
        wav_writer,
        stt_samples,
        stop_flag,
        channels,
        from_rate: sample_rate,
        to_rate: target_rate,
    };
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => build_stream_f32(&device, &stream_config, capture)?,
        cpal::SampleFormat::I16 => build_stream_i16(&device, &stream_config, capture)?,
        cpal::SampleFormat::U16 => build_stream_u16(&device, &stream_config, capture)?,
        other => anyhow::bail!("unsupported sample format: {other:?}"),
    };
    stream.play()?;
    {
        let mut d = diag.lock().unwrap();
        d.push_log("mic stream playing (mixed with system when available)");
    }
    Ok((Some(stream), sample_rate, channels))
}

/// Downmix + optional resample, then append into shared mono buffer.
struct CaptureContext {
    wav_writer: LiveWavWriter,
    stt_samples: Arc<Mutex<Vec<f32>>>,
    stop_flag: Arc<AtomicBool>,
    channels: u16,
    from_rate: u32,
    to_rate: u32,
}

fn push_mixed_mono(
    wav_writer: &LiveWavWriter,
    stt_samples: &Arc<Mutex<Vec<f32>>>,
    interleaved: &[f32],
    channels: u16,
    from_rate: u32,
    to_rate: u32,
) {
    let mono = crate::stt::downmix_interleaved(interleaved, channels);
    let mono = if from_rate != to_rate && to_rate > 0 {
        crate::stt::resample_linear(&mono, from_rate, to_rate)
    } else {
        mono
    };
    append_wav_samples(wav_writer, &mono);
    if let Ok(mut feed) = stt_samples.lock() {
        feed.extend_from_slice(&mono);
    }
}

fn build_stream_f32(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    capture: CaptureContext,
) -> anyhow::Result<cpal::Stream> {
    let err_fn = |e| log::error!("audio stream error: {e}");
    let stream = device.build_input_stream(
        config,
        move |data: &[f32], _| {
            if capture.stop_flag.load(Ordering::Relaxed) {
                return;
            }
            push_mixed_mono(
                &capture.wav_writer,
                &capture.stt_samples,
                data,
                capture.channels,
                capture.from_rate,
                capture.to_rate,
            );
        },
        err_fn,
        None,
    )?;
    Ok(stream)
}

fn build_stream_i16(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    capture: CaptureContext,
) -> anyhow::Result<cpal::Stream> {
    let err_fn = |e| log::error!("audio stream error: {e}");
    let stream = device.build_input_stream(
        config,
        move |data: &[i16], _| {
            if capture.stop_flag.load(Ordering::Relaxed) {
                return;
            }
            let f: Vec<f32> = data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
            push_mixed_mono(
                &capture.wav_writer,
                &capture.stt_samples,
                &f,
                capture.channels,
                capture.from_rate,
                capture.to_rate,
            );
        },
        err_fn,
        None,
    )?;
    Ok(stream)
}

fn build_stream_u16(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    capture: CaptureContext,
) -> anyhow::Result<cpal::Stream> {
    let err_fn = |e| log::error!("audio stream error: {e}");
    let stream = device.build_input_stream(
        config,
        move |data: &[u16], _| {
            if capture.stop_flag.load(Ordering::Relaxed) {
                return;
            }
            let f: Vec<f32> = data
                .iter()
                .map(|&s| (s as f32 - 32768.0) / 32768.0)
                .collect();
            push_mixed_mono(
                &capture.wav_writer,
                &capture.stt_samples,
                &f,
                capture.channels,
                capture.from_rate,
                capture.to_rate,
            );
        },
        err_fn,
        None,
    )?;
    Ok(stream)
}

/// Drain STT worker queue into DB + live segments. Call from TUI tick.
pub async fn drain_stt_segments(
    pool: &SqlitePool,
    handle: &RecordingHandle,
) -> anyhow::Result<Vec<LiveSegment>> {
    let pending: Vec<SttSegment> = {
        let mut q = handle
            .pending_stt
            .lock()
            .map_err(|_| anyhow::anyhow!("stt queue lock"))?;
        std::mem::take(&mut *q)
    };
    let mut out = Vec::new();
    for p in pending {
        let seg = append_text_segment(
            pool,
            handle,
            &p.text,
            Some(p.audio_start),
            Some(p.audio_end),
        )
        .await?;
        out.push(seg);
    }
    Ok(out)
}

/// Append a transcript line during a live session (also writes to DB).
pub async fn append_text_segment(
    pool: &SqlitePool,
    handle: &RecordingHandle,
    text: &str,
    audio_start: Option<f64>,
    audio_end: Option<f64>,
) -> anyhow::Result<LiveSegment> {
    let ts = Utc::now().to_rfc3339();
    let start = audio_start.unwrap_or_else(|| handle.elapsed_secs());
    let end = audio_end.unwrap_or(start);
    let duration = (end - start).max(0.0);
    append_transcript_segment(
        pool,
        &handle.meeting_id,
        text,
        &ts,
        Some(start),
        Some(end),
        Some(duration),
        None,
    )
    .await?;
    let seg = LiveSegment {
        text: text.to_string(),
        timestamp: ts,
        audio_start: start,
        audio_end: end,
    };
    if let Ok(mut g) = handle.live.lock() {
        g.push(seg.clone());
    }
    if let Ok(mut d) = handle.diagnostics.lock() {
        d.push_log(format!(
            "segment saved: {}",
            text.chars().take(60).collect::<String>()
        ));
    }
    Ok(seg)
}

/// Inject transcript text (tests / dry-run without real STT).
pub async fn inject_live_transcript(
    pool: &SqlitePool,
    handle: &RecordingHandle,
    lines: &[&str],
) -> anyhow::Result<Vec<LiveSegment>> {
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let start = i as f64 * 2.0;
        let seg = append_text_segment(pool, handle, line, Some(start), Some(start + 2.0)).await?;
        out.push(seg);
    }
    Ok(out)
}

/// Stop capture, flush WAV, finalize transcript chunk blob.
pub async fn stop_recording(
    pool: &SqlitePool,
    mut handle: RecordingHandle,
    selection: &ModelSelection,
) -> anyhow::Result<Meeting> {
    // Drain any remaining STT first
    let _ = drain_stt_segments(pool, &handle).await;

    handle.stop_flag.store(true, Ordering::SeqCst);
    // Drop capture streams so callbacks stop feeding the buffer
    handle._mic_stream = None;
    #[cfg(target_os = "macos")]
    {
        handle._system_audio = None;
    }
    // Join STT worker (bounded wait so UI never hangs forever)
    if let Some(join) = handle.stt_join.take() {
        let _ = std::thread::spawn(move || {
            let _ = join.join();
        });
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    } else {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    let writer = handle
        .wav_writer
        .lock()
        .map_err(|_| anyhow::anyhow!("WAV writer lock poisoned"))?
        .take();
    if let Some(writer) = writer {
        writer.finalize()?;
        if let Ok(mut d) = handle.diagnostics.lock() {
            d.push_log(format!("finalized WAV {}", handle.wav_path.display()));
        }
    } else {
        if let Ok(mut d) = handle.diagnostics.lock() {
            d.push_log("WARN: WAV writer was unavailable");
        }
    }

    let full_text = handle
        .live
        .lock()
        .map(|g| {
            g.iter()
                .map(|s| s.text.clone())
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();

    upsert_transcript_chunk(
        pool,
        &handle.meeting_id,
        &handle.title,
        &full_text,
        selection.provider.as_str(),
        &selection.model,
    )
    .await?;

    let meeting = crate::db::get_meeting(pool, &handle.meeting_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("meeting missing after stop"))?;
    Ok(meeting)
}

pub(crate) fn create_wav_writer(
    path: &Path,
    sample_rate: u32,
    channels: u16,
) -> anyhow::Result<WavWriter<BufWriter<File>>> {
    let spec = WavSpec {
        channels: channels.max(1),
        sample_rate: sample_rate.max(1),
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    // Larger buffer: audio callbacks append continuously for long meetings.
    let file = File::create(path)?;
    let buffered = BufWriter::with_capacity(256 * 1024, file);
    Ok(WavWriter::new(buffered, spec)?)
}

/// Append mono f32 samples to the live meeting WAV. Unbounded — no rolling cap.
pub(crate) fn append_wav_samples(writer: &LiveWavWriter, samples: &[f32]) {
    let Ok(mut guard) = writer.lock() else {
        return;
    };
    let Some(writer) = guard.as_mut() else {
        return;
    };
    for &s in samples {
        let clipped = s.clamp(-1.0, 1.0);
        let i = (clipped * i16::MAX as f32) as i16;
        if writer.write_sample(i).is_err() {
            break;
        }
    }
}

#[cfg(test)]
fn write_wav(path: &Path, samples: &[f32], sample_rate: u32, channels: u16) -> anyhow::Result<()> {
    let mut writer = create_wav_writer(path, sample_rate, channels)?;
    for &sample in samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Import / re-transcribe: create a meeting from an audio file path + transcript lines.
pub async fn import_audio_file(
    pool: &SqlitePool,
    paths: &MeetilyPaths,
    audio_path: &Path,
    title: Option<&str>,
    transcript_lines: &[&str],
    selection: &ModelSelection,
) -> anyhow::Result<String> {
    if !audio_path.exists() {
        anyhow::bail!("audio file not found: {}", audio_path.display());
    }
    let started_at = Utc::now();
    let stamp = started_at.format("%Y-%m-%d_%H-%M-%S");
    let title = title.map(|s| s.to_string()).unwrap_or_else(|| {
        audio_path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("Import {stamp}"))
    });

    let folder_name = format!("Import {stamp}");
    let folder_path = paths.recordings_dir.join(&folder_name);
    std::fs::create_dir_all(&folder_path)?;
    let dest = folder_path.join(
        audio_path
            .file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new("audio.bin")),
    );
    std::fs::copy(audio_path, &dest)?;

    let meeting_id = create_meeting(pool, &title, Some(&folder_path.to_string_lossy())).await?;
    for (i, line) in transcript_lines.iter().enumerate() {
        let ts = Utc::now().to_rfc3339();
        let start = i as f64 * 2.0;
        append_transcript_segment(
            pool,
            &meeting_id,
            line,
            &ts,
            Some(start),
            Some(start + 2.0),
            Some(2.0),
            None,
        )
        .await?;
    }
    let full = transcript_lines.join("\n");
    upsert_transcript_chunk(
        pool,
        &meeting_id,
        &title,
        &full,
        selection.provider.as_str(),
        &selection.model,
    )
    .await?;
    Ok(meeting_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{list_meetings, load_transcript_text, open_database};
    use crate::paths::MeetilyPaths;

    /// Regression: the old in-memory capture buffer drained to a 3-minute rolling
    /// window (`48_000 * 60 * 3`). Live WAV append must keep the full session.
    #[test]
    fn live_wav_writer_keeps_more_than_three_minutes() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("long.wav");
        let sample_rate = 48_000u32;
        // 3 minutes + 30 seconds of mono silence (as i16 frames via f32).
        let total_secs = 210u32;
        let total_samples = sample_rate as usize * total_secs as usize;

        let writer: LiveWavWriter = Arc::new(Mutex::new(Some(
            create_wav_writer(&path, sample_rate, 1).unwrap(),
        )));

        // Append in chunks to mimic the capture callback.
        let chunk = vec![0.1f32; sample_rate as usize]; // 1s
        for _ in 0..total_secs {
            append_wav_samples(&writer, &chunk);
        }

        let w = writer.lock().unwrap().take().unwrap();
        w.finalize().unwrap();

        let reader = hound::WavReader::open(&path).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, sample_rate);
        assert_eq!(spec.channels, 1);
        let n = reader.len() as usize;
        assert_eq!(
            n,
            total_samples,
            "expected {total_secs}s of audio, got {:.1}s (still capped?)",
            n as f64 / sample_rate as f64
        );
        // Explicitly beyond the historical 3-minute bug.
        assert!(
            n > sample_rate as usize * 180,
            "WAV must exceed 3 minutes; got {:.1}s",
            n as f64 / sample_rate as f64
        );
    }

    #[tokio::test]
    async fn start_stop_recording_appends_segments_to_db() {
        // Avoid hanging the suite on ONNX load / mic permission dialogs.
        std::env::set_var("MEETICULOUS_DISABLE_STT", "1");
        let tmp = tempfile::tempdir().unwrap();
        let paths = MeetilyPaths::with_dirs(
            tmp.path().join("com.meetily.ai"),
            tmp.path().join("Movies").join("meetily-recordings"),
        );
        paths.ensure_dirs().unwrap();
        let pool = open_database(&paths.db_path).await.unwrap();
        let sel = ModelSelection::default();

        let handle = start_recording(&pool, &paths, Some("Unit Rec"), &sel)
            .await
            .unwrap();
        assert!(handle.meeting_id.starts_with("meeting-"));
        assert!(handle.folder_path.starts_with(&paths.recordings_dir));
        // Verbose diagnostics always available
        let v = handle.verbose_status();
        assert!(v.contains("mic:") || v.contains("stt:"));

        inject_live_transcript(&pool, &handle, &["Hello world", "Second line"])
            .await
            .unwrap();

        let meeting = stop_recording(&pool, handle, &sel).await.unwrap();
        assert_eq!(meeting.title, "Unit Rec");

        let text = load_transcript_text(&pool, &meeting.id).await.unwrap();
        assert!(text.contains("Hello world"));
        assert!(text.contains("Second line"));

        let meetings = list_meetings(&pool).await.unwrap();
        assert!(meetings.iter().any(|m| m.id == meeting.id));
    }

    #[tokio::test]
    async fn import_audio_creates_meeting_under_recordings_root() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = MeetilyPaths::with_dirs(
            tmp.path().join("com.meetily.ai"),
            tmp.path().join("Movies").join("meetily-recordings"),
        );
        paths.ensure_dirs().unwrap();
        let pool = open_database(&paths.db_path).await.unwrap();

        let audio = tmp.path().join("clip.wav");
        write_wav(&audio, &[0.1, -0.1, 0.05], 16_000, 1).unwrap();

        let mid = import_audio_file(
            &pool,
            &paths,
            &audio,
            Some("Imported"),
            &["Imported line one"],
            &ModelSelection::default(),
        )
        .await
        .unwrap();

        let text = load_transcript_text(&pool, &mid).await.unwrap();
        assert!(text.contains("Imported line one"));
        let m = crate::db::get_meeting(&pool, &mid).await.unwrap().unwrap();
        let folder = m.folder_path.unwrap();
        assert!(
            folder.contains("meetily-recordings") || folder.contains("Import"),
            "folder={folder}"
        );
    }
}
