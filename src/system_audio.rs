//! macOS system-audio capture via a Swift Core Audio process-tap helper.
//!
//! Captures what the Mac plays (Zoom / Meet / browser) using the same process-tap
//! approach as Meetily. Requires **Audio Capture** permission for the host app
//! (Terminal / iTerm when launched from a shell).

#![cfg(target_os = "macos")]

use crate::stt::DiagHandle;
use anyhow::{anyhow, Context, Result};
use log::{info, warn};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Live system-audio session. Drop (or stop flag) kills the helper.
pub struct SystemAudioSession {
    child: Option<Child>,
    stop: Arc<AtomicBool>,
    _reader: Option<thread::JoinHandle<()>>,
    pub sample_rate: u32,
    pub device_name: String,
}

impl Drop for SystemAudioSession {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(j) = self._reader.take() {
            let _ = j.join();
        }
        info!("system audio: session dropped");
    }
}

/// RAII guard that kills + reaps a spawned child on drop, so no error path in
/// `start_system_audio_capture` / `wait_ready` leaks a helper process.
struct ChildGuard {
    child: Option<Child>,
}

impl ChildGuard {
    fn new(child: Child) -> Self {
        Self { child: Some(child) }
    }

    fn kill(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
        }
    }

    fn into_child(mut self) -> Child {
        self.child
            .take()
            .expect("child only handed out once at success")
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Resolve the compiled Swift helper binary.
pub fn helper_path() -> Option<PathBuf> {
    // 1. build.rs embeds absolute path at compile time
    if let Some(p) = option_env!("MEETICULOUS_SYSTEM_AUDIO_HELPER") {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Some(path);
        }
    }
    // 2. Same directory as the running binary
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("meeticulous-system-audio");
            if p.is_file() {
                return Some(p);
            }
        }
    }
    // 3. target/{debug,release}
    for profile in ["release", "debug"] {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(profile)
            .join("meeticulous-system-audio");
        if p.is_file() {
            return Some(p);
        }
    }
    // 4. PATH
    which("meeticulous-system-audio")
}

fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// Start system audio capture, streaming PCM into the live WAV + STT feed.
///
/// Opens `wav_writer` for `wav_path` **after** the helper reports its sample rate
/// and **before** the PCM reader thread starts, so no audio is dropped and the
/// full meeting is written (not a rolling in-memory window).
pub fn start_system_audio_capture(
    wav_path: &Path,
    wav_writer: crate::recording::LiveWavWriter,
    stt_samples: Arc<Mutex<Vec<f32>>>,
    stop_flag: Arc<AtomicBool>,
    diag: DiagHandle,
) -> Result<SystemAudioSession> {
    let helper = helper_path().ok_or_else(|| {
        anyhow!(
            "meeticulous-system-audio helper not found — rebuild with `cargo build` \
             (needs swiftc). System Settings → Privacy & Security → Audio Capture must \
             allow your Terminal after first launch."
        )
    })?;

    info!("system audio: starting helper {}", helper.display());

    let mut guard = ChildGuard::new(
        Command::new(&helper)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null())
            .spawn()
            .with_context(|| format!("spawn {}", helper.display()))?,
    );

    // Dedicated stderr reader thread so a hung helper (e.g. waiting on the
    // Audio Capture permission prompt) can never block the caller.
    let stderr = guard
        .child
        .as_mut()
        .expect("child alive")
        .stderr
        .take()
        .ok_or_else(|| anyhow!("helper has no stderr"))?;
    let (err_tx, err_rx) = mpsc::channel();
    thread::spawn(move || {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    let t = line.trim().to_string();
                    if !t.is_empty() && err_tx.send(t).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    let (sample_rate, device_name) = wait_ready(&mut guard, &err_rx, &helper)?;

    // Install the WAV writer before any PCM is read so the full session is saved.
    {
        let mut wav_guard = wav_writer
            .lock()
            .map_err(|_| anyhow!("WAV writer lock poisoned"))?;
        *wav_guard = Some(crate::recording::create_wav_writer(
            wav_path,
            sample_rate.max(1),
            1,
        )?);
    }

    // Drain remaining stderr into logs until the session stops or helper exits.
    let stop_err = stop_flag.clone();
    thread::spawn(move || {
        while !stop_err.load(Ordering::SeqCst) {
            match err_rx.recv_timeout(Duration::from_millis(200)) {
                Ok(line) => {
                    if !line.is_empty() {
                        warn!("system-audio helper: {line}");
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
    });

    let stdout = guard
        .child
        .as_mut()
        .expect("child alive")
        .stdout
        .take()
        .ok_or_else(|| anyhow!("helper has no stdout"))?;

    let stop = stop_flag.clone();
    let reader = thread::Builder::new()
        .name("meeticulous-sysaudio-reader".into())
        .spawn(move || {
            read_pcm_loop(stdout, wav_writer, stt_samples, stop, sample_rate, diag);
        })
        .ok();

    Ok(SystemAudioSession {
        child: Some(guard.into_child()),
        stop: stop_flag,
        _reader: reader,
        sample_rate,
        device_name,
    })
}

fn wait_ready(
    guard: &mut ChildGuard,
    rx: &mpsc::Receiver<String>,
    helper: &Path,
) -> Result<(u32, String)> {
    let deadline = std::time::Instant::now() + Duration::from_secs(8);
    loop {
        if std::time::Instant::now() > deadline {
            guard.kill();
            return Err(anyhow!(
                "system audio helper timed out waiting for READY from {}",
                helper.display()
            ));
        }
        if let Some(child) = guard.child.as_mut() {
            if let Ok(Some(status)) = child.try_wait() {
                guard.kill();
                return Err(anyhow!(
                    "system audio helper exited early ({status}) before READY"
                ));
            }
        }
        // Bounded wait: never blocks past the deadline even if the helper hangs.
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(line) => {
                let t = line.trim();
                if t.starts_with("READY ") {
                    let rate = parse_kv(t, "sample_rate")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(48_000);
                    let device = parse_device(t);
                    info!("system audio READY: {rate} Hz device={device}");
                    return Ok((rate, device));
                }
                if t.starts_with("ERROR ") {
                    guard.kill();
                    return Err(anyhow!(t.to_string()));
                }
                if !t.is_empty() {
                    warn!("system-audio helper: {t}");
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                guard.kill();
                return Err(anyhow!("system audio helper stderr closed before READY"));
            }
        }
    }
}

fn parse_kv(line: &str, key: &str) -> Option<String> {
    for part in line.split_whitespace() {
        if let Some(rest) = part.strip_prefix(&format!("{key}=")) {
            return Some(rest.to_string());
        }
    }
    None
}

/// Device names may contain spaces: take everything after `device=` (last field).
fn parse_device(line: &str) -> String {
    line.split_once("device=")
        .and_then(|(_, rest)| {
            let name = rest.trim();
            (!name.is_empty()).then(|| name.to_string())
        })
        .unwrap_or_else(|| "system".into())
}

fn read_pcm_loop(
    mut stdout: impl Read,
    wav_writer: crate::recording::LiveWavWriter,
    stt_samples: Arc<Mutex<Vec<f32>>>,
    stop: Arc<AtomicBool>,
    sample_rate: u32,
    diag: DiagHandle,
) {
    let mut buf = vec![0u8; 8192 * 4];
    let mut pending: Vec<u8> = Vec::new();
    while !stop.load(Ordering::SeqCst) {
        let n = match stdout.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => {
                report_stream_lost(&diag);
                break;
            }
        };
        // Carry a partial 4-byte frame across reads instead of discarding it.
        pending.extend_from_slice(&buf[..n]);
        let complete = pending.len() - (pending.len() % 4);
        if complete > 0 {
            let mut floats = Vec::with_capacity(complete / 4);
            for chunk in pending[..complete].chunks_exact(4) {
                let bytes: [u8; 4] = chunk.try_into().unwrap();
                floats.push(f32::from_le_bytes(bytes));
            }
            pending.drain(..complete);
            crate::recording::append_wav_samples(&wav_writer, &floats);
            if let Ok(mut feed) = stt_samples.lock() {
                crate::recording::extend_stt_feed(&mut feed, &floats, sample_rate);
            }
        }
    }
    // EOF/error while not stopping means the tap died mid-meeting.
    if !stop.load(Ordering::SeqCst) {
        report_stream_lost(&diag);
    }
}

fn report_stream_lost(diag: &DiagHandle) {
    if let Ok(mut d) = diag.lock() {
        d.system_ok = false;
        d.system_error = Some("stream ended".into());
        d.push_log("system audio stream ended — meeting continues on mic only");
    }
}
