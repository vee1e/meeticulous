//! macOS system-audio capture via a Swift Core Audio process-tap helper.
//!
//! Captures what the Mac plays (Zoom / Meet / browser) using the same process-tap
//! approach as Meetily. Requires **Audio Capture** permission for the host app
//! (Terminal / iTerm when launched from a shell).

#![cfg(target_os = "macos")]

use anyhow::{anyhow, Context, Result};
use log::{info, warn};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
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

/// Start system audio capture into a shared mono f32 sample buffer.
pub fn start_system_audio_capture(
    wav_writer: crate::recording::LiveWavWriter,
    stt_samples: Arc<Mutex<Vec<f32>>>,
    stop_flag: Arc<AtomicBool>,
) -> Result<SystemAudioSession> {
    let helper = helper_path().ok_or_else(|| {
        anyhow!(
            "meeticulous-system-audio helper not found — rebuild with `cargo build` \
             (needs swiftc). System Settings → Privacy & Security → Audio Capture must \
             allow your Terminal after first launch."
        )
    })?;

    info!("system audio: starting helper {}", helper.display());

    let mut child = Command::new(&helper)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .spawn()
        .with_context(|| format!("spawn {}", helper.display()))?;

    // Read READY line from stderr (with timeout via try_read loop)
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("helper has no stderr"))?;
    let mut stderr_reader = BufReader::new(stderr);

    let (sample_rate, device_name) = wait_ready(&mut stderr_reader, &mut child, &helper)?;

    // Drain remaining stderr into logs on a side thread
    let stop_err = stop_flag.clone();
    thread::spawn(move || {
        let mut line = String::new();
        while !stop_err.load(Ordering::SeqCst) {
            line.clear();
            match stderr_reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    let t = line.trim();
                    if !t.is_empty() {
                        warn!("system-audio helper: {t}");
                    }
                }
                Err(_) => break,
            }
        }
    });

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("helper has no stdout"))?;

    let stop = stop_flag.clone();
    let reader = thread::Builder::new()
        .name("meeticulous-sysaudio-reader".into())
        .spawn(move || {
            read_pcm_loop(stdout, wav_writer, stt_samples, stop);
        })
        .ok();

    Ok(SystemAudioSession {
        child: Some(child),
        stop: stop_flag,
        _reader: reader,
        sample_rate,
        device_name,
    })
}

fn wait_ready(
    stderr: &mut BufReader<impl Read>,
    child: &mut Child,
    helper: &Path,
) -> Result<(u32, String)> {
    let deadline = std::time::Instant::now() + Duration::from_secs(8);
    let mut line = String::new();
    loop {
        if std::time::Instant::now() > deadline {
            let _ = child.kill();
            return Err(anyhow!(
                "system audio helper timed out waiting for READY from {}",
                helper.display()
            ));
        }
        // Non-blocking-ish: set short read via poll of process
        if let Ok(Some(status)) = child.try_wait() {
            // Collect remaining stderr
            let mut rest = String::new();
            let _ = stderr.read_to_string(&mut rest);
            return Err(anyhow!(
                "system audio helper exited early ({status}): {}",
                rest.trim()
            ));
        }
        line.clear();
        // Blocking read_line — helper should print READY quickly or ERROR
        match stderr.read_line(&mut line) {
            Ok(0) => {
                thread::sleep(Duration::from_millis(20));
                continue;
            }
            Ok(_) => {
                let t = line.trim();
                if t.starts_with("READY ") {
                    let rate = parse_kv(t, "sample_rate")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(48_000);
                    let device = parse_kv(t, "device").unwrap_or_else(|| "system".into());
                    info!("system audio READY: {rate} Hz device={device}");
                    return Ok((rate, device));
                }
                if t.starts_with("ERROR ") {
                    let _ = child.kill();
                    return Err(anyhow!(t.to_string()));
                }
                if !t.is_empty() {
                    warn!("system-audio helper: {t}");
                }
            }
            Err(e) => {
                let _ = child.kill();
                return Err(anyhow!("reading helper stderr: {e}"));
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

fn read_pcm_loop(
    mut stdout: impl Read,
    wav_writer: crate::recording::LiveWavWriter,
    stt_samples: Arc<Mutex<Vec<f32>>>,
    stop: Arc<AtomicBool>,
) {
    let mut buf = vec![0u8; 8192 * 4];
    while !stop.load(Ordering::SeqCst) {
        match stdout.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                let usable = n - (n % 4);
                if usable == 0 {
                    continue;
                }
                let mut floats = Vec::with_capacity(usable / 4);
                for chunk in buf[..usable].chunks_exact(4) {
                    let bytes: [u8; 4] = chunk.try_into().unwrap();
                    floats.push(f32::from_le_bytes(bytes));
                }
                crate::recording::append_wav_samples(&wav_writer, &floats);
                if let Ok(mut feed) = stt_samples.lock() {
                    feed.extend_from_slice(&floats);
                }
            }
            Err(_) => break,
        }
    }
}
