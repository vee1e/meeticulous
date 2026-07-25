use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=native/system_audio_capture.swift");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let dest = out_dir.join("meeticulous-system-audio");
    let src = PathBuf::from("native/system_audio_capture.swift");

    let status = Command::new("swiftc")
        .args([
            "-O",
            "-framework",
            "CoreAudio",
            "-framework",
            "AudioToolbox",
            "-framework",
            "AVFoundation",
            "-framework",
            "Foundation",
            "-o",
        ])
        .arg(&dest)
        .arg(&src)
        .status();

    match status {
        Ok(s) if s.success() => {
            println!(
                "cargo:rustc-env=MEETICULOUS_SYSTEM_AUDIO_HELPER={}",
                dest.display()
            );
            // Also copy next to the profile target dir for runtime discovery
            if let Ok(profile) = env::var("PROFILE") {
                if let Ok(manifest) = env::var("CARGO_MANIFEST_DIR") {
                    let bin_dir = PathBuf::from(manifest).join("target").join(profile);
                    let _ = std::fs::create_dir_all(&bin_dir);
                    let sidecar = bin_dir.join("meeticulous-system-audio");
                    let _ = std::fs::copy(&dest, &sidecar);
                }
            }
        }
        Ok(s) => {
            println!(
                "cargo:warning=swiftc failed building system audio helper (status={s}); system audio will be unavailable"
            );
        }
        Err(e) => {
            println!("cargo:warning=swiftc not runnable ({e}); system audio helper not built");
        }
    }
}
