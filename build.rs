use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=native/system_audio_capture.swift");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let dest = out_dir.join("meeticulous-system-audio");
    let src = PathBuf::from("native/system_audio_capture.swift");

    let mut args = vec![
        "-O".to_string(),
        "-framework".to_string(),
        "CoreAudio".to_string(),
        "-framework".to_string(),
        "AudioToolbox".to_string(),
        "-framework".to_string(),
        "AVFoundation".to_string(),
        "-framework".to_string(),
        "Foundation".to_string(),
        "-o".to_string(),
        dest.display().to_string(),
        src.display().to_string(),
    ];
    // Match swiftc's architecture to the target (Rust uses aarch64 for arm64).
    let arch = swift_arch();
    if let Some(arch) = &arch {
        args.insert(0, "-arch".to_string());
        args.insert(1, arch.clone());
    }

    let status = Command::new("swiftc").args(&args).status();
    // Some swiftc (swift-driver) rejects `-arch`; retry without it so the
    // helper still builds on those toolchains (native arch is then default).
    let status = match status {
        Ok(s) if s.success() => Ok(s),
        _ if arch.is_some() => Command::new("swiftc").args(&args[2..]).status(),
        other => other,
    };

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

/// Map Rust target arch to swiftc's `-arch` value (skip when unknown).
fn swift_arch() -> Option<String> {
    match env::var("CARGO_CFG_TARGET_ARCH").ok()?.as_str() {
        "aarch64" => Some("arm64".into()),
        "x86_64" => Some("x86_64".into()),
        _ => None,
    }
}
