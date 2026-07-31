//! meeticulous binary — macOS-only minimal TUI for Meetily data.

#[cfg(not(target_os = "macos"))]
compile_error!("meeticulous is macOS-only");

use std::io::Write;

use clap::Parser;
use meeticulous::{run, Cli};

#[tokio::main]
async fn main() {
    std::panic::set_hook(Box::new(|info| {
        let _ = crossterm::terminal::disable_raw_mode();
        let mut out = std::io::stdout();
        let _ = out.write_all(b"\x1b[?1049l");
        let _ = out.flush();
        eprintln!("panic: {info:?}");
    }));

    // Avoid polluting TUI with logs unless RUST_LOG is set
    if std::env::var_os("RUST_LOG").is_some() {
        env_logger::init();
    }

    let cli = Cli::parse();
    if let Err(e) = run(cli).await {
        eprintln!("meeticulous error: {e:#}");
        std::process::exit(1);
    }
}
