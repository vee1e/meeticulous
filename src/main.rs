//! meeticulous binary — macOS-only minimal TUI for Meetily data.

#[cfg(not(target_os = "macos"))]
compile_error!("meeticulous is macOS-only");

use clap::Parser;
use meeticulous::{run, Cli};

#[tokio::main]
async fn main() {
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
