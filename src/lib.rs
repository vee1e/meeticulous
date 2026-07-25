//! meeticulous — minimal macOS TUI for Meetily, sharing the exact same data paths.

#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

#[cfg(not(target_os = "macos"))]
compile_error!("meeticulous is macOS-only");

pub mod db;
pub mod markdown_view;
pub mod models;
pub mod paths;
pub mod recording;
pub mod stt;
pub mod summary;
#[cfg(target_os = "macos")]
pub mod system_audio;
pub mod tui;

use clap::{Parser, Subcommand};
use paths::{format_paths_report, MeetilyPaths};

#[derive(Debug, Parser)]
#[command(name = "meeticulous", about = "Minimal macOS TUI for Meetily meetings")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Print Meetily-compatible data paths and exit
    #[arg(long)]
    pub paths: bool,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Print shared Meetily data paths
    Paths,
    /// List meetings from the shared database
    List,
    /// Show transcript for a meeting id
    Show { meeting_id: String },
    /// Import an audio file as a new meeting (optional --text segments)
    Import {
        audio: std::path::PathBuf,
        #[arg(long)]
        title: Option<String>,
        #[arg(long)]
        text: Vec<String>,
    },
    /// Dry-run recording pipeline: create meeting, inject lines, stop
    #[command(name = "record-dry-run")]
    RecordDryRun {
        #[arg(long, default_value = "meeticulous dry-run")]
        title: String,
        #[arg(long)]
        line: Vec<String>,
    },
    /// Delete a meeting by id from the shared database
    Delete {
        meeting_id: String,
        /// Skip confirmation
        #[arg(long)]
        yes: bool,
    },
    /// Summarize a meeting via opencode / antigravity / http
    Summarize {
        meeting_id: String,
        /// Backend: opencode | antigravity | claude
        #[arg(long, default_value = "opencode")]
        backend: String,
        /// Extra context for the system prompt
        #[arg(long)]
        context: Option<String>,
    },
}

/// Library entry used by the binary and integration tests.
pub async fn run(cli: Cli) -> anyhow::Result<()> {
    let paths = MeetilyPaths::resolve()?;

    if cli.paths || matches!(cli.command, Some(Commands::Paths)) {
        print!("{}", format_paths_report(&paths));
        return Ok(());
    }

    match cli.command {
        None => {
            let pool = db::open_meetily_database(&paths).await?;
            tui::run_tui(paths, pool).await?;
        }
        Some(Commands::Paths) => {
            print!("{}", format_paths_report(&paths));
        }
        Some(Commands::List) => {
            let pool = db::open_meetily_database(&paths).await?;
            let meetings = db::list_meetings(&pool).await?;
            if meetings.is_empty() {
                println!("No meetings found in {}", paths.db_path.display());
            } else {
                println!(
                    "Meetings in {} ({}):",
                    paths.db_path.display(),
                    meetings.len()
                );
                for m in meetings {
                    println!("{}  {}  {}", m.id, m.created_at, m.title);
                }
            }
            db::cleanup(&pool).await?;
        }
        Some(Commands::Show { meeting_id }) => {
            let pool = db::open_meetily_database(&paths).await?;
            match db::get_meeting(&pool, &meeting_id).await? {
                None => {
                    println!("Meeting not found: {meeting_id}");
                }
                Some(m) => {
                    println!("{} — {}", m.id, m.title);
                    let text = db::load_transcript_text(&pool, &meeting_id).await?;
                    if text.trim().is_empty() {
                        println!("(empty transcript)");
                    } else {
                        println!("{text}");
                    }
                }
            }
            db::cleanup(&pool).await?;
        }
        Some(Commands::Import { audio, title, text }) => {
            let pool = db::open_meetily_database(&paths).await?;
            let sel = models::load_selection_from_app_data(&paths.app_data_dir)
                .unwrap_or_default();
            let lines: Vec<&str> = text.iter().map(|s| s.as_str()).collect();
            let id = recording::import_audio_file(
                &pool,
                &paths,
                &audio,
                title.as_deref(),
                &lines,
                &sel,
            )
            .await?;
            println!("Imported meeting {id}");
            db::cleanup(&pool).await?;
        }
        Some(Commands::RecordDryRun { title, line }) => {
            let pool = db::open_meetily_database(&paths).await?;
            let sel = models::load_selection_from_app_data(&paths.app_data_dir)
                .unwrap_or_default();
            let handle =
                recording::start_recording(&pool, &paths, Some(&title), &sel).await?;
            let lines: Vec<&str> = if line.is_empty() {
                vec!["dry-run segment"]
            } else {
                line.iter().map(|s| s.as_str()).collect()
            };
            recording::inject_live_transcript(&pool, &handle, &lines).await?;
            let meeting = recording::stop_recording(&pool, handle, &sel).await?;
            println!(
                "Dry-run meeting {} title={} segments={}",
                meeting.id,
                meeting.title,
                lines.len()
            );
            db::cleanup(&pool).await?;
        }
        Some(Commands::Delete { meeting_id, yes }) => {
            if !yes {
                eprintln!("Refusing to delete without --yes (meeting_id={meeting_id})");
                std::process::exit(2);
            }
            let pool = db::open_meetily_database(&paths).await?;
            let ok = db::delete_meeting(&pool, &meeting_id).await?;
            if ok {
                println!("Deleted {meeting_id}");
            } else {
                println!("Not found: {meeting_id}");
            }
            db::cleanup(&pool).await?;
        }
        Some(Commands::Summarize {
            meeting_id,
            backend,
            context,
        }) => {
            let pool = db::open_meetily_database(&paths).await?;
            let backend = summary::SummaryCliBackend::from_str_loose(&backend)
                .ok_or_else(|| anyhow::anyhow!("unknown backend: {backend}"))?;
            let transport = summary::transport_for_backend(backend)
                .map_err(|e| anyhow::anyhow!(e))?;
            let transcript = db::load_transcript_text_plain(&pool, &meeting_id).await?;
            // Never pass fake model labels like "agy" / "opencode" as --model.
            let res = summary::generate_meeting_summary_with_context(
                Some(&pool),
                Some(&meeting_id),
                &transcript,
                transport.as_ref(),
                match backend {
                    summary::SummaryCliBackend::Opencode => Some("opencode"),
                    summary::SummaryCliBackend::Antigravity => Some("antigravity"),
                    summary::SummaryCliBackend::Claude => Some("claude"),
                },
                None,
                context.as_deref(),
            )
            .await
            .map_err(|e| anyhow::anyhow!(e))?;
            print!("{}", summary::format_summary_for_display(&res));
            db::cleanup(&pool).await?;
        }
    }

    Ok(())
}
