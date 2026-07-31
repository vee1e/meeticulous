//! Minimal ratatui TUI for meeticulous.

use crate::db::{self, Meeting};
use crate::markdown_view::markdown_to_lines;
use crate::models::{
    discover_models_for_paths, load_selection_from_app_data, save_selection_to_app_data,
    ModelSelection, TranscriptionProvider,
};
use crate::paths::MeetilyPaths;
use crate::recording::{self, RecordingHandle};
use crate::summary::{
    backend_status_line, format_context_badge, generate_meeting_summary_with_context,
    load_summary_prefs, save_summary_prefs, ClaudeTransport, SummaryCliBackend, SummaryPrefs,
    SummaryResult, DEFAULT_ANTIGRAVITY_MODEL, DEFAULT_OPENCODE_MODEL,
};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Terminal;
use sqlx::SqlitePool;
use std::io::stdout;
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;

/// Braille spinner only (no bars / dots / emoji).
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// How many lines/chars of context to show before collapsing the rest.
const CONTEXT_PREVIEW_LINES: usize = 3;
const CONTEXT_PREVIEW_CHARS: usize = 240;
const CONTEXT_FIRST_LINE_CHARS: usize = 100;

/// Collapse a large pasted context blob for the TUI while keeping full text in `input_buf`.
fn format_context_preview(text: &str) -> String {
    if text.is_empty() {
        return "> ".to_string();
    }
    let lines: Vec<&str> = text.lines().collect();
    let n_lines = lines.len().max(1);
    let n_chars = text.chars().count();
    let is_large = n_lines > CONTEXT_PREVIEW_LINES || n_chars > CONTEXT_PREVIEW_CHARS;

    if !is_large {
        return format!("> {text}");
    }

    let first = lines.first().copied().unwrap_or("");
    let mut head: String = first.chars().take(CONTEXT_FIRST_LINE_CHARS).collect();
    if first.chars().count() > CONTEXT_FIRST_LINE_CHARS {
        head.push('…');
    }

    // Prefer line-based collapse; fall back to char count for a single giant line.
    if n_lines > 1 {
        let hidden = n_lines.saturating_sub(1);
        format!("> {head}\n  [+{hidden} lines]  ({n_chars} chars total)")
    } else {
        let hidden_chars = n_chars.saturating_sub(CONTEXT_FIRST_LINE_CHARS);
        format!("> {head}\n  [+{hidden_chars} chars]")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Screen {
    Meetings,
    Transcript,
    Recording,
    Summary,
    Settings,
    /// Type optional context, then generate summary.
    SummaryPrep,
    /// In-flight summary with spinner animation.
    Summarizing,
    /// Confirm delete of selected meeting.
    DeleteConfirm,
}

pub struct App {
    paths: MeetilyPaths,
    pool: SqlitePool,
    screen: Screen,
    meetings: Vec<Meeting>,
    list_state: ListState,
    transcript_text: String,
    /// Raw plain-text / markdown body of the summary (for copy + source of truth).
    summary_text: String,
    /// Pre-rendered markdown lines for the summary pane.
    summary_lines: Vec<Line<'static>>,
    /// Meta line under the header (provider / context badge).
    summary_meta: String,
    status: String,
    model_selection: ModelSelection,
    models_list: Vec<String>,
    models_state: ListState,
    recording: Option<RecordingHandle>,
    live_lines: Vec<String>,
    should_quit: bool,
    summary_prefs: SummaryPrefs,
    /// Meeting id pending delete confirmation.
    pending_delete_id: Option<String>,
    pending_delete_title: String,
    /// Text buffer for summary context / recording append (shared input).
    input_buf: String,
    /// Background summarize job.
    summary_job: Option<JoinHandle<Result<SummaryResult, String>>>,
    summary_spin: usize,
    summary_started: Option<Instant>,
    summary_meeting_title: String,
    summary_ctx_note: String,
    summary_model_label: String,
    /// Vertical scroll for transcript / summary / live panes (line offset).
    scroll: u16,
    /// Last content area height (for page scroll).
    content_height: u16,
    content_width: u16,
    /// Pending `g` for vim `gg` (go top).
    pending_g: bool,
    /// When true, live transcript sticks to the bottom as new lines arrive.
    live_follow: bool,
    /// Wrapped line count for summary pane (set while drawing).
    summary_wrapped_len: u16,
}

impl App {
    pub async fn new(paths: MeetilyPaths, pool: SqlitePool) -> anyhow::Result<Self> {
        let mut model_selection =
            load_selection_from_app_data(&paths.app_data_dir).unwrap_or_default();
        if let Ok(Some(ts)) = db::get_transcript_config(&pool).await {
            if let Some(p) = TranscriptionProvider::from_str_loose(&ts.provider) {
                model_selection = ModelSelection {
                    provider: p,
                    model: ts.model,
                };
            }
        }
        let summary_prefs = load_summary_prefs(&paths.app_data_dir);

        let mut app = Self {
            paths,
            pool,
            screen: Screen::Meetings,
            meetings: Vec::new(),
            list_state: ListState::default(),
            transcript_text: String::new(),
            summary_text: String::new(),
            summary_lines: Vec::new(),
            summary_meta: String::new(),
            status: "hjkl · Enter/l open (summary) · t transcript · s regen · c copy · d del · q"
                .into(),
            model_selection,
            models_list: Vec::new(),
            models_state: ListState::default(),
            recording: None,
            live_lines: Vec::new(),
            should_quit: false,
            summary_prefs,
            pending_delete_id: None,
            pending_delete_title: String::new(),
            input_buf: String::new(),
            summary_job: None,
            summary_spin: 0,
            summary_started: None,
            summary_meeting_title: String::new(),
            summary_ctx_note: String::new(),
            summary_model_label: String::new(),
            scroll: 0,
            content_height: 20,
            content_width: 80,
            pending_g: false,
            live_follow: true,
            summary_wrapped_len: 0,
        };
        app.refresh_meetings().await?;
        app.refresh_models();
        Ok(app)
    }

    fn resolved_summary_model(&self) -> String {
        if !self.summary_prefs.model.trim().is_empty() {
            return self.summary_prefs.model.clone();
        }
        match self.summary_prefs.backend {
            SummaryCliBackend::Opencode => DEFAULT_OPENCODE_MODEL.to_string(),
            SummaryCliBackend::Antigravity => DEFAULT_ANTIGRAVITY_MODEL.to_string(),
            SummaryCliBackend::Claude => {
                if self.summary_prefs.model.trim().is_empty() {
                    "claude (default)".into()
                } else {
                    self.summary_prefs.model.clone()
                }
            }
        }
    }

    async fn refresh_meetings(&mut self) -> anyhow::Result<()> {
        self.meetings = db::list_meetings(&self.pool).await?;
        if self.meetings.is_empty() {
            self.list_state.select(None);
        } else if self.list_state.selected().is_none() {
            self.list_state.select(Some(0));
        } else if let Some(i) = self.list_state.selected() {
            if i >= self.meetings.len() {
                self.list_state.select(Some(self.meetings.len() - 1));
            }
        }
        Ok(())
    }

    fn refresh_models(&mut self) {
        let found = discover_models_for_paths(&self.paths);
        self.models_list = found
            .into_iter()
            .map(|m| {
                let mark = if m.available { "✓" } else { " " };
                format!("[{mark}] {} / {} ({} MB)", m.provider, m.name, m.size_mb)
            })
            .collect();
        if self.models_list.is_empty() {
            self.models_list
                .push("(no models found under shared models/)".into());
        }
        if self.models_state.selected().is_none() {
            self.models_state.select(Some(0));
        }
    }

    fn selected_meeting(&self) -> Option<&Meeting> {
        self.list_state
            .selected()
            .and_then(|i| self.meetings.get(i))
    }

    async fn open_transcript(&mut self) -> anyhow::Result<()> {
        if let Some(m) = self.selected_meeting() {
            let id = m.id.clone();
            let title = m.title.clone();
            self.transcript_text = db::load_transcript_text(&self.pool, &id).await?;
            if self.transcript_text.trim().is_empty() {
                self.transcript_text = "(no transcript segments)".into();
            }
            self.scroll = 0;
            self.pending_g = false;
            self.screen = Screen::Transcript;
            self.status = format!(
                "Transcript: {title}  j/k scroll · gg/G · C-d/u · s sum · d del · h/Esc back"
            );
        }
        Ok(())
    }

    fn scroll_content_len(&self) -> u16 {
        let wrap_width = self.content_width.max(1) as usize;
        match self.screen {
            Screen::Transcript => {
                wrapped_line_count(&self.transcript_text, wrap_width).max(1) as u16
            }
            Screen::Summary => {
                if self.summary_wrapped_len > 0 {
                    self.summary_wrapped_len
                } else {
                    let meta = if self.summary_meta.is_empty() { 0 } else { 2 };
                    (self.summary_lines.len() + meta).max(1) as u16
                }
            }
            Screen::Summarizing => wrapped_line_count(&self.summary_text, wrap_width).max(1) as u16,
            Screen::Recording => {
                wrapped_line_count(&self.live_lines.join("\n"), wrap_width).max(1) as u16
            }
            _ => 1,
        }
    }

    fn scroll_by(&mut self, delta: i32) {
        if !matches!(
            self.screen,
            Screen::Transcript | Screen::Summary | Screen::Summarizing | Screen::Recording
        ) {
            return;
        }
        let max = self
            .scroll_content_len()
            .saturating_sub(self.content_height.max(1));
        let next = self.scroll as i32 + delta;
        self.scroll = next.clamp(0, max as i32) as u16;
        if self.screen == Screen::Recording {
            // User moved away from bottom → stop auto-follow until they G / jump bottom.
            self.live_follow = self.scroll >= max;
        }
    }

    fn scroll_to(&mut self, pos: u16) {
        if !matches!(
            self.screen,
            Screen::Transcript | Screen::Summary | Screen::Summarizing | Screen::Recording
        ) {
            return;
        }
        let max = self
            .scroll_content_len()
            .saturating_sub(self.content_height.max(1));
        self.scroll = pos.min(max);
        if self.screen == Screen::Recording {
            self.live_follow = self.scroll >= max;
        }
    }

    fn set_summary_body(&mut self, body: String, meta: String) {
        self.summary_text = body;
        self.summary_meta = meta;
        self.summary_lines = markdown_to_lines(&self.summary_text);
        self.scroll = 0;
    }

    fn copy_summary_plaintext(&mut self) {
        let text = self.summary_text.clone();
        if text.trim().is_empty() {
            self.status = "Nothing to copy".into();
            return;
        }
        match copy_to_clipboard(&text) {
            Ok(()) => {
                self.status = format!("Copied {} chars to clipboard", text.chars().count());
            }
            Err(e) => {
                self.status = format!("Copy failed: {e}");
            }
        }
    }

    fn page_step(&self) -> i32 {
        (self.content_height.max(2) as i32).saturating_sub(1)
    }

    fn half_page(&self) -> i32 {
        ((self.content_height.max(2) as i32) / 2).max(1)
    }

    /// List navigation helpers (vim-ish).
    fn list_move(&mut self, delta: i32) {
        let len = self.meetings.len();
        if len == 0 {
            return;
        }
        let cur = self.list_state.selected().unwrap_or(0) as i32;
        let next = (cur + delta).clamp(0, (len - 1) as i32) as usize;
        self.list_state.select(Some(next));
    }

    fn list_goto_top(&mut self) {
        if !self.meetings.is_empty() {
            self.list_state.select(Some(0));
        }
    }

    fn list_goto_bottom(&mut self) {
        if !self.meetings.is_empty() {
            self.list_state.select(Some(self.meetings.len() - 1));
        }
    }

    fn models_move(&mut self, delta: i32) {
        let len = self.models_list.len();
        if len == 0 {
            return;
        }
        let cur = self.models_state.selected().unwrap_or(0) as i32;
        let next = (cur + delta).clamp(0, (len - 1) as i32) as usize;
        self.models_state.select(Some(next));
    }

    fn begin_summary_prep(&mut self) {
        if self.selected_meeting().is_none() {
            self.status = "Select a meeting first".into();
            return;
        }
        self.input_buf = self.summary_prefs.context.clone();
        self.screen = Screen::SummaryPrep;
        self.status = format!(
            "Summary backend: {}  [Tab] cycle  [Enter] run  [Esc] cancel",
            backend_status_line(self.summary_prefs.backend)
        );
    }

    /// Kick off summary in the background so the UI can animate.
    async fn start_summary_job(&mut self) -> anyhow::Result<()> {
        let Some(m) = self.selected_meeting() else {
            self.status = "Select a meeting first".into();
            return Ok(());
        };
        let id = m.id.clone();
        let title = m.title.clone();

        self.summary_prefs.context = self.input_buf.clone();
        let _ = save_summary_prefs(&self.paths.app_data_dir, &self.summary_prefs);

        let model_label = self.resolved_summary_model();
        let ctx_note = if self.input_buf.trim().is_empty() {
            "(no extra context)".to_string()
        } else {
            let n_lines = self.input_buf.lines().count().max(1);
            let n_chars = self.input_buf.chars().count();
            if n_lines > CONTEXT_PREVIEW_LINES || n_chars > CONTEXT_PREVIEW_CHARS {
                format!("context: [+{n_lines} lines · {n_chars} chars]")
            } else {
                format!(
                    "context: {}",
                    self.input_buf.chars().take(72).collect::<String>()
                )
            }
        };

        // Plain text for the model (timestamps stay for UI display only).
        let transcript = match db::load_transcript_text_plain(&self.pool, &id).await {
            Ok(t) => t,
            Err(e) => {
                let prev = if self.screen == Screen::SummaryPrep {
                    Screen::SummaryPrep
                } else {
                    Screen::Meetings
                };
                self.screen = prev;
                self.status = format!("Could not load transcript: {e}");
                return Ok(());
            }
        };
        let ctx = self.summary_prefs.context.clone();
        let extra = if ctx.trim().is_empty() {
            None
        } else {
            Some(ctx)
        };
        let backend = self.summary_prefs.backend;
        let prefs_model = self.summary_prefs.model.clone();
        let pool = self.pool.clone();

        self.summary_meeting_title = title.clone();
        self.summary_ctx_note = ctx_note;
        self.summary_model_label = model_label.clone();
        self.summary_spin = 0;
        self.summary_started = Some(Instant::now());
        self.screen = Screen::Summarizing;
        self.status = format!(
            "Summarizing “{title}” · {} · {model_label}",
            self.summary_prefs.backend
        );
        self.refresh_summarizing_frame();

        let job = tokio::spawn(async move {
            let model_for_cli = if prefs_model.trim().is_empty() {
                match backend {
                    SummaryCliBackend::Opencode => DEFAULT_OPENCODE_MODEL.to_string(),
                    SummaryCliBackend::Antigravity => DEFAULT_ANTIGRAVITY_MODEL.to_string(),
                    SummaryCliBackend::Claude => String::new(),
                }
            } else {
                prefs_model
            };

            let transport: Box<dyn crate::summary::LlmTransport> = match backend {
                SummaryCliBackend::Opencode => {
                    let mut t = crate::summary::OpencodeTransport::discover()?;
                    t.model = Some(model_for_cli.clone());
                    Box::new(t)
                }
                SummaryCliBackend::Antigravity => {
                    let mut t = crate::summary::AntigravityTransport::discover()?;
                    t.model = Some(model_for_cli.clone());
                    Box::new(t)
                }
                SummaryCliBackend::Claude => {
                    let mut t = ClaudeTransport::discover()?;
                    if !model_for_cli.is_empty() {
                        t.model = Some(model_for_cli.clone());
                    }
                    Box::new(t)
                }
            };

            let (prov, model_override) = match backend {
                SummaryCliBackend::Opencode => (Some("opencode"), Some(model_for_cli.as_str())),
                SummaryCliBackend::Antigravity => {
                    (Some("antigravity"), Some(model_for_cli.as_str()))
                }
                SummaryCliBackend::Claude => (
                    Some("claude"),
                    if model_for_cli.is_empty() {
                        None
                    } else {
                        Some(model_for_cli.as_str())
                    },
                ),
            };

            generate_meeting_summary_with_context(
                Some(&pool),
                Some(&id),
                &transcript,
                transport.as_ref(),
                prov,
                model_override,
                extra.as_deref(),
            )
            .await
        });

        self.summary_job = Some(job);
        Ok(())
    }

    fn refresh_summarizing_frame(&mut self) {
        let spin = SPINNER[self.summary_spin % SPINNER.len()];
        let elapsed = self
            .summary_started
            .map(|t| t.elapsed().as_secs())
            .unwrap_or(0);
        // Keep as plain text (Summarizing screen uses draw_scroll_text, not markdown).
        self.summary_text = format!(
            "{spin} summarizing\n\n\
             meeting   {title}\n\
             backend   {backend}\n\
             model     {model}\n\
             {ctx}\n\n\
             {elapsed}s · waiting on CLI / API",
            title = self.summary_meeting_title,
            backend = backend_status_line(self.summary_prefs.backend),
            model = self.summary_model_label,
            ctx = self.summary_ctx_note,
        );
        self.summary_lines.clear();
        self.summary_meta.clear();
        self.status = format!(
            "{spin} {} · {} · {elapsed}s",
            self.summary_prefs.backend, self.summary_model_label
        );
    }

    async fn poll_summary_job(&mut self) -> anyhow::Result<()> {
        let Some(job) = self.summary_job.as_mut() else {
            return Ok(());
        };
        if !job.is_finished() {
            self.summary_spin = self.summary_spin.wrapping_add(1);
            self.refresh_summarizing_frame();
            return Ok(());
        }
        let job = self.summary_job.take().unwrap();
        let ctx = self.summary_prefs.context.clone();
        match job.await {
            Ok(Ok(res)) => {
                let badge = if ctx.trim().is_empty() {
                    String::new()
                } else {
                    format_context_badge(&ctx)
                };
                let meta = format!(
                    "{} / {} · {:.1}s{}",
                    res.provider,
                    res.model,
                    res.processing_time_secs,
                    if badge.is_empty() {
                        String::new()
                    } else {
                        format!(" · {badge}")
                    }
                );
                // Body is plain markdown/text saved in Meetily DB — render as markdown.
                self.set_summary_body(res.text.clone(), meta);
                self.screen = Screen::Summary;
                let title_note = res
                    .title
                    .as_deref()
                    .map(|t| format!(" · “{t}”"))
                    .unwrap_or_default();
                self.status = format!(
                    "Saved · {} / {} · {:.1}s{title_note}  s regen · c copy · j/k scroll",
                    res.provider, res.model, res.processing_time_secs
                );
                let _ = self.refresh_meetings().await;
            }
            Ok(Err(e)) => {
                self.set_summary_body(format!("Summary failed: {e}"), "error".into());
                self.screen = Screen::Summary;
                self.status = format!("Summary error: {e}");
            }
            Err(e) => {
                self.set_summary_body(format!("Summary task panicked: {e}"), "error".into());
                self.screen = Screen::Summary;
                self.status = "Summary task failed".into();
            }
        }
        self.summary_started = None;
        Ok(())
    }

    /// Open meeting: prefer stored summary in Meetily DB, else transcript.
    async fn open_meeting(&mut self) -> anyhow::Result<()> {
        let Some(m) = self.selected_meeting() else {
            return Ok(());
        };
        let id = m.id.clone();
        let title = m.title.clone();
        if self.load_summary_for_meeting(&id, &title).await? {
            return Ok(());
        }
        self.open_transcript().await
    }

    /// Load summary from Meetily `summary_processes` into the summary pane.
    /// Returns true if a summary was shown.
    async fn load_summary_for_meeting(&mut self, id: &str, title: &str) -> anyhow::Result<bool> {
        match db::load_summary_plain_text(&self.pool, id).await? {
            Some((body, status)) => {
                self.set_summary_body(body, format!("saved in Meetily DB · status={status}"));
                self.pending_g = false;
                self.screen = Screen::Summary;
                self.status = format!(
                    "Summary: {title}  j/k scroll · s regenerate · c copy · t transcript · h back"
                );
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// `s` always means regenerate (prep → generate).
    async fn open_or_prep_summary(&mut self) -> anyhow::Result<()> {
        if self.selected_meeting().is_none() {
            self.status = "Select a meeting first".into();
            return Ok(());
        }
        self.begin_summary_prep();
        Ok(())
    }

    async fn view_cached_summary(&mut self) -> anyhow::Result<()> {
        let Some(m) = self.selected_meeting() else {
            self.status = "Select a meeting first".into();
            return Ok(());
        };
        let id = m.id.clone();
        let title = m.title.clone();
        if !self.load_summary_for_meeting(&id, &title).await? {
            self.status = "No summary yet — press s to generate".into();
        }
        Ok(())
    }

    fn begin_delete_confirm(&mut self) {
        let Some((id, title)) = self
            .selected_meeting()
            .map(|m| (m.id.clone(), m.title.clone()))
        else {
            self.status = "Select a meeting to delete".into();
            return;
        };
        self.pending_delete_id = Some(id);
        self.pending_delete_title = title.clone();
        self.screen = Screen::DeleteConfirm;
        self.status = format!("Delete “{title}”?  [y] yes  [n]/Esc] cancel");
    }

    async fn confirm_delete(&mut self) -> anyhow::Result<()> {
        if let Some(id) = self.pending_delete_id.take() {
            let title = std::mem::take(&mut self.pending_delete_title);
            match db::delete_meeting(&self.pool, &id).await {
                Ok(true) => {
                    self.status = format!("Deleted “{title}”");
                    self.refresh_meetings().await?;
                }
                Ok(false) => {
                    self.status = format!("Meeting not found: {id}");
                }
                Err(e) => {
                    self.status = format!("Delete failed: {e}");
                }
            }
        }
        self.screen = Screen::Meetings;
        Ok(())
    }

    async fn start_or_stop_recording(&mut self) -> anyhow::Result<()> {
        if self.recording.is_some() {
            let handle = self.recording.take().unwrap();
            let sel = self.model_selection.clone();
            let meeting = recording::stop_recording(&self.pool, handle, &sel).await?;
            self.live_lines.clear();
            self.status = format!("Stopped recording: {}", meeting.title);
            self.refresh_meetings().await?;
            self.screen = Screen::Meetings;
        } else {
            let sel = self.model_selection.clone();
            let handle = recording::start_recording(&self.pool, &self.paths, None, &sel).await?;
            let diag = handle.diagnostics_snapshot();
            self.status = format!(
                "REC {} · sys={} mic={} · STT {} — r stop",
                handle.title,
                if diag.system_ok { "ok" } else { "FAIL" },
                if diag.mic_ok { "ok" } else { "FAIL" },
                diag.stt_status
            );
            self.live_lines.clear();
            self.input_buf.clear();
            self.scroll = 0;
            self.live_follow = true;
            self.pending_g = false;
            self.recording = Some(handle);
            self.screen = Screen::Recording;
        }
        Ok(())
    }

    async fn tick_recording(&mut self) -> anyhow::Result<()> {
        if let Some(handle) = self.recording.as_ref() {
            let new_segs = recording::drain_stt_segments(&self.pool, handle).await?;
            for seg in new_segs {
                self.live_lines
                    .push(format!("[{:.1}s] {}", seg.audio_start, seg.text));
            }
            let diag = handle.diagnostics_snapshot();
            self.status = format!(
                "REC {:.0}s · rms={:.3} · sys={} mic={} · segs={} · {}",
                handle.elapsed_secs(),
                diag.rms,
                if diag.system_ok { "ok" } else { "FAIL" },
                if diag.mic_ok { "ok" } else { "FAIL" },
                diag.segments_emitted,
                diag.stt_status
            );
        }
        Ok(())
    }

    async fn append_live_line(&mut self, text: String) -> anyhow::Result<()> {
        if let Some(handle) = self.recording.as_ref() {
            let seg = recording::append_text_segment(&self.pool, handle, &text, None, None).await?;
            self.live_lines
                .push(format!("[{:.1}s] {}", seg.audio_start, seg.text));
            self.status = format!("Appended segment ({} lines)", self.live_lines.len());
        }
        Ok(())
    }

    async fn select_model_at_cursor(&mut self) -> anyhow::Result<()> {
        let found = discover_models_for_paths(&self.paths);
        if let Some(i) = self.models_state.selected() {
            if let Some(m) = found.get(i) {
                if !m.available {
                    self.status = format!("Model {} not downloaded", m.name);
                    return Ok(());
                }
                self.model_selection = ModelSelection {
                    provider: m.provider,
                    model: m.name.clone(),
                };
                save_selection_to_app_data(&self.paths.app_data_dir, &self.model_selection)?;
                db::save_transcript_config(
                    &self.pool,
                    self.model_selection.provider.as_str(),
                    &self.model_selection.model,
                )
                .await?;
                self.status = format!(
                    "Selected {} / {}",
                    self.model_selection.provider, self.model_selection.model
                );
            }
        }
        Ok(())
    }

    fn cycle_summary_backend(&mut self) {
        self.summary_prefs.backend = self.summary_prefs.backend.cycle();
        let _ = save_summary_prefs(&self.paths.app_data_dir, &self.summary_prefs);
        self.status = format!(
            "Summary backend: {}",
            backend_status_line(self.summary_prefs.backend)
        );
    }

    fn cancel_summary_job(&mut self) {
        if let Some(job) = self.summary_job.take() {
            job.abort();
        }
        self.summary_started = None;
        self.screen = Screen::Meetings;
        self.status = "Summary cancelled".into();
    }
}

struct TermGuard {
    active: bool,
}

impl TermGuard {
    fn arm() -> Self {
        let _ = enable_raw_mode();
        let _ = stdout().execute(EnterAlternateScreen);
        TermGuard { active: true }
    }
}

impl Drop for TermGuard {
    fn drop(&mut self) {
        if self.active {
            let _ = disable_raw_mode();
            let _ = stdout().execute(LeaveAlternateScreen);
        }
    }
}

/// Run the interactive TUI (macOS terminal).
pub async fn run_tui(paths: MeetilyPaths, pool: SqlitePool) -> anyhow::Result<()> {
    let guard = TermGuard::arm();
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let mut app = App::new(paths, pool).await?;

    let result = async {
        loop {
            if app.screen == Screen::Recording {
                let _ = app.tick_recording().await;
            }
            if app.screen == Screen::Summarizing {
                let _ = app.poll_summary_job().await;
            }

            terminal.draw(|f| draw_ui(f, &mut app))?;

            // Faster poll while animating summary spinner.
            let poll_ms = if app.screen == Screen::Summarizing {
                80
            } else {
                200
            };

            if event::poll(Duration::from_millis(poll_ms))? {
                if let Event::Key(key) = event::read()? {
                    if let Err(e) = handle_key(&mut app, key).await {
                        app.status = format!("error: {e}");
                    }
                }
            }

            if app.should_quit {
                if let Some(job) = app.summary_job.take() {
                    job.abort();
                }
                if app.recording.is_some() {
                    app.start_or_stop_recording().await?;
                }
                break;
            }
        }
        Ok::<(), anyhow::Error>(())
    }
    .await;

    drop(guard);
    result
}

async fn handle_key(app: &mut App, key: event::KeyEvent) -> anyhow::Result<()> {
    if key.kind != KeyEventKind::Press {
        return Ok(());
    }

    if key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char('c') | KeyCode::Char('C'))
    {
        match app.screen {
            Screen::Summarizing => app.cancel_summary_job(),
            Screen::Recording => {
                app.start_or_stop_recording().await?;
                app.should_quit = true;
            }
            _ => app.should_quit = true,
        }
        return Ok(());
    }

    match app.screen {
        Screen::Summarizing => match key.code {
            KeyCode::Char('q') | KeyCode::Esc => {
                app.cancel_summary_job();
            }
            _ => {}
        },
        Screen::Recording => {
            let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
            let typing = !app.input_buf.is_empty();
            match key.code {
                KeyCode::Char('r') | KeyCode::Char('R') if !typing => {
                    app.start_or_stop_recording().await?;
                }
                KeyCode::Esc if !typing => {
                    app.start_or_stop_recording().await?;
                }
                // Scroll live transcript when not typing a manual segment
                KeyCode::Char('j') | KeyCode::Down if !typing => {
                    app.scroll_by(1);
                }
                KeyCode::Char('k') | KeyCode::Up if !typing => {
                    app.scroll_by(-1);
                }
                KeyCode::Char('d') if ctrl && !typing => {
                    app.scroll_by(app.half_page());
                }
                KeyCode::Char('u') if ctrl && !typing => {
                    app.scroll_by(-app.half_page());
                }
                KeyCode::Char('u') if ctrl => {
                    app.input_buf.clear();
                }
                KeyCode::Char('f') if ctrl && !typing => {
                    app.scroll_by(app.page_step());
                }
                KeyCode::Char('b') if ctrl && !typing => {
                    app.scroll_by(-app.page_step());
                }
                KeyCode::PageDown if !typing => {
                    app.scroll_by(app.page_step());
                }
                KeyCode::PageUp if !typing => {
                    app.scroll_by(-app.page_step());
                }
                KeyCode::Char('g') if !typing => {
                    if app.pending_g {
                        app.scroll_to(0);
                        app.pending_g = false;
                    } else {
                        app.pending_g = true;
                    }
                }
                KeyCode::Char('G') if !typing => {
                    app.pending_g = false;
                    app.scroll_to(u16::MAX);
                    app.live_follow = true;
                }
                KeyCode::Enter if !app.input_buf.is_empty() => {
                    let line = std::mem::take(&mut app.input_buf);
                    app.append_live_line(line).await?;
                }
                KeyCode::Backspace => {
                    app.input_buf.pop();
                }
                KeyCode::Char(c) if !ctrl => {
                    app.pending_g = false;
                    app.input_buf.push(c);
                }
                _ => {}
            }
        }
        Screen::SummaryPrep => {
            // Text entry mode: only Esc cancels — never steal h/j/k/etc.
            let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
            match key.code {
                KeyCode::Esc => {
                    app.screen = Screen::Meetings;
                    app.status = "Cancelled summary".into();
                }
                KeyCode::Tab => {
                    app.cycle_summary_backend();
                }
                KeyCode::Enter => {
                    app.start_summary_job().await?;
                }
                KeyCode::Backspace => {
                    app.input_buf.pop();
                }
                // Ctrl-u: clear line (vim-ish, still useful while typing)
                KeyCode::Char('u') if ctrl => {
                    app.input_buf.clear();
                }
                KeyCode::Char(c) if !ctrl => {
                    app.input_buf.push(c);
                }
                _ => {}
            }
        }
        Screen::DeleteConfirm => match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                app.confirm_delete().await?;
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc | KeyCode::Char('q') => {
                app.pending_delete_id = None;
                app.screen = Screen::Meetings;
                app.status = "Delete cancelled".into();
            }
            _ => {}
        },
        Screen::Transcript | Screen::Summary => {
            let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
            match (ctrl, key.code) {
                // Ctrl chords first
                (true, KeyCode::Char('d')) => app.scroll_by(app.half_page()),
                (true, KeyCode::Char('u')) => app.scroll_by(-app.half_page()),
                (true, KeyCode::Char('f')) => app.scroll_by(app.page_step()),
                (true, KeyCode::Char('b')) => app.scroll_by(-app.page_step()),
                // leave
                (false, KeyCode::Esc)
                | (false, KeyCode::Char('h'))
                | (false, KeyCode::Char('b'))
                | (false, KeyCode::Char('q')) => {
                    app.pending_g = false;
                    app.scroll = 0;
                    app.screen = Screen::Meetings;
                    app.status = "Meetings".into();
                }
                // scroll
                (false, KeyCode::Char('j')) | (_, KeyCode::Down) => {
                    app.pending_g = false;
                    app.scroll_by(1);
                }
                (false, KeyCode::Char('k')) | (_, KeyCode::Up) => {
                    app.pending_g = false;
                    app.scroll_by(-1);
                }
                (_, KeyCode::PageDown) => {
                    app.pending_g = false;
                    app.scroll_by(app.page_step());
                }
                (_, KeyCode::PageUp) => {
                    app.pending_g = false;
                    app.scroll_by(-app.page_step());
                }
                (_, KeyCode::Home) => {
                    app.pending_g = false;
                    app.scroll_to(0);
                }
                (_, KeyCode::End) => {
                    app.pending_g = false;
                    app.scroll_to(u16::MAX);
                }
                (false, KeyCode::Char('g')) => {
                    if app.pending_g {
                        app.scroll_to(0);
                        app.pending_g = false;
                    } else {
                        app.pending_g = true;
                    }
                }
                (false, KeyCode::Char('G')) => {
                    app.pending_g = false;
                    app.scroll_to(u16::MAX);
                }
                (false, KeyCode::Char('s')) => {
                    app.pending_g = false;
                    app.begin_summary_prep();
                }
                (false, KeyCode::Char('c')) => {
                    app.pending_g = false;
                    app.copy_summary_plaintext();
                }
                (false, KeyCode::Char('t')) if app.screen == Screen::Summary => {
                    app.pending_g = false;
                    app.open_transcript().await?;
                }
                (false, KeyCode::Char('d')) | (_, KeyCode::Delete) => {
                    app.pending_g = false;
                    app.begin_delete_confirm();
                }
                _ => {
                    app.pending_g = false;
                }
            }
        }
        Screen::Settings => match key.code {
            KeyCode::Esc | KeyCode::Char('h') | KeyCode::Char('b') => {
                app.pending_g = false;
                app.screen = Screen::Meetings;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                app.pending_g = false;
                app.models_move(-1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                app.pending_g = false;
                app.models_move(1);
            }
            KeyCode::Char('g') => {
                if app.pending_g {
                    app.models_state.select(Some(0));
                    app.pending_g = false;
                } else {
                    app.pending_g = true;
                }
            }
            KeyCode::Char('G') => {
                app.pending_g = false;
                if !app.models_list.is_empty() {
                    app.models_state.select(Some(app.models_list.len() - 1));
                }
            }
            KeyCode::Enter | KeyCode::Char('l') => {
                app.pending_g = false;
                app.select_model_at_cursor().await?;
            }
            KeyCode::Tab => {
                app.pending_g = false;
                app.cycle_summary_backend();
            }
            KeyCode::Char('q') => app.should_quit = true,
            _ => {
                app.pending_g = false;
            }
        },
        Screen::Meetings => {
            let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
            match key.code {
                KeyCode::Char('q') => app.should_quit = true,
                KeyCode::Up | KeyCode::Char('k') => {
                    app.pending_g = false;
                    app.list_move(-1);
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    app.pending_g = false;
                    app.list_move(1);
                }
                KeyCode::Char('d') if ctrl => {
                    app.pending_g = false;
                    app.list_move(10);
                }
                KeyCode::Char('u') if ctrl => {
                    app.pending_g = false;
                    app.list_move(-10);
                }
                KeyCode::Char('g') => {
                    if app.pending_g {
                        app.list_goto_top();
                        app.pending_g = false;
                    } else {
                        app.pending_g = true;
                    }
                }
                KeyCode::Char('G') => {
                    app.pending_g = false;
                    app.list_goto_bottom();
                }
                KeyCode::Enter | KeyCode::Char('l') => {
                    app.pending_g = false;
                    app.open_meeting().await?;
                }
                KeyCode::Char('t') => {
                    app.pending_g = false;
                    app.open_transcript().await?;
                }
                KeyCode::Char('h') => {
                    // stay on list
                    app.pending_g = false;
                }
                KeyCode::Char('r') => {
                    app.pending_g = false;
                    app.start_or_stop_recording().await?;
                }
                KeyCode::Char('s') => {
                    app.pending_g = false;
                    app.open_or_prep_summary().await?;
                }
                KeyCode::Char('v') => {
                    app.pending_g = false;
                    app.view_cached_summary().await?;
                }
                KeyCode::Char('c') => {
                    app.pending_g = false;
                    app.view_cached_summary().await?;
                }
                KeyCode::Char('d') | KeyCode::Delete if !ctrl => {
                    app.pending_g = false;
                    app.begin_delete_confirm();
                }
                KeyCode::Char('m') => {
                    app.pending_g = false;
                    app.refresh_models();
                    app.screen = Screen::Settings;
                    app.status = format!(
                        "Models — {}/{}  j/k · Enter select · Tab backend · h back",
                        app.model_selection.provider, app.model_selection.model
                    );
                }
                KeyCode::Char('R') => {
                    // capital R = refresh list (vim-ish reload)
                    app.pending_g = false;
                    app.refresh_meetings().await?;
                    app.status = format!("Refreshed ({} meetings)", app.meetings.len());
                }
                KeyCode::Tab => {
                    app.pending_g = false;
                    app.cycle_summary_backend();
                }
                _ => {
                    app.pending_g = false;
                }
            }
        }
    }
    Ok(())
}

fn draw_ui(f: &mut ratatui::Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(3),
        ])
        .split(f.area());

    let title = Paragraph::new(Line::from(vec![
        Span::styled(
            " meeticulous ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(
            "  Meetily data · sum={}",
            app.summary_prefs.backend
        )),
    ]))
    .block(Block::default().borders(Borders::ALL).title("meeticulous"));
    f.render_widget(title, chunks[0]);

    // Content height inside border for page scrolling.
    app.content_height = chunks[1].height.saturating_sub(2);
    app.content_width = chunks[1].width.saturating_sub(2);

    match app.screen {
        Screen::Meetings => draw_meetings(f, app, chunks[1]),
        Screen::Transcript => draw_scroll_text(
            f,
            chunks[1],
            "Transcript",
            &app.transcript_text,
            app.scroll,
            "j/k · gg/G · C-d/u · s regen · d del · h back",
        ),
        Screen::Summary => draw_summary_markdown(f, app, chunks[1]), // mut app via &mut
        Screen::Summarizing => draw_scroll_text(
            f,
            chunks[1],
            "Summary",
            &app.summary_text,
            app.scroll,
            "generating…",
        ),
        Screen::Recording => draw_recording(f, app, chunks[1]),
        Screen::Settings => draw_settings(f, app, chunks[1]),
        Screen::SummaryPrep => draw_summary_prep(f, app, chunks[1]),
        Screen::DeleteConfirm => draw_delete_confirm(f, app, chunks[1]),
    }

    let status = Paragraph::new(app.status.as_str())
        .block(Block::default().borders(Borders::ALL).title("status"));
    f.render_widget(status, chunks[2]);
}

fn draw_meetings(f: &mut ratatui::Frame, app: &mut App, area: Rect) {
    let items: Vec<ListItem> = if app.meetings.is_empty() {
        vec![ListItem::new("(no meetings yet — press r to record)")]
    } else {
        app.meetings
            .iter()
            .map(|m| {
                ListItem::new(format!(
                    "{}  {}",
                    m.created_at.get(..10).unwrap_or(&m.created_at),
                    m.title
                ))
            })
            .collect()
    };
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(
            "Meetings  j/k · l/Enter summary · t transcript · r rec · s regen · c copy · d del · q",
        ))
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");
    f.render_stateful_widget(list, area, &mut app.list_state);
}

fn draw_scroll_text(
    f: &mut ratatui::Frame,
    area: Rect,
    title: &str,
    text: &str,
    scroll: u16,
    hints: &str,
) {
    let wrap_width = area.width.saturating_sub(2).max(1) as usize;
    let line_count = wrapped_line_count(text, wrap_width).max(1) as u16;
    let view_h = area.height.saturating_sub(2).max(1);
    let max_scroll = line_count.saturating_sub(view_h);
    let scroll = scroll.min(max_scroll);
    let p = Paragraph::new(text.to_string())
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!("{title}  {hints}  ({scroll}/{max_scroll})")),
        );
    f.render_widget(p, area);
}

fn draw_summary_markdown(f: &mut ratatui::Frame, app: &mut App, area: Rect) {
    let wrap_width = area.width.saturating_sub(2).max(1) as usize;
    let mut lines: Vec<Line> = Vec::new();
    if !app.summary_meta.is_empty() {
        for wl in soft_wrap_line(
            &Line::from(Span::styled(
                app.summary_meta.clone(),
                Style::default().fg(Color::DarkGray),
            )),
            wrap_width,
        ) {
            lines.push(wl);
        }
        lines.push(Line::from(""));
    }
    for line in &app.summary_lines {
        lines.extend(soft_wrap_line(line, wrap_width));
    }

    let line_count = lines.len().max(1) as u16;
    app.summary_wrapped_len = line_count;
    let view_h = area.height.saturating_sub(2).max(1);
    let max_scroll = line_count.saturating_sub(view_h);
    let scroll = app.scroll.min(max_scroll);

    let p = Paragraph::new(lines).scroll((scroll, 0)).block(
        Block::default().borders(Borders::ALL).title(format!(
            "Summary  j/k · s regen · c copy · t transcript · h back  ({scroll}/{max_scroll})"
        )),
    );
    f.render_widget(p, area);
}

/// Soft-wrap a styled line to `width` columns, breaking at word boundaries
/// (spaces). Only splits mid-word if a single token is wider than `width`.
fn soft_wrap_line(line: &Line<'_>, width: usize) -> Vec<Line<'static>> {
    fn owned_line(line: &Line<'_>) -> Line<'static> {
        Line::from(
            line.spans
                .iter()
                .map(|s| Span::styled(s.content.to_string(), s.style))
                .collect::<Vec<_>>(),
        )
    }

    if width == 0 {
        return vec![owned_line(line)];
    }
    let total: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
    if total <= width {
        return vec![owned_line(line)];
    }

    // Flatten to (char, style), then pack words.
    let mut chars: Vec<(char, Style)> = Vec::with_capacity(total);
    for span in &line.spans {
        for ch in span.content.chars() {
            chars.push((ch, span.style));
        }
    }

    // Split into tokens: runs of non-space, or single spaces.
    let mut tokens: Vec<Vec<(char, Style)>> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].0.is_whitespace() {
            tokens.push(vec![chars[i]]);
            i += 1;
        } else {
            let start = i;
            while i < chars.len() && !chars[i].0.is_whitespace() {
                i += 1;
            }
            tokens.push(chars[start..i].to_vec());
        }
    }

    let mut out: Vec<Line<'static>> = Vec::new();
    let mut cur: Vec<(char, Style)> = Vec::new();
    let mut cur_len = 0usize;

    let flush = |cur: &mut Vec<(char, Style)>, out: &mut Vec<Line<'static>>| {
        // Trim trailing spaces on a wrapped line
        while cur.last().is_some_and(|(c, _)| c.is_whitespace()) {
            cur.pop();
        }
        out.push(chars_to_line(std::mem::take(cur)));
    };

    for token in tokens {
        let tlen = token.len();
        if tlen == 0 {
            continue;
        }
        // Space that doesn't fit → new line (drop leading spaces on new lines)
        if token.iter().all(|(c, _)| c.is_whitespace()) {
            if cur_len == 0 {
                continue; // no leading spaces after wrap
            }
            if cur_len + tlen <= width {
                cur.extend(token);
                cur_len += tlen;
            } else {
                flush(&mut cur, &mut out);
                cur_len = 0;
            }
            continue;
        }

        // Word longer than width: hard-break as last resort
        if tlen > width {
            if cur_len > 0 {
                flush(&mut cur, &mut out);
                cur_len = 0;
            }
            for chunk in token.chunks(width) {
                out.push(chars_to_line(chunk.to_vec()));
            }
            continue;
        }

        if cur_len + tlen > width {
            flush(&mut cur, &mut out);
            cur_len = 0;
        }
        cur.extend(token);
        cur_len += tlen;
    }
    if !cur.is_empty() {
        flush(&mut cur, &mut out);
    }
    if out.is_empty() {
        out.push(Line::from(""));
    }
    out
}

fn wrapped_line_count(text: &str, width: usize) -> usize {
    text.lines()
        .map(|l| soft_wrap_line(&Line::from(l), width).len())
        .sum()
}

fn chars_to_line(chars: Vec<(char, Style)>) -> Line<'static> {
    if chars.is_empty() {
        return Line::from("");
    }
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let mut style = chars[0].1;
    for (ch, st) in chars {
        if st == style {
            buf.push(ch);
        } else {
            spans.push(Span::styled(std::mem::take(&mut buf), style));
            style = st;
            buf.push(ch);
        }
    }
    if !buf.is_empty() {
        spans.push(Span::styled(buf, style));
    }
    Line::from(spans)
}

fn copy_to_clipboard(text: &str) -> Result<(), String> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new("pbcopy")
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| format!("pbcopy: {e}"))?;
    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| "pbcopy stdin missing".to_string())?;
        stdin
            .write_all(text.as_bytes())
            .map_err(|e| format!("write clipboard: {e}"))?;
    }
    let status = child.wait().map_err(|e| format!("pbcopy wait: {e}"))?;
    if !status.success() {
        return Err(format!("pbcopy exited {status}"));
    }
    Ok(())
}

fn draw_recording(f: &mut ratatui::Frame, app: &mut App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12),
            Constraint::Min(4),
            Constraint::Length(3),
        ])
        .split(area);

    // Live transcript pane height drives scroll math.
    app.content_height = chunks[1].height.saturating_sub(2);
    app.content_width = chunks[1].width.saturating_sub(2);

    let verbose = app
        .recording
        .as_ref()
        .map(|h| {
            format!(
                "REC ● {}  ({:.0}s)\nfolder: {}\n\n{}",
                h.title,
                h.elapsed_secs(),
                h.folder_path.display(),
                h.verbose_status()
            )
        })
        .unwrap_or_else(|| "no active session".into());

    let diag = Paragraph::new(verbose).wrap(Wrap { trim: false }).block(
        Block::default()
            .borders(Borders::ALL)
            .title("verbose · system / mic / STT / log"),
    );
    f.render_widget(diag, chunks[0]);

    let body = if app.live_lines.is_empty() {
        "(waiting for speech… j/k scroll when lines appear · G follow bottom · r stop)".to_string()
    } else {
        app.live_lines.join("\n")
    };
    // Wrapped line count so long lines stay reachable when scrolling.
    let wrap_width = chunks[1].width.saturating_sub(2).max(1) as usize;
    let line_count = wrapped_line_count(&body, wrap_width).max(1) as u16;
    let view_h = chunks[1].height.saturating_sub(2).max(1);
    let max_scroll = line_count.saturating_sub(view_h);
    let scroll = if app.live_follow {
        max_scroll
    } else {
        app.scroll.min(max_scroll)
    };
    // Keep app.scroll in sync when following so page math stays consistent.
    if app.live_follow {
        app.scroll = scroll;
    }

    let follow = if app.live_follow { "follow" } else { "manual" };
    let p = Paragraph::new(body)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0))
        .block(Block::default().borders(Borders::ALL).title(format!(
            "live transcript  j/k scroll · G follow · r stop  ({scroll}/{max_scroll} · {follow})"
        )));
    f.render_widget(p, chunks[1]);

    let input = Paragraph::new(format!("> {}", app.input_buf)).block(
        Block::default()
            .borders(Borders::ALL)
            .title("type to append segment · Enter send · (empty buffer: j/k scroll)"),
    );
    f.render_widget(input, chunks[2]);
}

fn draw_settings(f: &mut ratatui::Frame, app: &mut App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(5), Constraint::Min(3)])
        .split(area);

    let sum = Paragraph::new(format!(
        "Summary backend: {}\nContext: {}\n[Tab] cycle backend (opencode → agy → claude)",
        backend_status_line(app.summary_prefs.backend),
        if app.summary_prefs.context.is_empty() {
            "(none — set when summarizing with s)".to_string()
        } else {
            format_context_preview(&app.summary_prefs.context)
                .lines()
                .collect::<Vec<_>>()
                .join(" · ")
        }
    ))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("summary settings"),
    );
    f.render_widget(sum, chunks[0]);

    let items: Vec<ListItem> = app
        .models_list
        .iter()
        .map(|s| ListItem::new(s.as_str()))
        .collect();
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(format!(
            "STT Models ({})  [Enter] select  [Esc] back",
            app.paths.models_dir.display()
        )))
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");
    f.render_stateful_widget(list, chunks[1], &mut app.models_state);
}

fn draw_summary_prep(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let title = app
        .selected_meeting()
        .map(|m| m.title.as_str())
        .unwrap_or("?");
    let preview = format_context_preview(&app.input_buf);
    let n_lines = app.input_buf.lines().count();
    let n_chars = app.input_buf.chars().count();
    let size_hint = if app.input_buf.is_empty() {
        String::new()
    } else {
        format!("  ({n_lines} lines · {n_chars} chars in buffer)")
    };
    let body = format!(
        "Meeting: {title}\n\
         Backend: {}\n\
         Model:   {}\n\
         \n\
         Optional context (full text is kept; large pastes are collapsed in the UI):\n\
         \n\
         {preview}\n\
         {size_hint}\n\
         \n\
         [Tab] cycle backend   [Enter] generate   [Esc] cancel   (type/paste freely)",
        backend_status_line(app.summary_prefs.backend),
        app.resolved_summary_model(),
    );
    let p = Paragraph::new(body).wrap(Wrap { trim: false }).block(
        Block::default()
            .borders(Borders::ALL)
            .title("summarize · optional context"),
    );
    f.render_widget(p, area);
}

fn draw_delete_confirm(f: &mut ratatui::Frame, app: &App, area: Rect) {
    // Full content area, no opaque fill — same terminal bg as the rest of the TUI.
    let msg = format!(
        "Delete this meeting permanently?\n\n\
         “{}”\n\
         id: {}\n\n\
         This removes transcripts + summary from the shared Meetily database.\n\
         Recording files on disk are NOT deleted.\n\n\
         [y] yes, delete forever    [n] / Esc cancel",
        app.pending_delete_title,
        app.pending_delete_id.as_deref().unwrap_or("?")
    );
    let p = Paragraph::new(msg)
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(Color::Reset))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Red))
                .title(Span::styled(
                    " confirm delete ",
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                )),
        );
    f.render_widget(p, area);
}
