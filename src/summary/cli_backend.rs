//! Summarization backends that shell out to a logged-in CLI only:
//! - `opencode run`
//! - Antigravity `agy --print`
//! - Claude Code `claude -p`

use super::LlmTransport;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

/// Which CLI to use for summarization (no Meetily HTTP / settings path).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SummaryCliBackend {
    #[default]
    Opencode,
    Antigravity,
    Claude,
}

impl SummaryCliBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Opencode => "opencode",
            Self::Antigravity => "antigravity",
            Self::Claude => "claude",
        }
    }

    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "opencode" | "oc" => Some(Self::Opencode),
            "antigravity" | "agy" | "ag" => Some(Self::Antigravity),
            "claude" | "anthropic" | "cc" => Some(Self::Claude),
            // Legacy Meetily/http setting → treat as opencode
            "http" | "api" | "meetily" => Some(Self::Opencode),
            _ => None,
        }
    }

    pub fn cycle(self) -> Self {
        match self {
            Self::Opencode => Self::Antigravity,
            Self::Antigravity => Self::Claude,
            Self::Claude => Self::Opencode,
        }
    }
}

impl std::fmt::Display for SummaryCliBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Names that are labels, not real model IDs — never pass to `--model` / `-m`.
fn is_placeholder_model(m: &str) -> bool {
    matches!(
        m.trim().to_lowercase().as_str(),
        "" | "llama3.2"
            | "opencode"
            | "agy"
            | "antigravity"
            | "claude"
            | "http"
            | "meetily"
            | "default"
    )
}

fn resolve_model_flag(configured: Option<&str>, passed: &str) -> Option<String> {
    configured
        .map(str::trim)
        .filter(|s| !s.is_empty() && !is_placeholder_model(s))
        .map(|s| s.to_string())
        .or_else(|| {
            if is_placeholder_model(passed) {
                None
            } else {
                Some(passed.trim().to_string())
            }
        })
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

fn which_abs(p: &str) -> Option<PathBuf> {
    let path = PathBuf::from(p);
    path.is_file().then_some(path)
}

/// Resolve `opencode` binary (PATH + common locations).
pub fn find_opencode() -> Option<PathBuf> {
    which("opencode")
        .or_else(|| which_abs("/opt/homebrew/bin/opencode"))
        .or_else(|| which_abs("/usr/local/bin/opencode"))
}

/// Resolve Antigravity CLI (`agy`).
pub fn find_antigravity() -> Option<PathBuf> {
    which("agy")
        .or_else(|| which("antigravity"))
        .or_else(|| {
            let home = dirs::home_dir()?;
            let p = home.join(".local/bin/agy");
            if p.is_file() {
                Some(p)
            } else {
                None
            }
        })
}

/// Resolve Claude Code CLI.
pub fn find_claude() -> Option<PathBuf> {
    which("claude")
        .or_else(|| {
            let home = dirs::home_dir()?;
            let p = home.join(".local/bin/claude");
            if p.is_file() {
                Some(p)
            } else {
                None
            }
        })
        .or_else(|| which_abs("/usr/local/bin/claude"))
}

const CLI_TIMEOUT: Duration = Duration::from_secs(300);

/// Prefer stdout when non-empty even if exit code is non-zero.
fn pick_cli_output(
    status: std::process::ExitStatus,
    stdout: &str,
    stderr: &str,
    name: &str,
) -> Result<String, String> {
    let out = stdout.trim();
    let err = stderr.trim();
    if !out.is_empty() {
        return Ok(out.to_string());
    }
    if status.success() {
        return Err(format!("{name} returned empty stdout. stderr: {err}"));
    }
    Err(format!("{name} exited {status}: {err}"))
}

/// Default opencode model when the user's global default is broken/missing.
pub const DEFAULT_OPENCODE_MODEL: &str = "opencode-go/deepseek-v4-flash";

/// Sensible default for Antigravity when none configured.
pub const DEFAULT_ANTIGRAVITY_MODEL: &str = "gemini-3.5-flash-medium";

/// Optional Claude model pin (empty = CLI default / subscription model).
pub const DEFAULT_CLAUDE_MODEL: &str = "";

/// Transport that runs `opencode run` with the user's logged-in credentials.
pub struct OpencodeTransport {
    pub binary: PathBuf,
    pub model: Option<String>,
}

impl OpencodeTransport {
    pub fn discover() -> Result<Self, String> {
        let binary = find_opencode().ok_or_else(|| {
            "opencode not found on PATH (install opencode and log in via `opencode providers`)"
                .to_string()
        })?;
        Ok(Self {
            binary,
            model: Some(DEFAULT_OPENCODE_MODEL.to_string()),
        })
    }
}

#[async_trait::async_trait]
impl LlmTransport for OpencodeTransport {
    async fn complete(
        &self,
        _provider: &str,
        model: &str,
        _api_key: &str,
        system: &str,
        user: &str,
        _ollama_endpoint: Option<&str>,
    ) -> Result<String, String> {
        let tmp = tempfile::tempdir().map_err(|e| e.to_string())?;
        let prompt_path = tmp.path().join("meeticulous-summary-prompt.md");
        let body = format!(
            "{system}\n\n---\n\n{user}\n\n\
             Write plain-text meeting notes only (no JSON)."
        );
        std::fs::write(&prompt_path, &body).map_err(|e| e.to_string())?;

        let short_msg = "Read the attached meeticulous-summary-prompt.md file carefully and \
            write the plain-text meeting notes. No JSON.";

        let model_id = resolve_model_flag(self.model.as_deref(), model)
            .unwrap_or_else(|| DEFAULT_OPENCODE_MODEL.to_string());

        let mut cmd = Command::new(&self.binary);
        cmd.arg("run")
            .arg("-m")
            .arg(&model_id)
            .arg("--format")
            .arg("default")
            .arg(short_msg)
            .arg("--file")
            .arg(&prompt_path)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let output = timeout(CLI_TIMEOUT, cmd.output())
            .await
            .map_err(|_| "opencode timed out after 5 minutes".to_string())?
            .map_err(|e| format!("failed to spawn opencode: {e}"))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        pick_cli_output(output.status, &stdout, &stderr, "opencode")
    }
}

/// Transport for Antigravity CLI: `agy --print "…"`.
pub struct AntigravityTransport {
    pub binary: PathBuf,
    pub model: Option<String>,
}

impl AntigravityTransport {
    pub fn discover() -> Result<Self, String> {
        let binary = find_antigravity().ok_or_else(|| {
            "agy (Antigravity CLI) not found — install Antigravity CLI and ensure `agy` is on PATH"
                .to_string()
        })?;
        Ok(Self {
            binary,
            model: Some(DEFAULT_ANTIGRAVITY_MODEL.to_string()),
        })
    }
}

#[async_trait::async_trait]
impl LlmTransport for AntigravityTransport {
    async fn complete(
        &self,
        _provider: &str,
        model: &str,
        _api_key: &str,
        system: &str,
        user: &str,
        _ollama_endpoint: Option<&str>,
    ) -> Result<String, String> {
        // Always pass the prompt inline. Telling agy to "read a file" makes it
        // tool-call and dump monologue like "I am going to check permissions…".
        let prompt = format!(
            "{system}\n\n---\n\n{user}\n\n\
             Write the meeting notes as plain text only. \
             Do not narrate tools, permissions, or file access. \
             Do not say what you are about to do — start directly with the notes \
             (preferably a markdown title line beginning with #)."
        );

        let model_id = resolve_model_flag(self.model.as_deref(), model)
            .unwrap_or_else(|| DEFAULT_ANTIGRAVITY_MODEL.to_string());

        let mut cmd = Command::new(&self.binary);
        cmd.arg("--print")
            .arg(&prompt)
            .arg("--dangerously-skip-permissions")
            .arg("--model")
            .arg(&model_id)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let output = timeout(CLI_TIMEOUT, cmd.output())
            .await
            .map_err(|_| "agy timed out after 5 minutes".to_string())?
            .map_err(|e| format!("failed to spawn agy: {e}"))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let raw = pick_cli_output(output.status, &stdout, &stderr, "agy")?;
        Ok(strip_agent_chatter(&raw))
    }
}

/// Drop leading agent monologue (tool narration) that some CLIs prepend.
fn strip_agent_chatter(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return String::new();
    }

    let is_chatter = |t: &str| -> bool {
        let lower = t.to_lowercase();
        let t = lower.trim();
        if t.is_empty() {
            return true;
        }
        // Common agent preambles
        t.starts_with("i am ")
            || t.starts_with("i'm ")
            || t.starts_with("i will ")
            || t.starts_with("i'll ")
            || t.starts_with("i can ")
            || t.starts_with("i'll ")
            || t.starts_with("let me ")
            || t.starts_with("first,")
            || t.starts_with("first ")
            || t.starts_with("okay,")
            || t.starts_with("ok,")
            || t.starts_with("sure,")
            || t.contains("check the available permissions")
            || t.contains("view the contents")
            || t.contains("read the instructions")
            || t.contains("read the file")
            || t.contains("access the requested file")
            || t.contains("going to check")
            || t.contains("as an ai")
            || t.contains("i don't have access")
            || t.contains("i cannot access")
    };

    // Prefer starting at the first markdown heading.
    if let Some(i) = lines.iter().position(|l| l.trim_start().starts_with('#')) {
        // Only jump to heading if earlier lines look like chatter / empty.
        let prefix_is_chatter = lines[..i].iter().all(|l| {
            let t = l.trim();
            t.is_empty() || is_chatter(t)
        });
        if prefix_is_chatter {
            return lines[i..].join("\n").trim().to_string();
        }
    }

    // Otherwise drop leading chatter lines until a "real" line.
    let mut start = 0;
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if t.is_empty() {
            start = i + 1;
            continue;
        }
        if is_chatter(t) {
            start = i + 1;
            continue;
        }
        start = i;
        break;
    }
    lines[start..].join("\n").trim().to_string()
}

/// Transport for Claude Code CLI: `claude -p …` (logged-in subscription / CLI auth).
pub struct ClaudeTransport {
    pub binary: PathBuf,
    pub model: Option<String>,
}

impl ClaudeTransport {
    pub fn discover() -> Result<Self, String> {
        let binary = find_claude().ok_or_else(|| {
            "claude CLI not found on PATH (install Claude Code and log in with `claude`)"
                .to_string()
        })?;
        Ok(Self {
            binary,
            model: None,
        })
    }
}

#[async_trait::async_trait]
impl LlmTransport for ClaudeTransport {
    async fn complete(
        &self,
        _provider: &str,
        model: &str,
        _api_key: &str,
        system: &str,
        user: &str,
        _ollama_endpoint: Option<&str>,
    ) -> Result<String, String> {
        let prompt = format!(
            "{system}\n\n---\n\n{user}\n\n\
             Write plain-text meeting notes only (no JSON)."
        );

        let mut cmd = Command::new(&self.binary);
        cmd.arg("-p")
            .arg(&prompt)
            .arg("--output-format")
            .arg("text")
            // Avoid tool-use side effects for a pure text summary.
            .arg("--bare")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        if let Some(m) = resolve_model_flag(self.model.as_deref(), model) {
            cmd.arg("--model").arg(m);
        }

        let output = timeout(CLI_TIMEOUT, cmd.output())
            .await
            .map_err(|_| "claude timed out after 5 minutes".to_string())?
            .map_err(|e| format!("failed to spawn claude: {e}"))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        pick_cli_output(output.status, &stdout, &stderr, "claude")
    }
}

/// Build the preferred transport for a backend choice.
pub fn transport_for_backend(
    backend: SummaryCliBackend,
) -> Result<Box<dyn LlmTransport>, String> {
    match backend {
        SummaryCliBackend::Opencode => Ok(Box::new(OpencodeTransport::discover()?)),
        SummaryCliBackend::Antigravity => Ok(Box::new(AntigravityTransport::discover()?)),
        SummaryCliBackend::Claude => Ok(Box::new(ClaudeTransport::discover()?)),
    }
}

/// Human-readable availability line for the TUI.
pub fn backend_status_line(backend: SummaryCliBackend) -> String {
    match backend {
        SummaryCliBackend::Opencode => match find_opencode() {
            Some(p) => format!("opencode · {}", p.display()),
            None => "opencode · NOT FOUND".into(),
        },
        SummaryCliBackend::Antigravity => match find_antigravity() {
            Some(p) => format!("antigravity (agy) · {}", p.display()),
            None => "antigravity · NOT FOUND".into(),
        },
        SummaryCliBackend::Claude => match find_claude() {
            Some(p) => format!("claude · {}", p.display()),
            None => "claude · NOT FOUND".into(),
        },
    }
}

/// Persist summary UI prefs under Meetily app-data root.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SummaryPrefs {
    pub backend: SummaryCliBackend,
    /// Extra context appended into the system prompt.
    #[serde(default)]
    pub context: String,
    /// Optional real model id for CLI backends.
    #[serde(default)]
    pub model: String,
}

impl Default for SummaryPrefs {
    fn default() -> Self {
        let backend = if find_opencode().is_some() {
            SummaryCliBackend::Opencode
        } else if find_antigravity().is_some() {
            SummaryCliBackend::Antigravity
        } else if find_claude().is_some() {
            SummaryCliBackend::Claude
        } else {
            SummaryCliBackend::Opencode
        };
        Self {
            backend,
            context: String::new(),
            model: String::new(),
        }
    }
}

const PREFS_FILE: &str = "meeticulous-summary-prefs.json";

pub fn load_summary_prefs(app_data: &Path) -> SummaryPrefs {
    let p = app_data.join(PREFS_FILE);
    let Ok(raw) = std::fs::read_to_string(p) else {
        return SummaryPrefs::default();
    };
    // Migrate legacy "http" / unknown backends via Value first.
    if let Ok(mut v) = serde_json::from_str::<serde_json::Value>(&raw) {
        if let Some(b) = v.get("backend").and_then(|x| x.as_str()) {
            if let Some(mapped) = SummaryCliBackend::from_str_loose(b) {
                v["backend"] = serde_json::json!(mapped.as_str());
            } else {
                v["backend"] = serde_json::json!("opencode");
            }
        }
        if let Ok(prefs) = serde_json::from_value::<SummaryPrefs>(v) {
            return prefs;
        }
    }
    SummaryPrefs::default()
}

pub fn save_summary_prefs(app_data: &Path, prefs: &SummaryPrefs) -> anyhow::Result<()> {
    let p = app_data.join(PREFS_FILE);
    std::fs::write(p, serde_json::to_string_pretty(prefs)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_cycle_and_parse() {
        assert_eq!(
            SummaryCliBackend::from_str_loose("opencode"),
            Some(SummaryCliBackend::Opencode)
        );
        assert_eq!(
            SummaryCliBackend::from_str_loose("claude"),
            Some(SummaryCliBackend::Claude)
        );
        assert_eq!(
            SummaryCliBackend::from_str_loose("http"),
            Some(SummaryCliBackend::Opencode)
        );
        assert_eq!(
            SummaryCliBackend::Opencode.cycle(),
            SummaryCliBackend::Antigravity
        );
        assert_eq!(
            SummaryCliBackend::Antigravity.cycle(),
            SummaryCliBackend::Claude
        );
        assert_eq!(
            SummaryCliBackend::Claude.cycle(),
            SummaryCliBackend::Opencode
        );
    }

    #[test]
    fn placeholder_models_never_selected() {
        assert!(resolve_model_flag(None, "agy").is_none());
        assert!(resolve_model_flag(None, "opencode").is_none());
        assert!(resolve_model_flag(None, "claude").is_none());
        assert_eq!(
            resolve_model_flag(None, "opencode/big-pickle").as_deref(),
            Some("opencode/big-pickle")
        );
    }

    #[test]
    fn prefs_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let prefs = SummaryPrefs {
            backend: SummaryCliBackend::Claude,
            context: "Focus on hiring decisions".into(),
            model: String::new(),
        };
        save_summary_prefs(tmp.path(), &prefs).unwrap();
        let loaded = load_summary_prefs(tmp.path());
        assert_eq!(loaded.backend, SummaryCliBackend::Claude);
        assert!(loaded.context.contains("hiring"));
    }

    #[test]
    fn legacy_http_prefs_migrate() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join(PREFS_FILE);
        std::fs::write(
            &p,
            r#"{"backend":"http","context":"x","model":""}"#,
        )
        .unwrap();
        let loaded = load_summary_prefs(tmp.path());
        assert_eq!(loaded.backend, SummaryCliBackend::Opencode);
    }

    #[test]
    fn pick_cli_output_prefers_stdout_on_nonzero() {
        use std::os::unix::process::ExitStatusExt;
        let st = std::process::ExitStatus::from_raw(256); // exit 1
        let r = pick_cli_output(st, "  {\"summary\":\"hi\"}  ", "Error: boom", "opencode").unwrap();
        assert!(r.contains("summary"));
    }

    #[test]
    fn strips_agy_permission_monologue() {
        let raw = "I am going to check the available permissions to see if I can access the requested file. I will view the contents of the requested file to read the instructions.\n\n# Club Interview\n\nAryan discussed AI Agents.\n";
        let cleaned = strip_agent_chatter(raw);
        assert!(cleaned.starts_with("# Club Interview"));
        assert!(!cleaned.to_lowercase().contains("permissions"));
    }
}
