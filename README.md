<div align="center">

# meeticulous

**a macOS-only terminal UI for <a href="https://github.com/Zackriya-Solutions/meetily">Meetily</a> meetings.**
Record meetings, live-transcribe with your existing Parakeet/Whisper models, browse transcripts, and summarize with **opencode**, **Antigravity**, or **Claude Code**.

</div>


| Feature | Screenshot |
|---------|------------|
| **Meeting Transcription** | <img width="800" alt="Meeting Transcription" src="https://github.com/user-attachments/assets/00534af7-870f-477a-8135-c7fbbd7fa462" /> |
| **Meeting Summarization** | <img width="800" alt="Meeting Summarization" src="https://github.com/user-attachments/assets/2ebf3914-a9f7-45f0-8d1e-a095827f355d" /> |


## Why

Meetily is a full GUI (Tauri). **meeticulous** is a minimal TUI for the same core loop on macOS:

- list / open / delete meetings
- live record (system audio + mic)
- view timestamped transcripts
- summarize with your already-logged-in CLIs
- optional long context for the summarizer

## Requirements

| Need | Notes |
|------|--------|
| **macOS** | Only supported OS (`compile_error` elsewhere) |
| **Rust** | 1.77+ (`cargo`, `rustc`) |
| **swiftc** | Builds `meeticulous-system-audio` (Core Audio process tap) |
| **Mic + Audio Capture** | System Settings → Privacy & Security → Microphone / **Audio Capture** for Terminal/iTerm |
| **Optional CLIs** | `opencode`, `agy` (Antigravity), `claude` for summarization |
| **Optional Meetily data** | Existing models + DB under `com.meetily.ai` |

## Install / run

```bash
git clone <your-fork-or-path>
cd meeticulous
cargo build --release
./target/release/meeticulous
```

The release build also produces `target/release/meeticulous-system-audio` (keep it next to the binary).

```bash
# put on PATH (optional)
cp target/release/meeticulous target/release/meeticulous-system-audio ~/.local/bin/
```

## Shared data paths (frozen with Meetily GUI)

| What | Path |
|------|------|
| App data | `~/Library/Application Support/com.meetily.ai/` |
| Database | `…/meeting_minutes.sqlite` |
| Models | `…/models/` (`ggml-*.bin`, `parakeet/…`, `summary/…`) |
| Recordings | `~/Movies/meetily-recordings/` |

Durable identity is **`com.meetily.ai`**, not `meeticulous`. Prefs that are TUI-only live as small JSON files in that same folder (e.g. summary backend).

```bash
meeticulous --paths   # print resolved roots
```

## Features

### Recording
- **System audio** via Core Audio process tap (Zoom/Meet/etc. on default output)
- **Mic** via CPAL (secondary; STT prefers system when available)
- Live transcript with timestamps (`MM:SS` or `HH:MM:SS`)
- Scroll live lines (`j`/`k`, auto-follow with `G`)
- Segments written into the shared SQLite DB + WAV under `meetily-recordings`

### Meetings
- List / open / delete (confirm with `y`)
- Transcript view with scroll + timestamps
- Rename a title with `e`, or bulk-rename all titles with `E` — both open `$EDITOR` (nvim by default), one title per line
- Summaries stored in Meetily’s `summary_processes` table

### Summarization (CLI only)
| Backend | Command surface | Default model |
|---------|-----------------|---------------|
| **opencode** | `opencode run` | `opencode-go/deepseek-v4-flash` |
| **antigravity** | `agy --print` | `gemini-3.5-flash-medium` |
| **claude** | `claude -p` | CLI default |

## TUI keys

### Meetings list
| Key | Action |
|-----|--------|
| `j` / `k` | move |
| `Enter` / `l` | open **summary** if present, else transcript |
| `t` | transcript |
| `r` | start recording |
| `s` | regenerate summary (context screen) |
| `v` | open saved summary |
| `e` | rename meeting title (opens `$EDITOR`/nvim) |
| `E` | bulk-rename all meeting titles in `$EDITOR`/nvim (one per line) |
| `d` | delete (confirm `y`) |
| `c` | copy summary plaintext (when viewing) |
| `m` | models / backend |
| `Tab` | cycle summary backend |
| `R` | refresh list |
| `gg` / `G` | top / bottom |
| `q` | quit |

### Transcript / summary view
| Key | Action |
|-----|--------|
| `j`/`k`, `C-d`/`C-u`, `C-f`/`C-b` | scroll |
| `gg` / `G` | top / bottom |
| `s` | regenerate summary |
| `c` | copy summary body |
| `e` | rename this meeting's title (opens `$EDITOR`/nvim) |
| `t` | transcript (from summary) |
| `h` / `Esc` | back |

### recording
| Key | Action |
|-----|--------|
| `j`/`k` (empty input) | scroll live transcript |
| `G` | jump bottom + follow |
| type + `Enter` | append manual segment |
| `r` / `Esc` | stop |

### summary context screen
Type/paste freely (`h`/`j`/`k` are normal characters).
`Tab` cycles backend · `Enter` runs · `Esc` cancels.

## Non-goals

- **Windows/Linux support**
- Pixel parity with Meetily GUI
- Meetily PRO, licensing or calendar features
- Replacing Meetily’s Application Support identity

## License

MIT 
