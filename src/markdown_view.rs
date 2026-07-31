//! Lightweight markdown → ratatui lines for the summary pane.

use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// Render markdown into owned ratatui lines (scrollable).
pub fn markdown_to_lines(md: &str) -> Vec<Line<'static>> {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TABLES);

    let parser = Parser::new_ext(md, opts);
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut current: Vec<Span<'static>> = Vec::new();
    let mut style_stack: Vec<Style> = vec![Style::default()];
    let mut list_depth: usize = 0;
    let mut pending_prefix: Option<String> = None;
    let mut in_code_block = false;

    let push_line = |lines: &mut Vec<Line<'static>>, current: &mut Vec<Span<'static>>| {
        if current.is_empty() {
            lines.push(Line::from(""));
        } else {
            lines.push(Line::from(std::mem::take(current)));
        }
    };

    let current_style = |stack: &[Style]| *stack.last().unwrap_or(&Style::default());

    for ev in parser {
        match ev {
            Event::Start(tag) => match tag {
                Tag::Heading { level, .. } => {
                    if !current.is_empty() {
                        push_line(&mut lines, &mut current);
                    }
                    let style = match level {
                        pulldown_cmark::HeadingLevel::H1 => Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                        pulldown_cmark::HeadingLevel::H2 => Style::default()
                            .fg(Color::LightCyan)
                            .add_modifier(Modifier::BOLD),
                        _ => Style::default()
                            .fg(Color::Magenta)
                            .add_modifier(Modifier::BOLD),
                    };
                    style_stack.push(style);
                }
                Tag::Paragraph if !current.is_empty() => {
                    push_line(&mut lines, &mut current);
                }
                Tag::List(_) => {
                    list_depth += 1;
                }
                Tag::Item => {
                    if !current.is_empty() {
                        push_line(&mut lines, &mut current);
                    }
                    let indent = "  ".repeat(list_depth.saturating_sub(1));
                    pending_prefix = Some(format!("{indent}• "));
                }
                Tag::Emphasis => {
                    let s = current_style(&style_stack).add_modifier(Modifier::ITALIC);
                    style_stack.push(s);
                }
                Tag::Strong => {
                    let s = current_style(&style_stack).add_modifier(Modifier::BOLD);
                    style_stack.push(s);
                }
                Tag::Strikethrough => {
                    let s = current_style(&style_stack).add_modifier(Modifier::CROSSED_OUT);
                    style_stack.push(s);
                }
                Tag::CodeBlock(_) => {
                    if !current.is_empty() {
                        push_line(&mut lines, &mut current);
                    }
                    in_code_block = true;
                    style_stack.push(Style::default().fg(Color::DarkGray));
                }
                Tag::BlockQuote(_) => {
                    if !current.is_empty() {
                        push_line(&mut lines, &mut current);
                    }
                    style_stack.push(
                        Style::default()
                            .fg(Color::Gray)
                            .add_modifier(Modifier::ITALIC),
                    );
                    pending_prefix = Some("│ ".into());
                }
                Tag::Link { .. } | Tag::Image { .. } => {
                    style_stack.push(Style::default().fg(Color::LightBlue));
                }
                _ => {}
            },
            Event::End(tag) => match tag {
                TagEnd::Heading(_) => {
                    push_line(&mut lines, &mut current);
                    lines.push(Line::from(""));
                    style_stack.pop();
                }
                TagEnd::Paragraph => {
                    push_line(&mut lines, &mut current);
                    lines.push(Line::from(""));
                }
                TagEnd::Item => {
                    push_line(&mut lines, &mut current);
                }
                TagEnd::List(_) => {
                    list_depth = list_depth.saturating_sub(1);
                    lines.push(Line::from(""));
                }
                TagEnd::CodeBlock => {
                    push_line(&mut lines, &mut current);
                    lines.push(Line::from(""));
                    in_code_block = false;
                    style_stack.pop();
                }
                TagEnd::BlockQuote(_)
                | TagEnd::Emphasis
                | TagEnd::Strong
                | TagEnd::Strikethrough
                | TagEnd::Link
                | TagEnd::Image => {
                    style_stack.pop();
                }
                _ => {}
            },
            Event::Text(t) => {
                let style = current_style(&style_stack);
                let mut s = t.to_string();
                if let Some(prefix) = pending_prefix.take() {
                    s = format!("{prefix}{s}");
                }
                if in_code_block {
                    for (i, line) in s.split('\n').enumerate() {
                        if i > 0 {
                            push_line(&mut lines, &mut current);
                        }
                        current.push(Span::styled(line.to_string(), style));
                    }
                } else {
                    current.push(Span::styled(s, style));
                }
            }
            Event::Code(t) => {
                let style = if in_code_block {
                    current_style(&style_stack)
                } else {
                    Style::default().fg(Color::Yellow)
                };
                let mut s = t.to_string();
                if let Some(prefix) = pending_prefix.take() {
                    s = format!("{prefix}{s}");
                }
                current.push(Span::styled(s, style));
            }
            Event::SoftBreak => {
                current.push(Span::raw(" "));
            }
            Event::HardBreak => {
                push_line(&mut lines, &mut current);
            }
            Event::Rule => {
                if !current.is_empty() {
                    push_line(&mut lines, &mut current);
                }
                lines.push(Line::from(Span::styled(
                    "─".repeat(40),
                    Style::default().fg(Color::DarkGray),
                )));
                lines.push(Line::from(""));
            }
            Event::TaskListMarker(checked) => {
                pending_prefix = Some(if checked {
                    "[x] ".into()
                } else {
                    "[ ] ".into()
                });
            }
            Event::Html(t) | Event::InlineHtml(t) => {
                let style = current_style(&style_stack);
                for (i, line) in t.split('\n').enumerate() {
                    if i > 0 {
                        push_line(&mut lines, &mut current);
                    }
                    let mut s = line.to_string();
                    if i == 0 {
                        if let Some(prefix) = pending_prefix.take() {
                            s = format!("{prefix}{s}");
                        }
                    }
                    current.push(Span::styled(s, style));
                }
            }
            Event::FootnoteReference(name) => {
                current.push(Span::styled(
                    format!("[^{name}]"),
                    current_style(&style_stack),
                ));
            }
            _ => {}
        }
    }
    if !current.is_empty() {
        push_line(&mut lines, &mut current);
    }
    // Trim trailing empty lines
    while lines
        .last()
        .is_some_and(|l| l.spans.is_empty() || l.spans.iter().all(|s| s.content.is_empty()))
    {
        lines.pop();
    }
    if lines.is_empty() {
        lines.push(Line::from("(empty)"));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_heading_and_list() {
        let lines = markdown_to_lines("# Hello\n\n- one\n- two\n\n**bold** text\n");
        assert!(lines.len() >= 3);
        let joined: String = lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("Hello"));
        assert!(joined.contains("one"));
        assert!(joined.contains("bold"));
    }
}
