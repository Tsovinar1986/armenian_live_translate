"""Export local Claude Code session logs (~/.claude/projects/<this project>/*.jsonl) to Markdown in chats/.

    python scripts/export_chats.py            # one .md per session, oldest first

Everything in the conversation is written out unedited: user messages, assistant text, thinking, every tool call
with its full input, and every tool result. Only the logger's internal bookkeeping records (API request shapes,
cost state, file snapshots) are skipped. Re-run at the end so the current session is complete.
Sessions that ran on claude.ai (web) are not on this Mac; export those from claude.ai.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(ROOT))
OUT = ROOT / "chats"


def fence(text: str, lang: str = "") -> str:
    ticks = "```"
    while ticks in text:
        ticks += "`"
    return f"{ticks}{lang}\n{text}\n{ticks}"


def result_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if b.get("type") == "text":
            parts.append(b["text"])
        else:
            parts.append(f"[{b.get('type')} block]")
    return "\n".join(parts)


def render_block(role: str, b: dict) -> str:
    kind = b.get("type")
    if kind == "text":
        return b["text"]
    if kind == "thinking":
        return f"<details><summary>thinking</summary>\n\n{fence(b.get('thinking', ''))}\n\n</details>"
    if kind == "tool_use":
        return f"**Tool call: `{b['name']}`**\n\n{fence(json.dumps(b['input'], ensure_ascii=False, indent=2), 'json')}"
    if kind == "tool_result":
        err = " (error)" if b.get("is_error") else ""
        return f"**Tool result{err}:**\n\n{fence(result_text(b.get('content')))}"
    if kind == "document":
        src = b.get("source", {})
        return f"[attached document: {b.get('title') or src.get('media_type', 'document')}]"
    if kind == "image":
        return "[attached image]"
    return f"[{kind} block]"


def export(path: Path) -> tuple[dt.datetime, str]:
    lines, start, end = [], None, None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        d = json.loads(raw)
        if d.get("type") not in ("user", "assistant"):
            continue
        when = dt.datetime.fromisoformat(d["timestamp"].replace("Z", "+00:00")).astimezone()
        start, end = start or when, when
        content = d["message"]["content"]
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
        body = "\n\n".join(render_block(d["type"], b) for b in blocks)
        if body.strip():
            lines.append(f"### {d['type'].capitalize()} · {when:%Y-%m-%d %H:%M:%S}\n\n{body}\n")
    header = (f"# Claude Code session {path.stem}\n\n"
              f"{start:%Y-%m-%d %H:%M} – {end:%Y-%m-%d %H:%M} (local time). "
              f"Exported unedited from `{path.name}` by `scripts/export_chats.py`.\n\n---\n\n")
    return start, header + "\n".join(lines)


def main() -> None:
    sessions = sorted(LOGS.glob("*.jsonl"))
    if not sessions:
        raise SystemExit(f"no session logs in {LOGS}")
    exported = sorted((export(p) + (p,) for p in sessions), key=lambda x: x[0])
    for start, text, p in exported:
        out = OUT / f"claude-code-{start:%Y-%m-%d-%H%M}-{p.stem[:8]}.md"
        out.write_text(text, encoding="utf-8")
        print(f"{out.relative_to(ROOT)}  ({len(text) / 1000:.0f} kB)")


if __name__ == "__main__":
    main()
