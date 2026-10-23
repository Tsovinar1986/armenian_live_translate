# AI chat transcripts

Every AI session used for this project, exported in full and unedited, one Markdown file per session.

| File | When | Where | What |
|---|---|---|---|
| *(to add)* | 21 Oct | claude.ai | Plan, PLAN.md / DECISIONS.md / TESTS.md, walking skeleton (first commit) |
| *(to add)* | 22 Oct | claude.ai project chat | Coordinator conversations that started the two helper sessions below |
| `claude-code-2026-10-21-2130-f5b8830e.md` | 21–23 Oct | Claude Code (VS Code) | Setup on the Mac, GitHub, BlackHole, FLEURS test set, ASR comparison, switch to NVIDIA FastConformer, speaker threshold |
| `claude-code-2026-10-22-1840-a7d1c4cd.md` | 22 Oct | Claude Code, started from the project chat | Sync with GitHub, extend .gitignore |
| `claude-code-2026-10-22-2115-44398c8d.md` | 22 Oct | Claude Code, started from the project chat | Multi-Output Device setup in Audio MIDI Setup |

The Claude Code sessions are converted from the local session logs by `scripts/export_chats.py` (all messages,
thinking, tool calls and tool results; only the logger's internal bookkeeping records are left out). The script
is re-run at the end, so the ongoing session is complete in the submitted version.

## `program-output/`: what the program printed

Terminal output (time, speaker, Armenian, English, latency) from test runs, one Markdown file per run, with the
model and setup written at the top. These are not AI chats; they are kept here so the conversations above can be
read next to what the program actually produced.

| File | Run |
|---|---|
| `program-output/2026-10-23-live-test-1-turbo-hy.md` | YouTube conversation, call side only, old `turbo-hy` model: 30 phrases, latency p90 4.51 s |
| `program-output/2026-10-23-test-call-live-blackhole.md` | Test call (3 known voices, 9 turns) played live through BlackHole, current model: 9/9 speaker turns, latency max 1.6 s |
