# AI chat transcripts

Every AI session used for this project, exported in full and unedited, one Markdown file per session.

| File | When | Where | What |
|---|---|---|---|
| `claude-ai-2026-10-21-plan-and-skeleton.md` ([shared link](https://claude.ai/share/b1eac730-ad91-4f92-8f24-02865c9ad871)) | 21 Oct | claude.ai | Plan, PLAN.md / DECISIONS.md / TESTS.md, walking skeleton (first commit), BlackHole install |
| *(to add)* | 22 Oct | claude.ai project chat | Coordinator conversations that started the two helper sessions below |
| `claude-code-2026-10-21-2130-f5b8830e.md` | 21–23 Oct | Claude Code (VS Code) | Setup on the Mac, GitHub, BlackHole, FLEURS test set, ASR comparison, switch to NVIDIA FastConformer, speaker threshold |
| `claude-code-2026-10-22-1840-a7d1c4cd.md` | 22 Oct | Claude Code, started from the project chat | Sync with GitHub, extend .gitignore |
| `claude-code-2026-10-22-2115-44398c8d.md` | 22 Oct | Claude Code, started from the project chat | Multi-Output Device setup in Audio MIDI Setup |

The Claude Code sessions are converted from the local session logs by `scripts/export_chats.py` (all messages,
thinking, tool calls and tool results; only the logger's internal bookkeeping records are left out). The script
is re-run at the end, so the ongoing session is complete in the submitted version.
