# AI chat transcripts

Every AI session used for this project, one Markdown file per session, in the order they happened.
AI tools used: **Claude only**, on claude.ai (web) and in Claude Code (VS Code). No other AI assistant.

| # | File | When | Tool | What happened |
|---|---|---|---|---|
| 1 | `claude-ai-2026-10-21-plan-and-skeleton.md` ([shared link](https://claude.ai/share/b1eac730-ad91-4f92-8f24-02865c9ad871)) | 21 Oct | claude.ai | Brief read, setup questions answered, PLAN.md, walking skeleton with 33 unit tests (first commit), BlackHole install |
| 2 | `claude-code-2026-10-21-2130-f5b8830e.md` | 21–23 Oct | Claude Code | Main working session: Mac setup, GitHub, mic permission, FLEURS test set, comparison of 5 ASR models and switch to NVIDIA FastConformer, speaker threshold, test call, live tests, 15 s phrase limit, chat export |
| 3 | *(to add)* | 22 Oct | claude.ai project chat | The conversation that started sessions 4 and 5 |
| 4 | `claude-code-2026-10-22-1840-a7d1c4cd.md` | 22 Oct | Claude Code, started from 3 | Sync with GitHub, extend .gitignore |
| 5 | `claude-code-2026-10-22-2115-44398c8d.md` | 22 Oct | Claude Code, started from 3 | Multi-Output Device setup in Audio MIDI Setup |

How each file was made, and what it does or does not contain:
- **claude.ai sessions (1, 3):** the conversation copied from the page and pasted unchanged. The page collapses
  Claude's tool steps into one line ("Created 23 files, updated tasks, ran 10 commands"); the shared link shows
  the original. File 1 holds two copies (chat page and shared page) because each view leaves out something.
- **Claude Code sessions (2, 4, 5):** converted from the local session logs by `scripts/export_chats.py`, with
  all messages, Claude's reasoning, every tool call and every tool result. Only the logger's internal records
  (request metadata, cost counters, file snapshots) are left out. Session 2 is re-exported as the last step
  before submission, so it is complete.

The originals stay available in claude.ai and in Claude Code for the interview.
