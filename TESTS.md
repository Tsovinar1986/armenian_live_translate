# Tests and QA checklist

Derived from the brief's requirements, not from the code. Tick items by hand before the interview.
Automated tests: `pytest` (pure logic only, no models or audio devices needed).

## R1 — Two audio sources at once
- [x] `python -m livetr --list-devices` shows the mic and `BlackHole 2ch`. (2026-10-23)
- [x] `python -m livetr --meter` shows level bars moving for mic when I talk and for system when a video plays.
  System: OK (−14 dB). Mic: −90 dB at first (no microphone permission); after granting it −52 dB (2026-10-23).
- [x] I still hear the call while BlackHole captures it (Multi-Output Device selected as output).
- [x] Wrong device name → program stops with a clear error listing the input devices. (automated: `test_devices.py`)
- [x] Ambiguous device name (matches 2 devices) → clear error. (automated)

## R2 — Armenian transcription
- [x] CER/WER measured for every candidate on FLEURS (`run_asr.py --set fleurs`); own recordings not made
  (`eval/results/asr_results.csv`).
- [x] Fast speech in a full 8 s phrase is not truncated by the decoder length limit (FLEURS: 0 clips hit the
  cap after raising it to 224 text tokens; 4 of 40 were cut at 160).
- [ ] Silence and background noise do not produce invented text (5 min idle test: count printed lines).
- [ ] Numbers, names (Երևան, Գյումրի) and questions (՞) come out readable.
- [x] CER/WER metric is correct on known examples. (automated: `test_metrics.py`)

## R3 — English translation
- [x] Armenian quotes `«»` (unknown to NLLB): replacing them with `"` gives no gain (chrF 60.8 = 60.8 on 33 FLEURS sentences).
- [ ] `eval/run_translate.py` on the sentence list: every line understandable (judged by an Armenian speaker).
- [x] Empty ASR result → no translation call, nothing printed. (automated: `test_pipeline.py`)

## R4 — Speaker identification
- [x] Mic segments always labelled `Me`. (automated; live: 6/6 phrases `Me`, 0.8–1.5 s, 2026-10-23)
- [x] Two different call-side voices get different labels. (automated with synthetic embeddings; test call: 3 of 3
  voices kept apart, 9/9 turns, also live through BlackHole and through Opus 16 kbps)
- [ ] Same voice keeps its label for the whole session (10+ minutes). (automated: stable centroid test; live only 2.5 min so far)
- [x] Short segments do not create new speakers. (automated)
- [ ] `eval/run_speakers.py` similarity distributions: same-speaker vs different-speaker separate cleanly?
  FLEURS pairs (`eval/fleurs_speakers.py`): yes, valley at 0.4–0.5 for full clips and 3 s crops; threshold 0.45.
- [ ] Two call participants of the same gender / similar voice: are they kept apart? (known hard case)
- [ ] Video/call with 10+ speaker switches: every returning speaker gets its old label back.
  Test call: 8 switches, all returning speakers got their old label back.
  Live test 1: 2 switches, correct, too few to count.

## R5 — Live output
- [x] Each line shows time, speaker, HY, EN. (automated: `test_output.py`)
- [x] Lines appear while the conversation is running, not at the end. (live test call)

## C1 — Latency ≤ 5 s
- [x] `--stats` summary at exit: p50/p90/max latency per source under 5 s.
  Call side, NVIDIA model, live through BlackHole: p50 1.30 / p90 1.55 / max 1.59 s.
  Call side, live test 1: p50 4.16 / p90 4.51 / max 4.69 s, 0 over 5 s (margin only ~0.3 s). Mic side: not yet.
- [x] Long monologue (> 8 s) is force-cut and each part arrives in time. (automated + test call; cuts can split words)
- [ ] Both sides talking at once: queue wait visible, latency still acceptable?
- [ ] First segment after start is not slow (warm-up done at startup).
- [ ] Memory: Activity Monitor shows no swap during a 10-minute call.

## C2 — Local only
- [x] No network needed: runs inside a sandbox that blocks all outgoing connections (2026-10-23).

## Interview rehearsal
- [ ] Real call (Zoom/Meet) with 2 people on the call side, 5 minutes, Armenian.
- [ ] Start-up time from command to "listening" noted.
- [ ] Chat transcripts exported to `chats/`, one .md per session.
