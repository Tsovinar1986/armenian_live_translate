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
  Synthetic (`eval/make_noise.py`: room tone, fan, typing, thumps, breaths, 60 s each): 0 lines in 5 min, the
  VAD never fired (2026-10-24).
  Real idle room, MINISO mic, 13.5 min: 1 line, ASR «Ժ։» -> EN "What are you doing?". Fixed: results with fewer than
  2 letters are dropped like empty ones (automated: `test_pipeline.py`) -> 0 lines expected.
- [x] Numbers, names (Երևան, Գյումրի) and questions (՞) come out readable (except years).
  FLEURS (`eval/run_numbers.py`, 30 sentences with digits): ASR always writes numbers as words (readable, mostly
  right); English has the exact number in 12/30, years go wrong most (1889 -> 1899, 1940 -> "a hundred and forty").
  Only 2 FLEURS questions <= 15 s: ՞ lost in both. Names and questions: own recordings needed (s01 s05 s08 s09 s12 s15).
  Own recordings, first take (author, MINISO mic, 16 sentences, 2026-10-25):
  names: Երևան and Գյումրի right in every clip, in HY and EN (6/6 sentences);
  questions: ՞ kept 2 of 4, but the English is a correct question in 4/4;
  numbers: 25, 5000, March 15 at 3:00, 120 km right; both years wrong (1995: ASR «ին» for «ինը»;
  2026: ASR right, NLLB "the year 2600"). The take cut the last word in 6 clips (Enter pressed while speaking)
  and shifted s01/s05/s08; record.py now keeps 0.7 s after Enter.
  After re-taking s01 s05 s08 s24: CER 5.2 % / WER 12.4 % on all 16 (first take 27.9 % / 39.3 %); the 4 re-takes
  are exact. ՞ kept 4 of 7 questions, English a correct question 7/7. Still wrong: both years; s23, s26, s31 lost
  their last word in the first take (not re-recorded). 3 re-takes start with a dash «֊» (not seen in 300+ live lines).
- [x] CER/WER metric is correct on known examples. (automated: `test_metrics.py`)
- [x] Spontaneous fast speech with the current model: YouTube debate, 38 lines. Colloquial forms right, loanwords
  misheard, English gist in about half the lines. NLLB loops fixed with `no_repeat_ngram_size=4` (2026-10-24).

## R3 — English translation
- [x] Armenian quotes `«»` (unknown to NLLB): replacing them with `"` gives no gain (chrF 60.8 = 60.8 on 33 FLEURS sentences).
- [ ] `eval/run_translate.py` on the sentence list: every line understandable (judged by an Armenian speaker).
- [x] Empty ASR result → no translation call, nothing printed. (automated: `test_pipeline.py`)

## R4 — Speaker identification
- [x] Mic segments always labelled `Me`. (automated; live: 6/6 phrases `Me`, 0.8–1.5 s, 2026-10-23)
- [x] Two different call-side voices get different labels. (automated with synthetic embeddings; test call: 3 of 3
  voices kept apart, 9/9 turns, also live through BlackHole and through Opus 16 kbps)
- [x] Same voice keeps its label for the whole session (10+ minutes). (automated; `test_call_long`: 10.5 min,
  36 turns, 36/36 right, 2026-10-24; live only 2.5 min so far)
- [x] Short segments do not create new speakers. (automated)
- [ ] `eval/run_speakers.py` similarity distributions: same-speaker vs different-speaker separate cleanly?
  FLEURS pairs (`eval/fleurs_speakers.py`): yes, valley at 0.4–0.5 for full clips and 3 s crops; threshold 0.45.
- [x] Two call participants of the same gender / similar voice: are they kept apart? (known hard case)
  FLEURS two women: yes (36/36). YouTube debate: at least two men, 37 of 38 lines `Speaker 1` -> likely merged.
  Live check with two same-gender people still needed.
  In progress (2026-10-25): YouTube podcast, two men (Rearrange #310, host Narek Amirkhanyan, guest Hayk Margaryan,
  video 05:00-15:00, `--file`). First 6.5 min at threshold 0.45: 42 lines, 39 `Speaker 1`, 3 `Speaker 2` = an ad
  read by another voice at 07:11-07:39; after the ad the guest got `Speaker 1` back. Latency 0.9-3.7 s.
  Full 10 min (2026-10-25): host questions 3/3 `Speaker 2` (13:18, 13:32, 14:57), guest `Speaker 1` throughout,
  1 false new speaker (one-word interjection). Thresholds 0.55 / 0.65 only add false speakers (3 / 7): 0.45 stays.
  Roles read from the text, not yet confirmed by ear.
- [x] Video/call with 10+ speaker switches: every returning speaker gets its old label back.
  `test_call_long` (FLEURS, 3 voices, 35 switches): 36/36 turns, 3 speakers found, 0 `Speaker ?` (2026-10-24).
  Test call: 8 switches, all returning speakers got their old label back.
  Live test 1: 2 switches, correct, too few to count.

## R5 — Live output
- [x] Each line shows time, speaker, HY, EN. (automated: `test_output.py`)
- [x] Lines appear while the conversation is running, not at the end. (live test call)

## C1 — Latency ≤ 5 s
- [x] `--stats` summary at exit: p50/p90/max latency per source under 5 s.
  Call side, NVIDIA model, live through BlackHole: p50 1.30 / p90 1.55 / max 1.59 s.
  Call side, live test 1: p50 4.16 / p90 4.51 / max 4.69 s, 0 over 5 s (margin only ~0.3 s). Mic side: not yet.
- [x] Long monologue (> 15 s) is force-cut and each part arrives in time. (automated + test call)
- [x] Forced cut goes into a gap between words when there is one in the last 1 s (automated: `test_segmenter.py`;
  test call at 8 s limit: CER 5.7 % -> 5.1 %).
- [ ] Both sides talking at once: queue wait visible, latency still acceptable?
  Test: `python -m livetr --file eval/data/test_call.wav --mic-file eval/data/test_call_overlap_mic.wav` (mic = Opus copy
  of the call, 3 s later, so both sides overlap almost all the time).
  NLLB 1.3B: 4 of 23 over 5 s (max 6.6 s, queue wait up to 5.4 s). NLLB 600M: 0 over 5 s, max 3.5 s, wait max 1.9 s.
  -> default switched to 600M (2026-10-24). Wait is shown on every line (`wait`).
- [x] First segment after start is not slow (warm-up done at startup): first phrase 2.6 s; ready 15 s after the
  command on a quiet Mac (82-166 s while Ollama was swapping the Mac) (2026-10-24).
- [ ] Memory: Activity Monitor shows no swap during a 10-minute call.
  10.5 min replay, no call app: NLLB 1.3B peak 2.7 GB, swap grew ~1.3 GB over the tests; NLLB 600M peak 2.1 GB,
  swap grew ~0.2 GB (macOS also swaps out idle apps, so 'zero swap' is not reachable with VS Code + browser open).
  Still to check: real Zoom/Meet call open at the same time.

## C2 — Local only
- [x] No network needed: runs inside a sandbox that blocks all outgoing connections (2026-10-23).

## Interview rehearsal
- [ ] Real call (Zoom/Meet) with 2 people on the call side, 5 minutes, Armenian.
  Solo rehearsal instead (2026-10-25): author on the mic (wired headset) + YouTube interview Rearrange #310 (two men)
  through Multi-Output -> BlackHole, 1.5 min, 23 lines (12 `Me`, 11 call side). Latency mic p50 1.5 / max 2.6 s,
  call side p50 1.9 / max 3.9 s, 0 over 5 s. All `Me` lines are the author's own words (no call audio leaking into
  the mic). ASR repeated «միշտ» 12 times once; the English stayed clean (4-gram block).
- [x] Start-up time from command to "listening" noted: 15 s (2026-10-24).
- [x] Fresh clone from GitHub, README steps 1.1-1.2 exactly as written (new venv, pip install, model download):
  39 tests pass, test call p50 2.2 s / max 2.9 s; first start 30 s with cold caches (2026-10-25).
- [ ] Chat transcripts exported to `chats/`, one .md per session.
