# Live Armenian → English transcription

Listens to a call, transcribes spoken Armenian, translates it to English and prints each phrase in the
terminal with time and speaker. Your microphone is labelled `Me`; people on the call side are told apart as
`Speaker 1`, `Speaker 2`, … All models run locally.

```
[00:01:12] Speaker 1  (2.1s · wait 0.0 · asr 1.3 · spk 0.1 · mt 0.2)
HY: Բարև, ինչպե՞ս ես։
EN: Hi, how are you?

[00:01:15] Me  (1.8s · wait 0.0 · asr 1.2 · mt 0.2)
HY: Լավ եմ, շնորհակալություն։
EN: I'm fine, thank you.
```

The grey part of each header is the measured latency from the end of the phrase to the printed line,
followed by the time per stage (queue wait, ASR, speaker embedding, translation). It turns red above 5 s.

> **Status:** code and unit tests are complete. Model choice and thresholds are marked *provisional* until
> they have been measured on the target Mac (see [Measuring](#measuring) and `DECISIONS.md`).

---

## 1. Setup (macOS, Apple Silicon)

### 1.1 Python environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
`sounddevice` ships its own PortAudio on macOS; no Homebrew packages are needed for Python.

### 1.2 Download the models (once, needs internet)
```bash
python scripts/download_models.py                 # default ASR + translation + speaker + VAD
python scripts/download_models.py --asr all       # also the other ASR candidates, for benchmarking
```
After this the program runs **offline**: it sets `HF_HUB_OFFLINE=1` at start, so a missing model is an error,
never a silent download. (`--allow-download` turns that off.)

### 1.3 Capture system audio with BlackHole
macOS cannot record its own output, so the call audio is routed through a virtual loopback device.

1. Install BlackHole 2ch: `brew install blackhole-2ch` (or the installer from existential.audio). Reboot if it
   does not appear.
2. Open **Audio MIDI Setup** → `+` (bottom left) → **Create Multi-Output Device**. Tick your headphones/speakers
   **and** `BlackHole 2ch`. Put your headphones first (primary/clock device) and enable **Drift Correction** on
   BlackHole.
3. Select the **Multi-Output Device** as the output: either in **System Settings → Sound → Output**, or only in
   the call app's speaker setting. You still hear the call; BlackHole gets a copy.
   Note: the volume keys don't control a Multi-Output Device. Set the volume on the headphones first.
4. Keep your real microphone as the call app's input.
5. Use **headphones**. Otherwise the call is played out loud, the mic picks it up, and it is transcribed twice
   (echo handling is out of scope).

Check the routing before starting:
```bash
python -m livetr --list-devices          # find the indices / names
python -m livetr --meter                 # talk -> 'mic' bar moves; play a video -> 'system' bar moves
```

## 2. Run
```bash
python -m livetr                                   # default mic + "BlackHole" device
python -m livetr --mic "MacBook" --system 3        # pick devices by name part or index
python -m livetr --file call.wav                   # replay a recording in real time (no devices needed)
python -m livetr --no-mic                          # only the call side
```
Ctrl-C stops and prints session statistics (latency p50/p90/max per source, time per stage, dropped segments).

Useful knobs: `--min-silence-ms` (pause that ends a phrase, default 500), `--max-segment-s` (forced cut,
default 8), `--speaker-threshold` (default 0.45), `--vad-threshold` (default 0.5). Run `--help` for all of them.

## 3. Architecture

```
 mic ──► DeviceSource ─┐                                      ┌─► ASR (NVIDIA FastConformer hy, CPU)
                       ├─► Converter ─► Segmenter (Silero VAD) ├─► Speaker: mic → "Me" / system → ECAPA + online clustering
 BlackHole ► DeviceSource┘   48k→16k mono    per source   queue ─┤    (one worker thread)
                                                              └─► Translation (NLLB-600M, CTranslate2 int8, CPU) ─► terminal
```

| Module | Role | In → out |
|---|---|---|
| `livetr/audio/devices.py` | find devices by index or name | name/index → device index (error if 0 or 2+ matches) |
| `livetr/audio/capture.py` | live device or WAV file; down-mix + streaming resample | device → 16 kHz mono float32 blocks |
| `livetr/segmenter.py` | VAD state machine | blocks → `Segment` (audio + end-of-phrase time) |
| `livetr/asr.py` | candidate ASR models behind one interface | audio → Armenian text |
| `livetr/speakers.py` | `Me` / embeddings / online clustering | segment → label |
| `livetr/translate.py` | NLLB via CTranslate2 | Armenian → English |
| `livetr/output.py` | formatting, colours, latency stats | result → terminal |
| `livetr/pipeline.py` | threads and queues | wires everything |

Design points:
- **Utterance-level, not word streaming.** The VAD cuts at a 500 ms pause; each phrase is transcribed whole.
  ASR is more accurate on full phrases than on fragments, and a phrase still fits in 5 s: VAD 0.5 s, then ASR, speaker
  and translation. Phrases longer than 8 s are force-cut so ASR time stays bounded.
- **One worker for both sources.** Two copies of the models do not fit in 8 GB RAM. If both sides talk at the
  same time, one waits; the wait is shown in each line (`wait`).
- **Order inside the worker:** ASR first. If nothing is recognised, nothing is printed and the segment does
  not influence the speaker clusters. Then speaker, then translation.
- **Fail loudly.** A wrong device name, a missing model or an error in any thread stops the program with a clear
  message. No silent fallbacks, retries or timeouts that could hide where a problem is.
- Models are **warmed up** at start so the first real phrase is not slow.

## 4. Models and why

Target machine: MacBook Apple Silicon with **8 GB RAM**. Memory is the main constraint, then the 5 s budget.

### ASR (Armenian speech → text): NVIDIA FastConformer Armenian, chosen by measurement
Model: [`nvidia/stt_hy_fastconformer_hybrid_large_pc`](https://huggingface.co/nvidia/stt_hy_fastconformer_hybrid_large_pc)
(~115M parameters, NeMo, CC-BY-4.0). Trained by NVIDIA on 296 h of Armenian (Common Voice, FLEURS, audiobooks
and 214 h of NVIDIA's own data). It writes punctuation and capitals, which helps translation.

Five candidates measured on the same 40 Google FLEURS Armenian test clips (`eval/get_fleurs.py`), M1 8 GB:

| model | type | CER | WER | ASR time mean / p90 |
|---|---|---|---|---|
| **`nvidia/stt_hy_fastconformer_hybrid_large_pc`** (CPU) | FastConformer RNNT, Armenian | **3.3 %** | **10.0 %** | **0.25 / 0.27 s** |
| `BuzzASR/armenian` (MPS, fp16) | Whisper large-v3, Armenian fine-tune | 4.6 % | 11.6 % | 2.76 / 2.98 s |
| `facebook/mms-1b-all` + `hye` adapter (MPS) | CTC, 1,100 languages | 5.2 % | 21.9 % | 0.61 / 0.74 s |
| `Chillarmo/whisper-large-v3-turbo-armenian` (MPS) | Whisper turbo, Armenian fine-tune | 12.6 % | 45.6 % | 2.44 / 2.73 s |
| `openai/whisper-large-v3-turbo` (MPS) | Whisper turbo, no fine-tune | 21.5 % | 68.4 % | 2.37 / 2.68 s |

The NVIDIA model is the most accurate **and** about 11x faster than the next best, so it leaves most of the 5 s
budget free. It runs on the CPU: on this small model the CPU was as fast as the GPU (0.25 vs 0.28 s) without the
GPU's slow first call. The other four were removed from the code after the comparison; the code that measured
them is in git history (commit `fcc48a9`), the per-clip outputs in `eval/results/`.

Honest caveats:
- FLEURS is read speech, and both the NVIDIA model and BuzzASR list FLEURS in their training data (the test
  split we use is held out). Spontaneous speech in calls is harder; see the live tests in DECISIONS.md.
- Rejected without benchmarking: small/tiny Armenian Whisper fine-tunes (published WER 40–78 %); cloud ASR
  (not allowed).

### Translation: NLLB-200 distilled 600M, CTranslate2 int8, on CPU
Supports `hye_Armn → eng_Latn`, about 0.6 GB in int8, typically well under a second per sentence on CPU,
and it leaves the GPU to ASR. The 1.3B version is better but needs roughly twice the memory. It is the first
upgrade to try if translations are poor and memory allows. LLM-based translators were not considered:
too large for 8 GB next to the other models. Licence of NLLB weights: CC-BY-NC 4.0 (fine for this challenge, not for commercial use).

### Speakers: ECAPA-TDNN embeddings + online clustering
`speechbrain/spkrec-ecapa-voxceleb` (192-d) on CPU, ~0.1 s per segment. Each call-side segment is compared
by cosine similarity to the running mean of each known speaker. Below the threshold, it starts a new speaker.
Labels are never renamed, so a person keeps the same label for the session.
Segments shorter than 1 s give unreliable embeddings. They never create a speaker, and get a label only if they
are very close to a known one; otherwise they print as `Speaker ?`. An honest "unknown" is better than a
confident wrong label.
Considered: diart (pyannote segmentation + incremental clustering). It handles overlapping speech better but needs
gated pyannote models, more memory and its own streaming framework. In a call, people mostly talk one at a time,
and the VAD already gives us utterances.

### VAD: Silero VAD
Small, CPU-only, robust to background noise, one instance per source (it keeps state).

## 5. Measuring

Model and threshold choices come from numbers, not impressions. All of this runs on the Mac, and audio stays
local (`eval/data/` is git-ignored).

```bash
# 1. record the eval set: 24 sentences from eval/sentences.tsv (+ free speech for speaker tests)
python eval/record.py --speaker anna
python eval/record.py --speaker davit
python eval/record.py --speaker davit --free 6

# 2. ASR: CER/WER, latency and peak memory per model (one process per model)
python eval/run_asr.py --model nemo-hy
python eval/run_asr.py --summary

# 2b. the same on Google FLEURS Armenian (read speech, many readers, reference text; no recording needed)
python eval/get_fleurs.py                          # 40 test-split clips <= 10 s -> eval/data/fleurs/
python eval/run_asr.py --set fleurs --model nemo-hy

# 3. translation of the reference sentences, and of real ASR output (error propagation)
python eval/run_translate.py
python eval/run_translate.py --from-asr nemo-hy

# 4. speakers: same- vs different-speaker similarity, best threshold, simulated live clustering
python eval/run_speakers.py

# 5. end-to-end latency on a recording, without a call
python -m livetr --file some_call.wav
```

Unit tests (no models or audio devices needed): `pytest`. What is tested and what is checked by hand
is listed in `TESTS.md`.

## 6. Known limitations
- **Accuracy depends on the ASR fine-tune's training data** (read speech: Common Voice, FLEURS). Fast
  conversational speech, dialects, Western Armenian and code-switching with Russian or English will be worse.
- **Call audio is compressed** by the call app (Zoom/Meet codecs, noise suppression). This hurts both ASR and
  speaker embeddings compared with the clean clips in the eval set.
- **Speaker separation is per utterance.** If two people speak in one segment (interruptions, overlap), the
  segment gets one label. Similar voices (same gender, similar age) can be merged into one speaker. Very short
  replies show as `Speaker ?`.
- **One worker.** When both sides talk at the same time, one waits. That wait counts towards the 5 s.
- **Phrase-level output.** Nothing is printed until the speaker pauses (or 8 s pass). Long sentences that are
  force-cut are translated in pieces, so the translation can be worse at the cut.
- **ASR can still produce words from noise** that the VAD lets through (music, typing). There is no text filter,
  on purpose: such errors stay visible.
- **8 GB RAM.** Large models plus a call app and a browser can cause swapping, which shows up as latency spikes.
  Check Activity Monitor → Memory → Swap Used.
- No echo handling: use headphones.

## 7. What I would improve with more time
- Try the CTC decoder of the same model and an Armenian language model for decoding (card: RNNT is more
  accurate; an external LM may fix spelling of names).
- Fine-tune on conversational Armenian audio recorded through a real call codec.
- Stream partial results (show Armenian text while the person is still speaking, finalise at the pause).
- Process both sources in parallel if memory allows, or prioritise the source that has waited longer.
- Better diarization on the call side: overlap-aware segmentation (pyannote/diart) inside long segments, and
  re-clustering of the first minutes once more data is available.
- Translate with context (previous sentence) for pronouns and cut sentences.
- A small labelled set of real call recordings for regression testing of every change.

## 8. Repository layout
```
livetr/            program (python -m livetr)
eval/              recording + benchmark scripts, sentence list, results (no audio)
scripts/           model download
tests/             pytest unit tests
PLAN.md            plan written before coding
DECISIONS.md       decision log with reasons
TESTS.md           requirement-driven test cases and QA checklist
CLAUDE.md          project rules for AI-assisted work
chats/             exported AI chat transcripts (one .md per session)
```
