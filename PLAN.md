# PLAN — Live Armenian → English transcription

Target machine: **MacBook, Apple Silicon, 8 GB RAM, macOS.** Everything runs locally.

## Goal
Capture mic + system audio at the same time, transcribe Armenian, translate to English, label speakers
(`Me` for mic, `Speaker N` for the call side, stable for the session), and print each segment within
**5 s of the end of the phrase**.

## Hard constraints that drive the design
- **8 GB RAM** shared by OS, call app, and all models. Budget roughly ≤ 4.5 GB for models in total.
- **≤ 5 s latency** from end of phrase to printed line. Budget: VAD end-of-speech detection ~0.5 s, ASR ≤ 2.5 s,
  translation ≤ 0.5 s, speaker embedding ≤ 0.2 s, headroom ~1 s.
- **Armenian is low-resource.** Zero-shot Whisper large-v3 is poor on Armenian; fine-tuned checkpoints are much better.
  Model choice must be decided by *measurement on this Mac*, not by model cards.

## Modules (inputs → outputs)

| Module | File | In | Out |
|---|---|---|---|
| Device listing / selection | `audio/devices.py` | CLI flags | device indices for mic and loopback |
| Capture | `audio/capture.py` | two input devices | per-source stream of 16 kHz mono float32 chunks, tagged `mic` / `system` |
| Segmenter (VAD) | `segmenter.py` | chunk stream for one source | `Segment(source, audio, t_start, t_end)` when speech ends (silence ≥ ~500 ms) or hits max length (~8 s) |
| ASR | `asr/` (interface + backends) | `Segment.audio` | Armenian text |
| Translation | `translate.py` | Armenian text | English text |
| Speakers | `speakers.py` | `Segment` | label: `Me` for mic; for system: embedding → online clustering → `Speaker N` |
| Output | `output.py` | time, label, HY, EN, measured latency | terminal lines |
| Pipeline | `main.py` | everything above | wires threads + queues together |
| Evaluation | `eval/` | WAV files + reference text | CER/WER per model, latency stats |

Threading: one capture callback per device → one VAD per source → **one shared processing worker** (models loaded once;
8 GB can't hold two copies). Segments processed in order of `t_end`.

## Model candidates (to benchmark, not yet decided)

**ASR**
1. `BuzzASR/armenian` — Whisper large-v3 full fine-tune, card reports FLEURS WER 11.9 %, CV WER 2.4 %. ~3 GB in fp16.
   Risk: memory and latency on 8 GB; custom tokenizer means it must run via `transformers` (not mlx-whisper).
2. `Chillarmo/whisper-large-v3-turbo-armenian` — turbo fine-tune (4 decoder layers → much faster decoding), ~1.6 GB fp16.
   No reliable published WER found yet → we measure it.
3. `facebook/mms-1b-all` + `hye` adapter — CTC, fast, no hallucination loops; likely less accurate. Baseline.
4. `openai/whisper-large-v3-turbo` zero-shot — reference point only, to show why fine-tunes are needed.

**Translation**: `facebook/nllb-200-distilled-600M` converted to CTranslate2 int8 (~0.6 GB, fast on CPU),
`hye_Armn → eng_Latn`. Alternative if quality is poor: NLLB 1.3B distilled int8 (~1.4 GB) — measure memory/latency first.
Note: NLLB license is CC-BY-NC (fine for this challenge, state it in README).

**Speaker embeddings**: `speechbrain/spkrec-ecapa-voxceleb` (192-d) or a WeSpeaker ONNX model.
Clustering: online centroid clustering with a cosine-similarity threshold, tuned on recordings.

**VAD**: Silero VAD (small, CPU, reliable).

## Risks
- **Memory pressure** on 8 GB → swapping → latency blows up. Mitigation: measure RSS, prefer turbo if large-v3 doesn't fit.
- **Whisper hallucinations** on silence / noise (repeated phrases, invented text). Mitigation: only send VAD speech;
  `no_repeat_ngram_size`; drop segments < 0.4 s.
- **Long utterances** → latency > 5 s. Mitigation: force-cut at ~8 s.
- **Both sides talking at once** → queue builds up. Mitigation: measure; print queue delay; process oldest first.
- **Short call-side segments** give poor speaker embeddings. Open question below.
- **BlackHole routing**: user must still hear the call → Multi-Output Device (speakers + BlackHole).
- Call apps apply their own noise suppression / compression → audio differs from clean recordings.

## How we test
- `TESTS.md`: requirement-driven test cases and QA checklist, written as we go.
- Unit tests (pytest) for pure logic: segmenter on synthetic audio, speaker clustering on synthetic embeddings,
  output formatting, latency accounting.
- **Evaluation set** (`eval/data/`, not committed): ~10–20 short Armenian clips recorded by you, plus 2–3 voices
  for speaker tests, each with the correct transcript. `eval/run_asr.py` reports CER/WER and latency per model.
- End-to-end rehearsal: a real call with 2 people on the call side before the interview.

## Build order (walking skeleton first)
0. Env setup + BlackHole + `--list-devices`. Repo skeleton, PLAN/DECISIONS/TESTS files.
1. **Skeleton**: WAV file or mic → VAD → ASR (cheapest candidate) → NLLB → terminal, with latency printed.
2. **Measure**: record eval set, benchmark ASR candidates on your Mac → pick model (log in DECISIONS.md).
3. Second source: mic + BlackHole together, `Me` label.
4. Speaker clustering on system side, tune threshold on recordings.
5. Hardening (weakest link first), README, limitations.

## Open questions for you
1. Can you record the eval set (I'll give you a script + a list of sentences), ideally with 2–3 other people?
2. Short call-side segments (< ~1 s) where the voice embedding is unreliable: label them `Speaker ?` (honest)
   or assign to the closest existing speaker (looks cleaner but can be wrong)?
3. Python version on your Mac (`python3 --version`)? I'd target 3.11.
