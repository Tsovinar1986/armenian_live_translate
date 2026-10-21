# Decision log

Format: date — decision — why — status. "Provisional" means it still has to be confirmed by measurement on the Mac.

## 2026-10-21 — Target hardware: Apple Silicon, 8 GB RAM
The interview runs on this laptop. 8 GB is the main constraint: all models plus the call app must fit without
swapping, otherwise latency explodes. Budget ≈ 4.5 GB for models.

## 2026-10-21 — Pipeline shape: VAD-segmented utterances, one shared processing worker
- Each source (mic, system) gets its own capture stream and its own VAD (Silero VAD keeps internal state per stream).
- Segments from both sources go into one queue processed by **one** worker that holds the models.
- Why: two copies of the ASR model do not fit in 8 GB. Utterance-level processing (not word streaming) is simpler,
  gives Whisper full sentences (better accuracy), and still fits the 5 s budget if utterances are capped at ~8 s.
- Trade-off: if both sides talk at the same time, one waits. Queue wait is printed so we can see it.

## 2026-10-21 — Segment end = 500 ms of silence, force-cut at 8 s, drop < 0.4 s of speech
- 500 ms is a common pause length between phrases; shorter cuts mid-sentence, longer eats latency budget.
- Force-cut at 8 s keeps ASR time bounded.
- < 0.4 s segments are mostly coughs/clicks and are where Whisper hallucinates most.
- Status: provisional, tune on recordings (`--min-silence-ms`, `--max-segment-s`).

## 2026-10-21 — `Me` = everything from the microphone; no speaker model on the mic side
The brief defines it this way. Echo (call audio leaking into the mic) is out of scope; use headphones.

## 2026-10-21 — Call-side speakers: ECAPA embeddings + online centroid clustering
- `speechbrain/spkrec-ecapa-voxceleb` (192-d), cosine similarity to running centroids.
- New speaker if best similarity < threshold. Threshold default 0.45, **provisional** → tune with
  `eval/run_speakers.py` on real recordings.
- Considered diart (pyannote segmentation + clustering): more accurate with overlap, but needs pyannote gated
  models, more RAM and its own streaming framework. Our segments are already VAD utterances, and overlap is rare in
  a call (one person talks at a time through the call app), so per-utterance embeddings are enough.

## 2026-10-21 — Short call-side segments are labelled `Speaker ?` unless very close to a known speaker
Embeddings from < 1 s of audio are unreliable. Such segments never create a new speaker and never update a
centroid. They get a known label only if similarity ≥ `strong_threshold` (0.65), otherwise `Speaker ?`.
Why: an honest "unknown" is better than a confident wrong label (brief: honesty about limits beats hiding them).
Also: if the speaker limit (8) is reached, an unmatched segment is `Speaker ?` rather than forced into a cluster.

## 2026-10-21 — ASR candidates (decision pending measurement)
Benchmark on our own recordings with `eval/run_asr.py`:
1. `BuzzASR/armenian` (Whisper large-v3 fine-tune; card: FLEURS WER 11.9 %, CV WER 2.4 %). ~3 GB fp16.
2. `Chillarmo/whisper-large-v3-turbo-armenian` (turbo fine-tune, ~1.6 GB fp16, faster decoder). No trusted WER.
3. `facebook/mms-1b-all` with `hye` adapter (CTC baseline).
4. `openai/whisper-large-v3-turbo` zero-shot (reference: shows why a fine-tune is needed).
Runtime: Hugging Face transformers on MPS, fp16. Not mlx-whisper, because BuzzASR replaced the tokenizer.
**Provisional default: `turbo-hy`** (fits memory comfortably). Replace with the measured winner.

## 2026-10-21 — Translation: NLLB-200 distilled 600M, CTranslate2 int8, CPU
- Supports `hye_Armn → eng_Latn`, small, fast on CPU, leaves MPS free for ASR.
- Alternative: NLLB 1.3B distilled if quality is poor and memory allows (measure first).
- License CC-BY-NC 4.0: fine for this challenge, noted in README.

## 2026-10-21 — Resampling with soxr streaming resampler
Devices run at 44.1/48 kHz; models need 16 kHz. soxr's `ResampleStream` resamples block by block without edge
artifacts (plain per-block `resample_poly` clicks at block boundaries).

## 2026-10-21 — Python 3.11
Widest wheel support for torch / ctranslate2 / speechbrain on macOS arm64.
