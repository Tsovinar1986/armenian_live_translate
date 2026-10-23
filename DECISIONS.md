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

## 2026-10-23 — ASR test set from Google FLEURS (hy_am, test split)
Why: the brief sets no data amount; the tips ask for "a few short recordings with the correct text". FLEURS gives
read Armenian with reference text from many readers, without depending on other people to record.
- 40 clips, one recording per sentence, <= 10 s (live phrases are cut at 8 s), seed 0: `eval/get_fleurs.py`.
- Test split, because the fine-tunes may have trained on FLEURS train.
- Own manifest (`eval/data/fleurs_manifest.tsv`): FLEURS has no speaker ids, so these clips must not reach
  `run_speakers.py`.
- Limits: read speech, clean studio-ish audio, no call codec. It measures ASR only, not speakers or live latency.
First result, turbo-hy fp16 on MPS: CER 13.1 %, WER 45.4 %, 2.45 s mean ASR time. 8 of 40 outputs contain `�`:
the model emits incomplete UTF-8 byte tokens for Armenian letters (decoding is whole-sequence, so not our bug).

## 2026-10-23 — ASR length limit: max_length=228 (4 prompt + 224 text tokens)
Armenian costs ~15 Whisper tokens per second of speech (up to 21 on FLEURS). The old cap of 160 new tokens cut
4 of 40 FLEURS clips mid-letter (ending in `�`); live phrases are often a full 8 s, so fast speakers were cut too.
224 text tokens cover 8 s at 28 tokens/s; the repetition guard (`no_repeat_ngram_size=3`) still stops loops,
so the higher cap only costs time when the speech really is that long. Passed as `max_length` (not
`max_new_tokens`), because the model's saved config also sets `max_length` and both together printed a
transformers warning for every phrase. turbo-hy on FLEURS: CER 13.1 % -> 12.6 %, latency unchanged.

## 2026-10-23 — NLLB tokenizer warning at start-up is a false alarm (kept visible, not silenced)
transformers warns "incorrect regex pattern … will lead to incorrect tokenization" (a check written for Mistral).
Checked: our tokenizer and the original `sentencepiece.bpe.model` give identical tokens on 65 of 66 Armenian
sentences (24 own + 40 FLEURS + 2 edge cases). The one difference is `«`, which is `<unk>` in the original
model as well, so behaviour is the same. Side finding: `«` and `»` are unknown to NLLB, and the ASR often
outputs them. Replacing them with `"` before translation might help; measure before changing.

## 2026-10-23 — Live test 1: YouTube conversation through BlackHole (`--no-mic`), turbo-hy
Setup: video played to Multi-Output Device, `python -m livetr --no-mic --no-color | tee -i …` (log local only).
- 30 call-side phrases, ~3.5 min. Latency p50 4.16 s, p90 4.51 s, max 4.69 s, 0 over 5 s.
  Stages: ASR mean 2.70 s (max 3.43), translation 0.94 s (max 1.42), speaker 0.15 s, queue wait 0.21 s.
- Almost every phrase was force-cut at 8 s: the speakers never paused 0.5 s. Long segments make ASR slower
  and split sentences, which hurts translation.
- Speakers: 2 people, only 2 switches (S1 -> S2 -> S1); S1 got its label back correctly. Too few switches to
  judge separation; needs a second video with more back-and-forth.
- Text: spontaneous speech gives many ASR errors (`մջջաավյր` for `միջավայր`), and translation turns them into
  nonsense in places. Consistent with FLEURS WER ~45 %.
Conclusion: the 5 s limit holds, but with only ~0.3 s margin at worst. ASR time is the lever (faster runtime
such as MLX 8-bit, or a shorter input window), to be decided after the model comparison.

## 2026-10-23 — ASR comparison on FLEURS (40 test clips, M1 8 GB, fp16 on MPS)
| model | CER | WER | ASR time mean / p90 | outputs with `�` |
|---|---|---|---|---|
| buzz (`BuzzASR/armenian`, large-v3 fine-tune) | 4.6 % | 11.6 % | 2.76 / 2.98 s | 0 |
| mms (`facebook/mms-1b-all`, hye adapter, CTC) | 5.2 % | 21.9 % | 0.61 / 0.74 s | 0 |
| turbo-hy (`Chillarmo/whisper-large-v3-turbo-armenian`) | 12.6 % | 45.6 % | 2.44 / 2.73 s | 8 |
| turbo-zeroshot (`openai/whisper-large-v3-turbo`) | 21.5 % | 68.4 % | 2.37 / 2.68 s | 3 |
Reading:
- The provisional default turbo-hy is clearly beaten; it is no longer a candidate for the default.
- buzz is the most accurate, but ASR time is like turbo-hy, so live latency would again be ~4.5 s at p90.
  Its card reports FLEURS WER 11.9 %, which we reproduce; it may have trained on FLEURS train (same domain as
  this test), so its lead on spontaneous speech still has to be shown live.
- mms has almost the same CER, is 4–5x faster (5 s latency margin becomes large), but its WER is twice
  buzz's and it outputs no punctuation or capitals, which may hurt translation. Needs a translation check.
- Peak RSS is not reported here: on Apple Silicon, model weights on MPS live in Metal memory, which
  `ru_maxrss` does not count (buzz showed 0.44 GB for a ~3 GB model). Memory must be read from
  Activity Monitor or `torch.mps.driver_allocated_memory()` instead.
Next: replay the same recorded conversation with buzz and with mms (`--file`) to compare end-to-end latency and
translation, then pick the default.
