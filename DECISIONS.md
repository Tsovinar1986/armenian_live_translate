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

## 2026-10-23 — Default ASR: buzz (user's choice: "the most accurate")
- Lowest CER and WER on FLEURS (4.6 % / 11.6 %), no broken `�` outputs.
- Breaks our own first criterion (ASR p90 ≤ ~2.5 s): buzz p90 is 2.98 s. Accepted for accuracy, with a check:
  if the live test goes over 5 s, or the Mac swaps during a call, switch the default to `mms`.
- GPU memory (Metal, `torch.mps.driver_allocated_memory()` after an 8 s phrase): buzz 3.47 GB, mms 2.30 GB,
  turbo-hy 2.21 GB. Plus NLLB (~0.6 GB) and ECAPA: ~5 GB for the program on an 8 GB Mac.

## 2026-10-23 — Armenian quotes «» are NOT converted before translation
`«»` are the Armenian quotation marks (= English ""), and NLLB treats them as unknown tokens. Tested on all 33
FLEURS test sentences with «», against the official English FLEURS translations: chrF 60.8 with «», 60.8 with "",
16 sentences better, 15 worse. Quoted phrases are kept more often, but one sentence was dropped entirely.
No measurable gain, so no extra code. (sacrebleu used for the score; eval-only, not in requirements.)

## 2026-10-23 — ASR: NVIDIA FastConformer Armenian (`nemo-hy`); other candidates removed from the code
Supersedes the buzz default above. Same 40 FLEURS clips: CER 3.3 %, WER 10.0 %, 0.25 s mean / 0.27 s p90 on CPU
(MPS 0.28 s, with a slow first call). Better than buzz on accuracy (4.6 % / 11.6 %) and ~11x faster, so ASR no
longer eats the 5 s budget.
- Model card: trained on 296 h (Common Voice 17 48 h, FLEURS 12 h, Armenian audiobooks 22 h, NVIDIA proprietary
  214 h); card reports FLEURS test WER 12.3 %, consistent with our 10.0 % on shorter clips. Licence CC-BY-4.0.
- Caveat: FLEURS is in its training data (like buzz), so FLEURS favours both; live tests decide for real speech.
  The large non-FLEURS share (incl. spontaneous-ish proprietary data) is a reason to expect it to hold up better.
- Runs on CPU by default: as fast as MPS on this model, no MPS warm-up spike, and the GPU stays free.
- Writes punctuation and capitals (`,` `։` `՞`), unlike MMS; translation gets sentence structure.
- User asked to remove the other candidates from the code once the NVIDIA model won. Done: `livetr/asr.py` now
  only holds `nemo-hy`. Evidence for the comparison stays in `eval/results/` and git history (`fcc48a9`).
- Cost: `nemo_toolkit[asr]` is a large dependency (~70 extra packages); accepted for the accuracy/speed gain.
- End-to-end replay (6 FLEURS clips, `--file`): ASR 0.2–0.3 s per phrase when the Mac is quiet, phrases printed
  1.1–1.8 s after they end. Spikes (translation up to 10 s) were traced to the machine, not the code: a fixed numpy
  workload varied 5.8x in the same minutes, while another project was rendering speech (`say`) in the background,
  on battery, with ~4 GB swap. Latency must be measured on a quiet Mac (other jobs stopped, on power).

## 2026-10-23 — Speaker threshold 0.45 confirmed on FLEURS pairs (`eval/fleurs_speakers.py`)
ECAPA cosine similarity for all 832 pairs of FLEURS test recordings of the same sentence (932 clips).
Different-gender pairs (certainly different people, n=401): max 0.30 on full clips, 0.27 on 3 s crops.
Same-gender pairs (n=431) are bimodal: different people ~0.1–0.4, the same reader twice ~0.6–1.0 (full clips)
or ~0.5–0.9 (3 s crops). The valley is 0.4–0.5 in both cases, so 0.45 stays: no different-gender pair reaches it,
and it sits below the same-reader peak even for short phrases.
Limits: read speech, studio-like audio, no call codec. Two similar same-gender voices can reach 0.4–0.5 and may be
merged; that is the known hard case and needs a live check. Pair similarities are in `eval/results/`.

## 2026-10-23 — End-to-end test call with known speakers (`eval/make_test_call.py`, `eval/score_call.py`)
FLEURS Armenian has only **three voices**: one man and two women (every male test clip matches every other male
clip at >= 0.6; women form two groups). So the ASR test set also had only 3 speakers, and a 4-person test call is
not possible from FLEURS. A first version of the test call took "two men" who were the same man reading two
sentences (similarity 0.74); the program merged them, correctly. The builder now accepts a reader only if all
their clips are < 0.35 against the other readers.
Test call: 3 speakers (2 women, 1 man), 9 turns, 9 different sentences, 144 s; every speaker comes back twice.
| run | speaker turns right | speakers found | CER / WER | latency p50 / max |
|---|---|---|---|---|
| `--file`, clean | 9/9 | 3 of 3 (+1 `Speaker ?`) | 5.7 % / 13.0 % | 1.2 / 1.5 s |
| `--file`, through Opus 16 kbps | 9/9 | 3 of 3 (+1 `Speaker ?`) | 5.4 % / 12.2 % | 1.2 / 1.5 s |
| **live**: played through Multi-Output → BlackHole | 9/9 | 3 of 3 (+1 `Speaker ?`) | 5.8 % / 12.6 % | **1.3 / 1.6 s** |
Reading:
- Latency: ~1.3 s, far under 5 s; translation (0.7 s) is now the slowest stage, ASR 0.2 s.
- The call codec barely matters for ASR or speakers.
- CER 5.7 % vs 3.3 % on single clips: turns here are 10–29 s, so the VAD force-cuts at 8 s, sometimes inside a
  word, and both halves are transcribed and translated separately. Possible fix: cut at the quietest moment of
  the last ~1 s instead of exactly at 8 s.
- `Speaker ?` appeared once, on a short tail after a cut (< 1 s): by design (short segments don't get a guess).
- Not covered: two *very* similar voices (FLEURS has none), spontaneous speech, the mic side.

## 2026-10-23 — Local only, verified
The program ran end to end inside a macOS sandbox that blocks all outgoing network connections
(`sandbox-exec ... (deny network-outbound ...)`; a test request to huggingface.co failed inside it).

## 2026-10-23 — Microphone side live
Author speaking colloquial Armenian into the built-in mic (after granting microphone permission; before that
macOS delivered digital silence, −90 dB): 6 phrases, all `Me`, latency 0.8–1.5 s. ASR handled colloquial forms
well; NLLB mistranslated idioms and loanwords (`ինչ կա, չկա` → "there is no such thing", `չատերը` → "chessboard").
Translation of spoken, colloquial Armenian is now the weakest part. Options for later: NLLB 1.3B (memory),
or a small local LLM for translation (memory and latency); not tried within the time budget.

## 2026-10-23 — Phrase limit (forced cut) 8 s -> 15 s
The 8 s limit was set when ASR took 2–3 s. With the NVIDIA model (0.2–0.3 s) it only cost accuracy: cuts fell
inside words (`քարանձավ` -> "…մի" + "Անցավ…"). Test call (clean, 9 turns of 7–29 s), one run each:
| limit | forced cuts | CER / WER | latency p50 / max | speaker turns |
|---|---|---|---|---|
| 8 s | 11 | 5.7 % / 13.0 % | 1.2 / 1.5 s | 9/9 |
| 12 s | 6 | 5.0 % / 11.4 % | 1.9 / 3.0 s | 9/9 |
| 15 s | 3 | 4.1 % / 10.2 % | 2.1 / 2.8 s | 9/9 |
Chosen: 15 s. Trade-off: in a long monologue the first words appear later (up to ~15 s after they were said); the
brief measures latency from the end of a phrase, which stays < 3 s. Not done: cutting at the quietest moment.

## 2026-10-23 — Translation: NLLB 1.3B instead of 600M (600M stays as fallback)
`eval/run_mt_fleurs.py`: 150 FLEURS test sentences, Armenian -> English, chrF against the official English FLEURS
translations; plus the 6 colloquial phrases from the live mic test (`eval/colloquial.tsv`, judged by eye).
| model | chrF | MT time mean / p90 / max | memory (int8) |
|---|---|---|---|
| NLLB 600M | 60.1 | 1.00 / 1.43 / 2.59 s | 0.6 GB |
| NLLB 1.3B | 63.5 | 1.81 / 2.53 / 4.68 s | 1.4 GB |
Colloquial: 1.3B fixes the worst errors (`ինչպես ես ինչ կա, չկա` -> "Hi, how are you?" instead of "Hello, as I am,
there is no such thing"; `չատերը` -> "chat" instead of "chessboard").
Whole program with 1.3B, test call, 15 s phrase limit, on battery: latency p50 3.0 s, p90 3.6 s, max 4.0 s, 0 over
5 s; speakers 9/9; peak memory 3.6 GB (2.8 GB with 600M).
Chosen: 1.3B, because translation was the weakest stage (tip 2: improve the weakest link). Trade-off: the latency
margin shrinks from ~2 s to ~1 s. If the practice call (with a call app using CPU) goes over 5 s, the default goes
back to 600M. Conversion needs `load_as_float16=True, low_cpu_mem_usage=True` to fit in 8 GB RAM.

## 2026-10-24 — Forced cut at the quietest moment of the last second
A phrase longer than the limit used to be cut exactly at the limit, often inside a word. Now the segmenter looks at
the last 1 s (`cut_search_ms=1000`) and cuts after the quietest 32 ms frame (lowest RMS), if it is a clear dip
(below half the window's median RMS); without a clear dip it still cuts at the limit. The rest of the audio starts
the next segment, so nothing is lost or doubled (unit tests). Test call (`--file`, 9 turns), one run each:
| limit | cut | forced cuts | CER / WER | speaker turns |
|---|---|---|---|---|
| 8 s | at the limit (old) | 11 | 5.7 % / 13.0 % | 9/9 |
| 8 s | quietest moment | 11 | 5.1 % / 11.8 % | 9/9 |
| 15 s | at the limit (old) | 3 | 4.1 % / 10.2 % | 9/9 |
| 15 s | quietest moment | 3 | 4.0 % / 9.8 % | 9/9 |
Cost: a cut line now ends up to 1 s before the moment the cut is made, and its latency is measured from that
earlier point (honest: the line holds audio only up to there), so cut lines show up to +1 s in `wait`.
Latency of these four runs is not comparable: the Mac had 7 GB swap in use and translation took 2-4 s instead of
1-2 s (a 4.4 GB Ollama `llama-server` from another project started during the runs). Re-measure on a quiet Mac.

## 2026-10-24 — Test tools for the pre-meeting checks
- `--mic-file`: plays a second WAV as the mic side together with `--file`, to test both sides talking at once
  without a call. Test input only; live use is unchanged.
- `eval/make_noise.py`: 5 min of synthetic non-speech (room tone, fan, typing, thumps, breaths). Result: 0 lines,
  the VAD never fired. A real idle room through the mic is still needed.
- `eval/make_test_call.py --turns N --name X`: longer test call (e.g. 12 turns per speaker, ~10 min, no speaker twice
  in a row) for label stability and memory over a long session. `eval/score_call.py --tsv` scores it.
- `eval/run_numbers.py`: FLEURS sentences with digits and with ՞, checks numbers and question marks in HY and EN.
Timing results need a quiet Mac: during these runs a second local LLM (Ollama, another project, 4+ GB) relaunched
itself, the Mac swapped and one translation took 210 s. Before the meeting: Ollama must be quit.

## 2026-10-24 — Translation back to NLLB 600M (1.3B fails when both sides talk)
The 1.3B decision said: back to 600M if a test goes over 5 s. Stress tests on a quiet Mac (Ollama stopped), `--file`:
| test | NLLB 1.3B | NLLB 600M |
|---|---|---|
| both sides at once, 2.5 min (`--mic-file`), over 5 s | 4 of 23, max 6.6 s | 0 of 23, max 3.5 s |
| queue wait max | 5.4 s | 1.9 s |
| 10.5 min call, 36 turns: latency p50 / max | 3.4 / 4.8 s | 2.2 / 3.5 s |
| translation per line mean / max | 2.1–2.2 / 4.3 s | 1.2 / 2.3 s |
| peak memory (RSS) | 2.7 GB | 2.1 GB |
| swap growth over the runs | ~1.3 GB | ~0.2 GB |
| speaker turns (long call) | 36/36 | 36/36 |
Chosen: 600M as default. Trade-off: chrF 60.1 instead of 63.5 and worse on colloquial phrases (`ինչ կա, չկա`). 1.3B
stays available with `--nllb-dir models/nllb-200-distilled-1.3B-ct2-int8` for calls where one side mostly listens.
Start-up: 15 s from command to listening; first phrase 2.6 s (warm-up works).
Numbers (`eval/run_numbers.py`, FLEURS): ASR writes numbers as Armenian words; English has the exact number in 12/30
sentences, years are the weak spot. Not fixed (would need Armenian number-word parsing; not in the time budget).

## 2026-10-24 — Drop ASR results with fewer than 2 letters
Real idle room through the mic (MINISO headset, 13.5 min, nobody speaking): one line printed. The VAD passed a noise,
the ASR wrote a single letter «Ժ։», and NLLB invented "What are you doing?" from it. A single letter is never a
phrase, so `Processor` now treats fewer than 2 letters (punctuation ignored) like an empty result: no translation,
nothing printed, counted in the stats. Two-letter answers such as «Հա» still pass.

## 2026-10-24 — Translation: block repeated 4-grams (stops NLLB loops)
In the YouTube debate (spontaneous speech, current ASR) NLLB looped twice: "you can, you can, you can, …" and one line
of 130 words "no, no, no, …" until the 256-token cap (that line took 7.9 s to translate). CTranslate2
`no_repeat_ngram_size`, NLLB 600M:
| setting | FLEURS chrF (150) | MT mean / max | debate lines that loop (38) | longest line |
|---|---|---|---|---|
| off | 60.1 | 1.00 / 2.59 s | 2 | 130 words |
| 3 | 60.0 | 1.05 / 2.63 s | 0 | - |
| **4** | **60.2** | 1.01 / 2.51 s | **0** | 30 words |
| 5 | 60.1 | 1.01 / 2.58 s | 1 | 29 words |
Chosen: 4. Side effect: a word cannot appear twice in the same 4-token context ("a suspicious burden, a suspicion
situation"); with 3 it was worse ("suspicuous"). `eval/run_mt_fleurs.py --no-repeat-ngram N` reproduces the table.

## 2026-10-24 — Spontaneous speech and similar voices, from the YouTube debate (`eval/data/live_yt2.txt`)
Parliament hearing, fast spontaneous speech, current ASR (FastConformer), 38 lines (old 8 s cut).
- ASR keeps colloquial forms (`էս`, `ա`, `սենց`, `էլի`, `իրա`); loanwords are misheard (front office ->
  `Ֆրոնտովսում`, `ֆրոնտոֆիս`; `Մաշնային` for `մաքսային`). English gives the gist in about half the lines; the
  rest is partly or fully wrong. ASR latency stays 0.2 s, so speed is not the issue, accuracy is.
- Speakers: 37 lines `Speaker 1`, 1 line `Speaker 2`, although the text shows at least two men (the chair asking
  people to speak in turn, and the official answering). Likely two similar male voices merged at threshold 0.45.
  Not provable without the audio (not saved). FLEURS two women (max similarity 0.29) are kept apart (36/36).
  Similar voices stay a known weak point; a live check with two same-gender people is still needed.

## 2026-10-25 — Similar voices: two men on a podcast, threshold 0.45 confirmed
YouTube podcast "Rearrange #310" (host Narek Amirkhanyan, guest Hayk Margaryan, both men), video 05:00-15:00,
replayed with `--file` (audio downloaded with yt-dlp in a separate scratch environment, not committed). No written
ground truth: roles are read from the text (questions with ՞ from the host, long answers from the guest).
| threshold | lines | guest | host questions (3) | false new speakers | latency max |
|---|---|---|---|---|---|
| **0.45** | 71 | `Speaker 1` throughout | 3/3 `Speaker 2` | 1 (one-word interjection) | 3.7 s |
| 0.55 | 71 | `Speaker 1` | 3/3 `Speaker 2` | 3 | 4.5 s |
| 0.65 | 71 | `Speaker 1` | 3/3 `Speaker 4` | 7, plus 3 `Speaker ?` | 3.6 s |
A stricter threshold does not separate the host better (he is already apart at 0.45); it only splits the guest's
short lines into new speakers. 0.45 stays. A sponsor ad (07:11-07:39) gets the host's label at every threshold,
most likely because the host reads it himself. Together with the parliament debate (two men probably merged),
same-gender voices are kept apart when they differ enough, not always.

## 2026-10-25 — Own recordings: names and questions fine, years are the weak spot
16 sentences read by the author (eval/sentences.tsv s01-s32 subset). Names (Երևան, Գյումրի) were right in every
clip, in Armenian and English. Questions became English questions in 4/4 even where ՞ was dropped. Of six numbers,
the four non-year ones were right; both years were wrong, once from ASR (1995) and once from NLLB (2026 -> 2600).
Not fixed: converting Armenian number words to digits before translation would address the NLLB part, but it is a
new feature with a parser to test, not in the time budget. Kept as a known limitation.
After the fix and a re-take of 4 clips: CER 5.2 % / WER 12.4 % over the 16 sentences (`eval/results/asr_results.csv`).
Recorder fix: the first take cut the last word in 6 of 16 clips, because Enter was pressed while the last word was
still being said; `eval/record.py` now records 0.7 s more after Enter, and a re-recorded clip replaces its old
manifest row instead of adding a duplicate.

## 2026-10-25 — Drop the ASR noise output «Հովհ.»
In live runs the ASR wrote «Հովհ.» on noise 10 times (idle mic, a phone's echo-cancelled audio in a Meet call),
always as the whole output, never inside real speech (0 of 694 saved lines otherwise, 0 in FLEURS). It is the
abbreviation of Հովհաննես used in Bible references, likely learned from the audiobooks in the training data; NLLB
turns it into "John: What is it?", and on the call side each one could also create a new speaker. `Processor` now
treats it like a one-letter result: not translated, not printed, not used for speakers, counted in the stats.
Words that only start with it («Հովհաննեսը») pass. Other noise outputs stay visible on purpose.
Same session: Meet -> Multi-Output Device -> BlackHole verified (call side reached the program, latency < 1 s).
