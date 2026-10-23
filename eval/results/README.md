# Benchmark results

Written by `eval/run_asr.py` (and committed, because they are the evidence for the model choice):
- `asr_results.csv`: one row per model run: CER, WER, latency mean/p90/max, real-time factor, peak memory.
- `asr_<model>.tsv`: per-clip reference vs hypothesis, to see *what kind* of errors a model makes.

No audio is stored here.

`live-runs/`: terminal output of the program in test runs (time, speaker, Armenian, English, latency), one Markdown
file per run, with model and setup at the top:

| File | Run |
|---|---|
| `live-runs/2026-10-23-live-test-1-turbo-hy.md` | YouTube conversation, call side only, old `turbo-hy` model: 30 phrases, latency p90 4.51 s |
| `live-runs/2026-10-23-mic-live-me.md` | Author speaking into the MacBook mic: 6 phrases, all `Me`, latency 0.8–1.5 s |
| `live-runs/2026-10-23-test-call-live-blackhole.md` | Test call (3 known voices, 9 turns) played live through BlackHole: 9/9 speaker turns, latency max 1.6 s |
