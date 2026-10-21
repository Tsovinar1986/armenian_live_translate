# Benchmark results

Written by `eval/run_asr.py` (and committed, because they are the evidence for the model choice):
- `asr_results.csv`: one row per model run: CER, WER, latency mean/p90/max, real-time factor, peak memory.
- `asr_<model>.tsv`: per-clip reference vs hypothesis, to see *what kind* of errors a model makes.

No audio is stored here.
