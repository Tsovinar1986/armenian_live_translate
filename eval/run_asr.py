"""Benchmark one ASR model on the recorded clips: CER, WER, latency, memory.

    python eval/run_asr.py --model turbo-hy
    for m in buzz turbo-hy mms turbo-zeroshot; do python eval/run_asr.py --model $m; done
    python eval/run_asr.py --summary
    python eval/run_asr.py --set fleurs --model turbo-hy   # FLEURS clips from eval/get_fleurs.py

One model per process so peak memory is measured cleanly. Each run appends one row to
eval/results/asr_results.csv and writes per-clip output to eval/results/asr_<model>.tsv
(asr_<model>_fleurs.tsv for the FLEURS set).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

from common import DATA, RESULTS, ROOT, load_manifest, load_wav, peak_rss_gb

sys.path.insert(0, str(ROOT))
from livetr.asr import MODELS, load_asr, pick_device  # noqa: E402
from livetr.metrics import cer_counts, wer_counts  # noqa: E402

SUMMARY = RESULTS / "asr_results.csv"
SETS = {"own": DATA / "manifest.tsv", "fleurs": DATA / "fleurs_manifest.tsv"}
FIELDS = ["set", "model", "device", "dtype", "clips", "cer", "wer", "lat_mean", "lat_p90", "lat_max", "rtf", "peak_rss_gb",
          "load_s", "date"]


def run(model: str, device: str, dtype: str, clip_set: str) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    rows = [r for r in load_manifest(SETS[clip_set]) if r["reference"].strip()]
    if not rows:
        raise SystemExit("no clips with a reference transcript in the manifest")

    t0 = time.monotonic()
    asr = load_asr(model, pick_device(device), dtype)
    load_s = time.monotonic() - t0
    asr.transcribe(np.zeros(16000, dtype=np.float32))  # warm-up, not measured

    RESULTS.mkdir(parents=True, exist_ok=True)
    c_err = c_tot = w_err = w_tot = 0
    lats, audio_s = [], 0.0
    per_clip = f"asr_{model}.tsv" if clip_set == "own" else f"asr_{model}_{clip_set}.tsv"
    with open(RESULTS / per_clip, "w", encoding="utf-8", newline="") as f:
        out = csv.writer(f, delimiter="\t")
        out.writerow(["path", "seconds", "latency", "cer", "reference", "hypothesis"])
        for r in rows:
            audio = load_wav(DATA / r["path"])
            t = time.monotonic()
            hyp = asr.transcribe(audio)
            lat = time.monotonic() - t
            ce, ct = cer_counts(r["reference"], hyp)
            we, wt = wer_counts(r["reference"], hyp)
            c_err, c_tot, w_err, w_tot = c_err + ce, c_tot + ct, w_err + we, w_tot + wt
            lats.append(lat)
            audio_s += len(audio) / 16000
            out.writerow([r["path"], f"{len(audio) / 16000:.2f}", f"{lat:.2f}", f"{ce / max(ct, 1):.3f}",
                          r["reference"], hyp])
            print(f"{lat:5.2f}s  CER {ce / max(ct, 1):5.1%}  {hyp}")

    lats_a = np.array(lats)
    row = dict(
        set=clip_set, model=model, device=pick_device(device), dtype=dtype, clips=len(rows),
        cer=f"{c_err / c_tot:.4f}", wer=f"{w_err / w_tot:.4f}",
        lat_mean=f"{lats_a.mean():.2f}", lat_p90=f"{np.percentile(lats_a, 90):.2f}", lat_max=f"{lats_a.max():.2f}",
        rtf=f"{lats_a.sum() / audio_s:.3f}", peak_rss_gb=f"{peak_rss_gb():.2f}", load_s=f"{load_s:.0f}",
        date=time.strftime("%Y-%m-%d %H:%M"),
    )
    new = not SUMMARY.exists()
    with open(SUMMARY, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print("\n" + "  ".join(f"{k}={v}" for k, v in row.items()))


def summary() -> None:
    if not SUMMARY.exists():
        raise SystemExit("no results yet")
    with open(SUMMARY) as f:
        rows = list(csv.DictReader(f))
    print(f"{'set':<8}{'model':<16}{'CER':>7}{'WER':>7}{'lat mean':>10}{'p90':>7}{'max':>7}{'RSS GB':>8}")
    for r in sorted(rows, key=lambda r: (r["set"], float(r["cer"]))):
        print(f"{r['set']:<8}{r['model']:<16}{float(r['cer']):>7.1%}{float(r['wer']):>7.1%}{r['lat_mean']:>9}s"
              f"{r['lat_p90']:>6}s{r['lat_max']:>6}s{r['peak_rss_gb']:>8}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(MODELS))
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--set", default="own", choices=sorted(SETS), help="own recordings or FLEURS clips")
    p.add_argument("--summary", action="store_true")
    a = p.parse_args()
    if a.summary:
        summary()
    elif a.model:
        run(a.model, a.device, a.dtype, a.set)
    else:
        p.error("give --model or --summary")
