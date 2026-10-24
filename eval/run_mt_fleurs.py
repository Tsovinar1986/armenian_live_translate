"""Compare translation models: chrF against the official English FLEURS translations, plus latency.

    python eval/run_mt_fleurs.py --nllb-dir models/nllb-200-distilled-600M-ct2-int8
    python eval/run_mt_fleurs.py --nllb-dir models/nllb-200-distilled-1.3B-ct2-int8

FLEURS sentences have the same id in every language, so hy_am and en_us give Armenian/English pairs. Uses the
first --n sentences of the test split (one recording's text per sentence). Also translates the colloquial
phrases from the live microphone test (eval/colloquial.tsv) and prints them for a human to judge: no reference
score there, because a handful of sentences gives no reliable number.
Appends one row to eval/results/mt_results.csv; per-sentence output goes to eval/results/mt_<model>.tsv.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from common import RESULTS, ROOT

sys.path.insert(0, str(ROOT))
from livetr.translate import NllbTranslator  # noqa: E402


def fleurs_pairs(n: int) -> list[tuple[str, str]]:
    from huggingface_hub import hf_hub_download

    def load(lang: str) -> dict[str, str]:
        path = hf_hub_download("google/fleurs", f"data/{lang}/test.tsv", repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            return {r[0]: r[2] for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}

    hy, en = load("hy_am"), load("en_us")
    ids = sorted(i for i in hy if i in en)[:n]
    return [(hy[i], en[i]) for i in ids]


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    p = argparse.ArgumentParser()
    p.add_argument("--nllb-dir", required=True)
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--no-repeat-ngram", type=int, default=0, help="NllbTranslator no_repeat_ngram_size (0 = off)")
    args = p.parse_args()
    from sacrebleu.metrics import CHRF

    model = Path(args.nllb_dir).name + (f"-norepeat{args.no_repeat_ngram}" if args.no_repeat_ngram else "")
    pairs = fleurs_pairs(args.n)
    tr = NllbTranslator(args.nllb_dir, no_repeat_ngram_size=args.no_repeat_ngram)
    tr.translate("Բարև")  # warm-up
    hyps, lats = [], []
    for hy, _ in pairs:
        t = time.monotonic()
        hyps.append(tr.translate(hy))
        lats.append(time.monotonic() - t)
    chrf = CHRF().corpus_score(hyps, [[en for _, en in pairs]]).score

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"mt_{model}.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["hy", "reference_en", "output_en", "seconds"])
        for (hy, en), out, lat in zip(pairs, hyps, lats):
            w.writerow([hy, en, out, f"{lat:.2f}"])
    lats.sort()
    row = dict(model=model, sentences=len(pairs), chrf=f"{chrf:.1f}", lat_mean=f"{sum(lats) / len(lats):.2f}",
               lat_p90=f"{lats[int(len(lats) * 0.9)]:.2f}", lat_max=f"{lats[-1]:.2f}", date=time.strftime("%Y-%m-%d %H:%M"))
    summary = RESULTS / "mt_results.csv"
    new = not summary.exists()
    with open(summary, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print("  ".join(f"{k}={v}" for k, v in row.items()))

    print("\ncolloquial phrases (live mic test):")
    with open(ROOT / "eval" / "colloquial.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            print(f"  HY: {r['hy']}\n  MT: {tr.translate(r['hy'])}\n  meant: {r['meaning']}\n")


if __name__ == "__main__":
    main()
