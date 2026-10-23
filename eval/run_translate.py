"""Translate the reference sentences and show them next to the expected English, with latency.

    python eval/run_translate.py
    python eval/run_translate.py --from-asr turbo-hy   # translate that model's ASR output instead (error propagation)

Translation quality is judged by a person who speaks both languages; the numbers here are latency only.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

from common import RESULTS, ROOT

sys.path.insert(0, str(ROOT))
from livetr.translate import NllbTranslator  # noqa: E402


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    p = argparse.ArgumentParser()
    p.add_argument("--nllb-dir", default=str(ROOT / "models/nllb-200-distilled-1.3B-ct2-int8"))
    p.add_argument("--from-asr", default=None, help="use eval/results/asr_<model>.tsv hypotheses as input")
    a = p.parse_args()

    if a.from_asr:
        with open(RESULTS / f"asr_{a.from_asr}.tsv", encoding="utf-8") as f:
            items = [(r["hypothesis"], "") for r in csv.DictReader(f, delimiter="\t")]
    else:
        with open(ROOT / "eval/sentences.tsv", encoding="utf-8") as f:
            items = [(r["hy"], r["en"]) for r in csv.DictReader(f, delimiter="\t")]

    tr = NllbTranslator(a.nllb_dir)
    tr.translate("Բարև")  # warm-up
    lats = []
    for hy, ref in items:
        t = time.monotonic()
        en = tr.translate(hy)
        lats.append(time.monotonic() - t)
        print(f"HY : {hy}\nMT : {en}" + (f"\nREF: {ref}" if ref else "") + f"\n     ({lats[-1]:.2f}s)\n")
    print(f"mean {sum(lats) / len(lats):.2f}s  max {max(lats):.2f}s over {len(lats)} sentences")


if __name__ == "__main__":
    main()
