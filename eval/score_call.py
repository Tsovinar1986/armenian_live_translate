"""Score the program's output on the test call: speaker labels, transcription errors, latency.

    python eval/score_call.py eval/data/test_call_out.txt [--offset 0]

Each printed phrase is matched to the true turn that contains its start time ([hh:mm:ss] in the header, ±1 s).
Speaker labels are scored per turn: a turn is right if its label is the label its speaker got on its first
turn, and different speakers never share a label. --offset shifts the printed times (live playback starts a
little after the program, so its clock runs ahead).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter

from common import DATA, ROOT

sys.path.insert(0, str(ROOT))
from livetr.metrics import cer_counts, wer_counts  # noqa: E402

HEADER = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\] (Me|Speaker [0-9?]+)\s+\(([\d.]+)s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("output")
    p.add_argument("--offset", type=float, default=0.0)
    args = p.parse_args()

    with open(DATA / "test_call.tsv", encoding="utf-8") as f:
        turns = [dict(r, start=float(r["start"]), end=float(r["end"])) for r in csv.DictReader(f, delimiter="\t")]
    phrases, cur = [], None
    for line in open(args.output, encoding="utf-8"):
        m = HEADER.match(line)
        if m:
            h, mi, s, label, lat = m.groups()
            cur = dict(t=int(h) * 3600 + int(mi) * 60 + int(s) - args.offset, label=label, latency=float(lat), hy="")
            phrases.append(cur)
        elif cur is not None and line.startswith("HY:"):
            cur["hy"] = line[3:].strip()

    for ph in phrases:
        ph["turn"] = next((i for i, t in enumerate(turns) if t["start"] - 1.5 <= ph["t"] <= t["end"]), None)
    unmatched = [ph for ph in phrases if ph["turn"] is None]

    first_label: dict[str, str] = {}
    right = 0
    print(f"{'turn':>4} {'true':>4}  labels printed")
    for i, t in enumerate(turns):
        labels = [ph["label"] for ph in phrases if ph["turn"] == i]
        main_label = Counter(labels).most_common(1)[0][0] if labels else "-"
        first_label.setdefault(t["speaker"], main_label)
        ok = main_label == first_label[t["speaker"]] and labels != []
        right += ok
        print(f"{i + 1:>4} {t['speaker']:>4}  {', '.join(labels) or '(nothing)'}  {'ok' if ok else 'WRONG'}")
    shared = len(set(first_label.values())) < len(first_label)

    c_err = c_tot = w_err = w_tot = 0
    for i, t in enumerate(turns):
        hyp = " ".join(ph["hy"] for ph in phrases if ph["turn"] == i)
        ce, ct = cer_counts(t["reference"], hyp)
        we, wt = wer_counts(t["reference"], hyp)
        c_err, c_tot, w_err, w_tot = c_err + ce, c_tot + ct, w_err + we, w_tot + wt
    lats = sorted(ph["latency"] for ph in phrases)

    found = {ph["label"] for ph in phrases if ph["label"] != "Speaker ?"}
    unknown = sum(ph["label"] == "Speaker ?" for ph in phrases)
    print(f"\nspeaker turns right: {right}/{len(turns)}; speakers found: {len(found)} (true {len(first_label)}); "
          f"'Speaker ?' phrases: {unknown}{'; TWO SPEAKERS SHARE A LABEL' if shared else ''}")
    print(f"transcription: CER {c_err / c_tot:.1%}  WER {w_err / w_tot:.1%}")
    print(f"latency: n={len(lats)}  p50 {lats[len(lats) // 2]:.2f}s  max {lats[-1]:.2f}s  over 5 s: "
          f"{sum(x > 5 for x in lats)}")
    if unmatched:
        print(f"{len(unmatched)} phrase(s) outside any turn: {[round(ph['t'], 1) for ph in unmatched]}")


if __name__ == "__main__":
    main()
