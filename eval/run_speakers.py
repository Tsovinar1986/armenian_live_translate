"""Check whether speaker embeddings separate the people in the eval set, and pick a clustering threshold.

    python eval/run_speakers.py

Needs clips from at least 2 speakers in eval/data/manifest.tsv (eval/record.py --speaker <name>).
Prints:
  1. cosine similarity of same-speaker pairs vs different-speaker pairs,
  2. the threshold that best separates them on this data,
  3. a simulation of the live online clustering in a random order: are labels pure and stable?
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from common import DATA, ROOT, load_manifest, load_wav

sys.path.insert(0, str(ROOT))
from livetr.speakers import EcapaEmbedder, OnlineSpeakerClusterer  # noqa: E402


def best_threshold(same: np.ndarray, diff: np.ndarray) -> tuple[float, float]:
    best = (0.0, 0.0)
    for th in np.arange(0.0, 1.0, 0.01):
        acc = ((same >= th).sum() + (diff < th).sum()) / (len(same) + len(diff))
        if acc > best[1]:
            best = (float(th), float(acc))
    return best


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=None, help="threshold for the simulation (default: best found)")
    p.add_argument("--orders", type=int, default=20, help="random orders to simulate")
    a = p.parse_args()

    rows = load_manifest()
    people = sorted({r["speaker"] for r in rows})
    if len(people) < 2:
        raise SystemExit(f"need clips from >= 2 speakers, found {people}")

    emb_model = EcapaEmbedder(str(ROOT / "models/spkrec-ecapa-voxceleb"))
    embs, spk, durs = [], [], []
    for r in rows:
        audio = load_wav(DATA / r["path"])
        e = emb_model.embed(audio)
        embs.append(e / np.linalg.norm(e))
        spk.append(r["speaker"])
        durs.append(len(audio) / 16000)
    E = np.stack(embs)
    S = E @ E.T

    same, diff = [], []
    for i, j in itertools.combinations(range(len(rows)), 2):
        (same if spk[i] == spk[j] else diff).append(S[i, j])
    same_a, diff_a = np.array(same), np.array(diff)
    print(f"speakers: {dict(Counter(spk))}")
    print(f"same-speaker  similarity: mean {same_a.mean():.2f}  min {same_a.min():.2f}  p10 {np.percentile(same_a, 10):.2f}")
    print(f"diff-speaker  similarity: mean {diff_a.mean():.2f}  max {diff_a.max():.2f}  p90 {np.percentile(diff_a, 90):.2f}")
    th, acc = best_threshold(same_a, diff_a)
    print(f"best pairwise threshold on this data: {th:.2f} (pair accuracy {acc:.1%})")

    sim_th = a.threshold if a.threshold is not None else th
    rng = np.random.default_rng(0)
    purities, n_found, unknown = [], [], 0
    for _ in range(a.orders):
        order = rng.permutation(len(rows))
        cl = OnlineSpeakerClusterer(threshold=sim_th)
        by_label: dict[str, list[str]] = defaultdict(list)
        for i in order:
            label = cl.assign(E[i], durs[i])
            if label == "Speaker ?":
                unknown += 1
                continue
            by_label[label].append(spk[i])
        correct = sum(Counter(v).most_common(1)[0][1] for v in by_label.values())
        purities.append(correct / max(1, sum(len(v) for v in by_label.values())))
        n_found.append(cl.n_speakers)
    print(f"\nonline clustering at threshold {sim_th:.2f}, {a.orders} random orders:")
    print(f"  label purity mean {np.mean(purities):.1%}  min {np.min(purities):.1%}")
    print(f"  speakers found: {Counter(n_found)}  (true: {len(people)})")
    print(f"  'Speaker ?' labels: {unknown / a.orders:.1f} per run")


if __name__ == "__main__":
    main()
