"""Pick the speaker threshold from data: ECAPA similarity of FLEURS recordings of the same sentence.

    python eval/fleurs_speakers.py                 # whole clips
    python eval/fleurs_speakers.py --crop-s 3      # only the first 3 s of each clip (closer to short live phrases)

FLEURS has several readers per sentence but no speaker ids. Pairs with different gender are certainly two
different people (ground truth). Same-gender pairs are a mix of different people and the same reader recorded
twice. A good threshold sits above (almost) all different-gender pairs and in the gap of the same-gender
histogram. Writes eval/results/fleurs_speaker_pairs_<crop>.csv (ids and similarities, no audio).
"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import RESULTS, ROOT

sys.path.insert(0, str(ROOT))
from livetr.speakers import EcapaEmbedder  # noqa: E402

REPO = "google/fleurs"
LANG = "hy_am"
BINS = np.round(np.arange(-0.2, 1.01, 0.1), 1)


def histogram(name: str, sims: np.ndarray) -> None:
    counts, _ = np.histogram(sims, bins=BINS)
    print(f"\n{name}: n={len(sims)}")
    for lo, c in zip(BINS[:-1], counts):
        print(f"  {lo:+.1f}..{lo + 0.1:+.1f} {c:5d} {'#' * int(np.ceil(c / max(counts.max(), 1) * 50))}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--crop-s", type=float, default=0.0, help="use only the first N seconds of each clip (0 = all)")
    p.add_argument("--split", default="test", choices=["test", "dev"])
    args = p.parse_args()

    import soundfile as sf
    from huggingface_hub import hf_hub_download

    tsv = hf_hub_download(REPO, f"data/{LANG}/{args.split}.tsv", repo_type="dataset")
    with open(tsv, encoding="utf-8") as f:
        meta = {r[1]: (r[0], r[6]) for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}  # file -> (sentence, gender)
    archive = hf_hub_download(REPO, f"data/{LANG}/audio/{args.split}.tar.gz", repo_type="dataset")

    embedder = EcapaEmbedder()
    emb: dict[str, np.ndarray] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if not member.isfile() or name not in meta:
                continue
            audio, sr = sf.read(io.BytesIO(tar.extractfile(member).read()), dtype="float32")
            if sr != 16000 or audio.ndim != 1:
                raise ValueError(f"{name}: expected 16 kHz mono, got {sr} Hz, shape {audio.shape}")
            if args.crop_s:
                audio = audio[: int(args.crop_s * 16000)]
            e = embedder.embed(audio)
            emb[name] = e / np.linalg.norm(e)
            if len(emb) % 100 == 0:
                print(f"  embedded {len(emb)}/{len(meta)}", file=sys.stderr)

    by_sentence: dict[str, list[str]] = defaultdict(list)
    for name in emb:
        by_sentence[meta[name][0]].append(name)

    crop = f"{args.crop_s:g}s" if args.crop_s else "full"
    out = RESULTS / f"fleurs_speaker_pairs_{crop}.csv"
    cross, same_gender = [], []
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sentence", "file_a", "file_b", "gender_a", "gender_b", "similarity"])
        for sid, names in sorted(by_sentence.items()):
            for a, b in itertools.combinations(sorted(names), 2):
                sim = float(emb[a] @ emb[b])
                ga, gb = meta[a][1], meta[b][1]
                (same_gender if ga == gb else cross).append(sim)
                w.writerow([sid, a, b, ga, gb, f"{sim:.4f}"])
    cross_a, same_a = np.array(cross), np.array(same_gender)

    print(f"\nclips {len(emb)}, crop {crop}, pairs written to {out}")
    histogram("different gender (certainly different people)", cross_a)
    histogram("same gender (different people + same reader twice)", same_a)
    print(f"\ndifferent-gender pairs: max {cross_a.max():.2f}, p99 {np.percentile(cross_a, 99):.2f}, "
          f"p95 {np.percentile(cross_a, 95):.2f}")
    for t in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        print(f"  threshold {t:.2f}: different-gender pairs wrongly merged {np.mean(cross_a >= t):6.1%}   "
              f"same-gender pairs above {np.mean(same_a >= t):6.1%}")


if __name__ == "__main__":
    main()
