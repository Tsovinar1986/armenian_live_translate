"""Build an ASR test set from Google FLEURS Armenian (hy_am): read speech with reference text, many readers.

    python eval/get_fleurs.py                      # 40 clips <= 10 s from the test split
    python eval/get_fleurs.py --n 50 --max-seconds 8

Output: eval/data/fleurs/<file>.wav (16 kHz mono 16-bit) and eval/data/fleurs_manifest.tsv (path, speaker,
reference). It is a separate manifest on purpose: FLEURS has no speaker ids, so these clips must not go into
eval/data/manifest.tsv, which run_speakers.py reads. Benchmark with: python eval/run_asr.py --set fleurs --model ...

The test split is used because the ASR fine-tunes may have trained on FLEURS train. The archive (~520 MB) is
cached by huggingface_hub; only the chosen clips are written to eval/data/ (git-ignored).
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import tarfile
import wave
from pathlib import Path

import numpy as np

from common import DATA

REPO = "google/fleurs"
LANG = "hy_am"
OUT_DIR = DATA / "fleurs"
MANIFEST = DATA / "fleurs_manifest.tsv"


def pick(rows: list[list[str]], n: int, max_seconds: float, seed: int) -> list[list[str]]:
    """One recording per sentence (FLEURS has several readers per sentence), length <= max_seconds."""
    by_sentence: dict[str, list[str]] = {}
    for r in rows:
        if int(r[5]) / 16000 <= max_seconds:
            by_sentence.setdefault(r[0], r)
    if len(by_sentence) < n:
        raise SystemExit(f"only {len(by_sentence)} sentences <= {max_seconds}s, asked for {n}")
    return random.Random(seed).sample(sorted(by_sentence.values()), n)


def read_wav_16k_mono(data: bytes, name: str) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if sr != 16000 or audio.ndim != 1:
        raise ValueError(f"{name}: expected 16 kHz mono, got {sr} Hz, shape {audio.shape}")
    return audio


def save_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40, help="number of clips")
    p.add_argument("--max-seconds", type=float, default=10.0, help="skip longer clips (live phrases are cut at 8 s)")
    p.add_argument("--split", default="test", choices=["test", "dev"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from huggingface_hub import hf_hub_download

    tsv = hf_hub_download(REPO, f"data/{LANG}/{args.split}.tsv", repo_type="dataset")
    with open(tsv, encoding="utf-8") as f:
        # columns: id, file name, raw transcription, normalised transcription, phonemes, samples, gender
        rows = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    chosen = pick(rows, args.n, args.max_seconds, args.seed)
    wanted = {r[1]: r for r in chosen}

    print(f"downloading {args.split} audio archive (cached after the first run)…")
    archive = hf_hub_download(REPO, f"data/{LANG}/audio/{args.split}.tar.gz", repo_type="dataset")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    found: dict[str, np.ndarray] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if member.isfile() and name in wanted:
                found[name] = read_wav_16k_mono(tar.extractfile(member).read(), name)
    missing = set(wanted) - set(found)
    if missing:
        raise SystemExit(f"{len(missing)} chosen files not in the archive, e.g. {sorted(missing)[:3]}")

    total = 0.0
    with open(MANIFEST, "w", encoding="utf-8", newline="") as mf:
        w = csv.writer(mf, delimiter="\t")
        w.writerow(["path", "speaker", "reference"])
        for r in chosen:
            rel = Path("fleurs") / r[1]
            save_wav(DATA / rel, found[r[1]])
            w.writerow([str(rel), f"fleurs-{r[6].lower()}", r[2]])
            total += len(found[r[1]]) / 16000
    print(f"wrote {len(chosen)} clips ({total:.0f} s) to {OUT_DIR} and {MANIFEST}")


if __name__ == "__main__":
    main()
