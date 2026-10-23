"""Helpers shared by the evaluation scripts."""
from __future__ import annotations

import csv
import resource
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "eval" / "data"
RESULTS = ROOT / "eval" / "results"


def load_manifest(path: Path = DATA / "manifest.tsv") -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found. Create it first: eval/record.py (own clips) or eval/get_fleurs.py (FLEURS)")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit WAV")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9  # bytes on macOS, KiB on Linux
