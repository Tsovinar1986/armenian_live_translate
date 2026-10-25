"""Record the evaluation set: one WAV per sentence, 16 kHz mono 16-bit, plus a manifest.

    python eval/record.py --speaker anna                 # reads eval/sentences.tsv
    python eval/record.py --speaker anna --mic 2
    python eval/record.py --speaker davit --free 6       # 6 free-speech clips (no reference) for speaker tests
    python eval/record.py --speaker anna --ids s09 s15 s23  # only these sentences

Output: eval/data/<speaker>/<id>.wav and lines in eval/data/manifest.tsv (path, speaker, reference).
Audio stays local: eval/data/ is in .gitignore.
"""
from __future__ import annotations

import argparse
import csv
import queue
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from livetr.audio import devices as dev  # noqa: E402
from livetr.audio.capture import Converter, DeviceSource  # noqa: E402

DATA = ROOT / "eval" / "data"
TAIL_S = 0.7  # seconds still recorded after Enter (first take cut the last word in 6 of 16 clips)


def record_one(src: DeviceSource) -> np.ndarray:
    conv = Converter(src.sample_rate)
    input("  press Enter to START recording")
    while not src.blocks.empty():
        src.blocks.get()
    chunks: list[np.ndarray] = []
    stop = threading.Event()

    def wait_enter() -> None:
        input("  recording… press Enter to STOP")
        stop.set()

    t = threading.Thread(target=wait_enter, daemon=True)
    t.start()
    stop_at = None  # keep recording TAIL_S after Enter: people press it while saying the last word
    while stop_at is None or time.monotonic() < stop_at or not src.blocks.empty():
        if stop_at is None and stop.is_set():
            stop_at = time.monotonic() + TAIL_S
        try:
            chunks.append(conv(src.blocks.get(timeout=0.1).audio))
        except queue.Empty:
            continue
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def save_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--speaker", required=True, help="short name of the person speaking, e.g. anna")
    p.add_argument("--mic", default=None, help="input device index or name (default: system default)")
    p.add_argument("--free", type=int, default=0, help="record N free-speech clips instead of the sentence list")
    p.add_argument("--ids", nargs="+", default=None, help="record only these sentence ids, e.g. s09 s15")
    args = p.parse_args()

    devices = dev.query_devices()
    idx = dev.resolve_input_device(args.mic, devices) if args.mic else dev.default_input_index()
    src = DeviceSource("mic", idx)
    print(f"recording from {src.describe()}")
    src.start()

    if args.free:
        items = [(f"free{i + 1:02d}", "", "Say anything for 3–6 seconds in Armenian.") for i in range(args.free)]
    else:
        with open(ROOT / "eval" / "sentences.tsv", encoding="utf-8") as f:
            items = [(r["id"], r["hy"], r["hy"]) for r in csv.DictReader(f, delimiter="\t")]
        if args.ids:
            unknown = set(args.ids) - {i for i, _, _ in items}
            if unknown:
                raise SystemExit(f"unknown sentence ids: {sorted(unknown)}")
            items = [it for it in items if it[0] in args.ids]

    manifest = DATA / "manifest.tsv"
    DATA.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[str]] = {}  # path -> row; a re-recorded clip replaces its old row
    if manifest.exists():
        with open(manifest, encoding="utf-8") as f:
            rows = {r["path"]: [r["path"], r["speaker"], r["reference"]] for r in csv.DictReader(f, delimiter="\t")}

    def write_manifest() -> None:
        with open(manifest, "w", encoding="utf-8", newline="") as mf:
            w = csv.writer(mf, delimiter="\t")
            w.writerow(["path", "speaker", "reference"])
            w.writerows(rows.values())

    try:
        for clip_id, ref, prompt in items:
            print(f"\n[{clip_id}] {prompt}")
            audio = record_one(src)
            if len(audio) < 8000:
                print("  too short, skipped")
                continue
            rel = Path(args.speaker) / f"{clip_id}.wav"
            save_wav(DATA / rel, audio)
            rows[str(rel)] = [str(rel), args.speaker, ref]
            write_manifest()
            print(f"  saved {rel} ({len(audio) / 16000:.1f}s)")
    finally:
        src.stop()


if __name__ == "__main__":
    main()
