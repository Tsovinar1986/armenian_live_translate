"""Build a test "call" from FLEURS with known speakers, to replay through the whole program.

    python eval/make_test_call.py
    python -m livetr --file eval/data/test_call.wav --no-mic --no-color | tee eval/data/test_call_out.txt
    python eval/score_call.py eval/data/test_call_out.txt

FLEURS has no speaker ids, and its Armenian test set has only three voices: one man and two women (every male
clip matches every other male clip at >= 0.6). A reader is a clip plus its closest clips of other sentences,
all with ECAPA similarity >= --min-sim (0.7); different people never come close to that (man/woman pairs
max 0.30, see fleurs_speakers.py). A reader is only accepted if all its clips are below --max-cross (0.35)
against the readers already chosen, so the ground truth is reliable. A first version took "two men" that were
the same man reading two sentences (similarity 0.74); this check prevents that.
Three speakers, three turns each, in an order where every speaker comes back after others have spoken.
Limit: no two *very* similar voices, because FLEURS does not have them.

Writes eval/data/test_call.wav, test_call_opus16k.wav (through the 16 kbps Opus codec that calls use, needs
ffmpeg) and test_call.tsv (start, end, speaker, reference text).

Long session (10+ minutes, many speaker switches): `--turns 12 --name test_call_long` gives each speaker 12 turns
in a random order (fixed seed) with no speaker twice in a row.
"""
from __future__ import annotations

import argparse
import csv
import io
import subprocess
import tarfile
import wave
from pathlib import Path

import numpy as np

from common import DATA, ROOT

GAP_S = 1.2
ORDER = ["F1", "M1", "F2", "M1", "F1", "F2", "M1", "F2", "F1"]   # every speaker comes back after others


def turn_order(turns: int, seed: int = 0) -> list[str]:
    """The fixed 9-turn order for 3 turns each; otherwise random, each speaker `turns` times, never twice in a row."""
    if turns == 3:
        return ORDER
    rng = np.random.default_rng(seed)
    left = {"F1": turns, "F2": turns, "M1": turns}
    order: list[str] = []
    while any(left.values()):
        options = [s for s, n in left.items() if n and (not order or s != order[-1])]
        if not options:
            raise SystemExit(f"cannot place the last turns without a repeat: {order}")
        top = max(left[s] for s in options)  # prefer the speaker with most turns left, so none is stranded at the end
        spk = str(rng.choice([s for s in options if left[s] == top]))
        order.append(spk)
        left[spk] -= 1
    return order


def save(path: Path, audio: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--min-sim", type=float, default=0.7)
    p.add_argument("--max-cross", type=float, default=0.35)
    p.add_argument("--turns", type=int, default=3, help="turns per speaker")
    p.add_argument("--name", default="test_call", help="output file prefix in eval/data/")
    args = p.parse_args()
    TURNS = args.turns

    import soundfile as sf
    from huggingface_hub import hf_hub_download

    import sys
    sys.path.insert(0, str(ROOT))
    from livetr.speakers import EcapaEmbedder

    tsv = hf_hub_download("google/fleurs", "data/hy_am/test.tsv", repo_type="dataset")
    with open(tsv, encoding="utf-8") as f:
        meta = {r[1]: dict(sentence=r[0], text=r[2], gender=r[6]) for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}
    archive = hf_hub_download("google/fleurs", "data/hy_am/audio/test.tar.gz", repo_type="dataset")

    def read_tar(names: set[str] | None):
        with tarfile.open(archive, "r:gz") as tar:
            for m in tar:
                name = Path(m.name).name
                if m.isfile() and name in meta and (names is None or name in names):
                    a, sr = sf.read(io.BytesIO(tar.extractfile(m).read()), dtype="float32")
                    if sr != 16000 or a.ndim != 1:
                        raise ValueError(f"{name}: expected 16 kHz mono")
                    yield name, a

    embedder = EcapaEmbedder()
    names, vecs = [], []
    for name, a in read_tar(None):  # embed every clip, keep only the 192-d vectors in memory
        e = embedder.embed(a)
        names.append(name)
        vecs.append(e / np.linalg.norm(e))
    V = np.array(vecs)
    S = V @ V.T

    # FLEURS Armenian has only three voices (one man, two women: every male clip matches every other at >= 0.6).
    # A reader = a seed clip plus its closest clips of other sentences (all >= min_sim). A new reader is accepted
    # only if all its clips are below max_cross against every clip already taken.
    chosen: dict[str, list[list[int]]] = {"FEMALE": [], "MALE": []}
    need = {"FEMALE": 2, "MALE": 1}
    taken: list[int] = []
    for i, name in enumerate(names):
        g = meta[name]["gender"]
        if len(chosen[g]) >= need[g] or any(S[i, k] >= args.max_cross for k in taken):
            continue
        members, sentences = [], set()
        for j in np.argsort(-S[i]):
            if meta[names[j]]["gender"] == g and meta[names[j]]["sentence"] not in sentences and S[i, j] >= args.min_sim:
                members.append(int(j))
                sentences.add(meta[names[j]]["sentence"])
            if len(members) == TURNS:
                break
        if len(members) == TURNS and all(S[m, k] < args.max_cross for m in members for k in taken):
            chosen[g].append(members)
            taken += members
    if any(len(chosen[g]) < need[g] for g in need):
        raise SystemExit(f"not enough distinct readers: {[(k, len(v)) for k, v in chosen.items()]}")
    speakers = {"F1": chosen["FEMALE"][0], "F2": chosen["FEMALE"][1], "M1": chosen["MALE"][0]}
    cross = max(S[a, b] for sa, ma in speakers.items() for sb, mb in speakers.items() if sa != sb
                for a in ma for b in mb)
    same = {spk: min(S[a, b] for a in m for b in m if a != b) for spk, m in speakers.items()}
    taken_pairs = [(spk, k) for spk, m in speakers.items() for k in m]
    audio = dict(read_tar({names[k] for _, k in taken_pairs}))
    text = {n: meta[n]["text"] for n in audio}

    turn_of = {k: 0 for k in speakers}
    parts, rows, t = [np.zeros(int(GAP_S * 16000), np.float32)], [], GAP_S
    for spk in turn_order(TURNS):
        name = names[speakers[spk][turn_of[spk]]]
        turn_of[spk] += 1
        a = audio[name]
        rows.append((f"{t:.2f}", f"{t + len(a) / 16000:.2f}", spk, text[name]))
        parts += [a, np.zeros(int(GAP_S * 16000), np.float32)]
        t += len(a) / 16000 + GAP_S
    call = np.concatenate(parts)
    call *= 0.5 / max(np.abs(call).max(), 1e-6)  # same level for both versions

    out = DATA / f"{args.name}.wav"
    save(out, call)
    with open(DATA / f"{args.name}.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["start", "end", "speaker", "reference"])
        w.writerows(rows)
    opus = DATA / f"{args.name}_opus16k.wav"
    subprocess.run(f"ffmpeg -y -loglevel error -i {out} -c:a libopus -b:a 16k -f ogg - | "
                   f"ffmpeg -y -loglevel error -i - -ar 16000 -ac 1 -c:a pcm_s16le {opus}", shell=True, check=True)
    print(f"{len(call) / 16000:.0f} s, {len(rows)} turns, lowest similarity within a reader "
          + ", ".join(f"{k} {v:.2f}" for k, v in same.items()))
    print(f"largest similarity between two different chosen readers: {cross:.2f}")
    print(f"wrote {out}, {opus}, {DATA / (args.name + '.tsv')}")


if __name__ == "__main__":
    main()
