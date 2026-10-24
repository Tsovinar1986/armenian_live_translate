"""Build a 5-minute "nobody is talking" file to count invented lines (silence and noise test, R2).

    python eval/make_noise.py
    python -m livetr --file eval/data/noise.wav --no-color | tee eval/data/noise_out.txt

Synthetic, so it is repeatable; a real idle room through the mic is the other half of this test. Five sections of
60 s, each a different kind of sound that a call or a room produces while nobody speaks:
  1 room tone   quiet pink noise (-50 dBFS)
  2 fan         louder low-frequency rumble with 100 Hz hum (-35 dBFS)
  3 typing      key clicks at typing speed over room tone
  4 thumps      desk knocks, a door, a cup put down (loud, short)
  5 breaths     breath-like noise bursts of 0.4-1.2 s (the hardest case: the VAD may call them speech)
Writes eval/data/noise.wav and noise.tsv (start, end, section).
"""
from __future__ import annotations

import csv
import wave

import numpy as np

from common import DATA

SR = 16000
SECTION_S = 60
rng = np.random.default_rng(0)


def db(x: np.ndarray, level_db: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x ** 2))) or 1.0
    return x * (10 ** (level_db / 20) / rms)


def coloured(n: int, exponent: float) -> np.ndarray:
    """Noise with power ~ 1/f^exponent (1 = pink, 2 = brown)."""
    spec = np.fft.rfft(rng.normal(size=n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    return np.fft.irfft(spec / f ** (exponent / 2), n).astype(np.float32)


def bursts(n: int, times: np.ndarray, make) -> np.ndarray:
    out = np.zeros(n, np.float32)
    for t in times:
        b = make()
        i = int(t * SR)
        out[i:i + len(b)] += b[:max(0, n - i)]
    return out


def click() -> np.ndarray:
    m = int(0.012 * SR)
    return (rng.normal(size=m) * np.exp(-np.arange(m) / (0.002 * SR)) * 0.08).astype(np.float32)


def thump() -> np.ndarray:
    m = int(0.25 * SR)
    t = np.arange(m) / SR
    body = np.sin(2 * np.pi * rng.uniform(60, 180) * t) + 0.3 * rng.normal(size=m)
    return (body * np.exp(-t / 0.04) * rng.uniform(0.2, 0.5)).astype(np.float32)


def breath() -> np.ndarray:
    m = int(rng.uniform(0.4, 1.2) * SR)
    env = np.sin(np.linspace(0, np.pi, m)) ** 2
    noise = coloured(m, 0.5)
    return (db(noise, -30) * env).astype(np.float32)


def main() -> None:
    n = SECTION_S * SR
    room = lambda: db(coloured(n, 1.0), -50)  # noqa: E731
    keys = np.cumsum(rng.exponential(0.18, 400))
    sections = {
        "room tone": room(),
        "fan": db(coloured(n, 2.0), -38) + db(np.sin(2 * np.pi * 100 * np.arange(n) / SR).astype(np.float32), -42),
        "typing": room() + bursts(n, keys[keys < 59], click),
        "thumps": room() + bursts(n, np.sort(rng.uniform(1, 58, 15)), thump),
        "breaths": room() + bursts(n, np.arange(2, 58, 4.0) + rng.uniform(-0.5, 0.5, 14), breath),
    }
    rows, parts, t = [], [], 0.0
    for name, audio in sections.items():
        rows.append((f"{t:.0f}", f"{t + SECTION_S:.0f}", name))
        parts.append(audio)
        t += SECTION_S
    audio = np.clip(np.concatenate(parts), -1, 1)
    out = DATA / "noise.wav"
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
    with open(DATA / "noise.tsv", "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerows([("start", "end", "section"), *rows])
    print(f"wrote {out} ({len(audio) / SR:.0f} s) and noise.tsv")


if __name__ == "__main__":
    main()
