"""Terminal output and latency statistics."""
from __future__ import annotations

import sys
from collections import defaultdict
from typing import TextIO

import numpy as np

from livetr.types import Result

LATENCY_BUDGET_S = 5.0

_DIM, _RED, _BOLD, _RESET = "\033[2m", "\033[31m", "\033[1m", "\033[0m"


def fmt_clock(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def format_result(result: Result, session_start: float, color: bool = True, details: bool = True) -> str:
    seg = result.segment
    clock = fmt_clock(seg.start_wall - session_start)
    header = f"[{clock}] {result.speaker}"
    if color:
        header = f"{_BOLD}{header}{_RESET}"
    if details:
        t = result.timings
        parts = [f"{result.latency:.1f}s"] + [f"{k} {t[k]:.1f}" for k in ("wait", "asr", "spk", "mt") if k in t]
        if seg.forced_cut:
            parts.append("cut")
        info = "  (" + " · ".join(parts) + ")"
        if color:
            info = (_RED if result.latency > LATENCY_BUDGET_S else _DIM) + info + _RESET
        header += info
    return f"{header}\nHY: {result.hy}\nEN: {result.en}\n"


class Printer:
    def __init__(self, session_start: float, stream: TextIO = sys.stdout, color: bool = True, details: bool = True):
        self.session_start = session_start
        self.stream = stream
        self.color = color
        self.details = details

    def show(self, result: Result) -> None:
        self.stream.write(format_result(result, self.session_start, self.color, self.details) + "\n")
        self.stream.flush()


class Stats:
    """Collects latency per source and per stage; printed at exit."""

    def __init__(self) -> None:
        self.latencies: dict[str, list[float]] = defaultdict(list)
        self.stage_times: dict[str, list[float]] = defaultdict(list)
        self.empty_asr = 0
        self.dropped_short = 0
        self.overflows = 0

    def add(self, result: Result) -> None:
        self.latencies[result.segment.source].append(result.latency)
        for k, v in result.timings.items():
            self.stage_times[k].append(v)

    def summary(self) -> str:
        lines = ["", "── session stats ──"]
        if not self.latencies:
            lines.append("no segments printed")
        for src, vals in sorted(self.latencies.items()):
            a = np.array(vals)
            over = int((a > LATENCY_BUDGET_S).sum())
            lines.append(
                f"{src:<7} n={len(a):<4} latency p50 {np.percentile(a, 50):.2f}s  "
                f"p90 {np.percentile(a, 90):.2f}s  max {a.max():.2f}s  over {LATENCY_BUDGET_S:.0f}s: {over}"
            )
        for k, vals in sorted(self.stage_times.items()):
            lines.append(f"stage {k:<5} mean {np.mean(vals):.2f}s  max {np.max(vals):.2f}s")
        lines.append(
            f"dropped short segments: {self.dropped_short}   empty ASR results: {self.empty_asr}   "
            f"audio overflows: {self.overflows}"
        )
        return "\n".join(lines)
