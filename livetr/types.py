"""Data passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIC = "mic"
SYSTEM = "system"


@dataclass
class Segment:
    """One utterance cut out by the VAD."""

    source: str          # MIC or SYSTEM
    audio: np.ndarray    # float32, mono, 16 kHz
    start_wall: float    # time.monotonic() of the first sample
    end_wall: float      # time.monotonic() of the last speech sample = "end of phrase"
    forced_cut: bool = False  # True if cut at max length instead of at a pause

    @property
    def duration(self) -> float:
        from livetr import SAMPLE_RATE
        return len(self.audio) / SAMPLE_RATE


@dataclass
class Result:
    """What gets printed for one segment."""

    segment: Segment
    speaker: str
    hy: str
    en: str
    printed_wall: float
    timings: dict[str, float] = field(default_factory=dict)  # seconds per stage

    @property
    def latency(self) -> float:
        """End of phrase -> line printed."""
        return self.printed_wall - self.segment.end_wall
