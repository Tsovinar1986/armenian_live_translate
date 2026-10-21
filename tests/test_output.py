"""R5: each line has time, speaker, Armenian and English; latency over budget is highlighted."""
import io

import numpy as np

from livetr.output import Printer, Stats, fmt_clock, format_result
from livetr.types import Result, Segment


def result(latency: float, start: float = 72.4) -> Result:
    seg = Segment("system", np.zeros(16000, dtype=np.float32), start_wall=1000 + start, end_wall=1000 + start + 1)
    return Result(seg, "Speaker 1", "Բարև, ինչպե՞ս ես։", "Hi, how are you?",
                  printed_wall=seg.end_wall + latency, timings={"asr": 1.2, "mt": 0.2})


def test_clock():
    assert fmt_clock(72.9) == "00:01:12"
    assert fmt_clock(3725) == "01:02:05"


def test_contains_all_four_pieces():
    text = format_result(result(2.0), session_start=1000, color=False)
    assert text.startswith("[00:01:12] Speaker 1")
    assert "HY: Բարև, ինչպե՞ս ես։" in text
    assert "EN: Hi, how are you?" in text
    assert "2.0s" in text


def test_over_budget_is_red():
    assert "\033[31m" in format_result(result(6.0), 1000, color=True)
    assert "\033[31m" not in format_result(result(3.0), 1000, color=True)


def test_printer_writes_and_stats_summarise():
    buf = io.StringIO()
    Printer(1000, stream=buf, color=False).show(result(1.0))
    assert "EN: Hi, how are you?" in buf.getvalue()
    st = Stats()
    for lat in (1.0, 2.0, 6.0):
        st.add(result(lat))
    s = st.summary()
    assert "system" in s and "n=3" in s and "over 5s: 1" in s
