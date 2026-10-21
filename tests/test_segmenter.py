"""R5/C1: utterances are cut at pauses, long speech is force-cut, noise blips are dropped."""
import numpy as np
import pytest

from conftest import SR, energy_vad, silence, tone
from livetr.segmenter import Segmenter


def feed(seg: Segmenter, audio: np.ndarray, block: int = 512, t0: float = 100.0):
    """Push audio in blocks with consistent wall-clock timestamps, return all segments (incl. flush)."""
    out = []
    for start in range(0, len(audio), block):
        chunk = audio[start:start + block]
        out += seg.push(chunk, t0 + (start + len(chunk)) / SR)
    return out + seg.flush()


def test_one_phrase_between_silences():
    audio = np.concatenate([silence(1.0), tone(2.0), silence(1.0)])
    segs = feed(Segmenter("mic", energy_vad), audio)
    assert len(segs) == 1
    s = segs[0]
    assert s.source == "mic" and not s.forced_cut
    assert 2.0 <= s.duration <= 2.0 + 2 * 0.16 + 0.07  # speech + padding on both sides (frame rounding)
    # end of phrase = end of the tone at t0 + 3.0 s, within one frame
    assert s.end_wall == pytest.approx(100.0 + 3.0, abs=0.04)
    assert s.start_wall == pytest.approx(100.0 + 1.0 - 0.16, abs=0.05)


def test_two_phrases_with_short_pause_merge_long_pause_splits():
    short_pause = np.concatenate([tone(1.0), silence(0.3), tone(1.0), silence(1.0)])
    assert len(feed(Segmenter("mic", energy_vad, min_silence_ms=500), short_pause)) == 1
    long_pause = np.concatenate([tone(1.0), silence(0.8), tone(1.0), silence(1.0)])
    assert len(feed(Segmenter("mic", energy_vad, min_silence_ms=500), long_pause)) == 2


def test_short_blip_is_dropped_and_counted():
    seg = Segmenter("system", energy_vad, min_speech_ms=400)
    segs = feed(seg, np.concatenate([silence(0.5), tone(0.2), silence(1.0)]))
    assert segs == []
    assert seg.dropped_short == 1


def test_long_speech_is_force_cut():
    seg = Segmenter("system", energy_vad, max_segment_s=8.0)
    segs = feed(seg, np.concatenate([tone(20.0), silence(1.0)]))
    assert [s.forced_cut for s in segs] == [True, True, False]
    assert all(s.duration <= 8.0 + 0.2 for s in segs)
    assert sum(s.duration for s in segs) == pytest.approx(20.0, abs=0.5)


@pytest.mark.parametrize("block", [160, 512, 1000, 4800])
def test_block_size_does_not_change_result(block):
    audio = np.concatenate([silence(0.7), tone(1.5), silence(0.9), tone(1.2), silence(1.0)])
    ref = feed(Segmenter("mic", energy_vad), audio, block=512)
    got = feed(Segmenter("mic", energy_vad), audio, block=block)
    assert len(got) == len(ref) == 2
    for a, b in zip(ref, got):
        assert np.array_equal(a.audio, b.audio)
        assert a.end_wall == pytest.approx(b.end_wall, abs=1e-6)


def test_flush_emits_pending_speech():
    seg = Segmenter("mic", energy_vad)
    out = []
    audio = np.concatenate([silence(0.3), tone(1.0)])
    for start in range(0, len(audio), 512):
        out += seg.push(audio[start:start + 512], 1.0)
    assert out == []
    assert len(seg.flush()) == 1


def test_rejects_wrong_dtype():
    with pytest.raises(ValueError):
        Segmenter("mic", energy_vad).push(np.zeros(512, dtype=np.float64), 0.0)
