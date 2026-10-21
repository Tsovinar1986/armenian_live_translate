"""End-to-end wiring with fake models: WAV at 48 kHz -> resample -> VAD -> worker -> printed lines."""
import io
import wave

import numpy as np
import pytest

from conftest import energy_vad
from livetr.audio.capture import FileSource
from livetr.output import Printer, Stats
from livetr.pipeline import Pipeline, Processor
from livetr.speakers import OnlineSpeakerClusterer, SpeakerLabeler
from livetr.types import Segment


class FakeAsr:
    def __init__(self, text="Բարև"):
        self.text = text
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        return self.text


class FakeTranslator:
    def __init__(self):
        self.calls = 0

    def translate(self, text):
        self.calls += 1
        return f"<en:{text}>"


class FakeEmbedder:
    def embed(self, audio):
        return np.ones(192)


def make_processor(asr, tr, buf):
    stats = Stats()
    labeler = SpeakerLabeler(FakeEmbedder(), OnlineSpeakerClusterer())
    return Processor(asr, tr, labeler, Printer(0.0, stream=buf, color=False), stats), stats


def test_empty_asr_prints_nothing_and_skips_translation():
    buf, tr = io.StringIO(), FakeTranslator()
    proc, stats = make_processor(FakeAsr(text="  "), tr, buf)
    seg = Segment("mic", np.zeros(16000, dtype=np.float32), 0.0, 1.0)
    assert proc.process(seg) is None
    assert tr.calls == 0 and buf.getvalue() == "" and stats.empty_asr == 1


def test_processor_records_stage_timings():
    ticks = iter([10.0, 11.5, 11.6, 11.9])
    buf = io.StringIO()
    proc, _ = make_processor(FakeAsr(), FakeTranslator(), buf)
    proc.clock = lambda: next(ticks)
    r = proc.process(Segment("mic", np.zeros(16000, dtype=np.float32), 8.0, 9.0))
    assert r.speaker == "Me"
    assert r.timings == pytest.approx({"wait": 1.0, "asr": 1.5, "spk": 0.1, "mt": 0.3})
    assert r.latency == pytest.approx(2.9)  # printed at 11.9, phrase ended at 9.0


def write_wav(path, audio, sr):
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.repeat(pcm[:, None], 2, axis=1).tobytes())


def test_file_through_full_pipeline(tmp_path):
    sr = 48000
    t = lambda s, f=220: (0.3 * np.sin(2 * np.pi * f * np.arange(int(s * sr)) / sr)).astype(np.float32)
    z = lambda s: np.zeros(int(s * sr), dtype=np.float32)
    audio = np.concatenate([z(0.5), t(1.5), z(1.0), t(2.0), z(0.5)])
    path = tmp_path / "two_phrases.wav"
    write_wav(path, audio, sr)

    buf = io.StringIO()
    asr, tr = FakeAsr(), FakeTranslator()
    proc, stats = make_processor(asr, tr, buf)
    src = FileSource("system", str(path), realtime=False)
    pipe = Pipeline([src], vad_factory=lambda: energy_vad, processor=proc, stats=stats)
    pipe.start()
    pipe.run_until_stopped()

    out = buf.getvalue()
    assert asr.calls == 2 and tr.calls == 2
    assert out.count("Speaker 1") == 2  # same fake voice -> same label
    assert "HY: Բարև" in out and "EN: <en:Բարև>" in out
    assert sum(len(v) for v in stats.latencies.values()) == 2
