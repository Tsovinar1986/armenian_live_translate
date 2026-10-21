"""R4: Me for the mic; different call-side voices get different, stable labels; short segments stay honest."""
import numpy as np
import pytest

from livetr.speakers import UNKNOWN, OnlineSpeakerClusterer, SpeakerLabeler
from livetr.types import MIC, SYSTEM, Segment

DIM = 192


def voices(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, DIM))


def sample(base: np.ndarray, rng, noise: float = 0.6) -> np.ndarray:
    # noise 0.6 per dim -> cosine to own voice ≈ 0.85, to other voices ≈ 0 (like a clean ECAPA case)
    return base + rng.normal(scale=noise, size=base.shape)


def test_three_voices_get_three_stable_labels():
    rng = np.random.default_rng(1)
    v = voices(3)
    cl = OnlineSpeakerClusterer(threshold=0.45)
    seq = rng.integers(0, 3, size=60)
    seen: dict[int, str] = {}
    for who in seq:
        label = cl.assign(sample(v[who], rng), duration_s=2.0)
        assert seen.setdefault(int(who), label) == label  # same person -> same label every time
    assert cl.n_speakers == 3
    assert len(set(seen.values())) == 3


def test_labels_are_numbered_in_order_of_appearance():
    rng = np.random.default_rng(2)
    v = voices(2)
    cl = OnlineSpeakerClusterer()
    assert cl.assign(sample(v[1], rng), 2.0) == "Speaker 1"
    assert cl.assign(sample(v[0], rng), 2.0) == "Speaker 2"
    assert cl.assign(sample(v[1], rng), 2.0) == "Speaker 1"


def test_short_segment_never_creates_a_speaker():
    rng = np.random.default_rng(3)
    v = voices(2)
    cl = OnlineSpeakerClusterer(min_duration_s=1.0)
    assert cl.assign(sample(v[0], rng), duration_s=0.5) == UNKNOWN
    assert cl.n_speakers == 0
    cl.assign(sample(v[0], rng), 2.0)
    # short but clearly the same voice -> known label; short and a new voice -> unknown
    assert cl.assign(sample(v[0], rng, noise=0.2), 0.5) == "Speaker 1"
    assert cl.assign(sample(v[1], rng), 0.5) == UNKNOWN
    assert cl.n_speakers == 1


def test_speaker_limit_gives_unknown_not_a_forced_match():
    rng = np.random.default_rng(4)
    v = voices(3)
    cl = OnlineSpeakerClusterer(max_speakers=2)
    cl.assign(sample(v[0], rng), 2.0)
    cl.assign(sample(v[1], rng), 2.0)
    assert cl.assign(sample(v[2], rng), 2.0) == UNKNOWN


def test_zero_embedding_fails_loudly():
    with pytest.raises(ValueError):
        OnlineSpeakerClusterer().assign(np.zeros(DIM), 2.0)


class ExplodingEmbedder:
    def embed(self, audio):
        raise AssertionError("mic audio must not be embedded")


class FixedEmbedder:
    def __init__(self, vec):
        self.vec = vec

    def embed(self, audio):
        return self.vec


def seg(source: str, seconds: float = 2.0) -> Segment:
    return Segment(source, np.zeros(int(16000 * seconds), dtype=np.float32), 0.0, seconds)


def test_mic_is_always_me():
    lab = SpeakerLabeler(ExplodingEmbedder(), OnlineSpeakerClusterer())
    assert lab.label(seg(MIC)) == "Me"


def test_system_side_goes_through_clustering():
    lab = SpeakerLabeler(FixedEmbedder(voices(1)[0]), OnlineSpeakerClusterer())
    assert lab.label(seg(SYSTEM)) == "Speaker 1"
    assert lab.label(seg(SYSTEM, 0.5)) == "Speaker 1"  # identical vector -> similarity 1.0 >= strong threshold
