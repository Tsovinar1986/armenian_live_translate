"""Speaker labels: `Me` for the microphone, `Speaker N` for voices on the system-audio side."""
from __future__ import annotations

import numpy as np

from livetr import SAMPLE_RATE
from livetr.types import MIC, Segment

UNKNOWN = "Speaker ?"


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero-length speaker embedding")
    return v / n


class OnlineSpeakerClusterer:
    """Assign each embedding to the closest known speaker centroid, or start a new speaker.

    - Cosine similarity >= `threshold` -> same speaker; the centroid becomes the mean of its members.
    - Below threshold -> new speaker (`Speaker N`), up to `max_speakers`.
    - Short segments (< `min_duration_s`) give unreliable embeddings: they never create speakers or move
      centroids, and only get a known label if similarity >= `strong_threshold`. Otherwise `Speaker ?`.
    Labels never change once given, so the same person keeps the same label for the session.
    """

    def __init__(
        self,
        threshold: float = 0.45,
        strong_threshold: float = 0.65,
        min_duration_s: float = 1.0,
        max_speakers: int = 8,
    ):
        self.threshold = threshold
        self.strong_threshold = strong_threshold
        self.min_duration_s = min_duration_s
        self.max_speakers = max_speakers
        self._sums: list[np.ndarray] = []
        self._counts: list[int] = []

    @property
    def n_speakers(self) -> int:
        return len(self._sums)

    def _centroids(self) -> np.ndarray:
        return np.stack([_unit(s) for s in self._sums])

    def best_match(self, emb: np.ndarray) -> tuple[int, float]:
        """Index and cosine similarity of the closest speaker; (-1, -1.0) if there are none."""
        if not self._sums:
            return -1, -1.0
        sims = self._centroids() @ _unit(emb)
        i = int(np.argmax(sims))
        return i, float(sims[i])

    def assign(self, emb: np.ndarray, duration_s: float) -> str:
        emb = _unit(np.asarray(emb, dtype=np.float64))
        idx, sim = self.best_match(emb)

        if duration_s < self.min_duration_s:
            return f"Speaker {idx + 1}" if idx >= 0 and sim >= self.strong_threshold else UNKNOWN

        if idx >= 0 and sim >= self.threshold:
            self._sums[idx] = self._sums[idx] + emb
            self._counts[idx] += 1
            return f"Speaker {idx + 1}"

        if self.n_speakers >= self.max_speakers:
            return UNKNOWN
        self._sums.append(emb.copy())
        self._counts.append(1)
        return f"Speaker {self.n_speakers}"


class EcapaEmbedder:
    """speechbrain ECAPA-TDNN trained on VoxCeleb -> 192-d speaker embedding."""

    def __init__(self, savedir: str = "models/spkrec-ecapa-voxceleb"):
        import torch
        from speechbrain.inference.classifiers import EncoderClassifier

        self._torch = torch
        # CPU on purpose: small model (~20 MB), keeps the GPU free for ASR.
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir=savedir, run_opts={"device": "cpu"}
        )

    def embed(self, audio: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            wav = self._torch.from_numpy(audio).unsqueeze(0)
            emb = self.model.encode_batch(wav)  # (1, 1, 192)
        return emb.squeeze().cpu().numpy().astype(np.float64)


class SpeakerLabeler:
    def __init__(self, embedder, clusterer: OnlineSpeakerClusterer):
        self.embedder = embedder
        self.clusterer = clusterer

    def label(self, segment: Segment) -> str:
        if segment.source == MIC:
            return "Me"
        emb = self.embedder.embed(segment.audio)
        return self.clusterer.assign(emb, len(segment.audio) / SAMPLE_RATE)
