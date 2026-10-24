"""Cut a continuous 16 kHz stream into utterances using a voice-activity probability per frame.

Pure logic: the VAD is passed in as a function, so this module is testable without any model.
"""
from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np

from livetr import SAMPLE_RATE
from livetr.types import Segment

VadFn = Callable[[np.ndarray], float]  # frame (float32, frame_size samples) -> speech probability 0..1


class Segmenter:
    def __init__(
        self,
        source: str,
        vad: VadFn,
        *,
        frame_size: int = 512,          # Silero VAD requires exactly 512 samples at 16 kHz (32 ms)
        threshold: float = 0.5,         # prob >= threshold starts speech
        neg_threshold: float | None = None,  # prob < neg_threshold counts as silence (hysteresis)
        min_silence_ms: int = 500,      # pause that ends an utterance
        speech_pad_ms: int = 160,       # audio kept before speech start and after speech end
        min_speech_ms: int = 400,       # shorter utterances are dropped (clicks, coughs)
        max_segment_s: float = 15.0,    # force a cut so ASR time stays bounded (see DECISIONS.md)
        cut_search_ms: int = 1000,      # a forced cut goes at the quietest frame of this last stretch
    ):
        self.source = source
        self.vad = vad
        self.frame_size = frame_size
        self.frame_s = frame_size / SAMPLE_RATE
        self.threshold = threshold
        self.neg_threshold = threshold - 0.15 if neg_threshold is None else neg_threshold
        self.min_silence_frames = max(1, round(min_silence_ms / 1000 / self.frame_s))
        self.pad_frames = round(speech_pad_ms / 1000 / self.frame_s)
        self.min_speech_frames = round(min_speech_ms / 1000 / self.frame_s)
        self.max_frames = round(max_segment_s / self.frame_s)
        self.cut_search_frames = max(1, min(round(cut_search_ms / 1000 / self.frame_s), self.max_frames - 1))

        self.dropped_short = 0  # counter, reported in --stats
        self._buf = np.zeros(0, dtype=np.float32)
        self._pre: deque[np.ndarray] = deque(maxlen=max(self.pad_frames, 1))
        self._reset_segment()

    def _reset_segment(self) -> None:
        self._in_speech = False
        self._frames: list[np.ndarray] = []
        self._probs: list[float] = []
        self._start_wall = 0.0
        self._speech_frames = 0
        self._silence_run = 0
        self._last_speech_idx = -1
        self._last_speech_end = 0.0

    def push(self, audio: np.ndarray, end_wall: float) -> list[Segment]:
        """Feed audio of any length. `end_wall` is the monotonic time of its last sample."""
        if audio.dtype != np.float32 or audio.ndim != 1:
            raise ValueError(f"expected 1-D float32 audio, got {audio.dtype} with shape {audio.shape}")
        self._buf = np.concatenate([self._buf, audio])
        n_frames = len(self._buf) // self.frame_size
        leftover = len(self._buf) - n_frames * self.frame_size
        out: list[Segment] = []
        for i in range(n_frames):
            frame = self._buf[i * self.frame_size:(i + 1) * self.frame_size]
            samples_after = (n_frames - i - 1) * self.frame_size + leftover
            frame_end = end_wall - samples_after / SAMPLE_RATE
            seg = self._step(frame, frame_end)
            if seg is not None:
                out.append(seg)
        self._buf = self._buf[n_frames * self.frame_size:].copy()
        return out

    def flush(self) -> list[Segment]:
        """End of stream: emit whatever speech is pending."""
        if not self._in_speech:
            return []
        seg = self._emit(forced=False)
        return [seg] if seg is not None else []

    def _step(self, frame: np.ndarray, frame_end: float) -> Segment | None:
        prob = self.vad(frame)
        if not self._in_speech:
            if prob >= self.threshold:
                self._in_speech = True
                self._frames = list(self._pre) + [frame]
                self._probs = [0.0] * len(self._pre) + [prob]
                self._start_wall = frame_end - len(self._frames) * self.frame_s
                self._speech_frames = 1
                self._silence_run = 0
                self._last_speech_idx = len(self._frames) - 1
                self._last_speech_end = frame_end
                self._pre.clear()
            else:
                self._pre.append(frame)
            return None

        self._frames.append(frame)
        self._probs.append(prob)
        if prob >= self.neg_threshold:
            self._speech_frames += 1
            self._silence_run = 0
            self._last_speech_idx = len(self._frames) - 1
            self._last_speech_end = frame_end
        else:
            self._silence_run += 1

        if self._silence_run >= self.min_silence_frames:
            trailing = self._frames[self._last_speech_idx + 1:]
            seg = self._emit(forced=False)
            self._pre.extend(trailing)  # silence after the cut can pre-roll the next utterance
            return seg
        if len(self._frames) >= self.max_frames:
            return self._forced_cut(frame_end)
        return None

    def _forced_cut(self, frame_end: float) -> Segment | None:
        """The phrase is not over, but waiting longer breaks the latency budget. Cut after the quietest frame
        of the last `cut_search_ms` (lowest RMS: usually a gap between words) instead of at the current frame,
        so words are not split. Only a clear dip counts (below half the window's median); without one, cut here.
        The frames after the cut start the next segment."""
        window = self._frames[-self.cut_search_frames:]
        rms = np.array([float(np.sqrt(np.mean(f ** 2))) for f in window])
        quietest = len(rms) - 1 - int(np.argmin(rms[::-1]))  # latest frame on ties
        if rms[quietest] >= 0.5 * float(np.median(rms)):
            quietest = len(rms) - 1
        cut = len(self._frames) - len(window) + quietest  # last frame of the emitted segment
        rest, rest_probs = self._frames[cut + 1:], self._probs[cut + 1:]
        cut_wall = frame_end - len(rest) * self.frame_s

        self._frames, self._probs = self._frames[:cut + 1], self._probs[:cut + 1]
        self._speech_frames = sum(p >= self.neg_threshold for p in self._probs)
        self._last_speech_idx = cut
        self._last_speech_end = cut_wall
        seg = self._emit(forced=True)

        self._in_speech = True  # speech continues into the next segment
        self._frames, self._probs = rest, rest_probs
        self._start_wall = cut_wall
        self._speech_frames = sum(p >= self.neg_threshold for p in rest_probs)
        self._last_speech_idx = len(rest) - 1
        self._last_speech_end = frame_end
        return seg

    def _emit(self, forced: bool) -> Segment | None:
        end_idx = self._last_speech_idx + 1 + (0 if forced else self.pad_frames)
        frames = self._frames[:end_idx]
        speech_frames = self._speech_frames
        start_wall, end_wall = self._start_wall, self._last_speech_end
        self._reset_segment()
        if speech_frames < self.min_speech_frames:
            self.dropped_short += 1
            return None
        return Segment(
            source=self.source,
            audio=np.concatenate(frames).astype(np.float32),
            start_wall=start_wall,
            end_wall=end_wall,
            forced_cut=forced,
        )


class SileroVAD:
    """Silero VAD wrapper. Stateful: use one instance per audio source."""

    def __init__(self) -> None:
        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        self._model = load_silero_vad()

    def __call__(self, frame: np.ndarray) -> float:
        with self._torch.no_grad():
            return float(self._model(self._torch.from_numpy(frame), SAMPLE_RATE).item())
