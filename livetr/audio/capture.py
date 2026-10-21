"""Audio sources. Each source puts raw blocks on a queue; conversion to 16 kHz mono happens in `Converter`."""
from __future__ import annotations

import queue
import threading
import time
import wave
from dataclasses import dataclass

import numpy as np

from livetr import SAMPLE_RATE


@dataclass
class RawBlock:
    audio: np.ndarray   # float32, shape (frames, channels), at the source's own sample rate
    end_wall: float     # time.monotonic() when the block was delivered (~ its last sample)


class Converter:
    """Down-mix to mono and resample to 16 kHz without block-edge artifacts (streaming soxr)."""

    def __init__(self, in_rate: int):
        self.in_rate = in_rate
        self._rs = None
        if in_rate != SAMPLE_RATE:
            import soxr

            self._rs = soxr.ResampleStream(in_rate, SAMPLE_RATE, 1, dtype="float32")

    def __call__(self, block: np.ndarray) -> np.ndarray:
        mono = block.mean(axis=1).astype(np.float32) if block.ndim == 2 else block.astype(np.float32)
        if self._rs is None:
            return mono
        return self._rs.resample_chunk(mono)


class DeviceSource:
    """One live input device (microphone or BlackHole loopback)."""

    def __init__(self, name: str, device_index: int, blocksize_ms: int = 32):
        import sounddevice as sd

        info = sd.query_devices(device_index)
        self.name = name
        self.device_index = device_index
        self.device_name = info["name"]
        self.sample_rate = int(info["default_samplerate"])
        self.channels = min(int(info["max_input_channels"]), 2)
        self.blocks: queue.Queue[RawBlock] = queue.Queue()
        self.overflows = 0
        self._stream = sd.InputStream(
            device=device_index,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=int(self.sample_rate * blocksize_ms / 1000),
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status) -> None:
        # Runs on the PortAudio thread: only copy and enqueue, no heavy work here.
        if status.input_overflow:
            self.overflows += 1
        self.blocks.put(RawBlock(indata.copy(), time.monotonic()))

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()

    def describe(self) -> str:
        return f"{self.name}: [{self.device_index}] {self.device_name} ({self.channels} ch, {self.sample_rate} Hz)"


class FileSource:
    """Plays a 16-bit PCM WAV file into the pipeline at real-time speed, as if it came from a device.

    Used for the walking skeleton, for latency tests and for rehearsals without a call.
    """

    def __init__(self, name: str, path: str, blocksize_ms: int = 32, realtime: bool = True):
        with wave.open(path, "rb") as w:
            if w.getsampwidth() != 2:
                raise ValueError(
                    f"{path}: only 16-bit PCM WAV is supported. Convert with: "
                    f"afconvert -f WAVE -d LEI16 in.m4a out.wav"
                )
            self.sample_rate = w.getframerate()
            self.channels = w.getnchannels()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        self.name = name
        self.path = path
        self.audio = (pcm.reshape(-1, self.channels).astype(np.float32) / 32768.0)
        self.blocksize = int(self.sample_rate * blocksize_ms / 1000)
        self.realtime = realtime
        self.blocks: queue.Queue[RawBlock] = queue.Queue()
        self.overflows = 0
        self.done = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"file-{name}", daemon=True)
        self._stop = threading.Event()

    def _run(self) -> None:
        t0 = time.monotonic()
        for start in range(0, len(self.audio), self.blocksize):
            if self._stop.is_set():
                break
            block = self.audio[start:start + self.blocksize]
            if self.realtime:
                due = t0 + (start + len(block)) / self.sample_rate
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            self.blocks.put(RawBlock(block, time.monotonic()))
        # One second of silence so the VAD sees the final pause.
        tail = np.zeros((self.sample_rate, self.channels), dtype=np.float32)
        if self.realtime:
            time.sleep(1.0)
        self.blocks.put(RawBlock(tail, time.monotonic()))
        self.done.set()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def describe(self) -> str:
        return f"{self.name}: file {self.path} ({self.channels} ch, {self.sample_rate} Hz)"
