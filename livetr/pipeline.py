"""Wires sources -> converter -> VAD segmenter -> shared queue -> one worker (ASR, speaker, translation) -> printer.

Threads:
  - one PortAudio callback per device (inside sounddevice) puts raw blocks on the source's queue
  - one `source loop` thread per source: down-mix, resample, VAD, push finished segments to `segments`
  - one `worker` thread: holds the models and processes segments one at a time (models load once: 8 GB RAM)
Any exception in a thread stops the whole program with the traceback (fail loudly).
"""
from __future__ import annotations

import queue
import sys
import threading
import time
import traceback
from typing import Callable, Protocol

import numpy as np

from livetr.audio.capture import Converter
from livetr.output import Printer, Stats
from livetr.segmenter import Segmenter
from livetr.types import Result, Segment


class Asr(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...


class Translator(Protocol):
    def translate(self, text: str) -> str: ...


class Labeler(Protocol):
    def label(self, segment: Segment) -> str: ...


class Processor:
    """Turns one segment into one printed result. No threads here, so it can be tested directly."""

    def __init__(self, asr: Asr, translator: Translator, labeler: Labeler, printer: Printer, stats: Stats,
                 clock: Callable[[], float] = time.monotonic):
        self.asr = asr
        self.translator = translator
        self.labeler = labeler
        self.printer = printer
        self.stats = stats
        self.clock = clock

    def process(self, segment: Segment) -> Result | None:
        timings: dict[str, float] = {}
        t = self.clock()
        timings["wait"] = max(0.0, t - segment.end_wall)

        hy = self.asr.transcribe(segment.audio).strip()
        t2 = self.clock()
        timings["asr"] = t2 - t
        if not hy:
            self.stats.empty_asr += 1
            return None

        # Speaker after ASR: segments with no recognised speech must not shape the speaker clusters.
        speaker = self.labeler.label(segment)
        t3 = self.clock()
        timings["spk"] = t3 - t2

        en = self.translator.translate(hy)
        t4 = self.clock()
        timings["mt"] = t4 - t3

        result = Result(segment=segment, speaker=speaker, hy=hy, en=en, printed_wall=t4, timings=timings)
        self.printer.show(result)
        self.stats.add(result)
        return result


class Pipeline:
    def __init__(self, sources: list, vad_factory: Callable[[], Callable[[np.ndarray], float]],
                 processor: Processor, stats: Stats, segmenter_kwargs: dict | None = None):
        self.sources = sources
        self.processor = processor
        self.stats = stats
        self.segments: queue.Queue[Segment] = queue.Queue()
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.segmenters = {s.name: Segmenter(s.name, vad_factory(), **(segmenter_kwargs or {})) for s in sources}
        self._threads: list[threading.Thread] = []

    def _guard(self, fn: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            try:
                fn()
            except BaseException as e:  # noqa: BLE001 - re-raised in the main thread
                self.error = e
                traceback.print_exc()
                self.stop_event.set()
        return run

    def _source_loop(self, source) -> None:
        convert = Converter(source.sample_rate)
        seg = self.segmenters[source.name]
        while not self.stop_event.is_set():
            try:
                block = source.blocks.get(timeout=0.1)
            except queue.Empty:
                if getattr(source, "done", None) is not None and source.done.is_set():
                    for s in seg.flush():
                        self.segments.put(s)
                    return
                continue
            audio = convert(block.audio)
            if len(audio):
                for s in seg.push(audio, block.end_wall):
                    self.segments.put(s)

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                segment = self.segments.get(timeout=0.1)
            except queue.Empty:
                continue
            self.processor.process(segment)
            self.segments.task_done()

    def start(self) -> None:
        for src in self.sources:
            t = threading.Thread(target=self._guard(lambda s=src: self._source_loop(s)),
                                 name=f"src-{src.name}", daemon=True)
            self._threads.append(t)
        self._threads.append(threading.Thread(target=self._guard(self._worker), name="worker", daemon=True))
        for t in self._threads:
            t.start()
        for src in self.sources:
            src.start()

    def files_finished(self) -> bool:
        """True when every source is a file that has been fully played and processed."""
        if not all(getattr(s, "done", None) is not None and s.done.is_set() for s in self.sources):
            return False
        source_threads = [t for t in self._threads if t.name.startswith("src-")]
        return all(not t.is_alive() for t in source_threads) and self.segments.unfinished_tasks == 0

    def run_until_stopped(self) -> None:
        try:
            while not self.stop_event.is_set():
                if self.files_finished():
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nstopping…", file=sys.stderr)
        finally:
            self.stop_event.set()
            for src in self.sources:
                src.stop()
            self.stats.dropped_short = sum(s.dropped_short for s in self.segmenters.values())
            self.stats.overflows = sum(s.overflows for s in self.sources)
        if self.error is not None:
            raise SystemExit(f"stopped because of an error in a pipeline thread: {self.error!r}")
