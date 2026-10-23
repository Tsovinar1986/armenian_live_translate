"""Command line entry point: python -m livetr [options]"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from livetr.asr import MODELS

DEFAULT_ASR = "buzz"  # most accurate on FLEURS; see DECISIONS.md (2026-10-23)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m livetr",
        description="Live Armenian transcription + English translation with speaker labels (local models only).",
    )
    g = p.add_argument_group("audio")
    g.add_argument("--list-devices", action="store_true", help="print input devices and exit")
    g.add_argument("--meter", action="store_true", help="show input levels for both sources (setup check) and exit")
    g.add_argument("--mic", default=None, help="microphone: device index or part of its name (default: system default input)")
    g.add_argument("--system", default="BlackHole", help="system-audio loopback device (default: 'BlackHole')")
    g.add_argument("--no-mic", action="store_true", help="do not capture the microphone")
    g.add_argument("--no-system", action="store_true", help="do not capture system audio")
    g.add_argument("--file", default=None, help="play a 16-bit WAV file in real time instead of live devices")
    g.add_argument("--file-source", choices=["mic", "system"], default="system", help="treat --file as this source")

    m = p.add_argument_group("models")
    m.add_argument("--asr", choices=sorted(MODELS), default=DEFAULT_ASR, help=f"ASR model (default {DEFAULT_ASR})")
    m.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto", help="device for ASR")
    m.add_argument("--asr-dtype", choices=["float16", "float32"], default="float16")
    m.add_argument("--nllb-dir", default="models/nllb-200-distilled-600M-ct2-int8")
    m.add_argument("--allow-download", action="store_true",
                   help="allow Hugging Face downloads at runtime (default: offline, models must be downloaded first)")

    s = p.add_argument_group("segmentation and speakers")
    s.add_argument("--vad-threshold", type=float, default=0.5)
    s.add_argument("--min-silence-ms", type=int, default=500)
    s.add_argument("--max-segment-s", type=float, default=8.0)
    s.add_argument("--speaker-threshold", type=float, default=0.45, help="cosine similarity for 'same speaker'")
    s.add_argument("--max-speakers", type=int, default=8)

    o = p.add_argument_group("output")
    o.add_argument("--no-color", action="store_true")
    o.add_argument("--quiet-timings", action="store_true", help="hide per-line latency details")
    return p


def run_meter(sources: list, seconds: float = 15.0) -> None:
    from livetr.audio.capture import Converter

    converters = {s.name: Converter(s.sample_rate) for s in sources}
    levels = {s.name: -90.0 for s in sources}
    for s in sources:
        s.start()
    print("Speak, and play some audio on the computer. Ctrl-C to stop.\n")
    end = time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            for s in sources:
                while not s.blocks.empty():
                    a = converters[s.name](s.blocks.get().audio)
                    if len(a):
                        rms = float(np.sqrt(np.mean(a ** 2)) + 1e-9)
                        levels[s.name] = max(20 * np.log10(rms), -90.0)
            line = "   ".join(f"{n:<6} {'█' * int((lv + 60) / 2.5) if lv > -60 else '':<24} {lv:6.1f} dB"
                             for n, lv in levels.items())
            print("\r" + line, end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        for s in sources:
            s.stop()
        print()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.allow_download:
        # Local-only guarantee: no network at runtime. Models come from scripts/download_models.py.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from livetr.audio import devices as dev
    from livetr.types import MIC, SYSTEM

    if args.list_devices:
        print(dev.format_input_devices(dev.query_devices()))
        return

    # ---- sources -------------------------------------------------------------------------------
    sources = []
    if args.file:
        from livetr.audio.capture import FileSource

        sources.append(FileSource(MIC if args.file_source == "mic" else SYSTEM, args.file))
    else:
        from livetr.audio.capture import DeviceSource

        devices = dev.query_devices()
        try:
            mic_idx = None if args.no_mic else (
                dev.resolve_input_device(args.mic, devices) if args.mic else dev.default_input_index())
            sys_idx = None if args.no_system else dev.resolve_input_device(args.system, devices)
        except dev.DeviceError as e:
            sys.exit(f"error: {e}")
        if mic_idx is not None and mic_idx == sys_idx:
            sys.exit("error: mic and system audio are the same device. Pass --mic with your real microphone.")
        if mic_idx is not None:
            sources.append(DeviceSource(MIC, mic_idx))
        if sys_idx is not None:
            sources.append(DeviceSource(SYSTEM, sys_idx))
    if not sources:
        sys.exit("error: no audio source selected.")
    for s in sources:
        print("audio  " + s.describe(), file=sys.stderr)

    if args.meter:
        run_meter(sources)
        return

    # ---- models (loaded once, warmed up so the first real segment is not slow) --------------------
    from livetr.asr import load_asr, pick_device
    from livetr.output import Printer, Stats
    from livetr.pipeline import Pipeline, Processor
    from livetr.segmenter import SileroVAD
    from livetr.speakers import EcapaEmbedder, OnlineSpeakerClusterer, SpeakerLabeler
    from livetr.translate import NllbTranslator

    t0 = time.monotonic()
    device = pick_device(args.device)
    print(f"loading ASR '{args.asr}' ({MODELS[args.asr].repo}) on {device} …", file=sys.stderr)
    asr = load_asr(args.asr, device, args.asr_dtype)
    print("loading NLLB translator …", file=sys.stderr)
    translator = NllbTranslator(args.nllb_dir)
    embedder = EcapaEmbedder() if any(s.name == SYSTEM for s in sources) else None
    print("warming up …", file=sys.stderr)
    asr.transcribe(np.zeros(16000, dtype=np.float32))
    translator.translate("Բարև")
    if embedder is not None:
        embedder.embed(np.random.default_rng(0).normal(0, 0.01, 16000).astype(np.float32))
    print(f"ready in {time.monotonic() - t0:.0f}s — listening. Ctrl-C to stop.\n", file=sys.stderr)

    clusterer = OnlineSpeakerClusterer(threshold=args.speaker_threshold, max_speakers=args.max_speakers)
    stats = Stats()
    printer = Printer(session_start=time.monotonic(), color=not args.no_color, details=not args.quiet_timings)
    processor = Processor(asr, translator, SpeakerLabeler(embedder, clusterer), printer, stats)
    pipeline = Pipeline(
        sources,
        vad_factory=SileroVAD,
        processor=processor,
        stats=stats,
        segmenter_kwargs=dict(threshold=args.vad_threshold, min_silence_ms=args.min_silence_ms,
                              max_segment_s=args.max_segment_s),
    )
    pipeline.start()
    try:
        pipeline.run_until_stopped()
    finally:
        print(stats.summary(), file=sys.stderr)
