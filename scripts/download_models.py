"""Download every model once, so the program can run fully offline.

    python scripts/download_models.py                 # default ASR + NLLB + speaker + VAD
    python scripts/download_models.py --asr buzz mms  # extra ASR candidates for benchmarking
    python scripts/download_models.py --asr all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livetr.asr import MODELS  # noqa: E402
from livetr.cli import DEFAULT_ASR  # noqa: E402

NLLB_REPO = "facebook/nllb-200-distilled-600M"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--asr", nargs="+", default=[DEFAULT_ASR], help=f"keys from {list(MODELS)} or 'all'")
    p.add_argument("--nllb-dir", default="models/nllb-200-distilled-600M-ct2-int8")
    args = p.parse_args()
    keys = list(MODELS) if args.asr == ["all"] else args.asr

    from huggingface_hub import snapshot_download

    for key in keys:
        spec = MODELS[key]
        print(f"[asr] {key}: {spec.repo}")
        # MMS ships ~1,100 language adapters; fetch only the shared weights and the Armenian one.
        allow = None if spec.kind == "whisper" else ["*.json", "*.txt", "model*.safetensors", f"adapter.{spec.language}.*"]
        snapshot_download(spec.repo, allow_patterns=allow)

    out = Path(args.nllb_dir)
    if (out / "model.bin").exists():
        print(f"[mt] {out} already exists")
    else:
        import ctranslate2
        from transformers import AutoTokenizer

        print(f"[mt] converting {NLLB_REPO} to CTranslate2 int8 -> {out}")
        ctranslate2.converters.TransformersConverter(NLLB_REPO).convert(str(out), quantization="int8")
        AutoTokenizer.from_pretrained(NLLB_REPO).save_pretrained(str(out))

    print("[spk] speechbrain/spkrec-ecapa-voxceleb")
    from livetr.speakers import EcapaEmbedder

    EcapaEmbedder()

    print("[vad] silero (ships inside the pip package)")
    from livetr.segmenter import SileroVAD

    SileroVAD()
    print("done. The program now runs offline.")


if __name__ == "__main__":
    main()
