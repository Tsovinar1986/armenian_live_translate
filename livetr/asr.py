"""Armenian speech -> Armenian text with NVIDIA's Armenian FastConformer (NeMo).

Chosen by measurement on FLEURS (see DECISIONS.md, 2026-10-23). The Whisper and MMS candidates it was compared
against were removed from the code afterwards; their numbers stay in DECISIONS.md and eval/results/.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AsrSpec:
    repo: str
    note: str


MODELS: dict[str, AsrSpec] = {
    "nemo-hy": AsrSpec(
        "nvidia/stt_hy_fastconformer_hybrid_large_pc",
        "NVIDIA FastConformer hybrid RNNT/CTC (~115M) trained for Armenian, with punctuation and capitals",
    ),
}


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS is not available on this machine.")
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


class NemoASR:
    def __init__(self, repo: str, device: str):
        from pathlib import Path

        from huggingface_hub import snapshot_download
        from nemo.collections.asr.models import ASRModel

        files = sorted(Path(snapshot_download(repo, allow_patterns=["*.nemo"])).glob("*.nemo"))
        if len(files) != 1:
            raise RuntimeError(f"expected one .nemo file in {repo}, found {[f.name for f in files]}")
        self.model = ASRModel.restore_from(str(files[0]), map_location=device).eval()

    def transcribe(self, audio: np.ndarray) -> str:
        out = self.model.transcribe([audio], batch_size=1, verbose=False)
        return out[0].text.strip()


def load_asr(key: str, device: str) -> NemoASR:
    if key not in MODELS:
        raise ValueError(f"Unknown ASR model '{key}'. Choose from: {', '.join(MODELS)}")
    return NemoASR(MODELS[key].repo, device)
