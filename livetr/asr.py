"""Armenian speech -> Armenian text. Several candidate models behind one interface so they can be benchmarked."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from livetr import SAMPLE_RATE


@dataclass(frozen=True)
class AsrSpec:
    repo: str
    kind: str                 # "whisper" or "mms"
    language: str | None      # Whisper language code, MMS adapter id, or None if baked into the model
    note: str


MODELS: dict[str, AsrSpec] = {
    "buzz": AsrSpec(
        "BuzzASR/armenian", "whisper", None,
        "Whisper large-v3 full fine-tune for Armenian (language prompt is baked into its generation config)",
    ),
    "turbo-hy": AsrSpec(
        "Chillarmo/whisper-large-v3-turbo-armenian", "whisper", "hy",
        "Whisper large-v3-turbo fine-tuned on Common Voice Armenian",
    ),
    "turbo-zeroshot": AsrSpec(
        "openai/whisper-large-v3-turbo", "whisper", "hy",
        "Original Whisper turbo, no Armenian fine-tune (reference point)",
    ),
    "mms": AsrSpec(
        "facebook/mms-1b-all", "mms", "hye",
        "Meta MMS 1B CTC model with the Armenian adapter",
    ),
}


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS is not available on this machine.")
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


class WhisperASR:
    def __init__(self, repo: str, language: str | None, device: str, dtype: str = "float16"):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self._torch = torch
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.language = language
        self.processor = WhisperProcessor.from_pretrained(repo)
        self.model = WhisperForConditionalGeneration.from_pretrained(repo, dtype=self.dtype).to(device).eval()

    def transcribe(self, audio: np.ndarray) -> str:
        feats = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        feats = feats.to(self.device, self.dtype)
        kwargs = dict(
            num_beams=1,                 # greedy: beam search costs latency we don't have
            no_repeat_ngram_size=3,      # stops the classic Whisper repetition loop
            # 4 prompt tokens + 224 text tokens. Armenian needs ~15 tokens/s, up to 21 on FLEURS; 160 truncated
            # 4 of 40 clips mid-letter. 224 covers 8 s at 28 tokens/s. max_length, not max_new_tokens: the
            # saved config also sets max_length, and having both prints a warning for every phrase.
            max_length=228,
        )
        if self.language is not None:
            kwargs.update(language=self.language, task="transcribe")
        with self._torch.no_grad():
            ids = self.model.generate(feats, **kwargs)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


class MmsASR:
    def __init__(self, repo: str, adapter: str, device: str, dtype: str = "float16"):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self._torch = torch
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(repo)
        self.processor.tokenizer.set_target_lang(adapter)
        model = Wav2Vec2ForCTC.from_pretrained(repo)
        model.load_adapter(adapter)
        self.model = model.to(device, self.dtype).eval()

    def transcribe(self, audio: np.ndarray) -> str:
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        values = inputs.input_values.to(self.device, self.dtype)
        with self._torch.no_grad():
            logits = self.model(values).logits
        ids = logits.argmax(dim=-1)[0]
        return self.processor.decode(ids).strip()


def load_asr(key: str, device: str, dtype: str = "float16"):
    if key not in MODELS:
        raise ValueError(f"Unknown ASR model '{key}'. Choose from: {', '.join(MODELS)}")
    spec = MODELS[key]
    if spec.kind == "whisper":
        return WhisperASR(spec.repo, spec.language, device, dtype)
    if spec.kind == "mms":
        return MmsASR(spec.repo, spec.language, device, dtype)
    raise ValueError(f"Unknown model kind {spec.kind}")
