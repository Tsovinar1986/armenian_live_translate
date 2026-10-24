"""Armenian text -> English text with NLLB-200 (CTranslate2, int8, CPU)."""
from __future__ import annotations

from pathlib import Path

DEFAULT_MODEL_DIR = "models/nllb-200-distilled-600M-ct2-int8"  # 1.3B too slow when both sides talk (DECISIONS.md)


class NllbTranslator:
    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        src_lang: str = "hye_Armn",
        tgt_lang: str = "eng_Latn",
        threads: int = 4,
        beam_size: int = 2,
    ):
        if not (Path(model_dir) / "model.bin").exists():
            raise FileNotFoundError(
                f"No converted NLLB model in '{model_dir}'. Run: python scripts/download_models.py"
            )
        import ctranslate2
        from transformers import AutoTokenizer

        self.tgt_lang = tgt_lang
        self.beam_size = beam_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, src_lang=src_lang)
        self.translator = ctranslate2.Translator(
            model_dir, device="cpu", compute_type="int8", intra_threads=threads
        )

    def translate(self, text: str) -> str:
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        result = self.translator.translate_batch(
            [tokens],
            target_prefix=[[self.tgt_lang]],
            beam_size=self.beam_size,
            max_decoding_length=256,
        )
        out_tokens = result[0].hypotheses[0][1:]  # drop the target-language token
        return self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(out_tokens), skip_special_tokens=True
        ).strip()
