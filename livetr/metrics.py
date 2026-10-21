"""CER / WER for Armenian, with a fixed normalisation so numbers are comparable between models."""
from __future__ import annotations

import re
import unicodedata

# Armenian stress/exclamation/question marks (՚ ՛ ՜ ՞ ՟) are written INSIDE the word (ինչպե՞ս): delete them.
_IN_WORD = re.compile(r"[՚-՜՞՟]")
# Separating punctuation: ՝ (comma) ։ (full stop) ֊ (hyphen) « » and ASCII punctuation: replace with a space.
_PUNCT = re.compile(r"[՝։֊«»…\"'`.,!?;:()\[\]{}\-–—/\\]")
_SPACE = re.compile(r"\s+")


def normalize_hy(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace("և", "եւ")  # the ligature and the two letters are the same word
    text = _IN_WORD.sub("", text)
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def edit_distance(ref: list, hyp: list) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer_counts(ref: str, hyp: str) -> tuple[int, int]:
    """(errors, reference words) after normalisation."""
    r, h = normalize_hy(ref).split(), normalize_hy(hyp).split()
    return edit_distance(r, h), len(r)


def cer_counts(ref: str, hyp: str) -> tuple[int, int]:
    """(errors, reference characters) after normalisation; spaces are ignored."""
    r, h = list(normalize_hy(ref).replace(" ", "")), list(normalize_hy(hyp).replace(" ", ""))
    return edit_distance(r, h), len(r)
