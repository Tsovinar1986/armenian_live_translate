"""Numbers and questions (R2): do digits and ՞ come out readable, in Armenian and in English?

    python eval/run_numbers.py [--n 30]

FLEURS Armenian test sentences that contain digits (one recording per sentence, <= 15 s, random sample), plus all
sentences with the question mark ՞. FLEURS has no Երևան or Գյումրի: names are checked on own recordings
(eval/sentences.tsv s09, s15). Per clip: are all numbers of the reference in the ASR text and in the English text,
and is ՞ kept / does the English end in "?". Writes eval/results/numbers_fleurs.tsv (judge the text by eye too).
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import random
import re
import sys
import tarfile
from pathlib import Path

from common import RESULTS, ROOT

sys.path.insert(0, str(ROOT))
from livetr.metrics import cer_counts  # noqa: E402

NUM = re.compile(r"\d+(?:[.,:]\d+)*")


def numbers(text: str) -> list[str]:
    return [n.replace(",", "").replace(".", "") for n in NUM.findall(text)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30, help="sentences with digits to test")
    p.add_argument("--nllb-dir", default="models/nllb-200-distilled-1.3B-ct2-int8")
    args = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import soundfile as sf
    from huggingface_hub import hf_hub_download

    from livetr.asr import load_asr
    from livetr.translate import NllbTranslator

    tsv = hf_hub_download("google/fleurs", "data/hy_am/test.tsv", repo_type="dataset")
    with open(tsv, encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    by_sentence: dict[str, list[str]] = {}
    for r in rows:
        if int(r[5]) / 16000 <= 15:
            by_sentence.setdefault(r[0], r)
    digits = sorted(r for r in by_sentence.values() if NUM.search(r[2]))
    questions = [r for r in by_sentence.values() if "՞" in r[2]]
    chosen = random.Random(0).sample(digits, min(args.n, len(digits))) + [r for r in questions if r not in digits]
    wanted = {r[1]: r for r in chosen}

    archive = hf_hub_download("google/fleurs", "data/hy_am/audio/test.tar.gz", repo_type="dataset")
    audio = {}
    with tarfile.open(archive, "r:gz") as tar:
        for m in tar:
            name = Path(m.name).name
            if m.isfile() and name in wanted:
                audio[name], sr = sf.read(io.BytesIO(tar.extractfile(m).read()), dtype="float32")
                if sr != 16000:
                    raise ValueError(f"{name}: expected 16 kHz")
    if set(audio) != set(wanted):
        raise SystemExit(f"missing clips in the archive: {sorted(set(wanted) - set(audio))[:3]}")

    asr = load_asr("nemo-hy", "cpu")
    mt = NllbTranslator(str(ROOT / args.nllb_dir))
    out = RESULTS / "numbers_fleurs.tsv"
    hy_ok = en_ok = n_num = q_kept = q_en = n_q = 0
    c_err = c_tot = 0
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["file", "kind", "numbers_ref", "numbers_hy_ok", "numbers_en_ok", "question_kept", "question_en",
                    "reference", "hypothesis", "english"])
        for r in chosen:
            hy = asr.transcribe(audio[r[1]]).strip()
            en = mt.translate(hy)
            ce, ct = cer_counts(r[2], hy)
            c_err, c_tot = c_err + ce, c_tot + ct
            ref_nums = numbers(r[2])
            hy_nums, en_nums = numbers(hy), numbers(en)
            ok_hy = all(n in hy_nums for n in ref_nums)
            ok_en = all(n in en_nums for n in ref_nums)
            is_q = "՞" in r[2]
            kind = "question" if is_q else "digits"
            if ref_nums:
                n_num += 1
                hy_ok += ok_hy
                en_ok += ok_en
            if is_q:
                n_q += 1
                q_kept += "՞" in hy
                q_en += en.rstrip().endswith("?")
            w.writerow([r[1], kind, " ".join(ref_nums), ok_hy if ref_nums else "", ok_en if ref_nums else "",
                        ("՞" in hy) if is_q else "", en.rstrip().endswith("?") if is_q else "", r[2], hy, en])
            print(f"{kind:<8} nums {' '.join(ref_nums) or '-':<16} hy {'ok' if ok_hy else 'MISS'}  "
                  f"en {'ok' if ok_en else 'MISS'}  | {hy[:70]}")
    print(f"\nsentences with numbers: {n_num}; all numbers right in Armenian: {hy_ok}, in English: {en_ok}")
    print(f"questions: {n_q}; ՞ kept: {q_kept}; English ends with '?': {q_en}")
    print(f"CER over all {len(chosen)} clips: {c_err / c_tot:.1%}; wrote {out}")


if __name__ == "__main__":
    main()
