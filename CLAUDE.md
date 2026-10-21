# Project rules (read at the start of every chat)

Project: live Armenian speech → Armenian text → English text in the terminal, with speaker labels.
Target: MacBook Apple Silicon, 8 GB RAM, Python 3.11, local models only. See PLAN.md and DECISIONS.md.

## Hard requirements
- ≤ 5 s from end of phrase to printed English line. Every change must keep latency measurable (`--stats`).
- Local inference only. No cloud APIs, no API keys.
- Never commit model weights (`models/`) or audio (`eval/data/`).

## Code rules
- Fail loudly. No silent fallbacks, no broad `except:`, no hidden retries or timeouts.
  If a device or model is missing, stop with a clear message.
- Keep modules separate: `livetr/audio/` (capture), `segmenter.py` (VAD), `asr.py`, `translate.py`,
  `speakers.py`, `output.py`, `pipeline.py`. Each has plain inputs/outputs and can be tested alone.
- Heavy libraries (torch, transformers, ctranslate2, speechbrain, sounddevice) are imported inside the
  classes that need them, so the pure logic and its tests run with only numpy.
- No features outside the brief (no GUI, no saving transcripts, no echo cancellation).
- One change at a time; run `pytest` before every commit.

## Process rules
- Decisions with reasons go into DECISIONS.md.
- Test cases and QA questions go into TESTS.md as they come up.
- Model choices are made from numbers in `eval/results/`, not from model cards.
