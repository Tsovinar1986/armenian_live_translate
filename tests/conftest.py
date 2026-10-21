import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SR = 16000


def energy_vad(frame: np.ndarray) -> float:
    """Stand-in for Silero: 'speech' = loud frame. Lets us test segmentation logic without a model."""
    return 1.0 if float(np.abs(frame).mean()) > 0.01 else 0.0


def tone(seconds: float, amp: float = 0.3, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


@pytest.fixture
def vad():
    return energy_vad
