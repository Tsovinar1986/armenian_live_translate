"""R1: the user can pick devices by index or name; mistakes stop with a clear message."""
import pytest

from livetr.audio.devices import DeviceError, format_input_devices, resolve_input_device

DEVICES = [
    {"name": "MacBook Air Microphone", "max_input_channels": 1, "default_samplerate": 48000.0},
    {"name": "MacBook Air Speakers", "max_input_channels": 0, "default_samplerate": 48000.0},
    {"name": "BlackHole 2ch", "max_input_channels": 2, "default_samplerate": 48000.0},
    {"name": "BlackHole 16ch", "max_input_channels": 16, "default_samplerate": 48000.0},
    {"name": "Multi-Output Device", "max_input_channels": 0, "default_samplerate": 48000.0},
]


def test_by_name_and_index():
    assert resolve_input_device("microphone", DEVICES) == 0
    assert resolve_input_device("2ch", DEVICES) == 2
    assert resolve_input_device("3", DEVICES) == 3


def test_output_only_device_is_rejected():
    with pytest.raises(DeviceError, match="not an input device"):
        resolve_input_device("1", DEVICES)


def test_ambiguous_name_lists_matches():
    with pytest.raises(DeviceError, match="several"):
        resolve_input_device("blackhole", DEVICES)


def test_unknown_name_lists_inputs():
    with pytest.raises(DeviceError) as e:
        resolve_input_device("zoom", DEVICES)
    assert "BlackHole 2ch" in str(e.value) and "Speakers" not in str(e.value)


def test_listing_only_shows_inputs():
    text = format_input_devices(DEVICES)
    assert "Microphone" in text and "Multi-Output" not in text
