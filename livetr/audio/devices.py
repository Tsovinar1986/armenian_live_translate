"""Find input devices by index or by name. Pure functions over the list from sounddevice.query_devices()."""
from __future__ import annotations

from typing import Any, Sequence

DeviceList = Sequence[dict[str, Any]]


class DeviceError(RuntimeError):
    pass


def query_devices() -> DeviceList:
    import sounddevice as sd

    return list(sd.query_devices())


def default_input_index() -> int:
    import sounddevice as sd

    idx = sd.default.device[0]
    if idx is None or idx < 0:
        raise DeviceError("No default input device. Pass --mic explicitly (see --list-devices).")
    return int(idx)


def input_devices(devices: DeviceList) -> list[tuple[int, dict[str, Any]]]:
    return [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]


def format_input_devices(devices: DeviceList) -> str:
    lines = ["Input devices:"]
    for i, d in input_devices(devices):
        lines.append(
            f"  [{i:>2}] {d['name']}  ({d['max_input_channels']} ch, {int(d['default_samplerate'])} Hz)"
        )
    return "\n".join(lines)


def resolve_input_device(spec: str, devices: DeviceList) -> int:
    """`spec` is a device index ("3") or a case-insensitive part of the name ("blackhole").

    Raises DeviceError with the device list if nothing or more than one device matches.
    """
    candidates = input_devices(devices)
    if spec.strip().isdigit():
        idx = int(spec)
        if any(i == idx for i, _ in candidates):
            return idx
        raise DeviceError(f"Device {idx} is not an input device.\n{format_input_devices(devices)}")

    needle = spec.lower()
    matches = [(i, d) for i, d in candidates if needle in d["name"].lower()]
    if len(matches) == 1:
        return matches[0][0]
    if not matches:
        raise DeviceError(f"No input device matches '{spec}'.\n{format_input_devices(devices)}")
    names = ", ".join(f"[{i}] {d['name']}" for i, d in matches)
    raise DeviceError(f"'{spec}' matches several input devices: {names}. Use the index instead.")
