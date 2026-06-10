from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


PCM_SCALE = 32768.0


@dataclass
class DeemphasisFilter:
    sample_rate: int
    tau: float
    curve: NDArray[np.float32] = field(init=False)
    _history: NDArray[np.float32] = field(init=False)
    _pending_byte: bytes = b""

    def __post_init__(self) -> None:
        self.curve = generate_deemphasis_curve(self.sample_rate, self.tau)
        self._history = np.zeros(max(len(self.curve) - 1, 0), dtype=np.float32)

    @property
    def enabled(self) -> bool:
        return self.tau > 0

    def process(self, chunk: bytes) -> bytes:
        chunk = self._pending_byte + chunk
        if len(chunk) % 2:
            self._pending_byte = chunk[-1:]
            chunk = chunk[:-1]
        else:
            self._pending_byte = b""
        if not chunk:
            return b""

        samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / PCM_SCALE
        filtered = self._filter(samples)
        pcm = np.clip(filtered * PCM_SCALE, -32768, 32767).astype("<i2")
        return pcm.tobytes()

    def process_float(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        return self._filter(samples)

    def flush(self) -> bytes:
        self._pending_byte = b""
        return b""

    def _filter(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        if not self.enabled:
            return samples
        window = np.concatenate((self._history, samples))
        filtered = np.convolve(window, self.curve, mode="full")
        start = len(self._history)
        stop = start + len(samples)
        self._history = window[-len(self._history) :]
        return filtered[start:stop].astype(np.float32, copy=False)


def generate_deemphasis_curve(sample_rate: int, tau: float) -> NDArray[np.float32]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than 0")
    if tau <= 0:
        return np.array([1.0], dtype=np.float32)

    tau_seconds = tau / 1_000_000.0
    duration = max(1 / sample_rate, tau_seconds * 8)
    sample_count = max(1, int(np.ceil(duration * sample_rate)))
    times = np.arange(sample_count, dtype=np.float32) / sample_rate
    curve = np.exp(-times / tau_seconds).astype(np.float32)
    curve /= np.sum(curve)
    return curve
