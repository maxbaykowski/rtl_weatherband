from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


PCM_SCALE = 32768.0
NFM_DEVIATION_GAIN = 1.5


@dataclass
class NfmDemodulator:
    iq_format: str = "f32"
    _pending_bytes: bytes = b""
    _previous_sample: np.complex64 | None = None

    def process(self, chunk: bytes) -> NDArray[np.float32]:
        chunk = self._pending_bytes + chunk
        bytes_per_iq_sample = 8 if self.iq_format == "f32" else 4
        aligned_size = len(chunk) - (len(chunk) % bytes_per_iq_sample)
        self._pending_bytes = chunk[aligned_size:]
        chunk = chunk[:aligned_size]
        if not chunk:
            return np.array([], dtype=np.float32)

        iq = _iq_bytes_to_complex64(chunk, self.iq_format)
        if len(iq) < 2 and self._previous_sample is None:
            if len(iq) == 1:
                self._previous_sample = iq[-1]
            return np.array([], dtype=np.float32)

        if self._previous_sample is None:
            previous = iq[:-1]
            current = iq[1:]
        else:
            previous = np.concatenate((np.array([self._previous_sample]), iq[:-1]))
            current = iq

        self._previous_sample = iq[-1]
        demodulated = np.angle(current * np.conj(previous)).astype(np.float32)
        return (demodulated / np.pi * NFM_DEVIATION_GAIN).astype(
            np.float32,
            copy=False,
        )


def _iq_bytes_to_complex64(chunk: bytes, iq_format: str) -> NDArray[np.complex64]:
    if iq_format == "f32":
        iq_float = np.frombuffer(chunk, dtype="<f4")
        iq = iq_float[0::2].astype(np.complex64)
        iq += 1j * iq_float[1::2].astype(np.complex64)
        return iq
    if iq_format == "s16":
        iq_int = np.frombuffer(chunk, dtype="<i2").astype(np.float32)
        iq_float = iq_int / PCM_SCALE
        iq = iq_float[0::2].astype(np.complex64)
        iq += 1j * iq_float[1::2].astype(np.complex64)
        return iq
    raise ValueError(f"unsupported IQ format: {iq_format}")


def float_to_s16(samples: NDArray[np.float32]) -> bytes:
    pcm = np.clip(samples * PCM_SCALE, -32768, 32767).astype("<i2")
    return pcm.tobytes()
