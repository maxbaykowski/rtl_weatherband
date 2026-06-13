from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .config import AudioConfig, FilterConfig, IQ_SAMPLE_RATE
from .deemphasis import DeemphasisFilter


MIN_FILTER_TAPS = 5
MAX_FILTER_TAPS = 1025


@dataclass
class FirFilter:
    kernel: NDArray[np.float32]
    mix: float = 1.0
    history: NDArray[np.float32] = field(init=False)
    dry_history: NDArray[np.float32] = field(init=False)

    def __post_init__(self) -> None:
        self.history = np.zeros(max(len(self.kernel) - 1, 0), dtype=np.float32)
        self.dry_history = np.zeros(len(self.kernel) // 2, dtype=np.float32)

    def process(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        if len(samples) == 0:
            return samples
        window = np.concatenate((self.history, samples))
        filtered = np.convolve(window, self.kernel, mode="full")
        start = len(self.history)
        stop = start + len(samples)
        if len(self.history):
            self.history = window[-len(self.history) :]
        filtered = filtered[start:stop].astype(np.float32, copy=False)
        if self.mix >= 1.0:
            return filtered
        dry = self._delayed_dry(samples)
        return (dry * (1.0 - self.mix) + filtered * self.mix).astype(
            np.float32,
            copy=False,
        )

    def _delayed_dry(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        if len(self.dry_history) == 0:
            return samples
        window = np.concatenate((self.dry_history, samples))
        dry = window[: len(samples)]
        self.dry_history = window[-len(self.dry_history) :]
        return dry


@dataclass
class AudioEffectsProcessor:
    config: AudioConfig
    sample_rate: int = IQ_SAMPLE_RATE

    def __post_init__(self) -> None:
        self.deemphasis = DeemphasisFilter(
            self.sample_rate,
            self.config.deemphasis_tau,
        )
        self.highpass = _build_filter("highpass", self.config.highpass, self.sample_rate)
        self.lowpass = _build_filter("lowpass", self.config.lowpass, self.sample_rate)
        self.notch = _build_filter("notch", self.config.notch, self.sample_rate)

    def process(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        audio = self.deemphasis.process_float(samples)
        if self.highpass is not None:
            audio = self.highpass.process(audio)
        if self.lowpass is not None:
            audio = self.lowpass.process(audio)
        if self.notch is not None:
            audio = self.notch.process(audio)
        if self.config.volume.enabled:
            audio = audio * self.config.volume.multiplier
        return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def tap_count_for_sharpness(sharpness: float) -> int:
    normalized = max(0.0, min(1.0, sharpness / 10.0))
    taps = round(MIN_FILTER_TAPS + (MAX_FILTER_TAPS - MIN_FILTER_TAPS) * normalized**4)
    return taps if taps % 2 else taps + 1


def highpass_mix_for_sharpness(sharpness: float) -> float:
    normalized = max(0.0, min(1.0, sharpness / 10.0))
    return 0.35 + 0.65 * normalized**1.5


def _build_filter(
    kind: str,
    config: FilterConfig,
    sample_rate: int,
) -> FirFilter | None:
    if not config.enabled:
        return None
    taps = tap_count_for_sharpness(config.sharpness)
    if kind == "highpass":
        return FirFilter(
            _highpass_kernel(config.frequency, sample_rate, taps),
            mix=highpass_mix_for_sharpness(config.sharpness),
        )
    if kind == "lowpass":
        return FirFilter(_lowpass_kernel(config.frequency, sample_rate, taps))
    if kind == "notch":
        width = _notch_width(config.sharpness)
        low = max(1.0, config.frequency - width / 2)
        high = min(sample_rate / 2 - 1.0, config.frequency + width / 2)
        return FirFilter(_notch_kernel(low, high, sample_rate, taps))
    raise ValueError(f"unsupported filter kind: {kind}")


def _notch_width(sharpness: float) -> float:
    return 400.0 - 340.0 * (sharpness / 10.0)


def _lowpass_kernel(
    cutoff_hz: float,
    sample_rate: int,
    taps: int,
) -> NDArray[np.float32]:
    cutoff = cutoff_hz / sample_rate
    center = (taps - 1) / 2
    n = np.arange(taps, dtype=np.float64)
    kernel = 2 * cutoff * np.sinc(2 * cutoff * (n - center))
    kernel *= np.hamming(taps)
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def _highpass_kernel(
    cutoff_hz: float,
    sample_rate: int,
    taps: int,
) -> NDArray[np.float32]:
    lowpass = _lowpass_kernel(cutoff_hz, sample_rate, taps)
    highpass = -lowpass
    highpass[taps // 2] += 1.0
    return highpass.astype(np.float32)


def _notch_kernel(
    low_hz: float,
    high_hz: float,
    sample_rate: int,
    taps: int,
) -> NDArray[np.float32]:
    lowpass_low = _lowpass_kernel(low_hz, sample_rate, taps)
    lowpass_high = _lowpass_kernel(high_hz, sample_rate, taps)
    bandpass = lowpass_high - lowpass_low
    notch = -bandpass
    notch[taps // 2] += 1.0
    return notch.astype(np.float32)
