from __future__ import annotations

import unittest

import numpy as np

from rtl_weatherband.audio_effects import (
    AudioEffectsProcessor,
    DcBlocker,
    highpass_mix_for_sharpness,
    tap_count_for_sharpness,
)
from rtl_weatherband.config import (
    AudioConfig,
    DeemphasisConfig,
    FilterConfig,
    VolumeConfig,
)


def sine(frequency: float, sample_rate: int = 16000, seconds: float = 0.25):
    times = np.arange(round(sample_rate * seconds), dtype=np.float32) / sample_rate
    return np.sin(2 * np.pi * frequency * times).astype(np.float32) * 0.5


def rms(samples) -> float:
    return float(np.sqrt(np.mean(np.square(samples))))


def steady_rms(samples) -> float:
    return rms(samples[len(samples) // 2 :])


class AudioEffectsTests(unittest.TestCase):
    def test_sharpness_maps_to_gentle_then_steep_tap_counts(self) -> None:
        self.assertEqual(tap_count_for_sharpness(0), 5)
        self.assertLessEqual(tap_count_for_sharpness(5), 133)
        self.assertEqual(tap_count_for_sharpness(10), 1025)

    def test_highpass_low_sharpness_is_partially_blended(self) -> None:
        self.assertAlmostEqual(highpass_mix_for_sharpness(0), 0.35)
        self.assertLess(highpass_mix_for_sharpness(5), 0.6)
        self.assertEqual(highpass_mix_for_sharpness(10), 1.0)

    def test_highpass_low_sharpness_preserves_upper_voice_audio(self) -> None:
        processor = AudioEffectsProcessor(
            AudioConfig(
                deemphasis=DeemphasisConfig(enabled=False),
                highpass=FilterConfig(
                    enabled=True,
                    frequency=500,
                    sharpness=2,
                ),
            )
        )
        source = sine(2000, seconds=1.0)

        output = processor.process(source)

        self.assertGreater(steady_rms(output), steady_rms(source) * 0.7)

    def test_volume_multiplier_changes_level(self) -> None:
        processor = AudioEffectsProcessor(
            AudioConfig(
                deemphasis=DeemphasisConfig(enabled=False),
                volume=VolumeConfig(enabled=True, multiplier=2.0),
            )
        )
        reference = AudioEffectsProcessor(
            AudioConfig(
                deemphasis=DeemphasisConfig(enabled=False),
            )
        )
        source = sine(1000, seconds=1.0) * 0.25

        output = processor.process(source)
        reference_output = reference.process(source)

        self.assertGreater(steady_rms(output), steady_rms(reference_output) * 1.95)
        self.assertLess(steady_rms(output), steady_rms(reference_output) * 2.05)

    def test_dc_blocker_removes_constant_bias_across_chunks(self) -> None:
        blocker = DcBlocker()
        source = np.full(16000, 0.25, dtype=np.float32)

        output = np.concatenate(
            (
                blocker.process(source[:4000]),
                blocker.process(source[4000:]),
            )
        )

        self.assertLess(abs(float(np.mean(output[-4000:]))), 0.001)

    def test_audio_processor_always_removes_dc_when_deemphasis_is_disabled(self) -> None:
        processor = AudioEffectsProcessor(
            AudioConfig(
                deemphasis=DeemphasisConfig(enabled=False),
            )
        )
        source = np.full(16000, 0.2, dtype=np.float32)

        output = processor.process(source)

        self.assertLess(abs(float(np.mean(output[-4000:]))), 0.001)

    def test_audio_processor_removes_dc_after_deemphasis(self) -> None:
        processor = AudioEffectsProcessor(
            AudioConfig(
                deemphasis=DeemphasisConfig(enabled=True),
            )
        )
        source = np.full(16000, 0.2, dtype=np.float32)

        output = processor.process(source)

        self.assertLess(abs(float(np.mean(output[-4000:]))), 0.001)

    def test_highpass_reduces_low_frequency_audio(self) -> None:
        processor = AudioEffectsProcessor(
            AudioConfig(
                deemphasis=DeemphasisConfig(enabled=False),
                highpass=FilterConfig(
                    enabled=True,
                    frequency=500,
                    sharpness=8,
                ),
            )
        )
        source = sine(100)

        output = processor.process(source)

        self.assertLess(rms(output), rms(source) * 0.75)

    def test_lowpass_reduces_high_frequency_audio(self) -> None:
        processor = AudioEffectsProcessor(
            AudioConfig(
                deemphasis=DeemphasisConfig(enabled=False),
                lowpass=FilterConfig(
                    enabled=True,
                    frequency=3000,
                    sharpness=8,
                ),
            )
        )
        source = sine(5000)

        output = processor.process(source)

        self.assertLess(rms(output), rms(source) * 0.75)

    def test_lowpass_high_sharpness_is_steeper_than_low_sharpness(self) -> None:
        gentle = AudioEffectsProcessor(
            AudioConfig(
                deemphasis=DeemphasisConfig(enabled=False),
                lowpass=FilterConfig(
                    enabled=True,
                    frequency=4000,
                    sharpness=0,
                ),
            )
        )
        steep = AudioEffectsProcessor(
            AudioConfig(
                deemphasis=DeemphasisConfig(enabled=False),
                lowpass=FilterConfig(
                    enabled=True,
                    frequency=4000,
                    sharpness=10,
                ),
            )
        )
        source = sine(4200, seconds=1.0)

        gentle_output = gentle.process(source)
        steep_output = steep.process(source)

        self.assertLess(steady_rms(steep_output), steady_rms(gentle_output) * 0.25)

    def test_lowpass_mid_sharpness_leaves_room_for_higher_settings(self) -> None:
        middle = AudioEffectsProcessor(
            AudioConfig(
                deemphasis=DeemphasisConfig(enabled=False),
                lowpass=FilterConfig(
                    enabled=True,
                    frequency=4000,
                    sharpness=5,
                ),
            )
        )
        steep = AudioEffectsProcessor(
            AudioConfig(
                deemphasis=DeemphasisConfig(enabled=False),
                lowpass=FilterConfig(
                    enabled=True,
                    frequency=4000,
                    sharpness=10,
                ),
            )
        )
        source = sine(4200, seconds=1.0)

        middle_output = middle.process(source)
        steep_output = steep.process(source)

        self.assertLess(steady_rms(steep_output), steady_rms(middle_output) * 0.5)

    def test_notch_reduces_center_frequency_audio(self) -> None:
        processor = AudioEffectsProcessor(
            AudioConfig(
                deemphasis=DeemphasisConfig(enabled=False),
                notch=FilterConfig(
                    enabled=True,
                    frequency=3000,
                    sharpness=8,
                ),
            )
        )
        source = sine(3000)

        output = processor.process(source)

        self.assertLess(rms(output), rms(source) * 0.7)


if __name__ == "__main__":
    unittest.main()
