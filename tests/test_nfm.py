from __future__ import annotations

import unittest

import numpy as np

from rtl_weatherband.nfm import NfmDemodulator, float_to_s16


class NfmTests(unittest.TestCase):
    def test_demodulates_constant_phase_step(self) -> None:
        phase_step = np.float32(np.pi / 4)
        phases = np.arange(8, dtype=np.float32) * phase_step
        iq = np.exp(1j * phases).astype(np.complex64)
        interleaved = np.empty(len(iq) * 2, dtype="<f4")
        interleaved[0::2] = iq.real
        interleaved[1::2] = iq.imag

        demodulated = NfmDemodulator().process(interleaved.tobytes())

        self.assertEqual(len(demodulated), len(iq) - 1)
        np.testing.assert_allclose(demodulated, 0.25, rtol=1e-6, atol=1e-6)

    def test_demodulator_handles_split_iq_frames(self) -> None:
        iq = np.array([1 + 0j, 0 + 1j, -1 + 0j], dtype=np.complex64)
        interleaved = np.empty(len(iq) * 2, dtype="<f4")
        interleaved[0::2] = iq.real
        interleaved[1::2] = iq.imag
        payload = interleaved.tobytes()

        demodulator = NfmDemodulator()
        first = demodulator.process(payload[:5])
        second = demodulator.process(payload[5:])

        self.assertEqual(len(first), 0)
        np.testing.assert_allclose(second, [0.5, 0.5], rtol=1e-6, atol=1e-6)

    def test_float_to_s16_clips_output(self) -> None:
        pcm = np.frombuffer(
            float_to_s16(np.array([-2.0, 0.0, 2.0], dtype=np.float32)),
            dtype="<i2",
        )
        np.testing.assert_array_equal(pcm, np.array([-32768, 0, 32767], dtype="<i2"))


if __name__ == "__main__":
    unittest.main()

