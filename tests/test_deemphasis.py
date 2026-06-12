from __future__ import annotations

import unittest

import numpy as np

from rtl_weatherband.deemphasis import DeemphasisFilter, generate_deemphasis_curve


class DeemphasisTests(unittest.TestCase):
    def test_disabled_deemphasis_uses_identity_curve(self) -> None:
        curve = generate_deemphasis_curve(16000, 0)
        np.testing.assert_array_equal(curve, np.array([1.0], dtype=np.float32))

    def test_deemphasis_curve_is_normalized_exponential(self) -> None:
        curve = generate_deemphasis_curve(16000, 530)
        self.assertGreater(len(curve), 1)
        self.assertAlmostEqual(float(np.sum(curve)), 1.0, places=6)
        self.assertGreater(float(curve[0]), float(curve[-1]))

    def test_filter_preserves_pcm_frame_alignment(self) -> None:
        deemphasis = DeemphasisFilter(16000, 530)
        source = np.array([0, 12000, -12000, 4000], dtype="<i2").tobytes()
        output = deemphasis.process(source[:3]) + deemphasis.process(source[3:])
        self.assertEqual(len(output), len(source))


if __name__ == "__main__":
    unittest.main()

