from __future__ import annotations

import unittest

from rtl_weatherband.config import IcecastConfig
from rtl_weatherband.runner import _diff_icecast_destinations


def destination(mount: str, bitrate: int = 32) -> IcecastConfig:
    return IcecastConfig(
        host="127.0.0.1",
        port=8000,
        mount=mount,
        username="source",
        password="hackme",
        format="mp3",
        sample_rate=16000,
        bitrate=bitrate,
    )


class RunnerTests(unittest.TestCase):
    def test_icecast_diff_keeps_unchanged_destination(self) -> None:
        first = destination("/one.mp3")
        second = destination("/two.mp3")

        remove_configs, add_configs = _diff_icecast_destinations(
            (first, second),
            (first,),
        )

        self.assertEqual(remove_configs, [second])
        self.assertEqual(add_configs, [])

    def test_icecast_diff_replaces_changed_destination(self) -> None:
        old = destination("/one.mp3", bitrate=32)
        new = destination("/one.mp3", bitrate=40)

        remove_configs, add_configs = _diff_icecast_destinations((old,), (new,))

        self.assertEqual(remove_configs, [old])
        self.assertEqual(add_configs, [new])


if __name__ == "__main__":
    unittest.main()
