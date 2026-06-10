from __future__ import annotations

import unittest

from rtl_weatherband.config import ConfigError, parse_config


def valid_config() -> dict:
    return {
        "csdr_server": {
            "host": "127.0.0.1",
            "listen_port": 4951,
        },
        "station": {
            "frequency_mhz": 162.55,
        },
        "icecast": {
            "host": "127.0.0.1",
            "port": 8000,
            "mount": "nwr.mp3",
            "username": "source",
            "password": "hackme",
        },
        "audio": {
            "format": "mp3",
            "sample_rate": 16000,
            "bitrate": "32k",
        },
    }


class ConfigTests(unittest.TestCase):
    def test_parses_valid_config(self) -> None:
        config = parse_config(valid_config())
        self.assertEqual(config.station.frequency_hz, 162_550_000)
        self.assertEqual(config.csdr_server.control_port, 4952)
        self.assertEqual(config.icecast.mount, "/nwr.mp3")
        self.assertEqual(config.audio.content_type, "audio/mpeg")
        self.assertEqual(config.audio.deemphasis_tau, 530.0)

    def test_rejects_out_of_range_frequency(self) -> None:
        raw = valid_config()
        raw["station"]["frequency_mhz"] = 162.3
        with self.assertRaises(ConfigError):
            parse_config(raw)

    def test_rejects_unknown_format(self) -> None:
        raw = valid_config()
        raw["audio"]["format"] = "aac"
        with self.assertRaises(ConfigError):
            parse_config(raw)

    def test_accepts_disabled_deemphasis(self) -> None:
        raw = valid_config()
        raw["audio"]["deemphasis_tau"] = 0
        config = parse_config(raw)
        self.assertEqual(config.audio.deemphasis_tau, 0)

    def test_rejects_deemphasis_tau_above_maximum(self) -> None:
        raw = valid_config()
        raw["audio"]["deemphasis_tau"] = 531
        with self.assertRaises(ConfigError):
            parse_config(raw)


if __name__ == "__main__":
    unittest.main()
