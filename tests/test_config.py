from __future__ import annotations

import unittest

from rtl_weatherband.config import ConfigError, merge_valid_reload_config, parse_config


def valid_config() -> dict:
    return {
        "csdr_server": {
            "host": "127.0.0.1",
            "port": 4951,
        },
        "station": {
            "frequency": 162.55,
        },
        "icecast": {
            "host": "127.0.0.1",
            "port": 8000,
            "mount": "nwr.mp3",
            "username": "source",
            "password": "hackme",
            "format": "mp3",
            "sample_rate": 16000,
            "bitrate": 32,
        },
        "audio": {
        },
    }


class ConfigTests(unittest.TestCase):
    def test_parses_valid_config(self) -> None:
        config = parse_config(valid_config())
        self.assertEqual(config.station.frequency_hz, 162_550_000)
        self.assertEqual(config.csdr_server.control_port, 4952)
        self.assertEqual(config.icecast.mount, "/nwr.mp3")
        self.assertEqual(config.icecast.content_type, "audio/mpeg")
        self.assertEqual(config.icecast.sample_rate, 16000)
        self.assertEqual(config.icecast.bitrate, 32)
        self.assertEqual(config.audio.deemphasis_tau, 530.0)

    def test_rejects_out_of_range_frequency(self) -> None:
        raw = valid_config()
        raw["station"]["frequency"] = 162.3
        with self.assertRaises(ConfigError):
            parse_config(raw)

    def test_rejects_unknown_format(self) -> None:
        raw = valid_config()
        raw["icecast"]["format"] = "aac"
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

    def test_rejects_string_bitrate(self) -> None:
        raw = valid_config()
        raw["icecast"]["bitrate"] = "32k"
        with self.assertRaises(ConfigError):
            parse_config(raw)

    def test_rejects_non_positive_bitrate(self) -> None:
        raw = valid_config()
        raw["icecast"]["bitrate"] = 0
        with self.assertRaisesRegex(ConfigError, "invalid bitrate"):
            parse_config(raw)

    def test_rejects_invalid_output_sample_rate(self) -> None:
        raw = valid_config()
        raw["icecast"]["sample_rate"] = 12345
        with self.assertRaisesRegex(ConfigError, "invalid sample rate"):
            parse_config(raw)

    def test_rejects_mp3_bitrate_above_low_sample_rate_limit(self) -> None:
        raw = valid_config()
        raw["icecast"]["sample_rate"] = 8000
        raw["icecast"]["bitrate"] = 65
        with self.assertRaisesRegex(ConfigError, "8 and 64 Kbps"):
            parse_config(raw)

    def test_rejects_mp3_bitrate_above_high_sample_rate_limit(self) -> None:
        raw = valid_config()
        raw["icecast"]["sample_rate"] = 44100
        raw["icecast"]["bitrate"] = 321
        with self.assertRaisesRegex(ConfigError, "32 and 320 Kbps"):
            parse_config(raw)

    def test_accepts_ogg_with_shared_output_sample_rate(self) -> None:
        raw = valid_config()
        raw["icecast"]["format"] = "ogg"
        raw["icecast"]["sample_rate"] = 48000
        raw["icecast"]["bitrate"] = 64
        config = parse_config(raw)
        self.assertEqual(config.icecast.sample_rate, 48000)
        self.assertEqual(config.icecast.bitrate, 64)

    def test_rejects_ogg_managed_bitrate_above_sample_rate_limit(self) -> None:
        raw = valid_config()
        raw["icecast"]["format"] = "ogg"
        raw["icecast"]["sample_rate"] = 16000
        raw["icecast"]["bitrate"] = 128
        with self.assertRaisesRegex(ConfigError, "16 and 100 Kbps"):
            parse_config(raw)

    def test_reports_sample_rate_and_bitrate_together(self) -> None:
        raw = valid_config()
        raw["icecast"]["sample_rate"] = 12345
        raw["icecast"]["bitrate"] = 0
        with self.assertRaises(ConfigError) as context:
            parse_config(raw)
        message = str(context.exception)
        self.assertIn("invalid sample rate", message)
        self.assertIn("invalid bitrate", message)

    def test_reload_keeps_only_invalid_sections(self) -> None:
        current = parse_config(valid_config())
        raw = valid_config()
        raw["icecast"]["sample_rate"] = 12345
        raw["audio"]["deemphasis_tau"] = 0

        config, errors = merge_valid_reload_config(raw, current)

        self.assertEqual(config.icecast, current.icecast)
        self.assertEqual(config.audio.deemphasis_tau, 0)
        self.assertTrue(any("icecast:" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
