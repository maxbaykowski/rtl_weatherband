from __future__ import annotations

import tempfile
import unittest
import wave

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


def _write_wave(
    path: str,
    channels: int = 1,
    sample_width: int = 2,
    sample_rate: int = 16000,
    frames: int = 1600,
) -> None:
    with wave.open(path, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames * channels)


class ConfigTests(unittest.TestCase):
    def test_parses_valid_config(self) -> None:
        config = parse_config(valid_config())
        self.assertEqual(config.station.frequency_hz, 162_550_000)
        self.assertEqual(config.csdr_server.control_port, 4952)
        self.assertEqual(len(config.icecast), 1)
        self.assertEqual(config.icecast[0].mount, "/nwr.mp3")
        self.assertEqual(config.icecast[0].content_type, "audio/mpeg")
        self.assertEqual(config.icecast[0].sample_rate, 16000)
        self.assertEqual(config.icecast[0].bitrate, 32)
        self.assertTrue(config.audio.deemphasis.enabled)
        self.assertEqual(config.audio.deemphasis.tau, 530.0)
        self.assertEqual(config.fallback.silence_timeout_seconds, 30.0)
        self.assertEqual(config.fallback.loop_delay_seconds, 0.0)
        self.assertIsNone(config.fallback.path)

    def test_accepts_valid_custom_fallback_audio(self) -> None:
        raw = valid_config()
        with tempfile.NamedTemporaryFile(suffix=".wav") as fp:
            _write_wave(fp.name)
            raw["fallback"] = {
                "path": fp.name,
                "silence_timeout_seconds": 45,
                "loop_delay_seconds": 2.5,
            }

            config = parse_config(raw)

        self.assertEqual(config.fallback.path, fp.name)
        self.assertEqual(config.fallback.silence_timeout_seconds, 45)
        self.assertEqual(config.fallback.loop_delay_seconds, 2.5)

    def test_rejects_fallback_timeout_below_minimum(self) -> None:
        raw = valid_config()
        raw["fallback"] = {
            "silence_timeout_seconds": 29,
        }
        with self.assertRaisesRegex(ConfigError, "between 30 and 120"):
            parse_config(raw)

    def test_rejects_invalid_custom_fallback_audio_path(self) -> None:
        raw = valid_config()
        raw["fallback"] = {
            "path": "/tmp/rtl_weatherband_missing_fallback.wav",
        }
        with self.assertRaisesRegex(ConfigError, "No such file or directory"):
            parse_config(raw)

    def test_rejects_stereo_custom_fallback_audio(self) -> None:
        raw = valid_config()
        with tempfile.NamedTemporaryFile(suffix=".wav") as fp:
            _write_wave(fp.name, channels=2)
            raw["fallback"] = {
                "path": fp.name,
            }
            with self.assertRaisesRegex(ConfigError, "mono"):
                parse_config(raw)

    def test_parses_multiple_icecast_destinations(self) -> None:
        raw = valid_config()
        first = raw["icecast"]
        second = dict(first)
        second["mount"] = "nwr.ogg"
        second["format"] = "ogg"
        second["bitrate"] = 64
        raw["icecast"] = {"destinations": [first, second]}

        config = parse_config(raw)

        self.assertEqual(len(config.icecast), 2)
        self.assertEqual(config.icecast[0].mount, "/nwr.mp3")
        self.assertEqual(config.icecast[1].mount, "/nwr.ogg")
        self.assertEqual(config.icecast[1].content_type, "application/ogg")

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
        raw["audio"]["deemphasis"] = {
            "enabled": False,
            "tau": 530,
        }
        config = parse_config(raw)
        self.assertEqual(config.audio.deemphasis_tau, 0)

    def test_accepts_legacy_disabled_deemphasis(self) -> None:
        raw = valid_config()
        raw["audio"]["deemphasis_tau"] = 0
        config = parse_config(raw)
        self.assertFalse(config.audio.deemphasis.enabled)
        self.assertEqual(config.audio.deemphasis_tau, 0)

    def test_rejects_deemphasis_tau_above_maximum(self) -> None:
        raw = valid_config()
        raw["audio"]["deemphasis"] = {
            "enabled": True,
            "tau": 531,
        }
        with self.assertRaises(ConfigError):
            parse_config(raw)

    def test_rejects_highpass_too_close_to_1050_hz_attention_tone(self) -> None:
        raw = valid_config()
        raw["audio"]["highpass"] = {
            "enabled": True,
            "frequency": 950,
            "sharpness": 5,
        }
        with self.assertRaisesRegex(ConfigError, "900"):
            parse_config(raw)

    def test_rejects_lowpass_too_close_to_same_mark_tone(self) -> None:
        raw = valid_config()
        raw["audio"]["lowpass"] = {
            "enabled": True,
            "frequency": 2100,
            "sharpness": 5,
        }
        with self.assertRaisesRegex(ConfigError, "2200"):
            parse_config(raw)

    def test_rejects_notch_too_close_to_same_space_tone(self) -> None:
        raw = valid_config()
        raw["audio"]["notch"] = {
            "enabled": True,
            "frequency": 1500,
            "sharpness": 5,
        }
        with self.assertRaisesRegex(ConfigError, "1400.0-1600.0"):
            parse_config(raw)

    def test_rejects_notch_below_enabled_highpass(self) -> None:
        raw = valid_config()
        raw["audio"]["highpass"] = {
            "enabled": True,
            "frequency": 700,
            "sharpness": 5,
        }
        raw["audio"]["notch"] = {
            "enabled": True,
            "frequency": 600,
            "sharpness": 5,
        }
        with self.assertRaisesRegex(ConfigError, "greater than highpass"):
            parse_config(raw)

    def test_rejects_notch_above_enabled_lowpass(self) -> None:
        raw = valid_config()
        raw["audio"]["lowpass"] = {
            "enabled": True,
            "frequency": 3000,
            "sharpness": 5,
        }
        raw["audio"]["notch"] = {
            "enabled": True,
            "frequency": 3500,
            "sharpness": 5,
        }
        with self.assertRaisesRegex(ConfigError, "less than lowpass"):
            parse_config(raw)

    def test_accepts_notch_between_enabled_highpass_and_lowpass(self) -> None:
        raw = valid_config()
        raw["audio"]["highpass"] = {
            "enabled": True,
            "frequency": 700,
            "sharpness": 5,
        }
        raw["audio"]["lowpass"] = {
            "enabled": True,
            "frequency": 3000,
            "sharpness": 5,
        }
        raw["audio"]["notch"] = {
            "enabled": True,
            "frequency": 2500,
            "sharpness": 5,
        }
        config = parse_config(raw)
        self.assertEqual(config.audio.notch.frequency, 2500)

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
        self.assertEqual(config.icecast[0].sample_rate, 48000)
        self.assertEqual(config.icecast[0].bitrate, 64)

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
        raw["audio"]["volume"] = {
            "enabled": True,
            "multiplier": 2.0,
        }

        config, errors = merge_valid_reload_config(raw, current)

        self.assertEqual(config.icecast, current.icecast)
        self.assertEqual(config.audio.volume.multiplier, 2.0)
        self.assertTrue(any("icecast:" in error for error in errors))

    def test_reload_keeps_only_invalid_audio_subsection(self) -> None:
        current = parse_config(valid_config())
        raw = valid_config()
        raw["audio"]["volume"] = {
            "enabled": True,
            "multiplier": 2.0,
        }
        raw["audio"]["notch"] = {
            "enabled": True,
            "frequency": 1050,
            "sharpness": 3,
        }

        config, errors = merge_valid_reload_config(raw, current)

        self.assertEqual(config.audio.volume.multiplier, 2.0)
        self.assertEqual(config.audio.notch, current.audio.notch)
        self.assertTrue(any("audio.notch:" in error for error in errors))

    def test_reload_keeps_old_fallback_when_new_fallback_is_invalid(self) -> None:
        current = parse_config(valid_config())
        raw = valid_config()
        raw["fallback"] = {
            "path": "/tmp/rtl_weatherband_missing_fallback.wav",
        }

        config, errors = merge_valid_reload_config(raw, current)

        self.assertEqual(config.fallback, current.fallback)
        self.assertTrue(any("fallback:" in error for error in errors))

    def test_reload_keeps_changed_notch_when_it_crosses_highpass(self) -> None:
        current_raw = valid_config()
        current_raw["audio"]["highpass"] = {
            "enabled": True,
            "frequency": 500,
            "sharpness": 5,
        }
        current_raw["audio"]["notch"] = {
            "enabled": True,
            "frequency": 700,
            "sharpness": 5,
        }
        current = parse_config(current_raw)
        raw = valid_config()
        raw["audio"]["highpass"] = current_raw["audio"]["highpass"]
        raw["audio"]["notch"] = {
            "enabled": True,
            "frequency": 400,
            "sharpness": 5,
        }

        config, errors = merge_valid_reload_config(raw, current)

        self.assertEqual(config.audio.highpass.frequency, 500)
        self.assertEqual(config.audio.notch, current.audio.notch)
        self.assertTrue(any("audio.notch:" in error for error in errors))

    def test_reload_keeps_changed_highpass_when_it_crosses_notch(self) -> None:
        current_raw = valid_config()
        current_raw["audio"]["highpass"] = {
            "enabled": True,
            "frequency": 500,
            "sharpness": 5,
        }
        current_raw["audio"]["notch"] = {
            "enabled": True,
            "frequency": 700,
            "sharpness": 5,
        }
        current = parse_config(current_raw)
        raw = valid_config()
        raw["audio"]["highpass"] = {
            "enabled": True,
            "frequency": 800,
            "sharpness": 5,
        }
        raw["audio"]["notch"] = current_raw["audio"]["notch"]

        config, errors = merge_valid_reload_config(raw, current)

        self.assertEqual(config.audio.highpass, current.audio.highpass)
        self.assertEqual(config.audio.notch.frequency, 700)
        self.assertTrue(any("audio.highpass:" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
