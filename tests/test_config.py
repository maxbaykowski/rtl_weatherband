from __future__ import annotations

import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

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
        self.assertEqual(config.csdr_server.iq_format, "f32")
        self.assertEqual(config.csdr_server.buffer_seconds, 1.0)
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
        self.assertEqual(config.soundcard, ())
        self.assertFalse(config.eas_recording.enabled)
        self.assertEqual(config.eas_recording.directory, "")

    def test_accepts_eas_recording_config(self) -> None:
        raw = valid_config()
        raw["eas_recording"] = {
            "enabled": True,
            "pre_seconds": 3,
            "post_seconds": 4,
            "max_seconds": 90,
            "directory": "/tmp/eas-alerts",
            "format": "mp3",
            "local_time": True,
        }

        config = parse_config(raw)

        self.assertTrue(config.eas_recording.enabled)
        self.assertEqual(config.eas_recording.pre_seconds, 3)
        self.assertEqual(config.eas_recording.post_seconds, 4)
        self.assertEqual(config.eas_recording.max_seconds, 90)
        self.assertEqual(config.eas_recording.directory, "/tmp/eas-alerts")
        self.assertEqual(config.eas_recording.format, "mp3")
        self.assertTrue(config.eas_recording.local_time)

    def test_eas_recording_defaults_to_state_directory(self) -> None:
        raw = valid_config()
        raw["eas_recording"] = {
            "enabled": True,
        }
        with tempfile.TemporaryDirectory() as state_home:
            expected = Path(state_home) / "rtl_weatherband" / "alerts"
            with patch.dict(os.environ, {"XDG_STATE_HOME": state_home}):
                config = parse_config(raw)

            self.assertEqual(config.eas_recording.directory, str(expected))
            self.assertTrue(expected.is_dir())

    def test_eas_recording_defaults_to_systemd_state_directory(self) -> None:
        raw = valid_config()
        raw["eas_recording"] = {
            "enabled": True,
        }
        with tempfile.TemporaryDirectory() as state_directory:
            expected = Path(state_directory) / "alerts"
            with patch.dict(
                os.environ,
                {"STATE_DIRECTORY": state_directory},
                clear=False,
            ):
                config = parse_config(raw)

            self.assertEqual(config.eas_recording.directory, str(expected))
            self.assertTrue(expected.is_dir())

    def test_expands_eas_recording_directory_environment_variables(self) -> None:
        raw = valid_config()
        raw["eas_recording"] = {
            "enabled": True,
            "directory": "${STATE_DIRECTORY}/alerts",
        }
        with tempfile.TemporaryDirectory() as state_directory:
            expected = Path(state_directory) / "alerts"
            with patch.dict(
                os.environ,
                {"STATE_DIRECTORY": state_directory},
                clear=False,
            ):
                config = parse_config(raw)

            self.assertEqual(config.eas_recording.directory, str(expected))
            self.assertTrue(expected.is_dir())

    def test_rejects_missing_eas_recording_directory_environment_variable(self) -> None:
        raw = valid_config()
        raw["eas_recording"] = {
            "enabled": True,
            "directory": "$RTL_WEATHERBAND_MISSING_STATE/alerts",
        }
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ConfigError, "environment variable is not set"):
                parse_config(raw)

    def test_rejects_empty_eas_recording_directory_environment_variable(self) -> None:
        raw = valid_config()
        raw["eas_recording"] = {
            "enabled": True,
            "directory": "$STATE_DIRECTORY/alerts",
        }
        with patch.dict(os.environ, {"STATE_DIRECTORY": ""}, clear=True):
            with self.assertRaisesRegex(ConfigError, "environment variable is not set"):
                parse_config(raw)

    def test_disabled_eas_recording_does_not_expand_directory_variables(self) -> None:
        raw = valid_config()
        raw["eas_recording"] = {
            "enabled": False,
            "directory": "$RTL_WEATHERBAND_MISSING_STATE/alerts",
        }
        with patch.dict(os.environ, {}, clear=True):
            config = parse_config(raw)

        self.assertFalse(config.eas_recording.enabled)
        self.assertEqual(
            config.eas_recording.directory,
            "$RTL_WEATHERBAND_MISSING_STATE/alerts",
        )

    def test_absent_disabled_eas_recording_does_not_capture_environment_variables(self) -> None:
        raw = valid_config()
        with tempfile.TemporaryDirectory() as first_state_directory:
            with patch.dict(
                os.environ,
                {"STATE_DIRECTORY": first_state_directory},
                clear=False,
            ):
                first = parse_config(raw)

        with tempfile.TemporaryDirectory() as second_state_directory:
            with patch.dict(
                os.environ,
                {"STATE_DIRECTORY": second_state_directory},
                clear=False,
            ):
                second = parse_config(raw)

        self.assertEqual(first.eas_recording, second.eas_recording)
        self.assertEqual(first.eas_recording.directory, "")

    def test_rejects_invalid_eas_recording_format(self) -> None:
        raw = valid_config()
        raw["eas_recording"] = {
            "enabled": True,
            "format": "ogg",
        }
        with self.assertRaisesRegex(ConfigError, "eas_recording.format"):
            parse_config(raw)

    def test_accepts_s16_csdr_server_iq_format(self) -> None:
        raw = valid_config()
        raw["csdr_server"]["iq_format"] = "s16"
        config = parse_config(raw)
        self.assertEqual(config.csdr_server.iq_format, "s16")

    def test_rejects_invalid_csdr_server_iq_format(self) -> None:
        raw = valid_config()
        raw["csdr_server"]["iq_format"] = "u8"
        with self.assertRaisesRegex(ConfigError, "f32.*s16"):
            parse_config(raw)

    def test_rejects_invalid_csdr_server_timeout(self) -> None:
        raw = valid_config()
        raw["csdr_server"]["timeout"] = 0
        with self.assertRaisesRegex(ConfigError, "timeout"):
            parse_config(raw)

    def test_rejects_boolean_numeric_values(self) -> None:
        cases = [
            ("station frequency", ("station", "frequency")),
            ("csdr_server timeout", ("csdr_server", "timeout")),
            ("icecast port", ("icecast", "port")),
            ("icecast sample_rate", ("icecast", "sample_rate")),
            ("fallback timeout", ("fallback", "silence_timeout_seconds")),
            ("audio volume", ("audio", "volume", "multiplier")),
            ("EAS pre_seconds", ("eas_recording", "pre_seconds")),
        ]
        for label, path in cases:
            with self.subTest(label=label):
                raw = valid_config()
                if path[0] == "fallback":
                    raw["fallback"] = {}
                if path[0] == "audio" and path[1] == "volume":
                    raw["audio"]["volume"] = {"enabled": True}
                if path[0] == "eas_recording":
                    raw["eas_recording"] = {
                        "enabled": True,
                        "directory": "/tmp/eas-alerts",
                    }
                section = raw
                for key in path[:-1]:
                    section = section[key]
                section[path[-1]] = True

                with self.assertRaises(ConfigError):
                    parse_config(raw)

    def test_accepts_csdr_server_buffer_seconds(self) -> None:
        raw = valid_config()
        raw["csdr_server"]["buffer_seconds"] = 10
        raw["fallback"] = {
            "silence_timeout_seconds": 30,
        }
        config = parse_config(raw)
        self.assertEqual(config.csdr_server.buffer_seconds, 10.0)

    def test_rejects_buffer_that_is_not_less_than_fallback_timeout(self) -> None:
        raw = valid_config()
        raw["csdr_server"]["buffer_seconds"] = 30
        raw["fallback"] = {
            "silence_timeout_seconds": 30,
        }
        with self.assertRaisesRegex(ConfigError, "buffer_seconds.*less than"):
            parse_config(raw)

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

    def test_accepts_soundcard_device_string(self) -> None:
        raw = valid_config()
        raw["soundcard"] = {
            "enabled": True,
            "device": "plughw:1,0",
        }

        config = parse_config(raw)

        self.assertEqual(len(config.soundcard), 1)
        self.assertTrue(config.soundcard[0].enabled)
        self.assertEqual(config.soundcard[0].device, "plughw:1,0")

    def test_accepts_soundcard_device_index(self) -> None:
        raw = valid_config()
        raw["soundcard"] = {
            "enabled": True,
            "device": 2,
        }

        config = parse_config(raw)

        self.assertEqual(config.soundcard[0].device, 2)

    def test_accepts_multiple_soundcard_outputs(self) -> None:
        raw = valid_config()
        raw["soundcard"] = {
            "outputs": [
                {"enabled": True, "device": "plughw:1"},
                {"enabled": False, "device": "plughw:2"},
                {"enabled": True, "device": "pulse:61"},
            ],
        }

        config = parse_config(raw)

        self.assertEqual([output.device for output in config.soundcard], ["plughw:1", "pulse:61"])

    def test_disabled_soundcard_section_ignores_outputs(self) -> None:
        raw = valid_config()
        raw["soundcard"] = {
            "enabled": False,
            "outputs": [
                {"enabled": True, "device": "plughw:1"},
            ],
        }

        config = parse_config(raw)

        self.assertEqual(config.soundcard, ())

    def test_disabled_legacy_soundcard_output_is_ignored(self) -> None:
        raw = valid_config()
        raw["soundcard"] = {
            "enabled": False,
            "device": "plughw:1",
        }

        config = parse_config(raw)

        self.assertEqual(config.soundcard, ())

    def test_rejects_string_soundcard_enabled(self) -> None:
        raw = valid_config()
        raw["soundcard"] = {
            "enabled": "false",
        }
        with self.assertRaisesRegex(ConfigError, "enabled"):
            parse_config(raw)

    def test_rejects_string_soundcard_section_enabled_with_outputs(self) -> None:
        raw = valid_config()
        raw["soundcard"] = {
            "enabled": "false",
            "outputs": [],
        }
        with self.assertRaisesRegex(ConfigError, "enabled"):
            parse_config(raw)

    def test_rejects_empty_soundcard_device(self) -> None:
        raw = valid_config()
        raw["soundcard"] = {
            "enabled": True,
            "device": "",
        }
        with self.assertRaisesRegex(ConfigError, "device"):
            parse_config(raw)

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

    def test_disabled_icecast_destination_is_ignored(self) -> None:
        raw = valid_config()
        first = raw["icecast"]
        second = dict(first)
        second["mount"] = "disabled.mp3"
        second["enabled"] = False
        raw["icecast"] = {"destinations": [first, second]}

        config = parse_config(raw)

        self.assertEqual(len(config.icecast), 1)
        self.assertEqual(config.icecast[0].mount, "/nwr.mp3")

    def test_disabled_icecast_section_ignores_destinations(self) -> None:
        raw = valid_config()
        first = raw["icecast"]
        raw["icecast"] = {
            "enabled": False,
            "destinations": [first],
        }

        config = parse_config(raw)

        self.assertEqual(config.icecast, ())

    def test_rejects_string_icecast_section_enabled_with_destinations(self) -> None:
        raw = valid_config()
        first = raw["icecast"]
        raw["icecast"] = {
            "enabled": "false",
            "destinations": [first],
        }

        with self.assertRaisesRegex(ConfigError, "enabled"):
            parse_config(raw)

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

    def test_accepts_lowpass_and_notch_at_24khz_nyquist(self) -> None:
        lowpass_raw = valid_config()
        lowpass_raw["audio"]["lowpass"] = {
            "enabled": True,
            "frequency": 12000,
            "sharpness": 5,
        }
        notch_raw = valid_config()
        notch_raw["audio"]["notch"] = {
            "enabled": True,
            "frequency": 12000,
            "sharpness": 5,
        }

        lowpass_config = parse_config(lowpass_raw)
        notch_config = parse_config(notch_raw)

        self.assertEqual(lowpass_config.audio.lowpass.frequency, 12000)
        self.assertEqual(notch_config.audio.notch.frequency, 12000)

    def test_rejects_filter_above_24khz_nyquist(self) -> None:
        raw = valid_config()
        raw["audio"]["lowpass"] = {
            "enabled": True,
            "frequency": 12001,
            "sharpness": 5,
        }

        with self.assertRaisesRegex(ConfigError, "12000"):
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

    def test_rejects_string_icecast_booleans(self) -> None:
        raw = valid_config()
        raw["icecast"]["tls"] = "false"
        with self.assertRaisesRegex(ConfigError, "tls"):
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

    def test_reload_keeps_old_soundcard_when_new_soundcard_is_invalid(self) -> None:
        current_raw = valid_config()
        current_raw["soundcard"] = {
            "enabled": True,
            "device": "plughw:1,0",
        }
        current = parse_config(current_raw)
        raw = valid_config()
        raw["soundcard"] = {
            "enabled": "false",
        }

        config, errors = merge_valid_reload_config(raw, current)

        self.assertEqual(config.soundcard, current.soundcard)
        self.assertTrue(any("soundcard:" in error for error in errors))

    def test_reload_keeps_old_eas_recording_when_new_eas_recording_is_invalid(self) -> None:
        current_raw = valid_config()
        current_raw["eas_recording"] = {
            "enabled": True,
            "directory": "/tmp/eas-alerts",
        }
        current = parse_config(current_raw)
        raw = valid_config()
        raw["eas_recording"] = {
            "enabled": True,
            "format": "ogg",
        }

        config, errors = merge_valid_reload_config(raw, current)

        self.assertEqual(config.eas_recording, current.eas_recording)
        self.assertTrue(any("eas_recording:" in error for error in errors))

    def test_reload_keeps_old_eas_recording_when_directory_variable_is_missing(self) -> None:
        current_raw = valid_config()
        current_raw["eas_recording"] = {
            "enabled": True,
            "directory": "/tmp/eas-alerts",
        }
        current = parse_config(current_raw)
        raw = valid_config()
        raw["eas_recording"] = {
            "enabled": True,
            "directory": "$RTL_WEATHERBAND_MISSING_STATE/alerts",
        }

        with patch.dict(os.environ, {}, clear=True):
            config, errors = merge_valid_reload_config(raw, current)

        self.assertEqual(config.eas_recording, current.eas_recording)
        self.assertTrue(any("eas_recording:" in error for error in errors))

    def test_reload_reevaluates_eas_recording_directory_environment_variables(self) -> None:
        raw = valid_config()
        raw["eas_recording"] = {
            "enabled": True,
            "directory": "$STATE_DIRECTORY/alerts",
        }
        with tempfile.TemporaryDirectory() as old_state_directory:
            with patch.dict(
                os.environ,
                {"STATE_DIRECTORY": old_state_directory},
                clear=False,
            ):
                current = parse_config(raw)

        with tempfile.TemporaryDirectory() as new_state_directory:
            expected = Path(new_state_directory) / "alerts"
            with patch.dict(
                os.environ,
                {"STATE_DIRECTORY": new_state_directory},
                clear=False,
            ):
                config, errors = merge_valid_reload_config(raw, current)

            self.assertFalse(errors)
            self.assertEqual(config.eas_recording.directory, str(expected))
            self.assertNotEqual(config.eas_recording, current.eas_recording)

    def test_reload_can_disable_soundcard_section_with_outputs(self) -> None:
        current_raw = valid_config()
        current_raw["soundcard"] = {
            "outputs": [
                {"enabled": True, "device": "plughw:1"},
            ],
        }
        current = parse_config(current_raw)
        raw = valid_config()
        raw["soundcard"] = {
            "enabled": False,
            "outputs": [
                {"enabled": True, "device": "plughw:1"},
            ],
        }

        config, errors = merge_valid_reload_config(raw, current)

        self.assertEqual(errors, [])
        self.assertEqual(config.soundcard, ())

    def test_reload_can_disable_icecast_section_with_destinations(self) -> None:
        current = parse_config(valid_config())
        raw = valid_config()
        first = raw["icecast"]
        raw["icecast"] = {
            "enabled": False,
            "destinations": [first],
        }

        config, errors = merge_valid_reload_config(raw, current)

        self.assertEqual(errors, [])
        self.assertEqual(config.icecast, ())

    def test_reload_keeps_old_buffer_when_new_buffer_exceeds_fallback(self) -> None:
        current = parse_config(valid_config())
        raw = valid_config()
        raw["csdr_server"]["buffer_seconds"] = 30
        raw["fallback"] = {
            "silence_timeout_seconds": 30,
        }

        config, errors = merge_valid_reload_config(raw, current)

        self.assertEqual(config.csdr_server, current.csdr_server)
        self.assertTrue(any("buffer_seconds" in error for error in errors))

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
