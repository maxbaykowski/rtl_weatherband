from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from rtl_weatherband.config import IQ_SAMPLE_RATE, SoundcardConfig
from rtl_weatherband.pipeline import PCM_FRAME_BYTES, SILENCE_FRAME, SILENCE_FRAME_SAMPLES
from rtl_weatherband.soundcard import (
    SoundcardDependencyError,
    SoundcardError,
    SoundcardOutput,
)


def stereo_pcm(pcm: bytes) -> bytes:
    output = bytearray(len(pcm) * 2)
    for sample_index in range(0, len(pcm), 2):
        sample = pcm[sample_index : sample_index + 2]
        output_index = sample_index * 2
        output[output_index : output_index + 2] = sample
        output[output_index + 2 : output_index + 4] = sample
    return bytes(output)


class FakeRawOutputStream:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        import os
        self.pulse_sink = os.environ.get("PULSE_SINK")
        self.callback = kwargs.get("callback")
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class FakeSounddevice(types.SimpleNamespace):
    def __init__(self, devices=None, hostapis=None) -> None:
        super().__init__()
        self.streams: list[FakeRawOutputStream] = []
        self.devices = devices or [
            {
                "index": 0,
                "name": "default",
                "hostapi": 0,
                "max_output_channels": 2,
                "default_samplerate": IQ_SAMPLE_RATE,
            }
        ]
        self.hostapis = hostapis or [{"name": "ALSA"}]

    def RawOutputStream(self, *args, **kwargs) -> FakeRawOutputStream:
        stream = FakeRawOutputStream(*args, **kwargs)
        self.streams.append(stream)
        return stream

    def query_devices(self, device=None, kind=None):
        if device is None:
            if kind == "output":
                return self.devices[0]
            return self.devices
        if isinstance(device, int):
            for entry in self.devices:
                if entry["index"] == device:
                    return entry
            raise ValueError(f"Error querying device {device}")
        for entry in self.devices:
            if str(device) in entry["name"]:
                return entry
        raise ValueError(f"No output device matching {device!r}")

    def query_hostapis(self):
        return self.hostapis


class SoundcardTests(unittest.TestCase):
    def test_soundcard_prefills_before_starting_stream(self) -> None:
        fake_sd = FakeSounddevice()
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            output = SoundcardOutput(SoundcardConfig(enabled=True))

        for _ in range(9):
            output.write(SILENCE_FRAME)

        stream = fake_sd.streams[0]
        self.assertFalse(stream.started)

        output.write(SILENCE_FRAME)

        self.assertTrue(stream.started)

    def test_soundcard_callback_drains_prefilled_audio_continuously(self) -> None:
        fake_sd = FakeSounddevice()
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            output = SoundcardOutput(SoundcardConfig(enabled=True))

        for _ in range(10):
            output.write(SILENCE_FRAME)
        stream = fake_sd.streams[0]
        outdata = bytearray(PCM_FRAME_BYTES * 5 * 2)

        stream.callback(outdata, SILENCE_FRAME_SAMPLES * 5, None, None)

        self.assertEqual(bytes(outdata), stereo_pcm(SILENCE_FRAME * 5))
        self.assertEqual(stream.kwargs["channels"], 2)

    def test_soundcard_callback_pads_underrun_with_silence(self) -> None:
        fake_sd = FakeSounddevice()
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            output = SoundcardOutput(SoundcardConfig(enabled=True))

        stream = fake_sd.streams[0]
        outdata = bytearray(PCM_FRAME_BYTES * 2)

        stream.callback(outdata, SILENCE_FRAME_SAMPLES, None, None)

        self.assertEqual(bytes(outdata), stereo_pcm(SILENCE_FRAME))

    def test_soundcard_callback_keeps_mono_for_mono_device(self) -> None:
        fake_sd = FakeSounddevice(
            devices=[
                {
                    "index": 0,
                    "name": "mono",
                    "hostapi": 0,
                    "max_output_channels": 1,
                    "default_samplerate": IQ_SAMPLE_RATE,
                }
            ],
        )
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            output = SoundcardOutput(SoundcardConfig(enabled=True))

        stream = fake_sd.streams[0]
        outdata = bytearray(PCM_FRAME_BYTES)

        stream.callback(outdata, SILENCE_FRAME_SAMPLES, None, None)

        self.assertEqual(bytes(outdata), SILENCE_FRAME)
        self.assertEqual(stream.kwargs["channels"], 1)

    def test_soundcard_close_flushes_partial_batch(self) -> None:
        fake_sd = FakeSounddevice()
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            output = SoundcardOutput(SoundcardConfig(enabled=True))

        output.write(SILENCE_FRAME)
        output.close()

        stream = fake_sd.streams[0]
        self.assertFalse(stream.stopped)
        self.assertTrue(stream.closed)

    def test_alsa_plughw_device_resolves_to_matching_sounddevice_index(self) -> None:
        fake_sd = FakeSounddevice(
            devices=[
                {
                    "index": 3,
                    "name": "USB Audio: Audio (hw:1,0)",
                    "hostapi": 0,
                    "max_output_channels": 2,
                    "default_samplerate": IQ_SAMPLE_RATE,
                }
            ],
            hostapis=[{"name": "ALSA"}],
        )

        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            output = SoundcardOutput(
                SoundcardConfig(enabled=True, device="plughw:1,0")
            )

        stream = fake_sd.streams[0]
        self.assertEqual(output.device, 3)
        self.assertEqual(stream.kwargs["device"], 3)
        self.assertEqual(output.sample_rate, IQ_SAMPLE_RATE)

    def test_alsa_plughw_card_only_defaults_to_pcm_device_zero(self) -> None:
        fake_sd = FakeSounddevice(
            devices=[
                {
                    "index": 3,
                    "name": "USB Audio: Audio (hw:1,0)",
                    "hostapi": 0,
                    "max_output_channels": 2,
                    "default_samplerate": IQ_SAMPLE_RATE,
                }
            ],
            hostapis=[{"name": "ALSA"}],
        )

        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            output = SoundcardOutput(SoundcardConfig(enabled=True, device="plughw:1"))

        stream = fake_sd.streams[0]
        self.assertEqual(output.device, 3)
        self.assertEqual(stream.kwargs["device"], 3)

    def test_pulseaudio_sink_index_resolves_with_pulse_sink_environment(self) -> None:
        fake_sd = FakeSounddevice(
            devices=[
                {
                    "index": 18,
                    "name": "Tiger Lake-LP Smart Sound Technology Audio Controller Speaker",
                    "hostapi": 0,
                    "max_output_channels": 2,
                    "default_samplerate": IQ_SAMPLE_RATE,
                }
            ],
            hostapis=[{"name": "ALSA"}],
        )
        pactl_output = (
            "Sink #61\n"
            "\tName: alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic."
            "HiFi__Speaker__sink\n"
            "\tDescription: Tiger Lake-LP Smart Sound Technology Audio Controller Speaker\n"
            "\tProperties:\n"
            "\t\tnode.name = \"alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic."
            "HiFi__Speaker__sink\"\n"
            "\t\tnode.nick = \"Speaker\"\n"
        )

        with patch.dict(sys.modules, {"sounddevice": fake_sd}), patch(
            "rtl_weatherband.soundcard.subprocess.run",
            return_value=types.SimpleNamespace(stdout=pactl_output),
        ):
            output = SoundcardOutput(SoundcardConfig(enabled=True, device=61))

        stream = fake_sd.streams[0]
        self.assertEqual(output.device, 18)
        self.assertEqual(stream.kwargs["device"], 18)
        self.assertEqual(
            stream.pulse_sink,
            "alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic."
            "HiFi__Speaker__sink",
        )

    def test_pulseaudio_sink_name_resolves_with_pulse_sink_environment(self) -> None:
        fake_sd = FakeSounddevice(
            devices=[
                {
                    "index": 18,
                    "name": "Tiger Lake-LP Smart Sound Technology Audio Controller Speaker",
                    "hostapi": 0,
                    "max_output_channels": 2,
                    "default_samplerate": IQ_SAMPLE_RATE,
                }
            ],
            hostapis=[{"name": "ALSA"}],
        )
        sink_name = (
            "alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic."
            "HiFi__Speaker__sink"
        )
        pactl_output = (
            "Sink #61\n"
            f"\tName: {sink_name}\n"
            "\tDescription: Tiger Lake-LP Smart Sound Technology Audio Controller Speaker\n"
            "\tProperties:\n"
            f"\t\tnode.name = \"{sink_name}\"\n"
            "\t\tnode.nick = \"Speaker\"\n"
        )

        with patch.dict(sys.modules, {"sounddevice": fake_sd}), patch(
            "rtl_weatherband.soundcard.subprocess.run",
            return_value=types.SimpleNamespace(stdout=pactl_output),
        ):
            output = SoundcardOutput(SoundcardConfig(enabled=True, device=sink_name))

        stream = fake_sd.streams[0]
        self.assertEqual(output.device, 18)
        self.assertEqual(stream.kwargs["device"], 18)
        self.assertEqual(stream.pulse_sink, sink_name)

    def test_explicit_pulse_sink_index_resolves_with_prefix(self) -> None:
        fake_sd = FakeSounddevice(
            devices=[
                {
                    "index": 18,
                    "name": "Tiger Lake-LP Smart Sound Technology Audio Controller Speaker",
                    "hostapi": 0,
                    "max_output_channels": 2,
                    "default_samplerate": IQ_SAMPLE_RATE,
                }
            ],
            hostapis=[{"name": "ALSA"}],
        )
        sink_name = (
            "alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic."
            "HiFi__Speaker__sink"
        )
        pactl_output = (
            "Sink #61\n"
            f"\tName: {sink_name}\n"
            "\tDescription: Tiger Lake-LP Smart Sound Technology Audio Controller Speaker\n"
        )

        with patch.dict(sys.modules, {"sounddevice": fake_sd}), patch(
            "rtl_weatherband.soundcard.subprocess.run",
            return_value=types.SimpleNamespace(stdout=pactl_output),
        ):
            output = SoundcardOutput(SoundcardConfig(enabled=True, device="pulse:61"))

        stream = fake_sd.streams[0]
        self.assertEqual(output.device, 18)
        self.assertEqual(stream.kwargs["device"], 18)
        self.assertEqual(stream.pulse_sink, sink_name)

    def test_pulseaudio_sink_falls_back_to_pipewire_device(self) -> None:
        fake_sd = FakeSounddevice(
            devices=[
                {
                    "index": 10,
                    "name": "pipewire",
                    "hostapi": 0,
                    "max_output_channels": 2,
                    "default_samplerate": IQ_SAMPLE_RATE,
                }
            ],
            hostapis=[{"name": "ALSA"}],
        )
        sink_name = "alsa_output.example.sink"
        pactl_output = (
            "Sink #61\n"
            f"\tName: {sink_name}\n"
            "\tDescription: Unlisted Sink\n"
        )

        with patch.dict(sys.modules, {"sounddevice": fake_sd}), patch(
            "rtl_weatherband.soundcard.subprocess.run",
            return_value=types.SimpleNamespace(stdout=pactl_output),
        ):
            output = SoundcardOutput(SoundcardConfig(enabled=True, device=61))

        stream = fake_sd.streams[0]
        self.assertEqual(output.device, 10)
        self.assertEqual(stream.kwargs["device"], 10)
        self.assertEqual(stream.pulse_sink, sink_name)

    def test_pulseaudio_sink_selection_requires_pactl(self) -> None:
        fake_sd = FakeSounddevice(
            devices=[
                {
                    "index": 10,
                    "name": "pipewire",
                    "hostapi": 0,
                    "max_output_channels": 2,
                    "default_samplerate": IQ_SAMPLE_RATE,
                }
            ],
            hostapis=[{"name": "ALSA"}],
        )

        with patch.dict(sys.modules, {"sounddevice": fake_sd}), patch(
            "rtl_weatherband.soundcard.subprocess.run",
            side_effect=FileNotFoundError("pactl"),
        ):
            with self.assertRaises(SoundcardDependencyError):
                SoundcardOutput(SoundcardConfig(enabled=True, device="pulse:61"))

    def test_missing_pulseaudio_sink_is_retryable_when_pactl_works(self) -> None:
        fake_sd = FakeSounddevice(
            devices=[
                {
                    "index": 10,
                    "name": "pipewire",
                    "hostapi": 0,
                    "max_output_channels": 2,
                    "default_samplerate": IQ_SAMPLE_RATE,
                }
            ],
            hostapis=[{"name": "ALSA"}],
        )
        pactl_output = "Sink #61\n\tName: alsa_output.example.sink\n"

        with patch.dict(sys.modules, {"sounddevice": fake_sd}), patch(
            "rtl_weatherband.soundcard.subprocess.run",
            return_value=types.SimpleNamespace(stdout=pactl_output),
        ):
            with self.assertRaises(SoundcardError) as error:
                SoundcardOutput(SoundcardConfig(enabled=True, device="pulse:62"))

        self.assertNotIsInstance(error.exception, SoundcardDependencyError)

    def test_missing_plain_sounddevice_name_does_not_require_pactl(self) -> None:
        fake_sd = FakeSounddevice()

        with patch.dict(sys.modules, {"sounddevice": fake_sd}), patch(
            "rtl_weatherband.soundcard.subprocess.run",
            side_effect=AssertionError("pactl should not be called"),
        ):
            with self.assertRaises(SoundcardError) as error:
                SoundcardOutput(SoundcardConfig(enabled=True, device="missing device"))

        self.assertNotIsInstance(error.exception, SoundcardDependencyError)

    def test_alsa_device_does_not_require_pactl(self) -> None:
        fake_sd = FakeSounddevice()

        with patch.dict(sys.modules, {"sounddevice": fake_sd}), patch(
            "rtl_weatherband.soundcard.subprocess.run",
            side_effect=AssertionError("pactl should not be called"),
        ):
            with self.assertRaises(SoundcardError) as error:
                SoundcardOutput(SoundcardConfig(enabled=True, device="plughw:9"))

        self.assertNotIsInstance(error.exception, SoundcardDependencyError)


if __name__ == "__main__":
    unittest.main()
