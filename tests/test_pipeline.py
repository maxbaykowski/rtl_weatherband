from __future__ import annotations

import threading
import time
import unittest
import queue
from unittest.mock import patch

import numpy as np

from rtl_weatherband.config import (
    AppConfig,
    AudioConfig,
    CsdrServerConfig,
    DeemphasisConfig,
    FallbackConfig,
    IcecastConfig,
    SoundcardConfig,
    StationConfig,
    VolumeConfig,
)
from rtl_weatherband.csdr_server import IqStream
from rtl_weatherband.fallback_audio import FallbackAudio
from rtl_weatherband.pipeline import (
    AUDIO_FRAME_BYTES,
    PCM_FRAME_BYTES,
    SILENCE_FRAME,
    SILENCE_FRAME_SAMPLES,
    EncoderWorker,
    FallbackPlaybackState,
    OutputWorker,
    SoundcardWorker,
    StreamPipeline,
    _next_fallback_frame,
)
from rtl_weatherband.soundcard import SoundcardError
from rtl_weatherband.soundcard import SoundcardDependencyError


class PcmSink:
    def __init__(self) -> None:
        self.data = bytearray()
        self.closed = False

    def write(self, chunk: bytes) -> int:
        self.data.extend(chunk)
        return len(chunk)

    def close(self) -> None:
        self.closed = True


class FakeEncoder:
    header = b""

    def __init__(self) -> None:
        self.closed = False

    def encode(self, pcm: bytes) -> bytes:
        return pcm

    def flush(self) -> bytes:
        return b""

    def close(self) -> None:
        self.closed = True


class FakeSoundcardOutput:
    def __init__(self, config: SoundcardConfig) -> None:
        self.config = config
        self.device = config.device
        self.frames: list[bytes] = []
        self.closed = False

    def write(self, pcm: bytes) -> None:
        self.frames.append(pcm)

    def close(self) -> None:
        self.closed = True


class FakeIqSocket:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def recv(self, count: int, flags: int = 0) -> bytes:
        payload = self.payload[:count]
        self.payload = self.payload[count:]
        return payload


def audio_frame(value: float) -> bytes:
    return np.full(SILENCE_FRAME_SAMPLES, value, dtype="<f4").tobytes()


class PipelineTests(unittest.TestCase):
    def make_pipeline(self) -> StreamPipeline:
        return StreamPipeline(
            AudioConfig(),
            (
                IcecastConfig(
                    host="127.0.0.1",
                    port=8000,
                    mount="/nwr.mp3",
                    username="source",
                    password="hackme",
                    format="mp3",
                    sample_rate=16000,
                    bitrate=32,
                ),
            ),
            CsdrServerConfig(host="127.0.0.1", port=4951),
            162_550_000,
        )

    def app_config(self, pipeline: StreamPipeline, fallback: FallbackConfig) -> AppConfig:
        return AppConfig(
            csdr_server=pipeline.csdr_server,
            station=StationConfig(frequency=pipeline.frequency_hz / 1_000_000),
            icecast=pipeline.icecast,
            audio=pipeline.audio,
            fallback=fallback,
        )

    def test_pcm_playback_outputs_silence_when_no_audio_is_available(self) -> None:
        pipeline = self.make_pipeline()
        encoder = FakeEncoder()
        encoder_group = EncoderWorker(
            key=("mp3", 16000, 32),
            config=pipeline.icecast[0],
            encoder=encoder,
            pcm_queue=queue.Queue(),
            stop_event=threading.Event(),
            thread=threading.Thread(),
        )
        pipeline.encoder_groups.append(encoder_group)
        try:
            thread = threading.Thread(
                target=pipeline._run_pcm_playback_worker,
            )
            thread.start()
            time.sleep(0.03)
            pipeline.stop_event.set()
            thread.join(timeout=1)
        finally:
            pipeline.stop_event.set()

        self.assertFalse(thread.is_alive())
        self.assertEqual(encoder_group.pcm_queue.get_nowait(), SILENCE_FRAME)

    def test_icecast_writer_sends_cached_encoder_header_before_audio(self) -> None:
        sink = PcmSink()
        pipeline = self.make_pipeline()
        encoded_queue: queue.Queue[bytes] = queue.Queue()
        encoder = FakeEncoder()
        encoder.header = b"ogg-header"
        encoder_group = EncoderWorker(
            key=("ogg", 16000, 32),
            config=pipeline.icecast[0],
            encoder=encoder,
            pcm_queue=queue.Queue(),
            stop_event=threading.Event(),
            thread=threading.Thread(),
        )
        output = OutputWorker(
            config=pipeline.icecast[0],
            encoder_group=encoder_group,
            encoded_queue=encoded_queue,
            stop_event=threading.Event(),
            thread=threading.Thread(),
        )
        encoded_queue.put_nowait(b"audio")

        try:
            thread = threading.Thread(
                target=pipeline._run_icecast_writer,
                args=(output, sink),
            )
            thread.start()
            time.sleep(0.03)
            output.stop_event.set()
            thread.join(timeout=1)
        finally:
            output.stop_event.set()

        self.assertFalse(thread.is_alive())
        self.assertTrue(bytes(sink.data).startswith(b"ogg-headeraudio"))
        self.assertTrue(sink.closed)

    def test_identical_encoder_settings_share_one_encoder_worker(self) -> None:
        pipeline = self.make_pipeline()
        first = pipeline.icecast[0]
        second = IcecastConfig(
            host="stream.example",
            port=8000,
            mount="/other.mp3",
            username="source",
            password="hackme",
            format="mp3",
            sample_rate=16000,
            bitrate=32,
        )

        with patch(
            "rtl_weatherband.pipeline.create_audio_encoder",
            side_effect=[FakeEncoder()],
        ) as create_encoder:
            first_group = pipeline._get_or_create_encoder_worker(first)
            second_group = pipeline._get_or_create_encoder_worker(second)

        self.assertIs(first_group, second_group)
        self.assertEqual(create_encoder.call_count, 1)

    def test_different_encoder_settings_use_separate_encoder_workers(self) -> None:
        pipeline = self.make_pipeline()
        first = pipeline.icecast[0]
        second = IcecastConfig(
            host="127.0.0.1",
            port=8000,
            mount="/other.mp3",
            username="source",
            password="hackme",
            format="mp3",
            sample_rate=24000,
            bitrate=32,
        )

        with patch(
            "rtl_weatherband.pipeline.create_audio_encoder",
            side_effect=[FakeEncoder(), FakeEncoder()],
        ) as create_encoder:
            first_group = pipeline._get_or_create_encoder_worker(first)
            second_group = pipeline._get_or_create_encoder_worker(second)

        self.assertIsNot(first_group, second_group)
        self.assertEqual(create_encoder.call_count, 2)

    def test_temporary_destination_removal_can_keep_encoder_alive(self) -> None:
        pipeline = self.make_pipeline()
        config = pipeline.icecast[0]
        sink = PcmSink()
        with patch(
            "rtl_weatherband.pipeline.create_audio_encoder",
            return_value=FakeEncoder(),
        ):
            output = pipeline._create_output_worker(config, sink)
        pipeline.outputs.append(output)
        output.thread.start()
        encoder_group = output.encoder_group

        pipeline.apply_icecast_outputs(
            (),
            [config],
            [],
            stop_unused_encoders=False,
        )

        self.assertEqual(pipeline.outputs, [])
        self.assertIn(encoder_group, pipeline.encoder_groups)
        self.assertFalse(encoder_group.stop_event.is_set())
        encoder_group.stop_event.set()

    def test_idle_frame_uses_silence_before_fallback_timeout(self) -> None:
        pipeline = self.make_pipeline()
        pipeline.fallback = FallbackConfig(silence_timeout_seconds=30)
        pipeline.fallback_audio = FallbackAudio(
            sample_rate=16000,
            pcm=b"\x01\x00" * 320,
            duration_seconds=0.02,
            source="test",
        )

        frame = pipeline._idle_pcm_frame()

        self.assertEqual(frame, SILENCE_FRAME)
        self.assertFalse(pipeline.fallback_state.active)

    def test_idle_frame_uses_fallback_after_timeout(self) -> None:
        pipeline = self.make_pipeline()
        pipeline.fallback = FallbackConfig(silence_timeout_seconds=30)
        pipeline.fallback_audio = FallbackAudio(
            sample_rate=16000,
            pcm=b"\x01\x00" * 320,
            duration_seconds=0.02,
            source="test",
        )
        pipeline.last_pcm_at = time.monotonic() - 31

        frame = pipeline._idle_pcm_frame()

        self.assertEqual(frame, b"\x01\x00" * 320)
        self.assertTrue(pipeline.fallback_state.active)

    def test_queued_audio_does_not_reset_fallback_state(self) -> None:
        pipeline = self.make_pipeline()
        pipeline.fallback_state.active = True

        pipeline._queue_audio(np.full(SILENCE_FRAME_SAMPLES, 0.25, dtype=np.float32))

        self.assertTrue(pipeline.fallback_state.active)
        self.assertEqual(pipeline.pcm_queue.get_nowait(), audio_frame(0.25))

    def test_output_pcm_resets_fallback_state(self) -> None:
        pipeline = self.make_pipeline()
        pipeline.fallback_state.active = True

        pipeline._mark_pcm_output()

        self.assertFalse(pipeline.fallback_state.active)

    def test_write_pcm_frame_fans_same_pcm_to_icecast_and_soundcard(self) -> None:
        pipeline = self.make_pipeline()
        encoder_group = EncoderWorker(
            key=("mp3", 16000, 32),
            config=pipeline.icecast[0],
            encoder=FakeEncoder(),
            pcm_queue=queue.Queue(),
            stop_event=threading.Event(),
            thread=threading.Thread(),
        )
        soundcard_worker = SoundcardWorker(
            config=SoundcardConfig(enabled=True),
            output=FakeSoundcardOutput(SoundcardConfig(enabled=True)),
        )
        pipeline.encoder_groups.append(encoder_group)
        pipeline.soundcard_workers = [soundcard_worker]

        pipeline._write_pcm_frame(b"\x03\x00" * 320)

        self.assertEqual(encoder_group.pcm_queue.get_nowait(), b"\x03\x00" * 320)
        self.assertEqual(soundcard_worker.output.frames, [b"\x03\x00" * 320])

    def test_write_pcm_frame_fans_same_pcm_to_multiple_soundcards(self) -> None:
        pipeline = self.make_pipeline()
        first = SoundcardWorker(
            config=SoundcardConfig(enabled=True, device="first"),
            output=FakeSoundcardOutput(SoundcardConfig(enabled=True, device="first")),
        )
        second = SoundcardWorker(
            config=SoundcardConfig(enabled=True, device="second"),
            output=FakeSoundcardOutput(SoundcardConfig(enabled=True, device="second")),
        )
        pipeline.soundcard_workers = [first, second]

        pipeline._write_pcm_frame(b"\x04\x00" * 320)

        self.assertEqual(first.output.frames, [b"\x04\x00" * 320])
        self.assertEqual(second.output.frames, [b"\x04\x00" * 320])

    def test_pcm_buffer_outputs_silence_until_target_is_filled(self) -> None:
        pipeline = self.make_pipeline()
        pipeline.csdr_server = CsdrServerConfig(
            host="127.0.0.1",
            port=4951,
            buffer_seconds=0.04,
        )
        first_frame = audio_frame(0.1)
        second_frame = audio_frame(0.2)
        pending = bytearray(first_frame)

        pipeline._apply_pcm_buffer_target(pending)
        frame = pipeline._next_buffered_pcm_frame(pending)

        self.assertIsNone(frame)
        pending.extend(second_frame)
        frame = pipeline._next_buffered_pcm_frame(pending)

        self.assertEqual(frame, first_frame)
        self.assertTrue(pipeline.pcm_buffer_ready)

    def test_lower_pcm_buffer_target_skips_old_audio(self) -> None:
        pipeline = self.make_pipeline()
        pipeline.csdr_server = CsdrServerConfig(
            host="127.0.0.1",
            port=4951,
            buffer_seconds=0.02,
        )
        pipeline.pcm_buffer_target_bytes = AUDIO_FRAME_BYTES * 3
        pipeline.pcm_buffer_ready = True
        pending = bytearray(
            audio_frame(0.1)
            + audio_frame(0.2)
            + audio_frame(0.3)
        )

        pipeline._apply_pcm_buffer_target(pending)
        frame = pipeline._next_buffered_pcm_frame(pending)

        self.assertEqual(frame, audio_frame(0.3))

    def test_pcm_buffer_refills_when_it_reaches_low_water_mark(self) -> None:
        pipeline = self.make_pipeline()
        pipeline.csdr_server = CsdrServerConfig(
            host="127.0.0.1",
            port=4951,
            buffer_seconds=0.08,
        )
        pipeline.pcm_buffer_target_bytes = AUDIO_FRAME_BYTES * 4
        pipeline.pcm_buffer_ready = True
        first_frame = audio_frame(0.1)
        pending = bytearray(first_frame)

        frame = pipeline._next_buffered_pcm_frame(pending)

        self.assertIsNone(frame)
        self.assertFalse(pipeline.pcm_buffer_ready)
        self.assertEqual(pending, first_frame)

        pending.extend(audio_frame(0.2) * 3)
        frame = pipeline._next_buffered_pcm_frame(pending)

        self.assertEqual(frame, first_frame)
        self.assertTrue(pipeline.pcm_buffer_ready)

    def test_buffer_only_reload_does_not_queue_csdr_hotswap(self) -> None:
        pipeline = self.make_pipeline()
        new_config = AppConfig(
            csdr_server=CsdrServerConfig(
                host="127.0.0.1",
                port=4951,
                buffer_seconds=5,
            ),
            station=StationConfig(frequency=pipeline.frequency_hz / 1_000_000),
            icecast=pipeline.icecast,
            audio=pipeline.audio,
            fallback=FallbackConfig(silence_timeout_seconds=30),
        )

        applied = pipeline.apply_runtime_config(new_config)

        self.assertEqual(applied.csdr_server.buffer_seconds, 5)
        self.assertEqual(pipeline.csdr_server.buffer_seconds, 5)
        self.assertIsNone(pipeline.pending_receiver)

    def test_soundcard_reload_starts_local_output(self) -> None:
        pipeline = self.make_pipeline()
        config = AppConfig(
            csdr_server=pipeline.csdr_server,
            station=StationConfig(frequency=pipeline.frequency_hz / 1_000_000),
            icecast=pipeline.icecast,
            audio=pipeline.audio,
            fallback=pipeline.fallback,
            soundcard=(SoundcardConfig(enabled=True, device="plughw:1,0"),),
        )

        with patch(
            "rtl_weatherband.pipeline.SoundcardOutput",
            side_effect=lambda soundcard_config: FakeSoundcardOutput(soundcard_config),
        ):
            applied = pipeline.apply_runtime_config(config)

        try:
            self.assertEqual(applied.soundcard, config.soundcard)
            self.assertEqual(len(pipeline.soundcard_workers), 1)
            self.assertEqual(pipeline.soundcard_workers[0].config.device, "plughw:1,0")
            self.assertIsNone(pipeline.soundcard_workers[0].error)
        finally:
            pipeline._stop_soundcard_workers(pipeline.soundcard_workers)
            pipeline.soundcard_workers = []

    def test_soundcard_reload_failure_keeps_existing_output(self) -> None:
        pipeline = self.make_pipeline()
        old_config = SoundcardConfig(enabled=True, device="default")
        old_output = FakeSoundcardOutput(old_config)
        old_worker = SoundcardWorker(
            config=old_config,
            output=old_output,
        )
        pipeline.soundcard = (old_config,)
        pipeline.soundcard_workers = [old_worker]
        new_config = AppConfig(
            csdr_server=pipeline.csdr_server,
            station=StationConfig(frequency=pipeline.frequency_hz / 1_000_000),
            icecast=pipeline.icecast,
            audio=pipeline.audio,
            fallback=pipeline.fallback,
            soundcard=(SoundcardConfig(enabled=True, device="missing"),),
        )

        with patch(
            "rtl_weatherband.pipeline.SoundcardOutput",
            side_effect=RuntimeError("no such device"),
        ):
            applied = pipeline.apply_runtime_config(new_config)

        self.assertEqual(applied.soundcard, (old_config,))
        self.assertEqual(pipeline.soundcard_workers, [old_worker])
        self.assertFalse(old_output.closed)

    def test_soundcard_reload_enable_failure_keeps_disabled_old_settings(self) -> None:
        pipeline = self.make_pipeline()
        new_config = AppConfig(
            csdr_server=pipeline.csdr_server,
            station=StationConfig(frequency=pipeline.frequency_hz / 1_000_000),
            icecast=pipeline.icecast,
            audio=pipeline.audio,
            fallback=pipeline.fallback,
            soundcard=(SoundcardConfig(enabled=True, device="missing"),),
        )

        with patch(
            "rtl_weatherband.pipeline.SoundcardOutput",
            side_effect=SoundcardError("missing device"),
        ):
            applied = pipeline.apply_runtime_config(new_config)

        self.assertEqual(applied.soundcard, ())
        self.assertEqual(pipeline.soundcard, ())
        self.assertEqual(pipeline.soundcard_workers, [])

    def test_soundcard_reload_duplicate_resolved_device_keeps_existing_output(self) -> None:
        pipeline = self.make_pipeline()
        old_config = SoundcardConfig(enabled=True, device="default")
        old_output = FakeSoundcardOutput(old_config)
        old_worker = SoundcardWorker(config=old_config, output=old_output)
        pipeline.soundcard = (old_config,)
        pipeline.soundcard_workers = [old_worker]
        new_config = AppConfig(
            csdr_server=pipeline.csdr_server,
            station=StationConfig(frequency=pipeline.frequency_hz / 1_000_000),
            icecast=pipeline.icecast,
            audio=pipeline.audio,
            fallback=pipeline.fallback,
            soundcard=(
                SoundcardConfig(enabled=True, device="first"),
                SoundcardConfig(enabled=True, device="second"),
            ),
        )

        def create_output(config: SoundcardConfig) -> FakeSoundcardOutput:
            output = FakeSoundcardOutput(config)
            output.device = "same-device"
            return output

        with patch(
            "rtl_weatherband.pipeline.SoundcardOutput",
            side_effect=create_output,
        ):
            applied = pipeline.apply_runtime_config(new_config)

        self.assertEqual(applied.soundcard, (old_config,))
        self.assertEqual(pipeline.soundcard_workers, [old_worker])
        self.assertFalse(old_output.closed)

    def test_soundcard_reload_can_disable_existing_output(self) -> None:
        pipeline = self.make_pipeline()
        old_config = SoundcardConfig(enabled=True, device="default")
        old_output = FakeSoundcardOutput(old_config)
        old_worker = SoundcardWorker(
            config=old_config,
            output=old_output,
        )
        pipeline.soundcard = (old_config,)
        pipeline.soundcard_workers = [old_worker]
        new_config = AppConfig(
            csdr_server=pipeline.csdr_server,
            station=StationConfig(frequency=pipeline.frequency_hz / 1_000_000),
            icecast=pipeline.icecast,
            audio=pipeline.audio,
            fallback=pipeline.fallback,
            soundcard=(),
        )

        applied = pipeline.apply_runtime_config(new_config)

        self.assertEqual(applied.soundcard, ())
        self.assertEqual(pipeline.soundcard_workers, [])
        self.assertTrue(old_output.closed)

    def test_soundcard_reload_disables_only_removed_output(self) -> None:
        pipeline = self.make_pipeline()
        pulse_config = SoundcardConfig(enabled=True, device="pulse:64")
        alsa_config = SoundcardConfig(enabled=True, device="plughw:1")
        pulse_output = FakeSoundcardOutput(pulse_config)
        alsa_output = FakeSoundcardOutput(alsa_config)
        pulse_worker = SoundcardWorker(config=pulse_config, output=pulse_output)
        alsa_worker = SoundcardWorker(config=alsa_config, output=alsa_output)
        pipeline.soundcard = (pulse_config, alsa_config)
        pipeline.soundcard_workers = [pulse_worker, alsa_worker]
        new_config = AppConfig(
            csdr_server=pipeline.csdr_server,
            station=StationConfig(frequency=pipeline.frequency_hz / 1_000_000),
            icecast=pipeline.icecast,
            audio=pipeline.audio,
            fallback=pipeline.fallback,
            soundcard=(alsa_config,),
        )

        with patch(
            "rtl_weatherband.pipeline.SoundcardOutput",
            side_effect=AssertionError("unchanged output should not be reopened"),
        ):
            applied = pipeline.apply_runtime_config(new_config)

        self.assertEqual(applied.soundcard, (alsa_config,))
        self.assertEqual(pipeline.soundcard_workers, [alsa_worker])
        self.assertTrue(pulse_output.closed)
        self.assertFalse(alsa_output.closed)

    def test_soundcard_reload_adds_output_without_reopening_unchanged_output(self) -> None:
        pipeline = self.make_pipeline()
        old_config = SoundcardConfig(enabled=True, device="plughw:1")
        new_config = SoundcardConfig(enabled=True, device="pulse:64")
        old_output = FakeSoundcardOutput(old_config)
        old_worker = SoundcardWorker(config=old_config, output=old_output)
        new_output = FakeSoundcardOutput(new_config)
        pipeline.soundcard = (old_config,)
        pipeline.soundcard_workers = [old_worker]
        app_config = AppConfig(
            csdr_server=pipeline.csdr_server,
            station=StationConfig(frequency=pipeline.frequency_hz / 1_000_000),
            icecast=pipeline.icecast,
            audio=pipeline.audio,
            fallback=pipeline.fallback,
            soundcard=(old_config, new_config),
        )

        with patch(
            "rtl_weatherband.pipeline.SoundcardOutput",
            return_value=new_output,
        ) as output_class:
            applied = pipeline.apply_runtime_config(app_config)

        self.assertEqual(applied.soundcard, (old_config, new_config))
        output_class.assert_called_once_with(new_config)
        self.assertEqual(len(pipeline.soundcard_workers), 2)
        self.assertIs(pipeline.soundcard_workers[0], old_worker)
        self.assertIs(pipeline.soundcard_workers[1].output, new_output)
        self.assertFalse(old_output.closed)

    def test_soundcard_startup_failure_keeps_icecast_running(self) -> None:
        pipeline = self.make_pipeline()
        pipeline.soundcard = (SoundcardConfig(enabled=True, device="missing"),)

        with patch(
            "rtl_weatherband.pipeline.create_audio_encoder",
            return_value=FakeEncoder(),
        ), patch(
            "rtl_weatherband.pipeline.SoundcardOutput",
            side_effect=SoundcardError("missing device"),
        ), patch(
            "rtl_weatherband.pipeline.open_iq_stream",
            side_effect=ConnectionError("offline"),
        ):
            pipeline.start([PcmSink()])
            self.assertEqual(pipeline.soundcard_workers, [])
            self.assertEqual(len(pipeline.encoder_groups), 1)
            self.assertTrue(pipeline.playback_thread.is_alive())
            pipeline.stop()

    def test_soundcard_dependency_failure_still_aborts_startup(self) -> None:
        pipeline = self.make_pipeline()
        pipeline.soundcard = (SoundcardConfig(enabled=True, device="pulse:61"),)

        with patch(
            "rtl_weatherband.pipeline.create_audio_encoder",
            return_value=FakeEncoder(),
        ), patch(
            "rtl_weatherband.pipeline.SoundcardOutput",
            side_effect=SoundcardDependencyError("pactl missing"),
        ):
            with self.assertRaises(SoundcardDependencyError):
                pipeline.start([PcmSink()])
            pipeline.stop()

    def test_soundcard_duplicate_resolved_device_aborts_startup(self) -> None:
        pipeline = self.make_pipeline()
        pipeline.soundcard = (
            SoundcardConfig(enabled=True, device="first"),
            SoundcardConfig(enabled=True, device="second"),
        )

        def create_output(config: SoundcardConfig) -> FakeSoundcardOutput:
            output = FakeSoundcardOutput(config)
            output.device = "same-device"
            return output

        with patch(
            "rtl_weatherband.pipeline.create_audio_encoder",
            return_value=FakeEncoder(),
        ), patch(
            "rtl_weatherband.pipeline.SoundcardOutput",
            side_effect=create_output,
        ):
            with self.assertRaises(SoundcardError):
                pipeline.start([PcmSink()])
            pipeline.stop()

    def test_queued_audio_is_sent_to_central_playback_queue(self) -> None:
        pipeline = self.make_pipeline()

        pipeline._queue_audio(np.full(SILENCE_FRAME_SAMPLES, 0.5, dtype=np.float32))

        self.assertEqual(pipeline.pcm_queue.get_nowait(), audio_frame(0.5))

    def test_audio_effects_apply_after_buffered_audio_is_drained(self) -> None:
        pipeline = self.make_pipeline()
        pipeline.audio = AudioConfig(
            deemphasis=DeemphasisConfig(enabled=False),
            volume=VolumeConfig(enabled=True, multiplier=2.0),
        )
        pipeline.csdr_server = CsdrServerConfig(host="127.0.0.1", port=4951, buffer_seconds=0)
        encoder_group = EncoderWorker(
            key=("mp3", 16000, 32),
            config=pipeline.icecast[0],
            encoder=FakeEncoder(),
            pcm_queue=queue.Queue(),
            stop_event=threading.Event(),
            thread=threading.Thread(),
        )
        pipeline.encoder_groups.append(encoder_group)
        pipeline._queue_audio(np.full(SILENCE_FRAME_SAMPLES, 0.25, dtype=np.float32))

        try:
            thread = threading.Thread(target=pipeline._run_pcm_playback_worker)
            thread.start()
            frame = encoder_group.pcm_queue.get(timeout=1)
            pipeline.stop_event.set()
            thread.join(timeout=1)
        finally:
            pipeline.stop_event.set()

        samples = np.frombuffer(frame, dtype="<i2")
        self.assertGreater(np.mean(samples), 16000)
        self.assertLess(np.mean(samples), 16500)

    def test_hotswap_probe_reads_only_available_iq_chunk(self) -> None:
        pipeline = self.make_pipeline()
        iq_socket = FakeIqSocket(b"\x00" * 16)
        iq_stream = IqStream(iq_socket, FakeIqSocket(b""), {}, "s16")

        with patch("rtl_weatherband.pipeline.select.select", return_value=([iq_socket], [], [])):
            chunk = pipeline._read_hotswap_iq_probe(iq_stream, 0.1)

        self.assertEqual(chunk, b"\x00" * 4)
        self.assertEqual(iq_socket.payload, b"\x00" * 12)

    def test_fallback_config_reload_does_not_reset_idle_timer(self) -> None:
        pipeline = self.make_pipeline()
        pipeline.fallback = FallbackConfig(silence_timeout_seconds=120)
        pipeline.fallback_audio = FallbackAudio(
            sample_rate=16000,
            pcm=b"\x01\x00" * 320,
            duration_seconds=0.02,
            source="test",
        )
        pipeline.last_pcm_at = time.monotonic() - 45

        pipeline.apply_runtime_config(
            self.app_config(
                pipeline,
                FallbackConfig(silence_timeout_seconds=30),
            )
        )
        frame = pipeline._idle_pcm_frame()

        self.assertEqual(len(frame), len(SILENCE_FRAME))
        self.assertTrue(pipeline.fallback_state.active)

    def test_fallback_config_reload_does_not_stop_active_fallback(self) -> None:
        pipeline = self.make_pipeline()
        pipeline.fallback = FallbackConfig(silence_timeout_seconds=30)
        pipeline.fallback_audio = FallbackAudio(
            sample_rate=16000,
            pcm=b"\x01\x00" * 320,
            duration_seconds=0.02,
            source="test",
        )
        pipeline.last_pcm_at = time.monotonic() - 60
        pipeline.fallback_state.active = True
        pipeline.fallback_state.position = 4

        pipeline.apply_runtime_config(
            self.app_config(
                pipeline,
                FallbackConfig(silence_timeout_seconds=120, loop_delay_seconds=1),
            )
        )
        frame = pipeline._idle_pcm_frame()

        self.assertTrue(pipeline.fallback_state.active)
        self.assertEqual(len(frame), len(SILENCE_FRAME))
        self.assertNotEqual(pipeline.fallback_state.position, 0)

    def test_fallback_loop_delay_controls_restart_gap(self) -> None:
        audio = FallbackAudio(
            sample_rate=16000,
            pcm=b"\x01\x00" * 160,
            duration_seconds=0.01,
            source="test",
        )

        no_delay = _next_fallback_frame(audio, FallbackPlaybackState(), 0.0)
        delayed_state = FallbackPlaybackState()
        delayed = _next_fallback_frame(audio, delayed_state, 0.01)

        self.assertEqual(no_delay, b"\x01\x00" * 320)
        self.assertEqual(delayed, b"\x01\x00" * 160 + b"\x00\x00" * 160)


if __name__ == "__main__":
    unittest.main()
