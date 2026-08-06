from __future__ import annotations

import sys
import types
import threading
import time
import unittest
import queue
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np

from rtl_weatherband.config import (
    AppConfig,
    AudioConfig,
    CsdrServerConfig,
    DeemphasisConfig,
    EasRecordingConfig,
    FallbackConfig,
    FilterConfig,
    IcecastConfig,
    IQ_SAMPLE_RATE,
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
    SILENCE_FRAME_SECONDS,
    SILENCE_FRAME_SAMPLES,
    EncoderWorker,
    EasRecorderWorker,
    FallbackPlaybackState,
    OutputWorker,
    SAME_TEST_INTER_ALERT_SILENCE_FRAMES,
    SoundcardWorker,
    StreamPipeline,
    _next_fallback_frame,
)
from rtl_weatherband.eas_recording import _same_test_header, generate_same_test_audio
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


class FakeEasRecorderOutput:
    def __init__(self, config: EasRecordingConfig) -> None:
        self.config = config
        self.frames: list[bytes] = []
        self.closed = False

    def write(self, pcm: bytes) -> None:
        self.frames.append(pcm)

    def close(self) -> None:
        self.closed = True


class FailingEasRecorderOutput(FakeEasRecorderOutput):
    def write(self, pcm: bytes) -> None:
        raise RuntimeError("recorder failed")


class FakeIqSocket:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def recv(self, count: int, flags: int = 0) -> bytes:
        payload = self.payload[:count]
        self.payload = self.payload[count:]
        return payload


class FakeSoxrResampleStream:
    def __init__(
        self,
        input_rate: int,
        output_rate: int,
        _channels: int,
        *,
        dtype: str,
    ) -> None:
        self.input_rate = input_rate
        self.output_rate = output_rate

    def resample_chunk(self, samples, last=False):
        if len(samples) == 0:
            return samples
        output_count = max(1, round(len(samples) * self.output_rate / self.input_rate))
        positions = np.linspace(0, len(samples) - 1, output_count)
        return np.interp(positions, np.arange(len(samples)), samples).astype(samples.dtype)


def audio_frame(value: float) -> bytes:
    return np.full(SILENCE_FRAME_SAMPLES, value, dtype="<f4").tobytes()


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.soxr_patch = patch.dict(
            sys.modules,
            {
                "soxr": types.SimpleNamespace(
                    ResampleStream=FakeSoxrResampleStream,
                )
            },
        )
        self.soxr_patch.start()

    def tearDown(self) -> None:
        self.soxr_patch.stop()

    def make_pipeline(self) -> StreamPipeline:
        fallback_audio = FallbackAudio(
            sample_rate=IQ_SAMPLE_RATE,
            pcm=b"\x00\x00" * SILENCE_FRAME_SAMPLES,
            duration_seconds=SILENCE_FRAME_SECONDS,
            source="test fallback",
        )
        with patch(
            "rtl_weatherband.pipeline.load_fallback_audio",
            return_value=fallback_audio,
        ):
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

    def test_icecast_reload_encoder_creation_failure_keeps_existing_output(self) -> None:
        pipeline = self.make_pipeline()
        old_config = pipeline.icecast[0]
        new_config = IcecastConfig(
            host="127.0.0.1",
            port=8000,
            mount="/new.mp3",
            username="source",
            password="hackme",
            format="ogg",
            sample_rate=22050,
            bitrate=32,
        )
        old_encoder_group = EncoderWorker(
            key=("mp3", 16000, 32),
            config=old_config,
            encoder=FakeEncoder(),
            pcm_queue=queue.Queue(),
            stop_event=threading.Event(),
            thread=threading.Thread(),
        )
        old_output = OutputWorker(
            config=old_config,
            encoder_group=old_encoder_group,
            encoded_queue=queue.Queue(),
            stop_event=threading.Event(),
            thread=threading.Thread(),
        )
        old_encoder_group.outputs.append(old_output)
        pipeline.encoder_groups = [old_encoder_group]
        pipeline.outputs = [old_output]

        with patch(
            "rtl_weatherband.pipeline.create_audio_encoder",
            side_effect=RuntimeError("encoder unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "encoder unavailable"):
                pipeline.apply_icecast_outputs(
                    (new_config,),
                    [old_config],
                    [(new_config, PcmSink())],
                )

        self.assertEqual(pipeline.outputs, [old_output])
        self.assertFalse(old_output.stop_event.is_set())
        self.assertEqual(old_encoder_group.outputs, [old_output])

    def test_idle_frame_uses_silence_before_fallback_timeout(self) -> None:
        pipeline = self.make_pipeline()
        pipeline.fallback = FallbackConfig(silence_timeout_seconds=30)
        pipeline.fallback_audio = FallbackAudio(
            sample_rate=16000,
            pcm=b"\x01\x00" * SILENCE_FRAME_SAMPLES,
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

        self.assertEqual(frame, b"\x01\x00" * SILENCE_FRAME_SAMPLES)
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

    def test_write_pcm_frame_fans_same_pcm_to_eas_recorder(self) -> None:
        pipeline = self.make_pipeline()
        eas_config = EasRecordingConfig(enabled=True)
        eas_output = FakeEasRecorderOutput(eas_config)
        pipeline.eas_recorder_worker = EasRecorderWorker(
            config=eas_config,
            output=eas_output,
        )

        pipeline._write_pcm_frame(b"\x03\x00" * 320)

        self.assertEqual(eas_output.frames, [b"\x03\x00" * 320])

    def test_eas_recorder_write_failure_does_not_stop_pipeline(self) -> None:
        pipeline = self.make_pipeline()
        eas_config = EasRecordingConfig(enabled=True)
        eas_output = FailingEasRecorderOutput(eas_config)
        pipeline.eas_recorder_worker = EasRecorderWorker(
            config=eas_config,
            output=eas_output,
        )

        pipeline._write_pcm_frame(b"\x03\x00" * 320)

        self.assertIsNone(pipeline.eas_recorder_worker)
        self.assertTrue(eas_output.closed)
        self.assertIsNone(pipeline.poll_exit())

    def test_same_test_audio_generator_produces_audio(self) -> None:
        audio = generate_same_test_audio(
            header="ZCZC-WXR-RWT-000000+0015-0010000-RTLWB-",
        )

        self.assertGreater(len(audio), IQ_SAMPLE_RATE)
        self.assertGreater(float(np.max(np.abs(audio))), 0.5)

    def test_same_test_audio_uses_valid_dmo_header_and_packaged_message(self) -> None:
        origin = datetime(2026, 6, 25, 20, 46, tzinfo=timezone.utc)
        header = _same_test_header(origin)
        audio = generate_same_test_audio(origin_time=origin)

        self.assertEqual(header, "ZCZC-WXR-DMO-999999+0015-1762046-RTLWB000-")
        self.assertEqual(len("RTLWB000"), 8)
        self.assertGreater(len(audio), round(IQ_SAMPLE_RATE * 25))
        self.assertLessEqual(float(np.max(np.abs(audio))), 1.0)

    def test_same_test_mode_disables_deemphasis_only(self) -> None:
        pipeline = self.make_pipeline()
        pipeline.same_test = True
        pipeline.audio = AudioConfig(
            deemphasis=DeemphasisConfig(enabled=True, tau=300),
            volume=VolumeConfig(enabled=True, multiplier=1.5),
            lowpass=FilterConfig(enabled=True, frequency=3000, sharpness=2),
        )

        audio = pipeline._audio_config()

        self.assertFalse(audio.deemphasis.enabled)
        self.assertTrue(audio.volume.enabled)
        self.assertTrue(audio.lowpass.enabled)

    def test_same_test_alert_is_queued_as_audio_frames(self) -> None:
        pipeline = self.make_pipeline()

        with patch("rtl_weatherband.pipeline.time.sleep"):
            pipeline._play_same_test_alert(
                datetime(2026, 6, 25, 20, 46, tzinfo=timezone.utc)
            )

        self.assertFalse(pipeline.pcm_queue.empty())

    def test_same_test_alert_requests_are_generated_immediately_and_queued(self) -> None:
        pipeline = self.make_pipeline()
        first = np.full(SILENCE_FRAME_SAMPLES, 0.1, dtype=np.float32)
        second = np.full(SILENCE_FRAME_SAMPLES, 0.2, dtype=np.float32)

        with patch(
            "rtl_weatherband.pipeline.generate_same_test_audio",
            side_effect=[first, second],
        ) as generate:
            pipeline.request_same_test_alert()
            pipeline.request_same_test_alert()

        self.assertEqual(generate.call_count, 2)
        self.assertIs(pipeline._pop_same_test_alert(), first)
        self.assertIs(pipeline._pop_same_test_alert(), second)

    def test_same_test_producer_outputs_silence_between_alerts(self) -> None:
        pipeline = self.make_pipeline()
        times = [
            datetime(2026, 6, 25, 20, 46, 30, tzinfo=timezone.utc),
            datetime(2026, 6, 25, 20, 46, 30, tzinfo=timezone.utc),
        ]

        def sleep_once(_seconds: float) -> None:
            pipeline.stop_event.set()

        with patch("rtl_weatherband.pipeline.datetime") as datetime_class, patch(
            "rtl_weatherband.pipeline.time.sleep",
            side_effect=sleep_once,
        ):
            datetime_class.now.side_effect = times
            pipeline._run_same_test_producer()

        frame = pipeline.pcm_queue.get_nowait()
        self.assertEqual(frame, np.zeros(SILENCE_FRAME_SAMPLES, dtype="<f4").tobytes())

    def test_same_test_producer_inserts_silence_between_queued_alerts(self) -> None:
        pipeline = self.make_pipeline()
        first = np.full(SILENCE_FRAME_SAMPLES, 0.1, dtype=np.float32)
        second = np.full(SILENCE_FRAME_SAMPLES, 0.2, dtype=np.float32)
        pipeline.same_test_alerts.append(first)
        pipeline.same_test_alerts.append(second)
        sleep_calls = 0

        def sleep_until_second_alert(_seconds: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls >= SAME_TEST_INTER_ALERT_SILENCE_FRAMES + 2:
                pipeline.stop_event.set()

        with patch(
            "rtl_weatherband.pipeline.time.sleep",
            side_effect=sleep_until_second_alert,
        ):
            pipeline._run_same_test_producer()

        frames = []
        while not pipeline.pcm_queue.empty():
            frames.append(np.frombuffer(pipeline.pcm_queue.get_nowait(), dtype="<f4"))

        self.assertTrue(np.allclose(frames[0], first))
        for frame in frames[1 : 1 + SAME_TEST_INTER_ALERT_SILENCE_FRAMES]:
            self.assertTrue(np.allclose(frame, 0.0))
        self.assertTrue(np.allclose(frames[1 + SAME_TEST_INTER_ALERT_SILENCE_FRAMES], second))

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

    def test_hotswap_preserves_iq_probe_bytes(self) -> None:
        pipeline = self.make_pipeline()
        receiver = (
            CsdrServerConfig(host="127.0.0.1", port=4952),
            pipeline.frequency_hz,
        )
        replacement = IqStream(FakeIqSocket(b""), FakeIqSocket(b""), {})
        probe = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        pipeline.pending_receiver = receiver

        with patch(
            "rtl_weatherband.pipeline.open_iq_stream",
            return_value=replacement,
        ), patch.object(
            pipeline,
            "_read_hotswap_iq_probe",
            return_value=probe,
        ):
            pipeline._connect_hotswap_iq_stream(receiver)

        self.assertEqual(pipeline.hotswap_result, (receiver, replacement, probe))

    def test_eas_recording_reload_starts_recorder(self) -> None:
        pipeline = self.make_pipeline()
        eas_config = EasRecordingConfig(enabled=True, directory="/tmp/eas-alerts")
        config = AppConfig(
            csdr_server=pipeline.csdr_server,
            station=StationConfig(frequency=pipeline.frequency_hz / 1_000_000),
            icecast=pipeline.icecast,
            audio=pipeline.audio,
            fallback=pipeline.fallback,
            eas_recording=eas_config,
        )

        with patch(
            "rtl_weatherband.pipeline.EasRecorderOutput",
            side_effect=lambda recorder_config: FakeEasRecorderOutput(recorder_config),
        ):
            applied = pipeline.apply_runtime_config(config)

        self.assertEqual(applied.eas_recording, eas_config)
        self.assertIsNotNone(pipeline.eas_recorder_worker)
        assert pipeline.eas_recorder_worker is not None
        self.assertEqual(pipeline.eas_recorder_worker.config, eas_config)
        pipeline._stop_eas_recorder_worker(pipeline.eas_recorder_worker)
        pipeline.eas_recorder_worker = None

    def test_eas_recording_reload_failure_keeps_existing_recorder(self) -> None:
        pipeline = self.make_pipeline()
        old_config = EasRecordingConfig(enabled=True, directory="/tmp/old-alerts")
        old_output = FakeEasRecorderOutput(old_config)
        old_worker = EasRecorderWorker(config=old_config, output=old_output)
        pipeline.eas_recording = old_config
        pipeline.eas_recorder_worker = old_worker
        new_config = AppConfig(
            csdr_server=pipeline.csdr_server,
            station=StationConfig(frequency=pipeline.frequency_hz / 1_000_000),
            icecast=pipeline.icecast,
            audio=pipeline.audio,
            fallback=pipeline.fallback,
            eas_recording=EasRecordingConfig(enabled=True, directory="/tmp/new-alerts"),
        )

        with patch(
            "rtl_weatherband.pipeline.EasRecorderOutput",
            side_effect=RuntimeError("multimon-ng missing"),
        ):
            applied = pipeline.apply_runtime_config(new_config)

        self.assertEqual(applied.eas_recording, old_config)
        self.assertIs(pipeline.eas_recorder_worker, old_worker)
        self.assertFalse(old_output.closed)

    def test_eas_recording_reload_can_disable_recorder(self) -> None:
        pipeline = self.make_pipeline()
        old_config = EasRecordingConfig(enabled=True, directory="/tmp/eas-alerts")
        old_output = FakeEasRecorderOutput(old_config)
        old_worker = EasRecorderWorker(config=old_config, output=old_output)
        pipeline.eas_recording = old_config
        pipeline.eas_recorder_worker = old_worker
        new_config = AppConfig(
            csdr_server=pipeline.csdr_server,
            station=StationConfig(frequency=pipeline.frequency_hz / 1_000_000),
            icecast=pipeline.icecast,
            audio=pipeline.audio,
            fallback=pipeline.fallback,
            eas_recording=EasRecordingConfig(enabled=False),
        )

        applied = pipeline.apply_runtime_config(new_config)

        self.assertFalse(applied.eas_recording.enabled)
        self.assertIsNone(pipeline.eas_recorder_worker)
        self.assertTrue(old_output.closed)

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
        times = np.arange(SILENCE_FRAME_SAMPLES, dtype=np.float32) / IQ_SAMPLE_RATE
        source = (np.sin(2 * np.pi * 1000 * times) * 0.25).astype(np.float32)
        pipeline._queue_audio(source)

        try:
            thread = threading.Thread(target=pipeline._run_pcm_playback_worker)
            thread.start()
            frame = encoder_group.pcm_queue.get(timeout=1)
            pipeline.stop_event.set()
            thread.join(timeout=1)
        finally:
            pipeline.stop_event.set()

        samples = np.frombuffer(frame, dtype="<i2")
        sample_rms = np.sqrt(np.mean(np.square(samples.astype(np.float32))))
        self.assertGreater(sample_rms, 11000)
        self.assertLess(sample_rms, 12000)

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
            sample_rate=IQ_SAMPLE_RATE,
            pcm=b"\x01\x00" * SILENCE_FRAME_SAMPLES,
            duration_seconds=0.02,
            source="test",
        )
        pipeline.last_pcm_at = time.monotonic() - 45

        with patch(
            "rtl_weatherband.pipeline.load_fallback_audio",
            return_value=pipeline.fallback_audio,
        ):
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
            sample_rate=IQ_SAMPLE_RATE,
            pcm=b"\x01\x00" * (SILENCE_FRAME_SAMPLES * 2),
            duration_seconds=0.04,
            source="test",
        )
        pipeline.last_pcm_at = time.monotonic() - 60
        pipeline.fallback_state.active = True
        pipeline.fallback_state.position = 4

        with patch(
            "rtl_weatherband.pipeline.load_fallback_audio",
            return_value=pipeline.fallback_audio,
        ):
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
            pcm=b"\x01\x00" * (SILENCE_FRAME_SAMPLES // 2),
            duration_seconds=0.01,
            source="test",
        )

        no_delay = _next_fallback_frame(audio, FallbackPlaybackState(), 0.0)
        delayed_state = FallbackPlaybackState()
        delayed = _next_fallback_frame(audio, delayed_state, 0.01)

        self.assertEqual(no_delay, b"\x01\x00" * SILENCE_FRAME_SAMPLES)
        self.assertEqual(
            delayed,
            b"\x01\x00" * (SILENCE_FRAME_SAMPLES // 2)
            + b"\x00\x00" * (SILENCE_FRAME_SAMPLES // 2),
        )


if __name__ == "__main__":
    unittest.main()
