from __future__ import annotations

import threading
import time
import unittest
import queue
from unittest.mock import patch

from rtl_weatherband.config import (
    AppConfig,
    AudioConfig,
    CsdrServerConfig,
    FallbackConfig,
    IcecastConfig,
    StationConfig,
)
from rtl_weatherband.csdr_server import IqStream
from rtl_weatherband.fallback_audio import FallbackAudio
from rtl_weatherband.pipeline import (
    PCM_FRAME_BYTES,
    SILENCE_FRAME,
    EncoderWorker,
    FallbackPlaybackState,
    OutputWorker,
    StreamPipeline,
    _next_fallback_frame,
)


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


class FakeIqSocket:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def recv(self, count: int, flags: int = 0) -> bytes:
        payload = self.payload[:count]
        self.payload = self.payload[count:]
        return payload


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

    def test_pcm_writer_outputs_silence_when_no_audio_is_available(self) -> None:
        pipeline = self.make_pipeline()
        pcm_queue: queue.Queue[bytes] = queue.Queue()
        encoded_queue: queue.Queue[bytes] = queue.Queue()
        encoder = FakeEncoder()
        encoder_group = EncoderWorker(
            key=("mp3", 16000, 32),
            config=pipeline.icecast[0],
            encoder=encoder,
            pcm_queue=pcm_queue,
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
        encoder_group.outputs.append(output)
        try:
            thread = threading.Thread(
                target=pipeline._run_encoder_worker,
                args=(encoder_group,),
            )
            thread.start()
            time.sleep(0.03)
            encoder_group.stop_event.set()
            thread.join(timeout=1)
        finally:
            encoder_group.stop_event.set()

        self.assertFalse(thread.is_alive())
        self.assertEqual(encoded_queue.get_nowait(), SILENCE_FRAME)

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
        encoder_group = EncoderWorker(
            key=("mp3", 16000, 32),
            config=pipeline.icecast[0],
            encoder=FakeEncoder(),
            pcm_queue=queue.Queue(),
            stop_event=threading.Event(),
            thread=threading.Thread(),
        )

        frame = pipeline._idle_pcm_frame(encoder_group)

        self.assertEqual(frame, SILENCE_FRAME)
        self.assertFalse(encoder_group.fallback_state.active)

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
        encoder_group = EncoderWorker(
            key=("mp3", 16000, 32),
            config=pipeline.icecast[0],
            encoder=FakeEncoder(),
            pcm_queue=queue.Queue(),
            stop_event=threading.Event(),
            thread=threading.Thread(),
        )

        frame = pipeline._idle_pcm_frame(encoder_group)

        self.assertEqual(frame, b"\x01\x00" * 320)
        self.assertTrue(encoder_group.fallback_state.active)

    def test_queued_pcm_does_not_reset_fallback_state(self) -> None:
        pipeline = self.make_pipeline()
        encoder_group = EncoderWorker(
            key=("mp3", 16000, 32),
            config=pipeline.icecast[0],
            encoder=FakeEncoder(),
            pcm_queue=queue.Queue(),
            stop_event=threading.Event(),
            thread=threading.Thread(),
        )
        encoder_group.fallback_state.active = True
        pipeline.encoder_groups.append(encoder_group)

        pipeline._queue_pcm(b"\x02\x00" * 320)

        self.assertTrue(encoder_group.fallback_state.active)
        self.assertEqual(encoder_group.pcm_queue.get_nowait(), b"\x02\x00" * 320)

    def test_output_pcm_resets_fallback_state(self) -> None:
        pipeline = self.make_pipeline()
        encoder_group = EncoderWorker(
            key=("mp3", 16000, 32),
            config=pipeline.icecast[0],
            encoder=FakeEncoder(),
            pcm_queue=queue.Queue(),
            stop_event=threading.Event(),
            thread=threading.Thread(),
        )
        encoder_group.fallback_state.active = True
        pipeline.encoder_groups.append(encoder_group)

        pipeline._mark_pcm_output()

        self.assertFalse(encoder_group.fallback_state.active)

    def test_pcm_buffer_outputs_silence_until_target_is_filled(self) -> None:
        pipeline = self.make_pipeline()
        pipeline.csdr_server = CsdrServerConfig(
            host="127.0.0.1",
            port=4951,
            buffer_seconds=0.04,
        )
        encoder_group = EncoderWorker(
            key=("mp3", 16000, 32),
            config=pipeline.icecast[0],
            encoder=FakeEncoder(),
            pcm_queue=queue.Queue(),
            stop_event=threading.Event(),
            thread=threading.Thread(),
        )
        pending = bytearray(b"\x01\x00" * 320)

        pipeline._apply_pcm_buffer_target(encoder_group, pending)
        frame = pipeline._next_buffered_pcm_frame(encoder_group, pending)

        self.assertIsNone(frame)
        pending.extend(b"\x02\x00" * 320)
        frame = pipeline._next_buffered_pcm_frame(encoder_group, pending)

        self.assertEqual(frame, b"\x01\x00" * 320)
        self.assertTrue(encoder_group.pcm_buffer_ready)

    def test_lower_pcm_buffer_target_skips_old_audio(self) -> None:
        pipeline = self.make_pipeline()
        pipeline.csdr_server = CsdrServerConfig(
            host="127.0.0.1",
            port=4951,
            buffer_seconds=0.02,
        )
        encoder_group = EncoderWorker(
            key=("mp3", 16000, 32),
            config=pipeline.icecast[0],
            encoder=FakeEncoder(),
            pcm_queue=queue.Queue(),
            stop_event=threading.Event(),
            thread=threading.Thread(),
            pcm_buffer_target_bytes=PCM_FRAME_BYTES * 3,
            pcm_buffer_ready=True,
        )
        pending = bytearray(
            b"\x01\x00" * 320
            + b"\x02\x00" * 320
            + b"\x03\x00" * 320
        )

        pipeline._apply_pcm_buffer_target(encoder_group, pending)
        frame = pipeline._next_buffered_pcm_frame(encoder_group, pending)

        self.assertEqual(frame, b"\x03\x00" * 320)

    def test_pcm_buffer_refills_when_it_reaches_low_water_mark(self) -> None:
        pipeline = self.make_pipeline()
        pipeline.csdr_server = CsdrServerConfig(
            host="127.0.0.1",
            port=4951,
            buffer_seconds=0.08,
        )
        encoder_group = EncoderWorker(
            key=("mp3", 16000, 32),
            config=pipeline.icecast[0],
            encoder=FakeEncoder(),
            pcm_queue=queue.Queue(),
            stop_event=threading.Event(),
            thread=threading.Thread(),
            pcm_buffer_target_bytes=PCM_FRAME_BYTES * 4,
            pcm_buffer_ready=True,
        )
        pending = bytearray(b"\x01\x00" * 320)

        frame = pipeline._next_buffered_pcm_frame(encoder_group, pending)

        self.assertIsNone(frame)
        self.assertFalse(encoder_group.pcm_buffer_ready)
        self.assertEqual(pending, b"\x01\x00" * 320)

        pending.extend(b"\x02\x00" * 320 * 3)
        frame = pipeline._next_buffered_pcm_frame(encoder_group, pending)

        self.assertEqual(frame, b"\x01\x00" * 320)
        self.assertTrue(encoder_group.pcm_buffer_ready)

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
        encoder_group = EncoderWorker(
            key=("mp3", 16000, 32),
            config=pipeline.icecast[0],
            encoder=FakeEncoder(),
            pcm_queue=queue.Queue(),
            stop_event=threading.Event(),
            thread=threading.Thread(),
        )

        pipeline.apply_runtime_config(
            self.app_config(
                pipeline,
                FallbackConfig(silence_timeout_seconds=30),
            )
        )
        frame = pipeline._idle_pcm_frame(encoder_group)

        self.assertEqual(len(frame), len(SILENCE_FRAME))
        self.assertTrue(encoder_group.fallback_state.active)

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
        encoder_group = EncoderWorker(
            key=("mp3", 16000, 32),
            config=pipeline.icecast[0],
            encoder=FakeEncoder(),
            pcm_queue=queue.Queue(),
            stop_event=threading.Event(),
            thread=threading.Thread(),
        )
        encoder_group.fallback_state.active = True
        encoder_group.fallback_state.position = 4
        pipeline.encoder_groups.append(encoder_group)

        pipeline.apply_runtime_config(
            self.app_config(
                pipeline,
                FallbackConfig(silence_timeout_seconds=120, loop_delay_seconds=1),
            )
        )
        frame = pipeline._idle_pcm_frame(encoder_group)

        self.assertTrue(encoder_group.fallback_state.active)
        self.assertEqual(len(frame), len(SILENCE_FRAME))
        self.assertNotEqual(encoder_group.fallback_state.position, 0)

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
