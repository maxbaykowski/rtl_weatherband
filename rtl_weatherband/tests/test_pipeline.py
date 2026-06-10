from __future__ import annotations

import threading
import time
import unittest

from rtl_weatherband.config import (
    AudioConfig,
    CsdrServerConfig,
    IcecastConfig,
)
from rtl_weatherband.pipeline import SILENCE_FRAME, StreamPipeline


class PcmSink:
    def __init__(self) -> None:
        self.data = bytearray()
        self.closed = False

    def write(self, chunk: bytes) -> int:
        self.data.extend(chunk)
        return len(chunk)

    def close(self) -> None:
        self.closed = True


class PipelineTests(unittest.TestCase):
    def make_pipeline(self) -> StreamPipeline:
        return StreamPipeline(
            AudioConfig(),
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
            CsdrServerConfig(host="127.0.0.1", port=4951),
            162_550_000,
        )

    def test_mp3_ffmpeg_command(self) -> None:
        pipeline = self.make_pipeline()
        self.assertEqual(
            pipeline._ffmpeg_command(),
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-f",
                "s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-i",
                "pipe:0",
                "-vn",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-b:a",
                "32k",
                "-f",
                "mp3",
                "-codec:a",
                "libmp3lame",
                "pipe:1",
            ],
        )

    def test_ogg_ffmpeg_command(self) -> None:
        pipeline = StreamPipeline(
            AudioConfig(),
            IcecastConfig(
                host="127.0.0.1",
                port=8000,
                mount="/nwr.ogg",
                username="source",
                password="hackme",
                format="ogg",
                sample_rate=22050,
                bitrate=40,
            ),
            CsdrServerConfig(host="127.0.0.1", port=4951),
            162_550_000,
        )
        self.assertIn("libvorbis", pipeline._ffmpeg_command())

    def test_pcm_writer_outputs_silence_when_no_audio_is_available(self) -> None:
        sink = PcmSink()
        pipeline = self.make_pipeline()
        try:
            thread = threading.Thread(
                target=pipeline._run_pcm_writer,
                args=(sink,),
            )
            thread.start()
            time.sleep(0.03)
            pipeline.stop_event.set()
            thread.join(timeout=1)
        finally:
            pipeline.stop_event.set()

        self.assertFalse(thread.is_alive())
        self.assertEqual(bytes(sink.data[: len(SILENCE_FRAME)]), SILENCE_FRAME)
        self.assertTrue(sink.closed)


if __name__ == "__main__":
    unittest.main()
