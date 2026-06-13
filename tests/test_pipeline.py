from __future__ import annotations

import threading
import time
import unittest
import queue

from rtl_weatherband.config import (
    AudioConfig,
    CsdrServerConfig,
    IcecastConfig,
)
from rtl_weatherband.pipeline import SILENCE_FRAME, OutputWorker, StreamPipeline


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

    def test_pcm_writer_outputs_silence_when_no_audio_is_available(self) -> None:
        sink = PcmSink()
        pipeline = self.make_pipeline()
        pcm_queue: queue.Queue[bytes] = queue.Queue()
        encoder = FakeEncoder()
        output = OutputWorker(
            config=pipeline.icecast[0],
            encoder=encoder,
            pcm_queue=pcm_queue,
            stop_event=threading.Event(),
            thread=threading.Thread(),
        )
        try:
            thread = threading.Thread(
                target=pipeline._run_encoded_writer,
                args=(output, sink),
            )
            thread.start()
            time.sleep(0.03)
            output.stop_event.set()
            thread.join(timeout=1)
        finally:
            output.stop_event.set()

        self.assertFalse(thread.is_alive())
        self.assertEqual(bytes(sink.data[: len(SILENCE_FRAME)]), SILENCE_FRAME)
        self.assertTrue(sink.closed)


if __name__ == "__main__":
    unittest.main()
