from __future__ import annotations

import unittest

from rtl_weatherband.config import AudioConfig, DspConfig
from rtl_weatherband.pipeline import StreamPipeline


class PipelineTests(unittest.TestCase):
    def test_mp3_ffmpeg_command(self) -> None:
        pipeline = StreamPipeline(
            DspConfig(ffmpeg_path="ffmpeg"),
            AudioConfig(format="mp3", sample_rate=16000, bitrate="32k"),
        )
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
            DspConfig(ffmpeg_path="ffmpeg"),
            AudioConfig(format="ogg", sample_rate=22050, bitrate="40k"),
        )
        self.assertIn("libvorbis", pipeline._ffmpeg_command())


if __name__ == "__main__":
    unittest.main()
