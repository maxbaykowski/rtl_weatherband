from __future__ import annotations

import unittest

import numpy as np

from rtl_weatherband.config import IcecastConfig
from rtl_weatherband.encoder import (
    Mp3Encoder,
    OggVorbisEncoder,
    PcmResampler,
    create_audio_encoder,
)


def icecast_config(
    format: str,
    sample_rate: int = 16000,
    bitrate: int = 32,
) -> IcecastConfig:
    return IcecastConfig(
        host="127.0.0.1",
        port=8000,
        mount=f"/nwr.{format}",
        username="source",
        password="hackme",
        format=format,
        sample_rate=sample_rate,
        bitrate=bitrate,
    )


class EncoderTests(unittest.TestCase):
    def test_resampler_downsamples_pcm(self) -> None:
        source = np.arange(320, dtype="<i2").tobytes()
        resampler = PcmResampler(16000, 8000)

        output = resampler.process(source) + resampler.flush()

        self.assertEqual(len(output), 160 * 2)

    def test_create_mp3_encoder(self) -> None:
        encoder = create_audio_encoder(icecast_config("mp3"))
        try:
            self.assertIsInstance(encoder, Mp3Encoder)
        finally:
            encoder.close()

    def test_mp3_encoder_outputs_bytes(self) -> None:
        encoder = Mp3Encoder(icecast_config("mp3"))
        try:
            output = bytearray()
            for _ in range(10):
                output.extend(encoder.encode(b"\x00\x00" * 320))
            output.extend(encoder.flush())
            self.assertGreater(len(output), 0)
        finally:
            encoder.close()

    def test_ogg_vorbis_encoder_outputs_ogg_pages(self) -> None:
        encoder = OggVorbisEncoder(icecast_config("ogg"))
        try:
            output = bytearray(encoder.header)
            for _ in range(10):
                output.extend(encoder.encode(b"\x00\x00" * 320))
            output.extend(encoder.flush())
            self.assertTrue(output.startswith(b"OggS"))
            self.assertGreater(output.count(b"OggS"), 1)
        finally:
            encoder.close()

    def test_ogg_vorbis_encoder_uses_managed_bitrate_mode(self) -> None:
        encoder = OggVorbisEncoder(
            icecast_config("ogg", sample_rate=16000, bitrate=96)
        )
        try:
            self.assertTrue(encoder.header.startswith(b"OggS"))
        finally:
            encoder.close()


if __name__ == "__main__":
    unittest.main()
