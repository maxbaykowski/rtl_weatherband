from __future__ import annotations

import ctypes
import random
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .config import IQ_SAMPLE_RATE, IcecastConfig


class EncoderError(RuntimeError):
    """Raised when an audio encoder cannot be initialized or used."""


class AudioEncoder(Protocol):
    def encode(self, pcm: bytes) -> bytes:
        ...

    def flush(self) -> bytes:
        ...

    def close(self) -> None:
        ...


def create_audio_encoder(config: IcecastConfig) -> AudioEncoder:
    if config.format == "mp3":
        return Mp3Encoder(config)
    if config.format == "ogg":
        return OggVorbisEncoder(config)
    raise EncoderError(f"unsupported icecast format: {config.format}")


class PcmResampler:
    def __init__(self, input_rate: int, output_rate: int) -> None:
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.stream = None
        if input_rate != output_rate:
            try:
                import soxr
            except ImportError as exc:
                raise EncoderError(
                    "sample-rate conversion requires the 'soxr' Python package"
                ) from exc
            self.stream = soxr.ResampleStream(
                input_rate,
                output_rate,
                1,
                dtype="int16",
            )

    def process(self, pcm: bytes) -> bytes:
        if not pcm:
            return b""
        if self.stream is None:
            return pcm
        return self._resample(pcm, last=False)

    def flush(self) -> bytes:
        if self.stream is None:
            return b""
        return self._resample(b"", last=True)

    def _resample(self, pcm: bytes, last: bool) -> bytes:
        samples = np.frombuffer(pcm, dtype="<i2")
        output = self.stream.resample_chunk(samples, last=last)
        return output.astype("<i2", copy=False).tobytes()


@dataclass
class Mp3Encoder:
    config: IcecastConfig

    def __post_init__(self) -> None:
        try:
            import lameenc
        except ImportError as exc:
            raise EncoderError(
                "MP3 output requires the 'lameenc' Python package"
            ) from exc

        self.resampler = PcmResampler(IQ_SAMPLE_RATE, self.config.sample_rate)
        self.header = b""
        self.encoder = lameenc.Encoder()
        self.encoder.set_bit_rate(self.config.bitrate)
        self.encoder.set_in_sample_rate(self.config.sample_rate)
        self.encoder.set_out_sample_rate(self.config.sample_rate)
        self.encoder.set_channels(1)
        self.encoder.set_quality(2)

    def encode(self, pcm: bytes) -> bytes:
        pcm = self.resampler.process(pcm)
        if not pcm:
            return b""
        return bytes(self.encoder.encode(pcm))

    def flush(self) -> bytes:
        output = bytearray()
        pcm = self.resampler.flush()
        if pcm:
            output.extend(self.encoder.encode(pcm))
        output.extend(self.encoder.flush())
        return bytes(output)

    def close(self) -> None:
        pass


class OggVorbisEncoder:
    def __init__(self, config: IcecastConfig) -> None:
        try:
            from pyogg import vorbis
        except ImportError as exc:
            raise EncoderError(
                "Ogg/Vorbis output requires the 'PyOgg' Python package"
            ) from exc

        self.config = config
        self.vorbis = vorbis
        self.libogg = vorbis.libogg
        self.libvorbis = vorbis.libvorbis
        self.libvorbisenc = vorbis.libvorbisenc
        self.resampler = PcmResampler(IQ_SAMPLE_RATE, config.sample_rate)
        self.closed = False
        self._configure_ctypes()
        self.vi = vorbis.vorbis_info()
        self.vc = vorbis.vorbis_comment()
        self.vd = vorbis.vorbis_dsp_state()
        self.vb = vorbis.vorbis_block()
        self.os = vorbis.ogg_stream_state()
        self._init_encoder()

    def _configure_ctypes(self) -> None:
        v = self.vorbis
        self.libvorbis.vorbis_info_init.argtypes = [ctypes.POINTER(v.vorbis_info)]
        self.libvorbis.vorbis_info_init.restype = None
        self.libvorbis.vorbis_info_clear.argtypes = [ctypes.POINTER(v.vorbis_info)]
        self.libvorbis.vorbis_info_clear.restype = None
        self.libvorbis.vorbis_comment_init.argtypes = [ctypes.POINTER(v.vorbis_comment)]
        self.libvorbis.vorbis_comment_init.restype = None
        self.libvorbis.vorbis_comment_add.argtypes = [
            ctypes.POINTER(v.vorbis_comment),
            ctypes.c_char_p,
        ]
        self.libvorbis.vorbis_comment_add.restype = None
        self.libvorbis.vorbis_comment_clear.argtypes = [
            ctypes.POINTER(v.vorbis_comment)
        ]
        self.libvorbis.vorbis_comment_clear.restype = None
        self.libvorbisenc.vorbis_encode_init.argtypes = [
            ctypes.POINTER(v.vorbis_info),
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_long,
        ]
        self.libvorbisenc.vorbis_encode_init.restype = ctypes.c_int
        self.libvorbis.vorbis_analysis_init.argtypes = [
            ctypes.POINTER(v.vorbis_dsp_state),
            ctypes.POINTER(v.vorbis_info),
        ]
        self.libvorbis.vorbis_analysis_init.restype = ctypes.c_int
        self.libvorbis.vorbis_analysis_buffer.restype = ctypes.POINTER(
            ctypes.POINTER(ctypes.c_float)
        )
        self.libvorbis.vorbis_analysis_wrote.argtypes = [
            ctypes.POINTER(v.vorbis_dsp_state),
            ctypes.c_int,
        ]
        self.libvorbis.vorbis_analysis_wrote.restype = ctypes.c_int
        self.libvorbis.vorbis_block_init.argtypes = [
            ctypes.POINTER(v.vorbis_dsp_state),
            ctypes.POINTER(v.vorbis_block),
        ]
        self.libvorbis.vorbis_block_init.restype = ctypes.c_int
        self.libvorbis.vorbis_block_clear.argtypes = [ctypes.POINTER(v.vorbis_block)]
        self.libvorbis.vorbis_block_clear.restype = ctypes.c_int
        self.libvorbis.vorbis_dsp_clear.argtypes = [
            ctypes.POINTER(v.vorbis_dsp_state)
        ]
        self.libvorbis.vorbis_dsp_clear.restype = None
        self.libvorbis.vorbis_analysis_headerout.argtypes = [
            ctypes.POINTER(v.vorbis_dsp_state),
            ctypes.POINTER(v.vorbis_comment),
            ctypes.POINTER(v.ogg_packet),
            ctypes.POINTER(v.ogg_packet),
            ctypes.POINTER(v.ogg_packet),
        ]
        self.libvorbis.vorbis_analysis_headerout.restype = ctypes.c_int
        self.libvorbis.vorbis_analysis_blockout.argtypes = [
            ctypes.POINTER(v.vorbis_dsp_state),
            ctypes.POINTER(v.vorbis_block),
        ]
        self.libvorbis.vorbis_analysis_blockout.restype = ctypes.c_int
        self.libvorbis.vorbis_analysis.argtypes = [
            ctypes.POINTER(v.vorbis_block),
            ctypes.POINTER(v.ogg_packet),
        ]
        self.libvorbis.vorbis_analysis.restype = ctypes.c_int
        self.libvorbis.vorbis_bitrate_addblock.argtypes = [
            ctypes.POINTER(v.vorbis_block)
        ]
        self.libvorbis.vorbis_bitrate_addblock.restype = ctypes.c_int
        self.libvorbis.vorbis_bitrate_flushpacket.argtypes = [
            ctypes.POINTER(v.vorbis_dsp_state),
            ctypes.POINTER(v.ogg_packet),
        ]
        self.libvorbis.vorbis_bitrate_flushpacket.restype = ctypes.c_int
        self.libogg.ogg_stream_init.argtypes = [
            ctypes.POINTER(v.ogg_stream_state),
            ctypes.c_int,
        ]
        self.libogg.ogg_stream_init.restype = ctypes.c_int
        self.libogg.ogg_stream_packetin.argtypes = [
            ctypes.POINTER(v.ogg_stream_state),
            ctypes.POINTER(v.ogg_packet),
        ]
        self.libogg.ogg_stream_packetin.restype = ctypes.c_int
        self.libogg.ogg_stream_pageout.argtypes = [
            ctypes.POINTER(v.ogg_stream_state),
            ctypes.POINTER(v.ogg_page),
        ]
        self.libogg.ogg_stream_pageout.restype = ctypes.c_int
        self.libogg.ogg_stream_flush.argtypes = [
            ctypes.POINTER(v.ogg_stream_state),
            ctypes.POINTER(v.ogg_page),
        ]
        self.libogg.ogg_stream_flush.restype = ctypes.c_int
        self.libogg.ogg_stream_clear.argtypes = [ctypes.POINTER(v.ogg_stream_state)]
        self.libogg.ogg_stream_clear.restype = ctypes.c_int

    def _init_encoder(self) -> None:
        v = self.vorbis
        self.libvorbis.vorbis_info_init(ctypes.byref(self.vi))
        result = self.libvorbisenc.vorbis_encode_init(
            ctypes.byref(self.vi),
            1,
            self.config.sample_rate,
            -1,
            self.config.bitrate * 1000,
            -1,
        )
        if result != 0:
            raise EncoderError(
                "vorbis encoder managed-bitrate initialization failed: "
                f"{result} for {self.config.bitrate} Kbps at "
                f"{self.config.sample_rate} Hz"
            )

        self.libvorbis.vorbis_comment_init(ctypes.byref(self.vc))
        self.libvorbis.vorbis_comment_add(
            ctypes.byref(self.vc), b"ENCODER=rtl_weatherband"
        )
        self.libvorbis.vorbis_analysis_init(ctypes.byref(self.vd), ctypes.byref(self.vi))
        self.libvorbis.vorbis_block_init(ctypes.byref(self.vd), ctypes.byref(self.vb))
        self.libogg.ogg_stream_init(ctypes.byref(self.os), random.randint(1, 2**31 - 1))

        header = v.ogg_packet()
        header_comment = v.ogg_packet()
        header_code = v.ogg_packet()
        result = self.libvorbis.vorbis_analysis_headerout(
            ctypes.byref(self.vd),
            ctypes.byref(self.vc),
            ctypes.byref(header),
            ctypes.byref(header_comment),
            ctypes.byref(header_code),
        )
        if result != 0:
            raise EncoderError(f"vorbis header creation failed: {result}")

        for packet in (header, header_comment, header_code):
            self.libogg.ogg_stream_packetin(ctypes.byref(self.os), ctypes.byref(packet))

        self.header = self._flush_pages()

    def encode(self, pcm: bytes) -> bytes:
        pcm = self.resampler.process(pcm)
        if not pcm:
            return b""
        return self._encode_resampled_pcm(pcm)

    def _encode_resampled_pcm(self, pcm: bytes) -> bytes:
        samples = _pcm_s16le_to_float(pcm)
        buffer = self.libvorbis.vorbis_analysis_buffer(
            ctypes.byref(self.vd), len(samples)
        )
        channel = buffer[0]
        for index, sample in enumerate(samples):
            channel[index] = float(sample)
        self.libvorbis.vorbis_analysis_wrote(ctypes.byref(self.vd), len(samples))
        return self._drain_packets()

    def flush(self) -> bytes:
        if self.closed:
            return b""
        output = bytearray()
        pcm = self.resampler.flush()
        if pcm:
            output.extend(self._encode_resampled_pcm(pcm))
        self.libvorbis.vorbis_analysis_wrote(ctypes.byref(self.vd), 0)
        output.extend(self._drain_packets())
        output += self._flush_pages()
        self.close()
        return bytes(output)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.libogg.ogg_stream_clear(ctypes.byref(self.os))
        self.libvorbis.vorbis_block_clear(ctypes.byref(self.vb))
        self.libvorbis.vorbis_dsp_clear(ctypes.byref(self.vd))
        self.libvorbis.vorbis_comment_clear(ctypes.byref(self.vc))
        self.libvorbis.vorbis_info_clear(ctypes.byref(self.vi))

    def _drain_packets(self) -> bytes:
        output = bytearray()
        v = self.vorbis
        packet = v.ogg_packet()
        while self.libvorbis.vorbis_analysis_blockout(
            ctypes.byref(self.vd), ctypes.byref(self.vb)
        ):
            self.libvorbis.vorbis_analysis(ctypes.byref(self.vb), None)
            self.libvorbis.vorbis_bitrate_addblock(ctypes.byref(self.vb))
            while self.libvorbis.vorbis_bitrate_flushpacket(
                ctypes.byref(self.vd), ctypes.byref(packet)
            ):
                self.libogg.ogg_stream_packetin(
                    ctypes.byref(self.os), ctypes.byref(packet)
                )
                output.extend(self._pageout_pages())
        return bytes(output)

    def _pageout_pages(self) -> bytes:
        output = bytearray()
        page = self.vorbis.ogg_page()
        while self.libogg.ogg_stream_pageout(
            ctypes.byref(self.os), ctypes.byref(page)
        ):
            output.extend(_page_bytes(page))
        return bytes(output)

    def _flush_pages(self) -> bytes:
        output = bytearray()
        page = self.vorbis.ogg_page()
        while self.libogg.ogg_stream_flush(ctypes.byref(self.os), ctypes.byref(page)):
            output.extend(_page_bytes(page))
        return bytes(output)


def _pcm_s16le_to_float(pcm: bytes) -> NDArray[np.float32]:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    return samples / 32768.0


def _page_bytes(page) -> bytes:
    header = ctypes.string_at(page.header, page.header_len)
    body = ctypes.string_at(page.body, page.body_len)
    return header + body
