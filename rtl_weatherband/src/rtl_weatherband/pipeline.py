from __future__ import annotations

import logging
import queue
import select
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import BinaryIO

from .config import (
    AudioConfig,
    CsdrServerConfig,
    DspConfig,
    IcecastConfig,
    IQ_SAMPLE_RATE,
)
from .csdr_server import open_iq_stream
from .deemphasis import DeemphasisFilter
from .nfm import NfmDemodulator, float_to_s16


LOG = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Raised when the DSP or encoder pipeline fails."""


SILENCE_FRAME_SECONDS = 0.02
SILENCE_FRAME_SAMPLES = round(IQ_SAMPLE_RATE * SILENCE_FRAME_SECONDS)
SILENCE_FRAME = b"\x00\x00" * SILENCE_FRAME_SAMPLES
PCM_FRAME_BYTES = len(SILENCE_FRAME)
PCM_QUEUE_CHUNKS = 100
RECONNECT_DELAY_SECONDS = 5


@dataclass(frozen=True)
class ProcessExit:
    stage: str
    returncode: int


@dataclass
class StreamPipeline:
    dsp: DspConfig
    audio: AudioConfig
    icecast: IcecastConfig
    csdr_server: CsdrServerConfig
    frequency_hz: int
    processes: list[tuple[str, subprocess.Popen[bytes]]] = field(default_factory=list)
    producer_thread: threading.Thread | None = None
    writer_thread: threading.Thread | None = None
    dsp_error: BaseException | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    pcm_queue: queue.Queue[bytes] = field(
        default_factory=lambda: queue.Queue(maxsize=PCM_QUEUE_CHUNKS)
    )

    def start(self, encoded_output: BinaryIO) -> None:
        ffmpeg = subprocess.Popen(
            self._ffmpeg_command(),
            stdin=subprocess.PIPE,
            stdout=encoded_output,
            stderr=None,
        )
        self.processes = [("ffmpeg encoder", ffmpeg)]
        self.producer_thread = threading.Thread(
            target=self._run_iq_producer,
            name="numpy-nfm-producer",
            daemon=True,
        )
        self.writer_thread = threading.Thread(
            target=self._run_pcm_writer,
            args=(ffmpeg.stdin,),
            name="pcm-silence-writer",
            daemon=True,
        )
        self.producer_thread.start()
        self.writer_thread.start()

    def wait(self) -> ProcessExit:
        if not self.processes:
            raise PipelineError("pipeline was not started")
        while True:
            if self.writer_thread is not None and not self.writer_thread.is_alive():
                return ProcessExit(
                    stage="pcm writer",
                    returncode=1 if self.dsp_error is not None else 0,
                )
            for stage, process in self.processes:
                returncode = process.poll()
                if returncode is not None:
                    return ProcessExit(stage=stage, returncode=returncode)
            time.sleep(0.25)

    def stop(self) -> None:
        self.stop_event.set()
        for _, process in reversed(self.processes):
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for _, process in reversed(self.processes):
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

    def _run_iq_producer(self) -> None:
        while not self.stop_event.is_set():
            try:
                iq_stream = open_iq_stream(self.csdr_server, self.frequency_hz)
                LOG.info("connected to csdr_server")
            except Exception as exc:
                LOG.warning(
                    "csdr_server connection failed: %s; sending silence and retrying "
                    "in %s seconds",
                    exc,
                    RECONNECT_DELAY_SECONDS,
                )
                self._sleep_until_reconnect()
                continue

            try:
                self._produce_from_iq_stream(iq_stream.stream_socket)
            except Exception as exc:
                LOG.warning(
                    "csdr_server stream failed: %s; sending silence and retrying in "
                    "%s seconds",
                    exc,
                    RECONNECT_DELAY_SECONDS,
                )
            finally:
                iq_stream.close()

            self._sleep_until_reconnect()

    def _produce_from_iq_stream(self, iq_socket: socket.socket) -> None:
        demodulator = NfmDemodulator()
        deemphasis = DeemphasisFilter(IQ_SAMPLE_RATE, self.audio.deemphasis_tau)
        while not self.stop_event.is_set():
            readable, _, _ = select.select([iq_socket], [], [], 0.5)
            if not readable:
                continue
            chunk = iq_socket.recv(65536)
            if not chunk:
                raise ConnectionError("IQ stream socket closed")
            audio = demodulator.process(chunk)
            if len(audio) == 0:
                continue
            audio = deemphasis.process_float(audio)
            self._queue_pcm(float_to_s16(audio))

    def _run_pcm_writer(self, pcm_sink: BinaryIO | None) -> None:
        if pcm_sink is None:
            return
        pending = bytearray()
        next_write_at = time.monotonic()
        try:
            while not self.stop_event.is_set():
                self._drain_pcm_queue(pending)
                if len(pending) >= PCM_FRAME_BYTES:
                    frame = bytes(pending[:PCM_FRAME_BYTES])
                    del pending[:PCM_FRAME_BYTES]
                else:
                    frame = SILENCE_FRAME
                pcm_sink.write(frame)

                next_write_at += SILENCE_FRAME_SECONDS
                delay = next_write_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_write_at = time.monotonic()
        except (BrokenPipeError, OSError) as exc:
            self.dsp_error = exc
        finally:
            try:
                pcm_sink.close()
            except OSError:
                pass

    def _queue_pcm(self, pcm: bytes) -> None:
        try:
            self.pcm_queue.put_nowait(pcm)
        except queue.Full:
            try:
                self.pcm_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.pcm_queue.put_nowait(pcm)
            except queue.Full:
                pass

    def _drain_pcm_queue(self, pending: bytearray) -> None:
        while True:
            try:
                pending.extend(self.pcm_queue.get_nowait())
            except queue.Empty:
                return

    def _sleep_until_reconnect(self) -> None:
        deadline = time.monotonic() + RECONNECT_DELAY_SECONDS
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.1)

    def _ffmpeg_command(self) -> list[str]:
        command = [
            self.dsp.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "s16le",
            "-ar",
            str(IQ_SAMPLE_RATE),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-vn",
            "-ar",
            str(self.audio.sample_rate),
            "-ac",
            "1",
            "-b:a",
            f"{self.icecast.bitrate}k",
        ]
        if self.audio.format == "mp3":
            command.extend(["-f", "mp3", "-codec:a", "libmp3lame", "pipe:1"])
        elif self.audio.format == "ogg":
            command.extend(["-f", "ogg", "-codec:a", "libvorbis", "pipe:1"])
        else:
            raise PipelineError(f"unsupported audio format: {self.audio.format}")
        return command
