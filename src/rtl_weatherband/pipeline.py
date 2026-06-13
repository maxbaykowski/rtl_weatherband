from __future__ import annotations

import logging
import queue
import select
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import BinaryIO

from .config import (
    AppConfig,
    AudioConfig,
    CsdrServerConfig,
    IcecastConfig,
    IQ_SAMPLE_RATE,
    StationConfig,
)
from .csdr_server import CsdrServerError, IqStream, open_iq_stream
from .deemphasis import DeemphasisFilter
from .encoder import AudioEncoder, create_audio_encoder
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
    audio: AudioConfig
    icecast: IcecastConfig
    csdr_server: CsdrServerConfig
    frequency_hz: int
    encoder: AudioEncoder | None = None
    producer_thread: threading.Thread | None = None
    writer_thread: threading.Thread | None = None
    stream_error: BaseException | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    active_iq_stream: IqStream | None = None
    pending_receiver: tuple[CsdrServerConfig, int] | None = None
    hotswap_thread: threading.Thread | None = None
    hotswap_result: tuple[tuple[CsdrServerConfig, int], IqStream, bytes] | None = None
    pcm_queue: queue.Queue[bytes] = field(
        default_factory=lambda: queue.Queue(maxsize=PCM_QUEUE_CHUNKS)
    )

    def start(self, encoded_output: BinaryIO) -> None:
        self.encoder = create_audio_encoder(self.icecast)
        self.producer_thread = threading.Thread(
            target=self._run_iq_producer,
            name="numpy-nfm-producer",
            daemon=True,
        )
        self.writer_thread = threading.Thread(
            target=self._run_encoded_writer,
            args=(encoded_output,),
            name="encoded-audio-writer",
            daemon=True,
        )
        self.producer_thread.start()
        self.writer_thread.start()

    def wait(self) -> ProcessExit:
        if self.writer_thread is None:
            raise PipelineError("pipeline was not started")
        while True:
            process_exit = self.poll_exit()
            if process_exit is not None:
                return process_exit
            time.sleep(0.25)

    def poll_exit(self) -> ProcessExit | None:
        if self.writer_thread is not None and not self.writer_thread.is_alive():
            return ProcessExit(
                stage="encoded audio writer",
                returncode=1 if self.stream_error is not None else 0,
            )
        return None

    def stop(self) -> None:
        self.stop_event.set()
        with self.state_lock:
            active_iq_stream = self.active_iq_stream
            self.active_iq_stream = None
            hotswap_result = self.hotswap_result
            self.hotswap_result = None
        if active_iq_stream is not None:
            active_iq_stream.close()
        if hotswap_result is not None:
            _, iq_stream, _ = hotswap_result
            iq_stream.close()
        if self.producer_thread is not None:
            self.producer_thread.join(timeout=2)
        if self.writer_thread is not None:
            self.writer_thread.join(timeout=2)
        if self.encoder is not None:
            self.encoder.close()

    def apply_runtime_config(self, config: AppConfig) -> AppConfig:
        applied = AppConfig(
            csdr_server=self.csdr_server,
            station=config.station,
            icecast=config.icecast,
            audio=config.audio,
        )
        with self.state_lock:
            self.audio = config.audio
            server_changed = config.csdr_server != self.csdr_server
            frequency_changed = config.station.frequency_hz != self.frequency_hz
            if server_changed:
                self.csdr_server = config.csdr_server
                self.frequency_hz = config.station.frequency_hz
                self.pending_receiver = (self.csdr_server, self.frequency_hz)
                LOG.info(
                    "queued csdr_server hotswap to %s:%s",
                    self.csdr_server.host,
                    self.csdr_server.port,
                )
                return config
            if self.pending_receiver is not None:
                self.frequency_hz = config.station.frequency_hz
                self.pending_receiver = (self.csdr_server, self.frequency_hz)
                return config
            if frequency_changed:
                stream = self.active_iq_stream
                timeout = self.csdr_server.timeout
                frequency_hz = config.station.frequency_hz
            else:
                stream = None
                timeout = 0
                frequency_hz = self.frequency_hz

        if stream is not None:
            if self._retune_stream(stream, frequency_hz, timeout):
                with self.state_lock:
                    self.frequency_hz = frequency_hz
                LOG.info("retuned csdr_server stream to %s Hz", frequency_hz)
            else:
                applied = AppConfig(
                    csdr_server=self.csdr_server,
                    station=_station_from_frequency_hz(self.frequency_hz),
                    icecast=config.icecast,
                    audio=config.audio,
                )
        else:
            with self.state_lock:
                self.frequency_hz = frequency_hz
        return applied

    def _run_iq_producer(self) -> None:
        while not self.stop_event.is_set():
            try:
                csdr_server, frequency_hz = self._receiver_config()
                iq_stream = open_iq_stream(csdr_server, frequency_hz)
                self._set_active_iq_stream(iq_stream, csdr_server, frequency_hz)
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
                active_stream = self._pop_active_iq_stream()
                if active_stream is not None:
                    active_stream.close()
                if active_stream is not iq_stream:
                    iq_stream.close()

            self._sleep_until_reconnect()

    def _produce_from_iq_stream(self, iq_socket: socket.socket) -> None:
        iq_stream = self.active_iq_stream
        if iq_stream is None:
            raise ConnectionError("IQ stream is not active")
        demodulator = NfmDemodulator()
        deemphasis_tau = self._deemphasis_tau()
        deemphasis = DeemphasisFilter(IQ_SAMPLE_RATE, deemphasis_tau)
        while not self.stop_event.is_set():
            replacement = self._try_hotswap_iq_stream(iq_stream)
            if replacement is not None:
                iq_stream, chunk = replacement
                iq_socket = iq_stream.stream_socket
                demodulator = NfmDemodulator()
                deemphasis_tau = self._deemphasis_tau()
                deemphasis = DeemphasisFilter(IQ_SAMPLE_RATE, deemphasis_tau)
                self._process_iq_chunk(chunk, demodulator, deemphasis)
                continue
            next_tau = self._deemphasis_tau()
            if next_tau != deemphasis_tau:
                deemphasis_tau = next_tau
                deemphasis = DeemphasisFilter(IQ_SAMPLE_RATE, deemphasis_tau)
                LOG.info("updated deemphasis tau to %s", deemphasis_tau)
            readable, _, _ = select.select([iq_socket], [], [], 0.5)
            if not readable:
                continue
            chunk = iq_socket.recv(65536)
            if not chunk:
                raise ConnectionError("IQ stream socket closed")
            self._process_iq_chunk(chunk, demodulator, deemphasis)

    def _process_iq_chunk(
        self,
        chunk: bytes,
        demodulator: NfmDemodulator,
        deemphasis: DeemphasisFilter,
    ) -> None:
        audio = demodulator.process(chunk)
        if len(audio) == 0:
            return
        audio = deemphasis.process_float(audio)
        self._queue_pcm(float_to_s16(audio))

    def _try_hotswap_iq_stream(
        self,
        current_stream: IqStream,
    ) -> tuple[IqStream, bytes] | None:
        self._start_hotswap_if_needed()
        ready = self._pop_ready_hotswap()
        if ready is None:
            return None
        _, replacement, chunk = ready
        csdr_server, frequency_hz = self._receiver_config()
        self._set_active_iq_stream(replacement, csdr_server, frequency_hz)
        current_stream.close()
        LOG.info(
            "switched csdr_server stream to %s:%s",
            csdr_server.host,
            csdr_server.port,
        )
        return replacement, chunk

    def _start_hotswap_if_needed(self) -> None:
        with self.state_lock:
            pending = self.pending_receiver
            if pending is None:
                return
            if self.hotswap_thread is not None and self.hotswap_thread.is_alive():
                return
            self.hotswap_thread = threading.Thread(
                target=self._connect_hotswap_iq_stream,
                args=(pending,),
                name="csdr-hotswap-connector",
                daemon=True,
            )
            self.hotswap_thread.start()

    def _connect_hotswap_iq_stream(
        self,
        receiver: tuple[CsdrServerConfig, int],
    ) -> None:
        csdr_server, frequency_hz = receiver
        try:
            replacement = open_iq_stream(csdr_server, frequency_hz)
            chunk = self._read_first_iq_chunk(replacement, csdr_server.timeout)
        except Exception as exc:
            LOG.warning(
                "csdr_server hotswap to %s:%s failed: %s; keeping current stream",
                csdr_server.host,
                csdr_server.port,
                exc,
            )
            return
        with self.state_lock:
            if self.pending_receiver != receiver or self.stop_event.is_set():
                close_replacement = True
            else:
                close_replacement = False
                self.hotswap_result = (receiver, replacement, chunk)
        if close_replacement:
            replacement.close()

    def _pop_ready_hotswap(
        self,
    ) -> tuple[tuple[CsdrServerConfig, int], IqStream, bytes] | None:
        with self.state_lock:
            result = self.hotswap_result
            self.hotswap_result = None
            if result is None:
                return None
            receiver, _, _ = result
            if self.pending_receiver != receiver:
                _, iq_stream, _ = result
                iq_stream.close()
                return None
            return result

    def _read_first_iq_chunk(self, iq_stream: IqStream, timeout: float) -> bytes:
        readable, _, _ = select.select([iq_stream.stream_socket], [], [], timeout)
        if not readable:
            iq_stream.close()
            raise TimeoutError("replacement IQ stream did not produce data")
        chunk = iq_stream.stream_socket.recv(65536)
        if not chunk:
            iq_stream.close()
            raise ConnectionError("replacement IQ stream socket closed")
        return chunk

    def _run_encoded_writer(self, encoded_sink: BinaryIO | None) -> None:
        if encoded_sink is None or self.encoder is None:
            return
        pending = bytearray()
        next_write_at = time.monotonic()
        try:
            header = getattr(self.encoder, "header", b"")
            if header:
                encoded_sink.write(header)
            while not self.stop_event.is_set():
                self._drain_pcm_queue(pending)
                if len(pending) >= PCM_FRAME_BYTES:
                    frame = bytes(pending[:PCM_FRAME_BYTES])
                    del pending[:PCM_FRAME_BYTES]
                else:
                    frame = SILENCE_FRAME
                encoded = self.encoder.encode(frame)
                if encoded:
                    encoded_sink.write(encoded)

                next_write_at += SILENCE_FRAME_SECONDS
                delay = next_write_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_write_at = time.monotonic()
            encoded = self.encoder.flush()
            if encoded:
                encoded_sink.write(encoded)
        except (BrokenPipeError, OSError) as exc:
            self.stream_error = exc
        except Exception as exc:
            self.stream_error = exc
        finally:
            try:
                encoded_sink.close()
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

    def _receiver_config(self) -> tuple[CsdrServerConfig, int]:
        with self.state_lock:
            return self.csdr_server, self.frequency_hz

    def _deemphasis_tau(self) -> float:
        with self.state_lock:
            return self.audio.deemphasis_tau

    def _pending_receiver(self) -> tuple[CsdrServerConfig, int] | None:
        with self.state_lock:
            return self.pending_receiver

    def _set_active_iq_stream(
        self,
        iq_stream: IqStream,
        csdr_server: CsdrServerConfig,
        frequency_hz: int,
    ) -> None:
        with self.state_lock:
            self.active_iq_stream = iq_stream
            self.csdr_server = csdr_server
            self.frequency_hz = frequency_hz
            if self.pending_receiver == (csdr_server, frequency_hz):
                self.pending_receiver = None

    def _pop_active_iq_stream(self) -> IqStream | None:
        with self.state_lock:
            active_iq_stream = self.active_iq_stream
            self.active_iq_stream = None
            return active_iq_stream

    def _retune_stream(
        self,
        iq_stream: IqStream,
        frequency_hz: int,
        timeout: float,
    ) -> bool:
        try:
            iq_stream.control_socket.settimeout(timeout)
            iq_stream.retune(frequency_hz)
            iq_stream.control_socket.settimeout(None)
            return True
        except (CsdrServerError, OSError, TimeoutError) as exc:
            LOG.warning("csdr_server retune to %s Hz failed: %s", frequency_hz, exc)
            try:
                iq_stream.control_socket.settimeout(None)
            except OSError:
                pass
            return False

def _station_from_frequency_hz(frequency_hz: int) -> StationConfig:
    return StationConfig(frequency=frequency_hz / 1_000_000)
