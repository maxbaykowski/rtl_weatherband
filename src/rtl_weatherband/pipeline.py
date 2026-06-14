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
from .audio_effects import AudioEffectsProcessor
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
class OutputWorker:
    config: IcecastConfig
    encoder_group: EncoderWorker
    encoded_queue: queue.Queue[bytes]
    stop_event: threading.Event
    thread: threading.Thread
    error: BaseException | None = None


@dataclass
class EncoderWorker:
    key: tuple[str, int, int]
    config: IcecastConfig
    encoder: AudioEncoder
    pcm_queue: queue.Queue[bytes]
    stop_event: threading.Event
    thread: threading.Thread
    outputs: list[OutputWorker] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    error: BaseException | None = None


@dataclass
class StreamPipeline:
    audio: AudioConfig
    icecast: tuple[IcecastConfig, ...]
    csdr_server: CsdrServerConfig
    frequency_hz: int
    outputs: list[OutputWorker] = field(default_factory=list)
    encoder_groups: list[EncoderWorker] = field(default_factory=list)
    producer_thread: threading.Thread | None = None
    stream_error: BaseException | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    active_iq_stream: IqStream | None = None
    pending_receiver: tuple[CsdrServerConfig, int] | None = None
    hotswap_thread: threading.Thread | None = None
    hotswap_result: tuple[tuple[CsdrServerConfig, int], IqStream, bytes] | None = None

    def start(self, encoded_outputs: list[BinaryIO]) -> None:
        if len(encoded_outputs) != len(self.icecast):
            raise PipelineError("encoded output count must match Icecast destinations")
        self.producer_thread = threading.Thread(
            target=self._run_iq_producer,
            name="numpy-nfm-producer",
            daemon=True,
        )
        self.outputs = []
        self.encoder_groups = []
        for destination, encoded_output in zip(
            self.icecast, encoded_outputs, strict=True
        ):
            self.outputs.append(
                self._create_output_worker(destination, encoded_output)
            )
        self.producer_thread.start()
        for encoder_group in self.encoder_groups:
            encoder_group.thread.start()
        for output in self.outputs:
            output.thread.start()

    def wait(self) -> ProcessExit:
        if not self.outputs:
            raise PipelineError("pipeline was not started")
        while True:
            process_exit = self.poll_exit()
            if process_exit is not None:
                return process_exit
            time.sleep(0.25)

    def poll_exit(self) -> ProcessExit | None:
        for index, encoder_group in enumerate(self.encoder_groups):
            if not encoder_group.thread.is_alive():
                return ProcessExit(
                    stage=f"audio encoder {index}",
                    returncode=1 if encoder_group.error is not None else 0,
                )
        for index, output in enumerate(self.outputs):
            if not output.thread.is_alive():
                return ProcessExit(
                    stage=f"encoded audio writer {index}",
                    returncode=1 if output.error is not None else 0,
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
        for output in list(self.outputs):
            self._stop_output_worker(output)
        self.outputs = []
        for encoder_group in list(self.encoder_groups):
            self._stop_encoder_worker(encoder_group)
        self.encoder_groups = []

    def apply_icecast_outputs(
        self,
        icecast: tuple[IcecastConfig, ...],
        remove_configs: list[IcecastConfig],
        additions: list[tuple[IcecastConfig, BinaryIO]],
        stop_unused_encoders: bool = True,
    ) -> None:
        LOG.debug(
            "applying Icecast outputs: remove=%s add=%s stop_unused_encoders=%s",
            [_icecast_label(config) for config in remove_configs],
            [_icecast_label(config) for config, _ in additions],
            stop_unused_encoders,
        )
        for config in remove_configs:
            self._remove_output(config)
        for config, encoded_output in additions:
            output = self._create_output_worker(config, encoded_output)
            self.outputs.append(output)
            output.thread.start()
            LOG.info("started Icecast writer for %s", _icecast_label(config))
        if stop_unused_encoders:
            self._stop_unused_encoder_workers()
        self.icecast = icecast
        LOG.debug(
            "Icecast outputs now active: %s",
            [_icecast_label(output.config) for output in self.outputs],
        )

    def _create_output_worker(
        self,
        config: IcecastConfig,
        encoded_output: BinaryIO,
    ) -> OutputWorker:
        encoder_group = self._get_or_create_encoder_worker(config)
        output = OutputWorker(
            config=config,
            encoder_group=encoder_group,
            encoded_queue=queue.Queue(maxsize=PCM_QUEUE_CHUNKS),
            stop_event=threading.Event(),
            thread=threading.Thread(),
        )
        with encoder_group.lock:
            encoder_group.outputs.append(output)
        output.thread = threading.Thread(
            target=self._run_icecast_writer,
            args=(output, encoded_output),
            name=f"encoded-audio-writer-{config.host}:{config.port}{config.mount}",
            daemon=True,
        )
        return output

    def _get_or_create_encoder_worker(self, config: IcecastConfig) -> EncoderWorker:
        key = _encoder_key(config)
        for encoder_group in self.encoder_groups:
            if encoder_group.key == key:
                LOG.debug(
                    "reusing encoder group %s for %s",
                    key,
                    _icecast_label(config),
                )
                return encoder_group

        LOG.info(
            "creating encoder group %s for %s",
            key,
            _icecast_label(config),
        )
        encoder_group = EncoderWorker(
            key=key,
            config=config,
            encoder=create_audio_encoder(config),
            pcm_queue=queue.Queue(maxsize=PCM_QUEUE_CHUNKS),
            stop_event=threading.Event(),
            thread=threading.Thread(),
        )
        encoder_group.thread = threading.Thread(
            target=self._run_encoder_worker,
            args=(encoder_group,),
            name=(
                f"audio-encoder-{config.format}-{config.sample_rate}-"
                f"{config.bitrate}"
            ),
            daemon=True,
        )
        self.encoder_groups.append(encoder_group)
        if self.producer_thread is not None and self.producer_thread.is_alive():
            encoder_group.thread.start()
        return encoder_group

    def _remove_output(self, config: IcecastConfig) -> None:
        for output in list(self.outputs):
            if output.config == config:
                self.outputs.remove(output)
                self._stop_output_worker(output)
                LOG.info("stopped Icecast writer for %s", _icecast_label(config))
                return
        LOG.debug(
            "requested removal for %s but no matching output worker was active",
            _icecast_label(config),
        )

    def _stop_output_worker(self, output: OutputWorker) -> None:
        output.stop_event.set()
        if output.thread.ident is not None:
            output.thread.join(timeout=2)
        with output.encoder_group.lock:
            if output in output.encoder_group.outputs:
                output.encoder_group.outputs.remove(output)

    def _stop_encoder_worker(self, encoder_group: EncoderWorker) -> None:
        encoder_group.stop_event.set()
        if encoder_group.thread.ident is not None:
            encoder_group.thread.join(timeout=2)
        encoder_group.encoder.close()
        LOG.info("stopped encoder group %s", encoder_group.key)

    def _stop_unused_encoder_workers(self) -> None:
        for encoder_group in list(self.encoder_groups):
            with encoder_group.lock:
                has_outputs = bool(encoder_group.outputs)
            if not has_outputs:
                self.encoder_groups.remove(encoder_group)
                LOG.debug("encoder group %s has no outputs; stopping", encoder_group.key)
                self._stop_encoder_worker(encoder_group)

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
        audio_config = self._audio_config()
        effects = AudioEffectsProcessor(audio_config)
        while not self.stop_event.is_set():
            replacement = self._try_hotswap_iq_stream(iq_stream)
            if replacement is not None:
                iq_stream, chunk = replacement
                iq_socket = iq_stream.stream_socket
                demodulator = NfmDemodulator()
                audio_config = self._audio_config()
                effects = AudioEffectsProcessor(audio_config)
                self._process_iq_chunk(chunk, demodulator, effects)
                continue
            next_audio_config = self._audio_config()
            if next_audio_config != audio_config:
                audio_config = next_audio_config
                effects = AudioEffectsProcessor(audio_config)
                LOG.info("updated audio effects")
            readable, _, _ = select.select([iq_socket], [], [], 0.5)
            if not readable:
                continue
            chunk = iq_socket.recv(65536)
            if not chunk:
                raise ConnectionError("IQ stream socket closed")
            self._process_iq_chunk(chunk, demodulator, effects)

    def _process_iq_chunk(
        self,
        chunk: bytes,
        demodulator: NfmDemodulator,
        effects: AudioEffectsProcessor,
    ) -> None:
        audio = demodulator.process(chunk)
        if len(audio) == 0:
            return
        audio = effects.process(audio)
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

    def _run_encoder_worker(
        self,
        encoder_group: EncoderWorker,
    ) -> None:
        pending = bytearray()
        next_write_at = time.monotonic()
        try:
            while not self.stop_event.is_set() and not encoder_group.stop_event.is_set():
                self._drain_pcm_queue(encoder_group.pcm_queue, pending)
                if len(pending) >= PCM_FRAME_BYTES:
                    frame = bytes(pending[:PCM_FRAME_BYTES])
                    del pending[:PCM_FRAME_BYTES]
                else:
                    frame = SILENCE_FRAME
                encoded = encoder_group.encoder.encode(frame)
                if encoded:
                    self._broadcast_encoded(encoder_group, encoded)

                next_write_at += SILENCE_FRAME_SECONDS
                delay = next_write_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_write_at = time.monotonic()
            encoded = encoder_group.encoder.flush()
            if encoded:
                self._broadcast_encoded(encoder_group, encoded)
        except Exception as exc:
            encoder_group.error = exc

    def _run_icecast_writer(
        self,
        output: OutputWorker,
        encoded_sink: BinaryIO | None,
    ) -> None:
        if encoded_sink is None:
            return
        try:
            header = getattr(output.encoder_group.encoder, "header", b"")
            if header:
                encoded_sink.write(header)
            while not self.stop_event.is_set() and not output.stop_event.is_set():
                try:
                    encoded = output.encoded_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if encoded:
                    encoded_sink.write(encoded)
        except (BrokenPipeError, OSError) as exc:
            output.error = exc
        except Exception as exc:
            output.error = exc
        finally:
            try:
                encoded_sink.close()
            except OSError:
                pass

    def _broadcast_encoded(
        self,
        encoder_group: EncoderWorker,
        encoded: bytes,
    ) -> None:
        with encoder_group.lock:
            outputs = list(encoder_group.outputs)
        for output in outputs:
            self._queue_encoded_for_output(output.encoded_queue, encoded)

    def _queue_pcm(self, pcm: bytes) -> None:
        for encoder_group in list(self.encoder_groups):
            self._queue_pcm_for_output(encoder_group.pcm_queue, pcm)

    def _queue_encoded_for_output(
        self,
        encoded_queue: queue.Queue[bytes],
        encoded: bytes,
    ) -> None:
        try:
            encoded_queue.put_nowait(encoded)
        except queue.Full:
            try:
                encoded_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                encoded_queue.put_nowait(encoded)
            except queue.Full:
                pass

    def _queue_pcm_for_output(self, pcm_queue: queue.Queue[bytes], pcm: bytes) -> None:
        try:
            pcm_queue.put_nowait(pcm)
        except queue.Full:
            try:
                pcm_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                pcm_queue.put_nowait(pcm)
            except queue.Full:
                pass

    def _drain_pcm_queue(
        self,
        pcm_queue: queue.Queue[bytes],
        pending: bytearray,
    ) -> None:
        while True:
            try:
                pending.extend(pcm_queue.get_nowait())
            except queue.Empty:
                return

    def _sleep_until_reconnect(self) -> None:
        deadline = time.monotonic() + RECONNECT_DELAY_SECONDS
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.1)

    def _receiver_config(self) -> tuple[CsdrServerConfig, int]:
        with self.state_lock:
            return self.csdr_server, self.frequency_hz

    def _audio_config(self) -> AudioConfig:
        with self.state_lock:
            return self.audio

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


def _encoder_key(config: IcecastConfig) -> tuple[str, int, int]:
    return config.format, config.sample_rate, config.bitrate


def _icecast_label(config: IcecastConfig) -> str:
    return (
        f"{config.format}@{config.sample_rate}Hz/{config.bitrate}kbps "
        f"{config.username}@{config.host}:{config.port}{config.mount}"
    )
