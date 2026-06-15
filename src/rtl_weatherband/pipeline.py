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
    FallbackConfig,
    IcecastConfig,
    IQ_SAMPLE_RATE,
    StationConfig,
)
from .csdr_server import CsdrServerError, IqStream, open_iq_stream
from .audio_effects import AudioEffectsProcessor
from .encoder import AudioEncoder, PcmResampler, create_audio_encoder
from .fallback_audio import FallbackAudio, load_fallback_audio
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
PCM_BUFFER_LOW_WATER_RATIO = 0.25


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
    fallback_state: FallbackPlaybackState = field(default_factory=lambda: FallbackPlaybackState())
    pcm_buffer_ready: bool = False
    pcm_buffer_target_bytes: int = 0
    error: BaseException | None = None


@dataclass
class FallbackPlaybackState:
    position: int = 0
    delay_samples_remaining: int = 0
    active: bool = False

    def reset(self) -> None:
        self.position = 0
        self.delay_samples_remaining = 0
        self.active = False


@dataclass
class StreamPipeline:
    audio: AudioConfig
    icecast: tuple[IcecastConfig, ...]
    csdr_server: CsdrServerConfig
    frequency_hz: int
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
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
    fallback_audio: FallbackAudio = field(init=False)
    last_pcm_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.fallback_audio = _load_pipeline_fallback_audio(self.fallback)

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
        applied = config
        with self.state_lock:
            self.audio = config.audio
            if config.fallback != self.fallback:
                self.fallback = config.fallback
                self.fallback_audio = _load_pipeline_fallback_audio(self.fallback)
                LOG.info("updated fallback audio settings")
            server_changed = _csdr_connection_changed(config.csdr_server, self.csdr_server)
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
            self.csdr_server = config.csdr_server
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
                    fallback=config.fallback,
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
        demodulator = NfmDemodulator(iq_stream.iq_format)
        audio_config = self._audio_config()
        effects = AudioEffectsProcessor(audio_config)
        while not self.stop_event.is_set():
            replacement = self._try_hotswap_iq_stream(iq_stream)
            if replacement is not None:
                iq_stream, chunk = replacement
                iq_socket = iq_stream.stream_socket
                demodulator = NfmDemodulator(iq_stream.iq_format)
                audio_config = self._audio_config()
                effects = AudioEffectsProcessor(audio_config)
                if chunk:
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
            self._read_hotswap_iq_probe(replacement, csdr_server.timeout)
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
                self.hotswap_result = (receiver, replacement, b"")
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

    def _read_hotswap_iq_probe(
        self,
        iq_stream: IqStream,
        timeout: float,
    ) -> bytes:
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
                self._apply_pcm_buffer_target(encoder_group, pending)
                frame = self._next_buffered_pcm_frame(encoder_group, pending)
                if frame is None:
                    frame = self._idle_pcm_frame(encoder_group)
                else:
                    self._mark_pcm_output()
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

    def _mark_pcm_output(self) -> None:
        with self.state_lock:
            self.last_pcm_at = time.monotonic()
        self._reset_fallback_states()

    def _idle_pcm_frame(self, encoder_group: EncoderWorker) -> bytes:
        with self.state_lock:
            idle_seconds = time.monotonic() - self.last_pcm_at
            fallback = self.fallback
            fallback_audio = self.fallback_audio
        state = encoder_group.fallback_state
        if not state.active and idle_seconds < fallback.silence_timeout_seconds:
            encoder_group.fallback_state.reset()
            return SILENCE_FRAME
        if not state.active:
            state.active = True
            LOG.info(
                "starting fallback audio for encoder group %s after %.1f seconds without PCM",
                encoder_group.key,
                idle_seconds,
            )
        return _next_fallback_frame(
            fallback_audio,
            state,
            fallback.loop_delay_seconds,
        )

    def _reset_fallback_states(self) -> None:
        for encoder_group in list(self.encoder_groups):
            if encoder_group.fallback_state.active:
                LOG.info("stopping fallback audio for encoder group %s", encoder_group.key)
            encoder_group.fallback_state.reset()

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

    def _next_buffered_pcm_frame(
        self,
        encoder_group: EncoderWorker,
        pending: bytearray,
    ) -> bytes | None:
        target_bytes = self._pcm_buffer_target_bytes()
        if not encoder_group.pcm_buffer_ready:
            required = max(PCM_FRAME_BYTES, target_bytes)
            if len(pending) < required:
                return None
            encoder_group.pcm_buffer_ready = True
            LOG.debug(
                "PCM buffer ready for encoder group %s with %.2f seconds buffered",
                encoder_group.key,
                len(pending) / 2 / IQ_SAMPLE_RATE,
            )
        low_water_bytes = self._pcm_buffer_low_water_bytes(target_bytes)
        if low_water_bytes and len(pending) <= low_water_bytes:
            encoder_group.pcm_buffer_ready = False
            LOG.debug(
                "PCM buffer low for encoder group %s with %.2f seconds remaining; refilling",
                encoder_group.key,
                len(pending) / 2 / IQ_SAMPLE_RATE,
            )
            return None
        if len(pending) < PCM_FRAME_BYTES:
            encoder_group.pcm_buffer_ready = False
            return None
        frame = bytes(pending[:PCM_FRAME_BYTES])
        del pending[:PCM_FRAME_BYTES]
        return frame

    def _apply_pcm_buffer_target(
        self,
        encoder_group: EncoderWorker,
        pending: bytearray,
    ) -> None:
        target_bytes = self._pcm_buffer_target_bytes()
        previous_target = encoder_group.pcm_buffer_target_bytes
        encoder_group.pcm_buffer_target_bytes = target_bytes
        if target_bytes > previous_target and len(pending) < target_bytes:
            encoder_group.pcm_buffer_ready = False
        if (
            target_bytes < previous_target
            and len(pending) > max(target_bytes, PCM_FRAME_BYTES)
        ):
            keep_bytes = max(target_bytes, PCM_FRAME_BYTES)
            del pending[: len(pending) - keep_bytes]

    def _pcm_buffer_target_bytes(self) -> int:
        with self.state_lock:
            buffer_seconds = self.csdr_server.buffer_seconds
        frame_count = round(buffer_seconds / SILENCE_FRAME_SECONDS)
        return max(0, frame_count * PCM_FRAME_BYTES)

    def _pcm_buffer_low_water_bytes(self, target_bytes: int) -> int:
        if target_bytes <= PCM_FRAME_BYTES:
            return 0
        low_water_frames = max(
            1,
            round(target_bytes * PCM_BUFFER_LOW_WATER_RATIO / PCM_FRAME_BYTES),
        )
        return low_water_frames * PCM_FRAME_BYTES

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


def _csdr_connection_changed(new: CsdrServerConfig, current: CsdrServerConfig) -> bool:
    return (
        new.host != current.host
        or new.port != current.port
        or new.iq_format != current.iq_format
    )


def _icecast_label(config: IcecastConfig) -> str:
    return (
        f"{config.format}@{config.sample_rate}Hz/{config.bitrate}kbps "
        f"{config.username}@{config.host}:{config.port}{config.mount}"
    )


def _load_pipeline_fallback_audio(config: FallbackConfig) -> FallbackAudio:
    audio = load_fallback_audio(config.path)
    if audio.sample_rate == IQ_SAMPLE_RATE:
        return audio
    resampler = PcmResampler(audio.sample_rate, IQ_SAMPLE_RATE)
    pcm = resampler.process(audio.pcm) + resampler.flush()
    return FallbackAudio(
        sample_rate=IQ_SAMPLE_RATE,
        pcm=pcm,
        duration_seconds=len(pcm) / 2 / IQ_SAMPLE_RATE,
        source=audio.source,
    )


def _next_fallback_frame(
    audio: FallbackAudio,
    state: FallbackPlaybackState,
    loop_delay_seconds: float,
) -> bytes:
    if not audio.pcm:
        return SILENCE_FRAME
    output = bytearray()
    delay_samples = round(loop_delay_seconds * IQ_SAMPLE_RATE)
    while len(output) < PCM_FRAME_BYTES:
        if state.delay_samples_remaining > 0:
            remaining_samples = (PCM_FRAME_BYTES - len(output)) // 2
            silence_samples = min(remaining_samples, state.delay_samples_remaining)
            output.extend(b"\x00\x00" * silence_samples)
            state.delay_samples_remaining -= silence_samples
            continue

        if state.position >= len(audio.pcm):
            state.position = 0
            if delay_samples > 0:
                state.delay_samples_remaining = delay_samples
                continue

        chunk_size = min(PCM_FRAME_BYTES - len(output), len(audio.pcm) - state.position)
        output.extend(audio.pcm[state.position : state.position + chunk_size])
        state.position += chunk_size
    return bytes(output)
