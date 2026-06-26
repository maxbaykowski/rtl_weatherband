from __future__ import annotations

import logging
import queue
import select
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import BinaryIO

import numpy as np

from .config import (
    AppConfig,
    AudioConfig,
    CsdrServerConfig,
    DeemphasisConfig,
    EasRecordingConfig,
    FallbackConfig,
    IcecastConfig,
    IQ_SAMPLE_RATE,
    SoundcardConfig,
    StationConfig,
)
from .csdr_server import CsdrServerError, IqStream, open_iq_stream
from .audio_effects import AudioEffectsProcessor
from .encoder import AudioEncoder, PcmResampler, create_audio_encoder
from .eas_recording import EasRecorderOutput, generate_same_test_audio
from .fallback_audio import FallbackAudio, load_fallback_audio
from .nfm import NfmDemodulator, float_to_s16
from .soundcard import (
    SoundcardDependencyError,
    SoundcardError,
    SoundcardFatalError,
    SoundcardOutput,
)


LOG = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Raised when the DSP or encoder pipeline fails."""


SILENCE_FRAME_SECONDS = 0.02
SILENCE_FRAME_SAMPLES = round(IQ_SAMPLE_RATE * SILENCE_FRAME_SECONDS)
SILENCE_FRAME = b"\x00\x00" * SILENCE_FRAME_SAMPLES
PCM_FRAME_BYTES = len(SILENCE_FRAME)
AUDIO_FRAME_BYTES = SILENCE_FRAME_SAMPLES * np.dtype("<f4").itemsize
PCM_QUEUE_CHUNKS = 100
RECONNECT_DELAY_SECONDS = 5
PCM_BUFFER_LOW_WATER_RATIO = 0.25
SAME_TEST_INTER_ALERT_SILENCE_SECONDS = 1.0
SAME_TEST_INTER_ALERT_SILENCE_FRAMES = round(
    SAME_TEST_INTER_ALERT_SILENCE_SECONDS / SILENCE_FRAME_SECONDS
)


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
class SoundcardWorker:
    config: SoundcardConfig
    output: SoundcardOutput
    key: str = "soundcard"
    error: BaseException | None = None


@dataclass
class EasRecorderWorker:
    config: EasRecordingConfig
    output: EasRecorderOutput
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
    soundcard: tuple[SoundcardConfig, ...] = field(default_factory=tuple)
    eas_recording: EasRecordingConfig = field(default_factory=EasRecordingConfig)
    same_test: bool = False
    outputs: list[OutputWorker] = field(default_factory=list)
    encoder_groups: list[EncoderWorker] = field(default_factory=list)
    soundcard_workers: list[SoundcardWorker] = field(default_factory=list)
    eas_recorder_worker: EasRecorderWorker | None = None
    same_test_alerts: deque[np.ndarray] = field(default_factory=deque)
    same_test_lock: threading.Lock = field(default_factory=threading.Lock)
    pcm_queue: queue.Queue[bytes] = field(default_factory=lambda: queue.Queue(maxsize=PCM_QUEUE_CHUNKS))
    playback_thread: threading.Thread | None = None
    producer_thread: threading.Thread | None = None
    stream_error: BaseException | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    soundcard_lock: threading.Lock = field(default_factory=threading.Lock)
    active_iq_stream: IqStream | None = None
    pending_receiver: tuple[CsdrServerConfig, int] | None = None
    hotswap_thread: threading.Thread | None = None
    hotswap_result: tuple[tuple[CsdrServerConfig, int], IqStream, bytes] | None = None
    fallback_audio: FallbackAudio = field(init=False)
    fallback_state: FallbackPlaybackState = field(default_factory=FallbackPlaybackState)
    pcm_buffer_ready: bool = False
    pcm_buffer_target_bytes: int = 0
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
        self.playback_thread = threading.Thread(
            target=self._run_pcm_playback_worker,
            name="pcm-playback",
            daemon=True,
        )
        self.outputs = []
        self.encoder_groups = []
        self.soundcard_workers = []
        self.eas_recorder_worker = None
        self.fallback_state.reset()
        self.pcm_buffer_ready = False
        self.pcm_buffer_target_bytes = 0
        for destination, encoded_output in zip(
            self.icecast, encoded_outputs, strict=True
        ):
            self.outputs.append(
                self._create_output_worker(destination, encoded_output)
            )
        self.soundcard_workers = self._create_soundcard_workers(
            self.soundcard,
            tolerate_output_failures=True,
        )
        self.eas_recorder_worker = self._create_eas_recorder_worker(
            self.eas_recording,
        )
        self.producer_thread.start()
        self.playback_thread.start()
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
        if self.playback_thread is not None and not self.playback_thread.is_alive():
            return ProcessExit(
                stage="PCM playback",
                returncode=1,
            )
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
        for index, soundcard_worker in enumerate(self.soundcard_workers):
            if soundcard_worker.error is not None:
                return ProcessExit(stage=f"soundcard output {index}", returncode=1)
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
        if self.producer_thread is not None and self.producer_thread.ident is not None:
            self.producer_thread.join(timeout=2)
        if self.playback_thread is not None and self.playback_thread.ident is not None:
            self.playback_thread.join(timeout=2)
        for output in list(self.outputs):
            self._stop_output_worker(output)
        self.outputs = []
        for encoder_group in list(self.encoder_groups):
            self._stop_encoder_worker(encoder_group)
        self.encoder_groups = []
        for soundcard_worker in list(self.soundcard_workers):
            self._stop_soundcard_worker(soundcard_worker)
        self.soundcard_workers = []
        if self.eas_recorder_worker is not None:
            self._stop_eas_recorder_worker(self.eas_recorder_worker)
        self.eas_recorder_worker = None
        self._clear_pcm_queue(self.pcm_queue)

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

    def _create_soundcard_worker(self, config: SoundcardConfig) -> SoundcardWorker:
        return SoundcardWorker(
            config=config,
            output=SoundcardOutput(config),
        )

    def _create_soundcard_workers(
        self,
        configs: tuple[SoundcardConfig, ...],
        *,
        tolerate_output_failures: bool,
    ) -> list[SoundcardWorker]:
        workers: list[SoundcardWorker] = []
        for config in configs:
            try:
                workers.append(self._create_soundcard_worker(config))
            except SoundcardDependencyError:
                self._stop_soundcard_workers(workers)
                raise
            except SoundcardError as exc:
                if not tolerate_output_failures:
                    self._stop_soundcard_workers(workers)
                    raise
                LOG.warning(
                    "soundcard output disabled after startup failure for %s: %s",
                    _soundcard_label(config),
                    exc,
                )
        try:
            self._validate_unique_soundcard_workers(workers)
        except Exception:
            self._stop_soundcard_workers(workers)
            raise
        return workers

    def _stop_soundcard_worker(self, worker: SoundcardWorker) -> None:
        worker.output.close()
        LOG.info("stopped soundcard output for %s", _soundcard_label(worker.config))

    def _stop_soundcard_workers(self, workers: list[SoundcardWorker]) -> None:
        for worker in list(workers):
            self._stop_soundcard_worker(worker)

    def _validate_unique_soundcard_workers(
        self,
        workers: list[SoundcardWorker],
    ) -> None:
        seen: dict[object, SoundcardWorker] = {}
        for worker in workers:
            key = getattr(worker.output, "device", worker.config.device)
            if key in seen:
                raise SoundcardFatalError(
                    "multiple soundcard outputs resolved to the same device: "
                    f"{_soundcard_label(seen[key].config)} and "
                    f"{_soundcard_label(worker.config)}"
                )
            seen[key] = worker

    def _create_eas_recorder_worker(
        self,
        config: EasRecordingConfig,
    ) -> EasRecorderWorker | None:
        if not config.enabled:
            return None
        return EasRecorderWorker(
            config=config,
            output=EasRecorderOutput(config),
        )

    def _stop_eas_recorder_worker(self, worker: EasRecorderWorker) -> None:
        worker.output.close()

    def apply_runtime_config(self, config: AppConfig) -> AppConfig:
        applied = config
        with self.state_lock:
            self.audio = config.audio
            if config.fallback != self.fallback:
                self.fallback = config.fallback
                self.fallback_audio = _load_pipeline_fallback_audio(self.fallback)
                LOG.info("updated fallback audio settings")
            old_soundcard = self.soundcard
            soundcard_changed = config.soundcard != self.soundcard
            old_eas_recording = self.eas_recording
            eas_recording_changed = config.eas_recording != self.eas_recording
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
                stream = None
                timeout = 0
                frequency_hz = self.frequency_hz
            elif self.pending_receiver is not None:
                self.csdr_server = config.csdr_server
                self.frequency_hz = config.station.frequency_hz
                self.pending_receiver = (self.csdr_server, self.frequency_hz)
                stream = None
                timeout = 0
                frequency_hz = self.frequency_hz
            elif frequency_changed:
                self.csdr_server = config.csdr_server
                stream = self.active_iq_stream
                timeout = self.csdr_server.timeout
                frequency_hz = config.station.frequency_hz
            else:
                self.csdr_server = config.csdr_server
                stream = None
                timeout = 0
                frequency_hz = self.frequency_hz

        if soundcard_changed:
            if not self._apply_soundcard_config(config.soundcard):
                applied = AppConfig(
                    csdr_server=config.csdr_server,
                    station=config.station,
                    icecast=config.icecast,
                    audio=config.audio,
                    fallback=config.fallback,
                    soundcard=old_soundcard,
                    eas_recording=config.eas_recording,
                )
                with self.state_lock:
                    self.soundcard = old_soundcard

        if eas_recording_changed:
            if not self._apply_eas_recording_config(config.eas_recording):
                applied = AppConfig(
                    csdr_server=config.csdr_server,
                    station=config.station,
                    icecast=config.icecast,
                    audio=config.audio,
                    fallback=config.fallback,
                    soundcard=applied.soundcard,
                    eas_recording=old_eas_recording,
                )
                with self.state_lock:
                    self.eas_recording = old_eas_recording

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
                    soundcard=applied.soundcard,
                    eas_recording=applied.eas_recording,
                )
        else:
            with self.state_lock:
                self.frequency_hz = frequency_hz
        return applied

    def _apply_soundcard_config(self, config: tuple[SoundcardConfig, ...]) -> bool:
        with self.soundcard_lock:
            current_workers = list(self.soundcard_workers)
        kept_workers: list[SoundcardWorker] = []
        removed_workers: list[SoundcardWorker] = []
        add_configs: list[SoundcardConfig] = []
        remaining_workers = list(current_workers)
        for soundcard_config in config:
            match = next(
                (
                    worker
                    for worker in remaining_workers
                    if worker.config == soundcard_config
                ),
                None,
            )
            if match is None:
                add_configs.append(soundcard_config)
                continue
            kept_workers.append(match)
            remaining_workers.remove(match)
        removed_workers = [
            worker for worker in current_workers if worker not in kept_workers
        ]

        new_workers: list[SoundcardWorker] = []
        try:
            for soundcard_config in add_configs:
                new_workers.append(self._create_soundcard_worker(soundcard_config))
            self._validate_unique_soundcard_workers(kept_workers + new_workers)
        except Exception as exc:
            self._stop_soundcard_workers(new_workers)
            LOG.warning("soundcard config reload failed; keeping existing output: %s", exc)
            return False
        final_workers = kept_workers + new_workers
        with self.soundcard_lock:
            self.soundcard_workers = final_workers
        self._stop_soundcard_workers(removed_workers)
        with self.state_lock:
            self.soundcard = config
        if add_configs or removed_workers:
            LOG.info("updated soundcard output settings")
        return True

    def _apply_eas_recording_config(self, config: EasRecordingConfig) -> bool:
        try:
            new_worker = self._create_eas_recorder_worker(config)
        except Exception as exc:
            LOG.warning("EAS recording config reload failed; keeping existing recorder: %s", exc)
            return False
        old_worker = self.eas_recorder_worker
        self.eas_recorder_worker = new_worker
        if old_worker is not None:
            self._stop_eas_recorder_worker(old_worker)
        with self.state_lock:
            self.eas_recording = config
        LOG.info("updated EAS recording settings")
        return True

    def _run_iq_producer(self) -> None:
        if self.same_test:
            self._run_same_test_producer()
            return
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

    def _run_same_test_producer(self) -> None:
        LOG.info("starting local SAME test audio generator")
        active_alert: np.ndarray | None = None
        position = 0
        inter_alert_silence_frames = 0
        while not self.stop_event.is_set():
            if active_alert is None and inter_alert_silence_frames <= 0:
                active_alert = self._pop_same_test_alert()
                position = 0

            if inter_alert_silence_frames > 0:
                self._queue_audio(np.zeros(SILENCE_FRAME_SAMPLES, dtype=np.float32))
                inter_alert_silence_frames -= 1
            elif active_alert is None:
                self._queue_audio(np.zeros(SILENCE_FRAME_SAMPLES, dtype=np.float32))
            else:
                frame = active_alert[position : position + SILENCE_FRAME_SAMPLES]
                if len(frame) < SILENCE_FRAME_SAMPLES:
                    padded = np.zeros(SILENCE_FRAME_SAMPLES, dtype=np.float32)
                    if len(frame):
                        padded[: len(frame)] = frame
                    self._queue_audio(padded)
                    active_alert = None
                    position = 0
                    inter_alert_silence_frames = SAME_TEST_INTER_ALERT_SILENCE_FRAMES
                else:
                    self._queue_audio(frame)
                    position += SILENCE_FRAME_SAMPLES
                    if position >= len(active_alert):
                        active_alert = None
                        position = 0
                        inter_alert_silence_frames = SAME_TEST_INTER_ALERT_SILENCE_FRAMES
            time.sleep(SILENCE_FRAME_SECONDS)
        LOG.info("local SAME test audio generator stopped")

    def request_same_test_alert(self) -> None:
        origin_time = datetime.now(timezone.utc)
        alert = generate_same_test_audio(origin_time=origin_time)
        with self.same_test_lock:
            self.same_test_alerts.append(alert)
        LOG.info("queued local SAME DMO test alert")

    def _pop_same_test_alert(self) -> np.ndarray | None:
        with self.same_test_lock:
            if not self.same_test_alerts:
                return None
            return self.same_test_alerts.popleft()

    def _play_same_test_alert(self, origin_time: datetime) -> None:
        audio = generate_same_test_audio(origin_time=origin_time)
        position = 0
        while not self.stop_event.is_set() and position < len(audio):
            frame = audio[position : position + SILENCE_FRAME_SAMPLES]
            if len(frame):
                self._queue_audio(frame)
            position += SILENCE_FRAME_SAMPLES
            time.sleep(SILENCE_FRAME_SECONDS)

    def _produce_from_iq_stream(self, iq_socket: socket.socket) -> None:
        iq_stream = self.active_iq_stream
        if iq_stream is None:
            raise ConnectionError("IQ stream is not active")
        demodulator = NfmDemodulator(iq_stream.iq_format)
        while not self.stop_event.is_set():
            replacement = self._try_hotswap_iq_stream(iq_stream)
            if replacement is not None:
                iq_stream, chunk = replacement
                iq_socket = iq_stream.stream_socket
                demodulator = NfmDemodulator(iq_stream.iq_format)
                if chunk:
                    self._process_iq_chunk(chunk, demodulator)
                continue
            readable, _, _ = select.select([iq_socket], [], [], 0.5)
            if not readable:
                continue
            chunk = iq_socket.recv(65536)
            if not chunk:
                raise ConnectionError("IQ stream socket closed")
            self._process_iq_chunk(chunk, demodulator)

    def _process_iq_chunk(
        self,
        chunk: bytes,
        demodulator: NfmDemodulator,
    ) -> None:
        audio = demodulator.process(chunk)
        if len(audio) == 0:
            return
        self._queue_audio(audio)

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
        bytes_per_iq_sample = 8 if iq_stream.iq_format == "f32" else 4
        chunk = iq_stream.stream_socket.recv(bytes_per_iq_sample)
        if not chunk:
            iq_stream.close()
            raise ConnectionError("replacement IQ stream socket closed")
        return chunk

    def _run_encoder_worker(
        self,
        encoder_group: EncoderWorker,
    ) -> None:
        try:
            while not self.stop_event.is_set() and not encoder_group.stop_event.is_set():
                try:
                    frame = encoder_group.pcm_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                encoded = encoder_group.encoder.encode(frame)
                if encoded:
                    self._broadcast_encoded(encoder_group, encoded)
            encoded = encoder_group.encoder.flush()
            if encoded:
                self._broadcast_encoded(encoder_group, encoded)
        except Exception as exc:
            encoder_group.error = exc

    def _run_pcm_playback_worker(self) -> None:
        pending = bytearray()
        audio_config = self._audio_config()
        effects = AudioEffectsProcessor(audio_config)
        next_write_at = time.monotonic()
        while not self.stop_event.is_set():
            self._drain_pcm_queue(self.pcm_queue, pending)
            self._apply_pcm_buffer_target(pending)
            audio_frame = self._next_buffered_pcm_frame(pending)
            if audio_frame is None:
                pcm_frame = self._idle_pcm_frame()
            else:
                next_audio_config = self._audio_config()
                if next_audio_config != audio_config:
                    audio_config = next_audio_config
                    effects = AudioEffectsProcessor(audio_config)
                    LOG.info("updated audio effects")
                audio = np.frombuffer(audio_frame, dtype="<f4")
                pcm_frame = float_to_s16(effects.process(audio))
                self._mark_pcm_output()
            self._write_pcm_frame(pcm_frame)

            next_write_at += SILENCE_FRAME_SECONDS
            delay = next_write_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_write_at = time.monotonic()

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

    def _queue_audio(self, audio: np.ndarray) -> None:
        audio_frame = audio.astype("<f4", copy=False).tobytes()
        self._queue_pcm_for_output(self.pcm_queue, audio_frame)

    def _write_pcm_frame(self, pcm: bytes) -> None:
        for encoder_group in list(self.encoder_groups):
            self._queue_pcm_for_output(encoder_group.pcm_queue, pcm)
        with self.soundcard_lock:
            soundcard_workers = list(self.soundcard_workers)
        for soundcard_worker in soundcard_workers:
            try:
                soundcard_worker.output.write(pcm)
            except Exception as exc:
                soundcard_worker.error = exc
        eas_worker = self.eas_recorder_worker
        if eas_worker is not None:
            try:
                eas_worker.output.write(pcm)
            except Exception as exc:
                if eas_worker.error is None:
                    LOG.warning("EAS recorder output failed; disabling recorder: %s", exc)
                eas_worker.error = exc
                self.eas_recorder_worker = None
                try:
                    eas_worker.output.close()
                except Exception:
                    pass

    def _mark_pcm_output(self) -> None:
        with self.state_lock:
            self.last_pcm_at = time.monotonic()
        self._reset_fallback_state()

    def _idle_pcm_frame(self) -> bytes:
        with self.state_lock:
            idle_seconds = time.monotonic() - self.last_pcm_at
            fallback = self.fallback
            fallback_audio = self.fallback_audio
        state = self.fallback_state
        if not state.active and idle_seconds < fallback.silence_timeout_seconds:
            state.reset()
            return SILENCE_FRAME
        if not state.active:
            state.active = True
            LOG.info(
                "starting fallback audio after %.1f seconds without PCM",
                idle_seconds,
            )
        return _next_fallback_frame(
            fallback_audio,
            state,
            fallback.loop_delay_seconds,
        )

    def _reset_fallback_state(self) -> None:
        if self.fallback_state.active:
            LOG.info("stopping fallback audio")
        self.fallback_state.reset()

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

    def _next_buffered_pcm_frame(self, pending: bytearray) -> bytes | None:
        target_bytes = self._pcm_buffer_target_bytes()
        if not self.pcm_buffer_ready:
            required = max(AUDIO_FRAME_BYTES, target_bytes)
            if len(pending) < required:
                return None
            self.pcm_buffer_ready = True
            LOG.debug(
                "audio buffer ready with %.2f seconds buffered",
                len(pending) / np.dtype("<f4").itemsize / IQ_SAMPLE_RATE,
            )
        low_water_bytes = self._pcm_buffer_low_water_bytes(target_bytes)
        if low_water_bytes and len(pending) <= low_water_bytes:
            self.pcm_buffer_ready = False
            LOG.debug(
                "audio buffer low with %.2f seconds remaining; refilling",
                len(pending) / np.dtype("<f4").itemsize / IQ_SAMPLE_RATE,
            )
            return None
        if len(pending) < AUDIO_FRAME_BYTES:
            self.pcm_buffer_ready = False
            return None
        frame = bytes(pending[:AUDIO_FRAME_BYTES])
        del pending[:AUDIO_FRAME_BYTES]
        return frame

    def _apply_pcm_buffer_target(self, pending: bytearray) -> None:
        target_bytes = self._pcm_buffer_target_bytes()
        previous_target = self.pcm_buffer_target_bytes
        self.pcm_buffer_target_bytes = target_bytes
        if target_bytes > previous_target and len(pending) < target_bytes:
            self.pcm_buffer_ready = False
        if (
            target_bytes < previous_target
            and len(pending) > max(target_bytes, AUDIO_FRAME_BYTES)
        ):
            keep_bytes = max(target_bytes, AUDIO_FRAME_BYTES)
            del pending[: len(pending) - keep_bytes]

    def _pcm_buffer_target_bytes(self) -> int:
        with self.state_lock:
            buffer_seconds = self.csdr_server.buffer_seconds
        frame_count = round(buffer_seconds / SILENCE_FRAME_SECONDS)
        return max(0, frame_count * AUDIO_FRAME_BYTES)

    def _pcm_buffer_low_water_bytes(self, target_bytes: int) -> int:
        if target_bytes <= AUDIO_FRAME_BYTES:
            return 0
        low_water_frames = max(
            1,
            round(target_bytes * PCM_BUFFER_LOW_WATER_RATIO / AUDIO_FRAME_BYTES),
        )
        return low_water_frames * AUDIO_FRAME_BYTES

    def _clear_pcm_queue(self, pcm_queue: queue.Queue[bytes]) -> None:
        while True:
            try:
                pcm_queue.get_nowait()
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
            audio = self.audio
        if not self.same_test:
            return audio
        return AudioConfig(
            deemphasis=DeemphasisConfig(enabled=False),
            volume=audio.volume,
            highpass=audio.highpass,
            lowpass=audio.lowpass,
            notch=audio.notch,
        )

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


def _soundcard_label(config: SoundcardConfig) -> str:
    return str(config.device) if config.device is not None else "default device"


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
