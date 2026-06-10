from __future__ import annotations

import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import BinaryIO

from .config import AudioConfig, DspConfig, IQ_SAMPLE_RATE
from .deemphasis import DeemphasisFilter
from .nfm import NfmDemodulator, float_to_s16


class PipelineError(RuntimeError):
    """Raised when the DSP or encoder pipeline fails."""


@dataclass(frozen=True)
class ProcessExit:
    stage: str
    returncode: int


@dataclass
class StreamPipeline:
    dsp: DspConfig
    audio: AudioConfig
    processes: list[tuple[str, subprocess.Popen[bytes]]] = field(default_factory=list)
    dsp_thread: threading.Thread | None = None
    dsp_error: BaseException | None = None

    def start(self, iq_socket: socket.socket, encoded_output: BinaryIO) -> None:
        ffmpeg = subprocess.Popen(
            self._ffmpeg_command(),
            stdin=subprocess.PIPE,
            stdout=encoded_output,
            stderr=None,
        )
        self.processes = [("ffmpeg encoder", ffmpeg)]
        self.dsp_thread = threading.Thread(
            target=self._run_numpy_dsp,
            args=(iq_socket, ffmpeg.stdin),
            name="numpy-nfm-demodulator",
            daemon=True,
        )
        self.dsp_thread.start()

    def wait(self) -> ProcessExit:
        if not self.processes:
            raise PipelineError("pipeline was not started")
        while True:
            if self.dsp_thread is not None and not self.dsp_thread.is_alive():
                return ProcessExit(
                    stage="numpy nfm demodulator",
                    returncode=1 if self.dsp_error is not None else 0,
                )
            for stage, process in self.processes:
                returncode = process.poll()
                if returncode is not None:
                    return ProcessExit(stage=stage, returncode=returncode)
            time.sleep(0.25)

    def stop(self) -> None:
        for _, process in reversed(self.processes):
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for _, process in reversed(self.processes):
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

    def _run_numpy_dsp(
        self,
        iq_socket: socket.socket,
        pcm_sink: BinaryIO | None,
    ) -> None:
        if pcm_sink is None:
            return
        demodulator = NfmDemodulator()
        deemphasis = DeemphasisFilter(IQ_SAMPLE_RATE, self.audio.deemphasis_tau)
        try:
            while True:
                chunk = iq_socket.recv(65536)
                if not chunk:
                    break
                audio = demodulator.process(chunk)
                if len(audio) == 0:
                    continue
                audio = deemphasis.process_float(audio)
                pcm_sink.write(float_to_s16(audio))
        except (BrokenPipeError, OSError) as exc:
            self.dsp_error = exc
        except BaseException as exc:
            self.dsp_error = exc
            raise
        finally:
            try:
                pcm_sink.close()
            except OSError:
                pass

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
            self.audio.bitrate,
        ]
        if self.audio.format == "mp3":
            command.extend(["-f", "mp3", "-codec:a", "libmp3lame", "pipe:1"])
        elif self.audio.format == "ogg":
            command.extend(["-f", "ogg", "-codec:a", "libvorbis", "pipe:1"])
        else:
            raise PipelineError(f"unsupported audio format: {self.audio.format}")
        return command
