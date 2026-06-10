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
    deemphasis_thread: threading.Thread | None = None

    def start(self, iq_socket: socket.socket, encoded_output: BinaryIO) -> None:
        fmdemod = subprocess.Popen(
            [self.dsp.csdr_path, "fmdemod"],
            stdin=iq_socket.fileno(),
            stdout=subprocess.PIPE,
            stderr=None,
        )
        convert = subprocess.Popen(
            [self.dsp.csdr_path, "convert", "--informat", "float", "--outformat", "s16"],
            stdin=fmdemod.stdout,
            stdout=subprocess.PIPE,
            stderr=None,
        )
        if fmdemod.stdout is not None:
            fmdemod.stdout.close()

        ffmpeg_stdin: int | BinaryIO | None
        if self.audio.deemphasis_tau > 0:
            ffmpeg_stdin = subprocess.PIPE
        else:
            ffmpeg_stdin = convert.stdout

        ffmpeg = subprocess.Popen(
            self._ffmpeg_command(),
            stdin=ffmpeg_stdin,
            stdout=encoded_output,
            stderr=None,
        )
        if self.audio.deemphasis_tau > 0:
            self.deemphasis_thread = threading.Thread(
                target=self._run_deemphasis,
                args=(convert.stdout, ffmpeg.stdin),
                name="deemphasis",
                daemon=True,
            )
            self.deemphasis_thread.start()
        elif convert.stdout is not None:
            convert.stdout.close()

        self.processes = [
            ("fmdemod", fmdemod),
            ("convert float to s16", convert),
            ("ffmpeg encoder", ffmpeg),
        ]

    def wait(self) -> ProcessExit:
        if not self.processes:
            raise PipelineError("pipeline was not started")
        while True:
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

    def _run_deemphasis(
        self,
        pcm_source: BinaryIO | None,
        pcm_sink: BinaryIO | None,
    ) -> None:
        if pcm_source is None or pcm_sink is None:
            return
        deemphasis = DeemphasisFilter(IQ_SAMPLE_RATE, self.audio.deemphasis_tau)
        try:
            while True:
                chunk = pcm_source.read(65536)
                if not chunk:
                    break
                filtered = deemphasis.process(chunk)
                if filtered:
                    pcm_sink.write(filtered)
            tail = deemphasis.flush()
            if tail:
                pcm_sink.write(tail)
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                pcm_source.close()
            except OSError:
                pass
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
