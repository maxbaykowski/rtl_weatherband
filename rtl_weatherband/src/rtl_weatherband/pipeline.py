from __future__ import annotations

import signal
import socket
import subprocess
from dataclasses import dataclass, field
from typing import BinaryIO

from .config import AudioConfig, DspConfig, IQ_SAMPLE_RATE


class PipelineError(RuntimeError):
    """Raised when the DSP or encoder pipeline fails."""


@dataclass
class StreamPipeline:
    dsp: DspConfig
    audio: AudioConfig
    processes: list[subprocess.Popen[bytes]] = field(default_factory=list)

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
        ffmpeg = subprocess.Popen(
            self._ffmpeg_command(),
            stdin=convert.stdout,
            stdout=encoded_output,
            stderr=None,
        )
        if convert.stdout is not None:
            convert.stdout.close()

        self.processes = [fmdemod, convert, ffmpeg]

    def wait(self) -> None:
        if not self.processes:
            raise PipelineError("pipeline was not started")
        encoder = self.processes[-1]
        return_code = encoder.wait()
        if return_code != 0:
            raise PipelineError(f"ffmpeg exited with status {return_code}")
        for process in self.processes[:-1]:
            process.wait(timeout=2)

    def stop(self) -> None:
        for process in reversed(self.processes):
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for process in reversed(self.processes):
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

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
