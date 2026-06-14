from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
import wave

from .config_errors import ConfigError


MAX_FALLBACK_AUDIO_SECONDS = 30.0
DEFAULT_FALLBACK_AUDIO = "assets/fallback.wav"


@dataclass(frozen=True)
class FallbackAudio:
    sample_rate: int
    pcm: bytes
    duration_seconds: float
    source: str


def load_fallback_audio(path: str | None) -> FallbackAudio:
    if path is None:
        resource = files("rtl_weatherband").joinpath(DEFAULT_FALLBACK_AUDIO)
        with as_file(resource) as fallback_path:
            return _load_wave_path(fallback_path, "packaged fallback.wav")
    return _load_wave_path(Path(path), path)


def _load_wave_path(path: Path, source: str) -> FallbackAudio:
    try:
        if path.is_dir():
            raise IsADirectoryError(21, "Is a directory", str(path))
        if not path.exists():
            raise FileNotFoundError(2, "No such file or directory", str(path))
        if not path.is_file():
            raise OSError(f"{path}: not a regular file")
        with path.open("rb") as fp:
            try:
                with wave.open(fp, "rb") as wav:
                    channels = wav.getnchannels()
                    sample_width = wav.getsampwidth()
                    sample_rate = wav.getframerate()
                    frame_count = wav.getnframes()
                    compression = wav.getcomptype()
                    duration = frame_count / sample_rate if sample_rate else 0.0
                    if channels != 1:
                        raise ConfigError("fallback audio must be mono")
                    if sample_width != 2 or compression != "NONE":
                        raise ConfigError(
                            "fallback audio must be PCM signed 16-bit little-endian"
                        )
                    if sample_rate <= 0 or frame_count <= 0:
                        raise ConfigError("fallback audio must contain audio samples")
                    if duration >= MAX_FALLBACK_AUDIO_SECONDS:
                        raise ConfigError("fallback audio must be under 30 seconds")
                    pcm = wav.readframes(frame_count)
            except wave.Error as exc:
                raise ConfigError(f"fallback audio is not a valid wave file: {exc}") from exc
    except OSError as exc:
        raise ConfigError(str(exc)) from exc
    return FallbackAudio(
        sample_rate=sample_rate,
        pcm=pcm,
        duration_seconds=duration,
        source=source,
    )
