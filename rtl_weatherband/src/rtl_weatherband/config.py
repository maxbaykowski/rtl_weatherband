from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json5


NOAA_MIN_MHZ = 162.4
NOAA_MAX_MHZ = 162.55
IQ_SAMPLE_RATE = 16000
VALID_OUTPUT_SAMPLE_RATES = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000)
MP3_BITRATE_LIMITS_BY_SAMPLE_RATE = {
    8000: (8, 64),
    11025: (8, 64),
    16000: (8, 160),
    22050: (8, 160),
    24000: (8, 160),
    32000: (32, 320),
    44100: (32, 320),
    48000: (32, 320),
}


class ConfigError(ValueError):
    """Raised when the JSON5 configuration is invalid."""


@dataclass(frozen=True)
class CsdrServerConfig:
    host: str
    listen_port: int
    timeout_seconds: float = 10.0

    @property
    def control_port(self) -> int:
        return self.listen_port + 1


@dataclass(frozen=True)
class StationConfig:
    frequency_mhz: float

    @property
    def frequency_hz(self) -> int:
        return round(self.frequency_mhz * 1_000_000)


@dataclass(frozen=True)
class IcecastConfig:
    host: str
    port: int
    mount: str
    username: str
    password: str
    bitrate: int
    tls: bool = False
    name: str | None = None
    genre: str | None = None
    description: str | None = None
    public: bool = False


@dataclass(frozen=True)
class AudioConfig:
    format: str
    sample_rate: int
    deemphasis_tau: float = 530.0

    @property
    def content_type(self) -> str:
        if self.format == "mp3":
            return "audio/mpeg"
        if self.format == "ogg":
            return "application/ogg"
        raise ConfigError(f"unsupported audio format: {self.format}")


@dataclass(frozen=True)
class DspConfig:
    ffmpeg_path: str = "ffmpeg"


@dataclass(frozen=True)
class AppConfig:
    csdr_server: CsdrServerConfig
    station: StationConfig
    icecast: IcecastConfig
    audio: AudioConfig
    dsp: DspConfig


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fp:
        raw = json5.load(fp)
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be an object")
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> AppConfig:
    csdr_server = _section(raw, "csdr_server")
    station = _section(raw, "station")
    icecast = _section(raw, "icecast")
    audio = _section(raw, "audio")
    dsp = raw.get("dsp", {})
    if not isinstance(dsp, dict):
        raise ConfigError("dsp must be an object")

    app = AppConfig(
        csdr_server=CsdrServerConfig(
            host=_str(csdr_server, "host"),
            listen_port=_int(csdr_server, "listen_port"),
            timeout_seconds=float(csdr_server.get("timeout_seconds", 10.0)),
        ),
        station=StationConfig(frequency_mhz=float(station["frequency_mhz"])),
        icecast=IcecastConfig(
            host=_str(icecast, "host"),
            port=_int(icecast, "port"),
            mount=_mount(_str(icecast, "mount")),
            username=_str(icecast, "username"),
            password=_str(icecast, "password"),
            bitrate=_int(icecast, "bitrate"),
            tls=bool(icecast.get("tls", False)),
            name=_optional_str(icecast, "name"),
            genre=_optional_str(icecast, "genre"),
            description=_optional_str(icecast, "description"),
            public=bool(icecast.get("public", False)),
        ),
        audio=AudioConfig(
            format=_str(audio, "format").lower(),
            sample_rate=_int(audio, "sample_rate"),
            deemphasis_tau=float(audio.get("deemphasis_tau", 530.0)),
        ),
        dsp=DspConfig(
            ffmpeg_path=str(dsp.get("ffmpeg_path", "ffmpeg")),
        ),
    )
    validate_config(app)
    return app


def validate_config(config: AppConfig) -> None:
    frequency = config.station.frequency_mhz
    if not NOAA_MIN_MHZ <= frequency <= NOAA_MAX_MHZ:
        raise ConfigError(
            f"frequency_mhz must be between {NOAA_MIN_MHZ} and {NOAA_MAX_MHZ}"
        )
    if config.csdr_server.listen_port <= 0 or config.csdr_server.listen_port > 65534:
        raise ConfigError("csdr_server.listen_port must be from 1 through 65534")
    if config.icecast.port <= 0 or config.icecast.port > 65535:
        raise ConfigError("icecast.port must be from 1 through 65535")
    if config.audio.format not in {"mp3", "ogg"}:
        raise ConfigError("audio.format must be either 'mp3' or 'ogg'")
    output_errors = _output_config_errors(config)
    if output_errors:
        raise ConfigError("; ".join(output_errors))
    if not 0 <= config.audio.deemphasis_tau <= 530:
        raise ConfigError("audio.deemphasis_tau must be between 0 and 530")


def _output_config_errors(config: AppConfig) -> list[str]:
    errors: list[str] = []
    sample_rate = config.audio.sample_rate
    bitrate = config.icecast.bitrate
    if sample_rate not in VALID_OUTPUT_SAMPLE_RATES:
        rates = ", ".join(str(rate) for rate in VALID_OUTPUT_SAMPLE_RATES)
        errors.append(f"invalid sample rate: audio.sample_rate must be one of {rates}")
    if bitrate <= 0:
        errors.append("invalid bitrate: icecast.bitrate must be greater than 0")
    elif config.audio.format == "mp3" and sample_rate in MP3_BITRATE_LIMITS_BY_SAMPLE_RATE:
        minimum, maximum = MP3_BITRATE_LIMITS_BY_SAMPLE_RATE[sample_rate]
        if not minimum <= bitrate <= maximum:
            errors.append(
                "invalid bitrate: icecast.bitrate must be between "
                f"{minimum} and {maximum} Kbps for MP3 at {sample_rate} Hz"
            )
    return errors


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be an object")
    return value


def _str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _optional_str(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string")
    return value


def _int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")
    return value


def _mount(value: str) -> str:
    return value if value.startswith("/") else f"/{value}"
