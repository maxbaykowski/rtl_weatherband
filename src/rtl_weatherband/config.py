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
    port: int
    timeout: float = 10.0

    @property
    def control_port(self) -> int:
        return self.port + 1


@dataclass(frozen=True)
class StationConfig:
    frequency: float

    @property
    def frequency_hz(self) -> int:
        return round(self.frequency * 1_000_000)


@dataclass(frozen=True)
class IcecastConfig:
    host: str
    port: int
    mount: str
    username: str
    password: str
    format: str
    sample_rate: int
    bitrate: int
    tls: bool = False
    name: str | None = None
    genre: str | None = None
    description: str | None = None
    public: bool = False

    @property
    def content_type(self) -> str:
        if self.format == "mp3":
            return "audio/mpeg"
        if self.format == "ogg":
            return "application/ogg"
        raise ConfigError(f"unsupported icecast format: {self.format}")


@dataclass(frozen=True)
class AudioConfig:
    deemphasis_tau: float = 530.0


@dataclass(frozen=True)
class AppConfig:
    csdr_server: CsdrServerConfig
    station: StationConfig
    icecast: IcecastConfig
    audio: AudioConfig


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    raw = load_raw_config(config_path)
    return parse_config(raw)


def load_raw_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fp:
        raw = json5.load(fp)
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be an object")
    return raw


def parse_config(raw: dict[str, Any]) -> AppConfig:
    csdr_server = _section(raw, "csdr_server")
    station = _section(raw, "station")
    icecast = _section(raw, "icecast")
    audio = _section(raw, "audio")

    app = AppConfig(
        csdr_server=parse_csdr_server_config(csdr_server),
        station=parse_station_config(station),
        icecast=parse_icecast_config(icecast),
        audio=parse_audio_config(audio),
    )
    validate_config(app)
    return app


def merge_valid_reload_config(
    raw: dict[str, Any],
    current: AppConfig,
) -> tuple[AppConfig, list[str]]:
    errors: list[str] = []

    try:
        csdr_server = parse_csdr_server_config(_section(raw, "csdr_server"))
        validate_csdr_server_config(csdr_server)
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"csdr_server: {exc}")
        csdr_server = current.csdr_server

    try:
        station = parse_station_config(_section(raw, "station"))
        validate_station_config(station)
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"station: {exc}")
        station = current.station

    try:
        icecast = parse_icecast_config(_section(raw, "icecast"))
        validate_icecast_config(icecast)
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"icecast: {exc}")
        icecast = current.icecast

    try:
        audio = parse_audio_config(_section(raw, "audio"))
        validate_audio_config(audio)
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"audio: {exc}")
        audio = current.audio

    return AppConfig(
        csdr_server=csdr_server,
        station=station,
        icecast=icecast,
        audio=audio,
    ), errors


def parse_csdr_server_config(raw: dict[str, Any]) -> CsdrServerConfig:
    return CsdrServerConfig(
        host=_str(raw, "host"),
        port=_int(raw, "port"),
        timeout=float(raw.get("timeout", 10.0)),
    )


def parse_station_config(raw: dict[str, Any]) -> StationConfig:
    return StationConfig(frequency=float(raw["frequency"]))


def parse_icecast_config(raw: dict[str, Any]) -> IcecastConfig:
    return IcecastConfig(
        host=_str(raw, "host"),
        port=_int(raw, "port"),
        mount=_mount(_str(raw, "mount")),
        username=_str(raw, "username"),
        password=_str(raw, "password"),
        format=_str(raw, "format").lower(),
        sample_rate=_int(raw, "sample_rate"),
        bitrate=_int(raw, "bitrate"),
        tls=bool(raw.get("tls", False)),
        name=_optional_str(raw, "name"),
        genre=_optional_str(raw, "genre"),
        description=_optional_str(raw, "description"),
        public=bool(raw.get("public", False)),
    )


def parse_audio_config(raw: dict[str, Any]) -> AudioConfig:
    return AudioConfig(deemphasis_tau=float(raw.get("deemphasis_tau", 530.0)))


def validate_config(config: AppConfig) -> None:
    validate_station_config(config.station)
    validate_csdr_server_config(config.csdr_server)
    validate_icecast_config(config.icecast)
    validate_audio_config(config.audio)


def validate_station_config(config: StationConfig) -> None:
    frequency = config.frequency
    if not NOAA_MIN_MHZ <= frequency <= NOAA_MAX_MHZ:
        raise ConfigError(
            f"station.frequency must be between {NOAA_MIN_MHZ} and {NOAA_MAX_MHZ}"
        )


def validate_csdr_server_config(config: CsdrServerConfig) -> None:
    if config.port <= 0 or config.port > 65534:
        raise ConfigError("csdr_server.port must be from 1 through 65534")


def validate_icecast_config(config: IcecastConfig) -> None:
    if config.port <= 0 or config.port > 65535:
        raise ConfigError("icecast.port must be from 1 through 65535")
    if config.format not in {"mp3", "ogg"}:
        raise ConfigError("icecast.format must be either 'mp3' or 'ogg'")
    output_errors = _output_config_errors(config)
    if output_errors:
        raise ConfigError("; ".join(output_errors))


def validate_audio_config(config: AudioConfig) -> None:
    if not 0 <= config.deemphasis_tau <= 530:
        raise ConfigError("audio.deemphasis_tau must be between 0 and 530")


def _output_config_errors(config: IcecastConfig) -> list[str]:
    errors: list[str] = []
    sample_rate = config.sample_rate
    bitrate = config.bitrate
    if sample_rate not in VALID_OUTPUT_SAMPLE_RATES:
        rates = ", ".join(str(rate) for rate in VALID_OUTPUT_SAMPLE_RATES)
        errors.append(f"invalid sample rate: icecast.sample_rate must be one of {rates}")
    if bitrate <= 0:
        errors.append("invalid bitrate: icecast.bitrate must be greater than 0")
    elif (
        config.format == "mp3"
        and sample_rate in MP3_BITRATE_LIMITS_BY_SAMPLE_RATE
    ):
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
