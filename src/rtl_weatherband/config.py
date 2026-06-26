from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json5

from .config_errors import ConfigError
from .fallback_audio import load_fallback_audio


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
OGG_BITRATE_LIMITS_BY_SAMPLE_RATE = {
    8000: (8, 42),
    11025: (12, 50),
    16000: (16, 100),
    22050: (16, 90),
    24000: (16, 90),
    32000: (30, 190),
    44100: (32, 240),
    48000: (32, 240),
}
SAME_ATTENTION_BAND_HZ = (900.0, 1100.0)
SAME_SPACE_BAND_HZ = (1400.0, 1600.0)
SAME_MARK_BAND_HZ = (2000.0, 2200.0)
PROTECTED_AUDIO_BANDS_HZ = (
    ("1050 Hz attention tone", SAME_ATTENTION_BAND_HZ),
    ("SAME space tone", SAME_SPACE_BAND_HZ),
    ("SAME mark tone", SAME_MARK_BAND_HZ),
)
AUDIO_NYQUIST_HZ = IQ_SAMPLE_RATE / 2


@dataclass(frozen=True)
class CsdrServerConfig:
    host: str
    port: int
    timeout: float = 10.0
    iq_format: str = "f32"
    buffer_seconds: float = 1.0

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
    enabled: bool = True
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
class DeemphasisConfig:
    enabled: bool = True
    tau: float = 530.0


@dataclass(frozen=True)
class VolumeConfig:
    enabled: bool = False
    multiplier: float = 1.0


@dataclass(frozen=True)
class FilterConfig:
    enabled: bool = False
    frequency: float = 0.0
    sharpness: float = 0.0


@dataclass(frozen=True)
class AudioConfig:
    deemphasis: DeemphasisConfig = field(default_factory=DeemphasisConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    highpass: FilterConfig = field(default_factory=FilterConfig)
    lowpass: FilterConfig = field(default_factory=FilterConfig)
    notch: FilterConfig = field(default_factory=FilterConfig)

    @property
    def deemphasis_tau(self) -> float:
        return self.deemphasis.tau if self.deemphasis.enabled else 0.0


@dataclass(frozen=True)
class FallbackConfig:
    silence_timeout_seconds: float = 30.0
    loop_delay_seconds: float = 0.0
    path: str | None = None


@dataclass(frozen=True)
class SoundcardConfig:
    enabled: bool = False
    device: str | int | None = None


@dataclass(frozen=True)
class EasRecordingConfig:
    enabled: bool = False
    pre_seconds: float = 2.0
    post_seconds: float = 5.0
    max_seconds: int = 120
    directory: str = "alerts"
    format: str = "wav"
    local_time: bool = False


@dataclass(frozen=True)
class AppConfig:
    csdr_server: CsdrServerConfig
    station: StationConfig
    icecast: tuple[IcecastConfig, ...]
    audio: AudioConfig
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    soundcard: tuple[SoundcardConfig, ...] = field(default_factory=tuple)
    eas_recording: EasRecordingConfig = field(default_factory=EasRecordingConfig)


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
    icecast = raw.get("icecast")
    audio = _section(raw, "audio")
    fallback = raw.get("fallback")
    soundcard = raw.get("soundcard")
    eas_recording = raw.get("eas_recording")

    app = AppConfig(
        csdr_server=parse_csdr_server_config(csdr_server),
        station=parse_station_config(station),
        icecast=parse_icecast_configs(icecast),
        audio=parse_audio_config(audio),
        fallback=parse_fallback_config(fallback),
        soundcard=parse_soundcard_config(soundcard),
        eas_recording=parse_eas_recording_config(eas_recording),
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
        icecast = parse_icecast_configs(raw.get("icecast"))
        validate_icecast_configs(icecast)
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"icecast: {exc}")
        icecast = current.icecast

    try:
        audio, audio_errors = merge_valid_audio_config(_section(raw, "audio"), current.audio)
        errors.extend(f"audio.{error}" for error in audio_errors)
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"audio: {exc}")
        audio = current.audio

    try:
        fallback = parse_fallback_config(raw.get("fallback"))
        validate_fallback_config(fallback)
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"fallback: {exc}")
        fallback = current.fallback

    try:
        validate_buffer_config(csdr_server, fallback)
    except ConfigError as exc:
        errors.append(str(exc))
        if csdr_server != current.csdr_server:
            csdr_server = current.csdr_server
        if fallback != current.fallback:
            fallback = current.fallback

    try:
        soundcard = parse_soundcard_config(raw.get("soundcard"))
        validate_soundcard_configs(soundcard)
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"soundcard: {exc}")
        soundcard = current.soundcard

    try:
        eas_recording = parse_eas_recording_config(raw.get("eas_recording"))
        validate_eas_recording_config(eas_recording)
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"eas_recording: {exc}")
        eas_recording = current.eas_recording

    return AppConfig(
        csdr_server=csdr_server,
        station=station,
        icecast=icecast,
        audio=audio,
        fallback=fallback,
        soundcard=soundcard,
        eas_recording=eas_recording,
    ), errors


def parse_csdr_server_config(raw: dict[str, Any]) -> CsdrServerConfig:
    return CsdrServerConfig(
        host=_str(raw, "host"),
        port=_int(raw, "port"),
        timeout=float(raw.get("timeout", 10.0)),
        iq_format=_str(raw, "iq_format", "f32").lower(),
        buffer_seconds=float(raw.get("buffer_seconds", 1.0)),
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
        enabled=_bool(raw, "enabled", True),
        tls=_bool(raw, "tls", False),
        name=_optional_str(raw, "name"),
        genre=_optional_str(raw, "genre"),
        description=_optional_str(raw, "description"),
        public=_bool(raw, "public", False),
    )


def parse_icecast_configs(raw: Any) -> tuple[IcecastConfig, ...]:
    if isinstance(raw, list):
        configs = tuple(parse_icecast_config(_object(value, "icecast")) for value in raw)
    elif isinstance(raw, dict) and "destinations" in raw:
        if not _bool(raw, "enabled", True):
            return ()
        destinations = raw["destinations"]
        if not isinstance(destinations, list):
            raise ConfigError("icecast.destinations must be an array")
        configs = tuple(
            parse_icecast_config(_object(value, "icecast destination"))
            for value in destinations
        )
    elif isinstance(raw, dict):
        configs = (parse_icecast_config(raw),)
    else:
        raise ConfigError("icecast must be an object or array")
    return tuple(config for config in configs if config.enabled)


def parse_audio_config(raw: dict[str, Any]) -> AudioConfig:
    if "deemphasis_tau" in raw and "deemphasis" not in raw:
        tau = _float(raw, "deemphasis_tau")
        audio = AudioConfig(
            deemphasis=DeemphasisConfig(enabled=tau > 0, tau=tau),
            volume=parse_volume_config(raw.get("volume")),
            highpass=parse_filter_config(raw.get("highpass")),
            lowpass=parse_filter_config(raw.get("lowpass")),
            notch=parse_filter_config(raw.get("notch")),
        )
    else:
        audio = AudioConfig(
            deemphasis=parse_deemphasis_config(raw.get("deemphasis")),
            volume=parse_volume_config(raw.get("volume")),
            highpass=parse_filter_config(raw.get("highpass")),
            lowpass=parse_filter_config(raw.get("lowpass")),
            notch=parse_filter_config(raw.get("notch")),
        )
    validate_audio_config(audio)
    return audio


def merge_valid_audio_config(
    raw: dict[str, Any],
    current: AudioConfig,
) -> tuple[AudioConfig, list[str]]:
    errors: list[str] = []
    changed_filters: set[str] = set()

    try:
        if "deemphasis_tau" in raw and "deemphasis" not in raw:
            tau = _float(raw, "deemphasis_tau")
            deemphasis = DeemphasisConfig(enabled=tau > 0, tau=tau)
        else:
            deemphasis = parse_deemphasis_config(raw.get("deemphasis"))
        validate_deemphasis_config(deemphasis)
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"deemphasis: {exc}")
        deemphasis = current.deemphasis

    try:
        volume = parse_volume_config(raw.get("volume"))
        validate_volume_config(volume)
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"volume: {exc}")
        volume = current.volume

    try:
        highpass = parse_filter_config(raw.get("highpass"))
        validate_filter_config("highpass", highpass)
        if highpass != current.highpass:
            changed_filters.add("highpass")
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"highpass: {exc}")
        highpass = current.highpass

    try:
        lowpass = parse_filter_config(raw.get("lowpass"))
        validate_filter_config("lowpass", lowpass)
        if lowpass != current.lowpass:
            changed_filters.add("lowpass")
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"lowpass: {exc}")
        lowpass = current.lowpass

    try:
        notch = parse_filter_config(raw.get("notch"))
        validate_filter_config("notch", notch)
        if notch != current.notch:
            changed_filters.add("notch")
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"notch: {exc}")
        notch = current.notch

    audio = AudioConfig(
        deemphasis=deemphasis,
        volume=volume,
        highpass=highpass,
        lowpass=lowpass,
        notch=notch,
    )
    audio, relationship_errors = _merge_valid_audio_filter_relationships(
        audio,
        current,
        changed_filters,
    )
    errors.extend(relationship_errors)
    return audio, errors


def parse_deemphasis_config(raw: Any) -> DeemphasisConfig:
    if raw is None:
        return DeemphasisConfig()
    raw = _object(raw, "audio.deemphasis")
    return DeemphasisConfig(
        enabled=_bool(raw, "enabled", True),
        tau=float(raw.get("tau", 530.0)),
    )


def parse_volume_config(raw: Any) -> VolumeConfig:
    if raw is None:
        return VolumeConfig()
    raw = _object(raw, "audio.volume")
    return VolumeConfig(
        enabled=_bool(raw, "enabled", False),
        multiplier=float(raw.get("multiplier", 1.0)),
    )


def parse_filter_config(raw: Any) -> FilterConfig:
    if raw is None:
        return FilterConfig()
    raw = _object(raw, "audio filter")
    return FilterConfig(
        enabled=_bool(raw, "enabled", False),
        frequency=float(raw.get("frequency", 0.0)),
        sharpness=float(raw.get("sharpness", 0.0)),
    )


def parse_fallback_config(raw: Any) -> FallbackConfig:
    if raw is None:
        return FallbackConfig()
    raw = _object(raw, "fallback")
    return FallbackConfig(
        silence_timeout_seconds=float(
            raw.get(
                "silence_timeout_seconds",
                raw.get("silence_allowed_seconds", 30.0),
            )
        ),
        loop_delay_seconds=float(raw.get("loop_delay_seconds", 0.0)),
        path=_optional_str(raw, "path"),
    )


def parse_soundcard_config(raw: Any) -> tuple[SoundcardConfig, ...]:
    if raw is None:
        return ()
    raw = _object(raw, "soundcard")
    section_enabled = _bool(raw, "enabled", True)
    if not section_enabled:
        return ()
    if "outputs" in raw:
        outputs = raw["outputs"]
        if not isinstance(outputs, list):
            raise ConfigError("soundcard.outputs must be an array")
        configs = tuple(
            parse_soundcard_output_config(_object(value, "soundcard output"))
            for value in outputs
        )
        return tuple(config for config in configs if config.enabled)
    config = parse_soundcard_output_config(raw)
    return (config,) if config.enabled else ()


def parse_soundcard_output_config(raw: dict[str, Any]) -> SoundcardConfig:
    device = raw.get("device")
    if device is not None and not isinstance(device, str | int):
        raise ConfigError("soundcard.device must be a string, integer, or null")
    if isinstance(device, str) and not device:
        raise ConfigError("soundcard.device must not be an empty string")
    return SoundcardConfig(
        enabled=_bool(raw, "enabled", False),
        device=device,
    )


def parse_eas_recording_config(raw: Any) -> EasRecordingConfig:
    if raw is None:
        return EasRecordingConfig()
    raw = _object(raw, "eas_recording")
    max_seconds = raw.get("max_seconds", 120)
    if not isinstance(max_seconds, int) or isinstance(max_seconds, bool):
        raise ConfigError("eas_recording.max_seconds must be an integer")
    return EasRecordingConfig(
        enabled=_bool(raw, "enabled", False),
        pre_seconds=float(raw.get("pre_seconds", 2.0)),
        post_seconds=float(raw.get("post_seconds", 5.0)),
        max_seconds=max_seconds,
        directory=_str(raw, "directory", "alerts"),
        format=_str(raw, "format", "wav").lower(),
        local_time=_bool(raw, "local_time", False),
    )


def validate_config(config: AppConfig) -> None:
    validate_station_config(config.station)
    validate_csdr_server_config(config.csdr_server)
    validate_icecast_configs(config.icecast)
    validate_audio_config(config.audio)
    validate_fallback_config(config.fallback)
    validate_soundcard_configs(config.soundcard)
    validate_eas_recording_config(config.eas_recording)
    validate_buffer_config(config.csdr_server, config.fallback)


def validate_station_config(config: StationConfig) -> None:
    frequency = config.frequency
    if not NOAA_MIN_MHZ <= frequency <= NOAA_MAX_MHZ:
        raise ConfigError(
            f"station.frequency must be between {NOAA_MIN_MHZ} and {NOAA_MAX_MHZ}"
        )


def validate_csdr_server_config(config: CsdrServerConfig) -> None:
    if config.port <= 0 or config.port > 65534:
        raise ConfigError("csdr_server.port must be from 1 through 65534")
    if not _finite(config.timeout) or config.timeout <= 0:
        raise ConfigError(
            "csdr_server.timeout must be a finite number greater than 0"
        )
    if config.iq_format not in {"f32", "s16"}:
        raise ConfigError("csdr_server.iq_format must be either 'f32' or 's16'")
    if not _finite(config.buffer_seconds) or config.buffer_seconds < 0:
        raise ConfigError(
            "csdr_server.buffer_seconds must be a finite number greater than or equal to 0"
        )


def validate_icecast_config(config: IcecastConfig) -> None:
    if config.port <= 0 or config.port > 65535:
        raise ConfigError("icecast.port must be from 1 through 65535")
    if config.format not in {"mp3", "ogg"}:
        raise ConfigError("icecast.format must be either 'mp3' or 'ogg'")
    output_errors = _output_config_errors(config)
    if output_errors:
        raise ConfigError("; ".join(output_errors))


def validate_icecast_configs(configs: tuple[IcecastConfig, ...]) -> None:
    errors: list[str] = []
    for index, config in enumerate(configs):
        try:
            validate_icecast_config(config)
        except ConfigError as exc:
            errors.append(f"icecast destination {index}: {exc}")
    if errors:
        raise ConfigError("; ".join(errors))


def validate_audio_config(config: AudioConfig) -> None:
    validate_deemphasis_config(config.deemphasis)
    validate_volume_config(config.volume)
    validate_filter_config("highpass", config.highpass)
    validate_filter_config("lowpass", config.lowpass)
    validate_filter_config("notch", config.notch)
    validate_audio_filter_relationships(config)


def validate_deemphasis_config(config: DeemphasisConfig) -> None:
    if not _finite(config.tau) or not 0 <= config.tau <= 530:
        raise ConfigError("tau must be between 0 and 530")


def validate_volume_config(config: VolumeConfig) -> None:
    if not _finite(config.multiplier) or config.multiplier < 0:
        raise ConfigError("multiplier must be a finite number greater than or equal to 0")


def validate_filter_config(name: str, config: FilterConfig) -> None:
    if not config.enabled:
        return
    if not _finite(config.frequency) or not 0 < config.frequency < AUDIO_NYQUIST_HZ:
        raise ConfigError(f"{name}.frequency must be between 0 and {AUDIO_NYQUIST_HZ}")
    if not _finite(config.sharpness) or not 0 <= config.sharpness <= 10:
        raise ConfigError(f"{name}.sharpness must be between 0 and 10")
    if name == "highpass" and config.frequency > SAME_ATTENTION_BAND_HZ[0]:
        raise ConfigError(
            f"highpass.frequency must be no higher than {SAME_ATTENTION_BAND_HZ[0]} Hz"
        )
    if name == "lowpass" and config.frequency < SAME_MARK_BAND_HZ[1]:
        raise ConfigError(
            f"lowpass.frequency must be no lower than {SAME_MARK_BAND_HZ[1]} Hz"
        )
    if name == "notch":
        for label, (minimum, maximum) in PROTECTED_AUDIO_BANDS_HZ:
            if minimum <= config.frequency <= maximum:
                raise ConfigError(
                    f"notch.frequency cannot be inside the protected {minimum}-{maximum} Hz "
                    f"band for the {label}"
                )


def validate_audio_filter_relationships(config: AudioConfig) -> None:
    errors = _audio_filter_relationship_errors(config)
    if errors:
        raise ConfigError("; ".join(errors))


def validate_fallback_config(config: FallbackConfig) -> None:
    if (
        not _finite(config.silence_timeout_seconds)
        or not 30 <= config.silence_timeout_seconds <= 120
    ):
        raise ConfigError(
            "fallback.silence_timeout_seconds must be between 30 and 120"
        )
    if not _finite(config.loop_delay_seconds) or config.loop_delay_seconds < 0:
        raise ConfigError(
            "fallback.loop_delay_seconds must be a finite number greater than or equal to 0"
        )
    load_fallback_audio(config.path)


def validate_soundcard_config(config: SoundcardConfig) -> None:
    if isinstance(config.device, bool):
        raise ConfigError("soundcard.device must be a string, integer, or null")


def validate_soundcard_configs(configs: tuple[SoundcardConfig, ...]) -> None:
    errors: list[str] = []
    for index, config in enumerate(configs):
        try:
            validate_soundcard_config(config)
        except ConfigError as exc:
            errors.append(f"soundcard output {index}: {exc}")
    if errors:
        raise ConfigError("; ".join(errors))


def validate_eas_recording_config(config: EasRecordingConfig) -> None:
    if not config.enabled:
        return
    if not _finite(config.pre_seconds) or not 0 <= config.pre_seconds <= 10:
        raise ConfigError("eas_recording.pre_seconds must be between 0 and 10")
    if not _finite(config.post_seconds) or not 0 <= config.post_seconds <= 10:
        raise ConfigError("eas_recording.post_seconds must be between 0 and 10")
    if config.max_seconds <= 0:
        raise ConfigError("eas_recording.max_seconds must be greater than 0")
    if config.format not in {"mp3", "wav"}:
        raise ConfigError("eas_recording.format must be either 'mp3' or 'wav'")
    directory = Path(config.directory).expanduser()
    if directory.exists() and not directory.is_dir():
        raise ConfigError("eas_recording.directory must be a directory")


def validate_buffer_config(
    csdr_server: CsdrServerConfig,
    fallback: FallbackConfig,
) -> None:
    if csdr_server.buffer_seconds >= fallback.silence_timeout_seconds:
        raise ConfigError(
            "csdr_server.buffer_seconds must be less than "
            "fallback.silence_timeout_seconds"
        )


def _audio_filter_relationship_errors(config: AudioConfig) -> list[str]:
    if not config.notch.enabled:
        return []
    errors: list[str] = []
    if (
        config.highpass.enabled
        and config.notch.frequency <= config.highpass.frequency
    ):
        errors.append(
            "notch.frequency must be greater than highpass.frequency "
            "when both filters are enabled"
        )
    if (
        config.lowpass.enabled
        and config.notch.frequency >= config.lowpass.frequency
    ):
        errors.append(
            "notch.frequency must be less than lowpass.frequency "
            "when both filters are enabled"
        )
    return errors


def _merge_valid_audio_filter_relationships(
    audio: AudioConfig,
    current: AudioConfig,
    changed_filters: set[str],
) -> tuple[AudioConfig, list[str]]:
    errors = _audio_filter_relationship_errors(audio)
    if not errors:
        return audio, []

    highpass = audio.highpass
    lowpass = audio.lowpass
    notch = audio.notch
    reload_errors: list[str] = []

    if notch.enabled and highpass.enabled and notch.frequency <= highpass.frequency:
        if "notch" in changed_filters:
            reload_errors.append(
                "notch: notch.frequency must be greater than highpass.frequency "
                "when both filters are enabled"
            )
            notch = current.notch
        elif "highpass" in changed_filters:
            reload_errors.append(
                "highpass: highpass.frequency must be less than notch.frequency "
                "when both filters are enabled"
            )
            highpass = current.highpass
        else:
            reload_errors.append(
                "notch: notch.frequency must be greater than highpass.frequency "
                "when both filters are enabled"
            )
            notch = current.notch

    candidate = AudioConfig(
        deemphasis=audio.deemphasis,
        volume=audio.volume,
        highpass=highpass,
        lowpass=lowpass,
        notch=notch,
    )
    if notch.enabled and lowpass.enabled and notch.frequency >= lowpass.frequency:
        if "notch" in changed_filters:
            reload_errors.append(
                "notch: notch.frequency must be less than lowpass.frequency "
                "when both filters are enabled"
            )
            notch = current.notch
        elif "lowpass" in changed_filters:
            reload_errors.append(
                "lowpass: lowpass.frequency must be greater than notch.frequency "
                "when both filters are enabled"
            )
            lowpass = current.lowpass
        else:
            reload_errors.append(
                "notch: notch.frequency must be less than lowpass.frequency "
                "when both filters are enabled"
            )
            notch = current.notch

    candidate = AudioConfig(
        deemphasis=audio.deemphasis,
        volume=audio.volume,
        highpass=highpass,
        lowpass=lowpass,
        notch=notch,
    )
    remaining_errors = _audio_filter_relationship_errors(candidate)
    if remaining_errors:
        reload_errors.extend(f"notch: {error}" for error in remaining_errors)
        candidate = AudioConfig(
            deemphasis=audio.deemphasis,
            volume=audio.volume,
            highpass=current.highpass,
            lowpass=current.lowpass,
            notch=current.notch,
        )
    return candidate, reload_errors


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
    elif (
        config.format == "ogg"
        and sample_rate in OGG_BITRATE_LIMITS_BY_SAMPLE_RATE
    ):
        minimum, maximum = OGG_BITRATE_LIMITS_BY_SAMPLE_RATE[sample_rate]
        if not minimum <= bitrate <= maximum:
            errors.append(
                "invalid bitrate: icecast.bitrate must be between "
                f"{minimum} and {maximum} Kbps for Ogg Vorbis at {sample_rate} Hz"
            )
    return errors


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be an object")
    return value


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be an object")
    return value


def _str(raw: dict[str, Any], key: str, default: str | None = None) -> str:
    value = raw.get(key, default)
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


def _float(raw: dict[str, Any], key: str) -> float:
    value = raw.get(key)
    if not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number")
    return float(value)


def _bool(raw: dict[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean")
    return value


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _mount(value: str) -> str:
    return value if value.startswith("/") else f"/{value}"
