from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .config import IQ_SAMPLE_RATE, SoundcardConfig
from .encoder import PcmResampler


LOG = logging.getLogger(__name__)
SOUNDCARD_PREFILL_SECONDS = 0.2
SOUNDCARD_MAX_BUFFER_SECONDS = 0.5
ALSA_PCM_RE = re.compile(
    r"^(?P<prefix>plug(?:hw)?|hw):(?P<card>[^,]+)(?:,(?P<device>[^,]+))?"
)
PULSE_PREFIX_RE = re.compile(r"^(?:pulse|pulseaudio|pipewire):(?P<device>.+)$")


class SoundcardError(RuntimeError):
    """Raised when local soundcard playback cannot be initialized or used."""


class SoundcardFatalError(SoundcardError):
    """Raised when a soundcard setup error should abort startup."""


class SoundcardDependencyError(SoundcardFatalError):
    """Raised when requested soundcard support is unavailable on this system."""


@dataclass(frozen=True)
class PulseSink:
    index: int
    name: str
    description: str
    properties: dict[str, str]


class SoundcardOutput:
    def __init__(self, config: SoundcardConfig) -> None:
        if not config.enabled:
            raise SoundcardError("soundcard output is disabled")
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise SoundcardError(
                "soundcard output requires the 'sounddevice' Python package"
            ) from exc

        self.config = config
        self.sounddevice = sd
        self.pulse_sink_name: str | None = None
        self.device, self.device_info = self._resolve_output_device(config.device)
        self.device_label = _device_label(config.device, self.device, self.device_info)
        self.sample_rate = self._device_sample_rate(self.device_info)
        self.channels = self._device_channels(self.device_info)
        self.resampler = PcmResampler(IQ_SAMPLE_RATE, self.sample_rate)
        self._buffer = bytearray()
        self._buffer_lock = threading.Lock()
        self._started = False
        self._prefill_bytes = max(1, round(self.sample_rate * SOUNDCARD_PREFILL_SECONDS)) * 2
        self._max_buffer_bytes = max(
            self._prefill_bytes,
            round(self.sample_rate * SOUNDCARD_MAX_BUFFER_SECONDS) * 2,
        )
        with _temporary_pulse_sink(self.pulse_sink_name):
            self.stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                device=self.device,
                callback=self._callback,
            )
        LOG.info(
            "prepared soundcard output on %s at %s Hz with %s channel(s)",
            self.device_label,
            self.sample_rate,
            self.channels,
        )

    def write(self, pcm: bytes) -> None:
        if not pcm:
            return
        pcm = self.resampler.process(pcm)
        if not pcm:
            return
        with self._buffer_lock:
            self._buffer.extend(pcm)
            if len(self._buffer) > self._max_buffer_bytes:
                extra = len(self._buffer) - self._max_buffer_bytes
                del self._buffer[: extra - (extra % 2)]
            should_start = not self._started and len(self._buffer) >= self._prefill_bytes
            if should_start:
                self._started = True
        if should_start:
            self.stream.start()
            LOG.info(
                "started soundcard output on %s with %.0f ms prebuffer",
                self.device_label,
                SOUNDCARD_PREFILL_SECONDS * 1000,
            )

    def close(self) -> None:
        try:
            pcm = self.resampler.flush()
            if pcm:
                with self._buffer_lock:
                    self._buffer.extend(pcm)
        finally:
            try:
                if self._started:
                    self.stream.stop()
            finally:
                self.stream.close()

    def _callback(self, outdata, frames: int, _time, _status) -> None:
        needed = frames * 2
        with self._buffer_lock:
            available = min(needed, len(self._buffer))
            chunk = bytes(self._buffer[:available])
            del self._buffer[:available]
        if available < needed:
            chunk += b"\x00" * (needed - available)
        outdata[:] = _interleave_mono_pcm(chunk, self.channels)

    def _resolve_output_device(
        self,
        configured_device: str | int | None,
    ) -> tuple[str | int | None, dict[str, Any]]:
        pulse_device = _pulse_device_selector(configured_device)
        if pulse_device is not None:
            return self._resolve_pulse_output_device(
                pulse_device,
                SoundcardError(
                    f"No Pulse/PipeWire sink matching {configured_device!r}"
                ),
            )
        try:
            device_info = self.sounddevice.query_devices(configured_device, "output")
        except Exception as exc:
            if isinstance(configured_device, str):
                try:
                    return self._resolve_alsa_output_device(configured_device, exc)
                except SoundcardError as alsa_exc:
                    if _alsa_hw_address(configured_device) is not None:
                        raise alsa_exc from exc
                if _looks_like_pulse_sink_name(configured_device):
                    return self._resolve_pulse_output_device(configured_device, exc)
                raise SoundcardError(
                    f"failed to query soundcard output device: {exc}"
                ) from exc
            if isinstance(configured_device, int):
                return self._resolve_pulse_output_device(configured_device, exc)
            else:
                raise SoundcardError(
                    f"failed to query soundcard output device: {exc}"
                ) from exc
        device = device_info.get("index", configured_device)
        return device, device_info

    def _resolve_alsa_output_device(
        self,
        configured_device: str,
        original_error: Exception,
    ) -> tuple[int, dict[str, Any]]:
        alsa_address = _alsa_hw_address(configured_device)
        if alsa_address is None:
            raise SoundcardError(
                f"failed to query soundcard output device: {original_error}"
            ) from original_error
        try:
            devices = self.sounddevice.query_devices()
            hostapis = self.sounddevice.query_hostapis()
        except Exception as exc:
            raise SoundcardError(
                f"failed to query soundcard output device list: {exc}"
            ) from exc
        matches = [
            device
            for device in devices
            if _is_output_device(device)
            and _hostapi_name(hostapis, device.get("hostapi")) == "ALSA"
            and f"({alsa_address})" in str(device.get("name", ""))
        ]
        if len(matches) == 1:
            device = matches[0]
            index = _device_index(device)
            LOG.info(
                "resolved ALSA device %s to sounddevice output %s",
                configured_device,
                _format_device(device),
            )
            return index, device
        if len(matches) > 1:
            formatted = ", ".join(_format_device(device) for device in matches)
            raise SoundcardError(
                f"ALSA device {configured_device!r} matched multiple output devices: "
                f"{formatted}; use a numeric sounddevice device index"
            ) from original_error
        available = ", ".join(
            _format_device(device)
            for device in devices
            if _is_output_device(device)
        )
        raise SoundcardError(
            f"No output device matching ALSA device {configured_device!r}; "
            f"sounddevice expects a numeric device index or a device name substring. "
            f"Available output devices: {available or 'none'}"
        ) from original_error

    def _resolve_pulse_output_device(
        self,
        configured_device: str | int,
        original_error: Exception,
    ) -> tuple[int, dict[str, Any]]:
        sink = _find_pulse_sink(configured_device)
        if sink is None:
            raise SoundcardError(
                f"failed to query soundcard output device: {original_error}"
            ) from original_error
        device = self._pulse_sounddevice_output(sink)
        index = _device_index(device)
        self.pulse_sink_name = sink.name
        LOG.info(
            "resolved Pulse/PipeWire sink %s to sounddevice output %s with PULSE_SINK=%s",
            configured_device,
            _format_device(device),
            sink.name,
        )
        return index, device

    def _pulse_sounddevice_output(self, sink: PulseSink) -> dict[str, Any]:
        try:
            devices = self.sounddevice.query_devices()
            hostapis = self.sounddevice.query_hostapis()
        except Exception as exc:
            raise SoundcardError(
                f"failed to query soundcard output device list: {exc}"
            ) from exc
        outputs = [
            device
            for device in devices
            if _is_output_device(device)
        ]
        sink_matches = [
            device for device in outputs if _sounddevice_matches_pulse_sink(device, sink)
        ]
        if sink_matches:
            return sink_matches[0]
        pulse_matches = [
            device
            for device in outputs
            if _hostapi_name(hostapis, device.get("hostapi")) == "PulseAudio"
        ]
        if pulse_matches:
            return pulse_matches[0]
        generic_matches = [
            device
            for device in outputs
            if str(device.get("name", "")).lower() in {"pipewire", "pulse", "default"}
        ]
        if generic_matches:
            return generic_matches[0]
        available = ", ".join(_format_device(device) for device in outputs)
        raise SoundcardError(
            "Pulse/PipeWire sink was found with pactl, but sounddevice did not "
            "report a matching output device. Available output devices: "
            f"{available or 'none'}"
        )

    def _device_sample_rate(self, device_info: dict[str, Any]) -> int:
        try:
            sample_rate = int(round(float(device_info["default_samplerate"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise SoundcardError(
                f"soundcard output device did not report a valid default sample rate: {device_info}"
            ) from exc
        if sample_rate <= 0:
            raise SoundcardError(
                f"soundcard output device reported invalid default sample rate: {sample_rate}"
            )
        return sample_rate

    def _device_channels(self, device_info: dict[str, Any]) -> int:
        try:
            channels = int(device_info.get("max_output_channels", 1))
        except (TypeError, ValueError) as exc:
            raise SoundcardError(
                f"soundcard output device did not report valid output channels: {device_info}"
            ) from exc
        if channels <= 0:
            raise SoundcardError(
                f"soundcard output device reported invalid output channels: {channels}"
            )
        return min(channels, 2)


def _alsa_hw_address(device: str) -> str | None:
    match = ALSA_PCM_RE.match(device)
    if match is None:
        return None
    card = match.group("card")
    pcm_device = match.group("device")
    if pcm_device is None:
        return f"hw:{card},0"
    return f"hw:{card},{pcm_device}"


def _pulse_device_selector(device: str | int | None) -> str | int | None:
    if not isinstance(device, str):
        return None
    match = PULSE_PREFIX_RE.match(device.strip())
    if match is None:
        return None
    selected = match.group("device").strip()
    if not selected:
        raise SoundcardError("Pulse/PipeWire device selector must not be empty")
    try:
        return int(selected)
    except ValueError:
        return selected


def _looks_like_pulse_sink_name(device: str) -> bool:
    normalized = device.strip()
    return (
        normalized.startswith(("alsa_output.", "bluez_output.", "raop_output."))
        or normalized.endswith(".sink")
        or "__sink" in normalized
    )


def _find_pulse_sink(device: str | int) -> PulseSink | None:
    sinks = _load_pactl_sinks(["pactl", "list", "sinks"], _parse_pactl_sinks)
    if not sinks:
        sinks = _load_pactl_sinks(
            ["pactl", "list", "sinks", "short"],
            _parse_pactl_short_sinks,
        )
    for sink in sinks:
        if _pulse_sink_matches(sink, device):
            return sink
    return None


def _load_pactl_sinks(
    command: list[str],
    parser,
) -> list[PulseSink]:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except FileNotFoundError as exc:
        raise SoundcardDependencyError(
            "Pulse/PipeWire sink selection requires the 'pactl' command, but "
            "pactl was not found. Install PulseAudio/PipeWire pactl support or "
            "select a sounddevice/ALSA output directly."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SoundcardDependencyError(
            "Pulse/PipeWire sink selection requires pactl, but pactl timed out "
            "while listing sinks."
        ) from exc
    except subprocess.SubprocessError as exc:
        raise SoundcardDependencyError(
            f"Pulse/PipeWire sink selection requires pactl, but pactl failed: {exc}"
        ) from exc
    return parser(result.stdout)


def _parse_pactl_sinks(output: str) -> list[PulseSink]:
    sinks: list[PulseSink] = []
    current_index: int | None = None
    current_name = ""
    current_description = ""
    current_properties: dict[str, str] = {}
    in_properties = False
    for line in output.splitlines():
        sink_match = re.match(r"^Sink #(?P<index>\d+)$", line)
        if sink_match is not None:
            if current_index is not None:
                sinks.append(
                    PulseSink(
                        current_index,
                        current_name,
                        current_description,
                        current_properties,
                    )
                )
            current_index = int(sink_match.group("index"))
            current_name = ""
            current_description = ""
            current_properties = {}
            in_properties = False
            continue
        if current_index is None:
            continue
        stripped = line.strip()
        if stripped == "Properties:":
            in_properties = True
            continue
        if line and not line.startswith("\t") and not line.startswith(" "):
            in_properties = False
        if stripped.startswith("Name:"):
            current_name = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Description:"):
            current_description = stripped.split(":", 1)[1].strip()
        elif in_properties and "=" in stripped:
            key, value = stripped.split("=", 1)
            current_properties[key.strip()] = value.strip().strip('"')
    if current_index is not None:
        sinks.append(
            PulseSink(
                current_index,
                current_name,
                current_description,
                current_properties,
            )
        )
    return sinks


def _parse_pactl_short_sinks(output: str) -> list[PulseSink]:
    sinks: list[PulseSink] = []
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            fields = line.split()
        if len(fields) < 2:
            continue
        try:
            index = int(fields[0])
        except ValueError:
            continue
        sinks.append(PulseSink(index=index, name=fields[1], description="", properties={}))
    return sinks


def _pulse_sink_matches(sink: PulseSink, device: str | int) -> bool:
    if isinstance(device, int):
        return sink.index == device
    return (
        device in {str(sink.index), sink.name}
        or device in sink.description
        or any(device == value for value in sink.properties.values())
    )


def _sounddevice_matches_pulse_sink(
    device: dict[str, Any],
    sink: PulseSink,
) -> bool:
    sounddevice_name = _normalize_device_name(str(device.get("name", "")))
    sink_names = {
        sink.name,
        sink.description,
        sink.properties.get("node.name", ""),
        sink.properties.get("node.nick", ""),
        sink.properties.get("device.description", ""),
    }
    normalized_sink_names = {
        _normalize_device_name(name)
        for name in sink_names
        if name
    }
    if sounddevice_name in normalized_sink_names:
        return True
    return any(
        sounddevice_name
        and sink_name
        and (sounddevice_name in sink_name or sink_name in sounddevice_name)
        for sink_name in normalized_sink_names
    )


def _normalize_device_name(name: str) -> str:
    return " ".join(name.casefold().split())


def _interleave_mono_pcm(pcm: bytes, channels: int) -> bytes:
    if channels == 1:
        return pcm
    frame = bytearray(len(pcm) * channels)
    for sample_index in range(0, len(pcm), 2):
        sample = pcm[sample_index : sample_index + 2]
        output_index = sample_index * channels
        for channel in range(channels):
            frame[output_index + channel * 2 : output_index + channel * 2 + 2] = sample
    return bytes(frame)


@contextmanager
def _temporary_pulse_sink(sink_name: str | None) -> Iterator[None]:
    if sink_name is None:
        yield
        return
    previous = os.environ.get("PULSE_SINK")
    os.environ["PULSE_SINK"] = sink_name
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PULSE_SINK", None)
        else:
            os.environ["PULSE_SINK"] = previous


def _is_output_device(device: dict[str, Any]) -> bool:
    try:
        return int(device.get("max_output_channels", 0)) > 0
    except (TypeError, ValueError):
        return False


def _hostapi_name(hostapis: Any, hostapi_index: Any) -> str:
    try:
        return str(hostapis[int(hostapi_index)]["name"])
    except (IndexError, KeyError, TypeError, ValueError):
        return ""


def _device_index(device: dict[str, Any]) -> int:
    try:
        return int(device["index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SoundcardError(
            f"soundcard output device did not report a valid index: {device}"
        ) from exc


def _format_device(device: dict[str, Any]) -> str:
    try:
        index = device["index"]
    except KeyError:
        index = "?"
    return f"{index}: {device.get('name', 'unknown')}"


def _device_label(
    configured_device: str | int | None,
    resolved_device: str | int | None,
    device_info: dict[str, Any],
) -> str:
    if configured_device is None:
        return "default device"
    name = device_info.get("name")
    if name is None:
        return str(configured_device)
    if configured_device == resolved_device:
        return f"{resolved_device}: {name}"
    return f"{configured_device} ({resolved_device}: {name})"
