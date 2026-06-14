from __future__ import annotations

import logging
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from typing import BinaryIO

from .config import AppConfig, IcecastConfig, load_raw_config, merge_valid_reload_config
from .icecast import IcecastSource
from .pipeline import ProcessExit, StreamPipeline


LOG = logging.getLogger(__name__)
RECONNECT_DELAY_SECONDS = 5


@dataclass(frozen=True)
class IcecastReloadPlan:
    unchanged: tuple[IcecastConfig, ...]
    replacements: tuple[tuple[IcecastConfig, IcecastConfig], ...]
    pure_removes: tuple[IcecastConfig, ...]
    pure_adds: tuple[IcecastConfig, ...]

    @property
    def has_changes(self) -> bool:
        return bool(self.replacements or self.pure_removes or self.pure_adds)


def run(config: AppConfig, config_path: str | Path | None = None) -> None:
    reload_event = threading.Event()
    pending_icecast_config: AppConfig | None = None
    next_icecast_retry_at = 0.0
    if config_path is not None and hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, lambda _signum, _frame: reload_event.set())
        LOG.info("SIGHUP config reload enabled for %s", config_path)

    while True:
        icecast_sources: list[IcecastSource] = []
        encoded_outputs = []
        pipeline = StreamPipeline(
            config.audio,
            config.icecast,
            config.csdr_server,
            config.station.frequency_hz,
            config.fallback,
        )
        try:
            for destination in config.icecast:
                icecast = IcecastSource(
                    destination,
                    destination.content_type,
                    timeout_seconds=config.csdr_server.timeout,
                )
                LOG.info(
                    "connecting to Icecast at %s:%s%s",
                    destination.host,
                    destination.port,
                    destination.mount,
                )
                encoded_outputs.append(icecast.connect())
                icecast_sources.append(icecast)
            pipeline.start(encoded_outputs)
            LOG.info("streaming started")
            reconnect_delay = True
            while True:
                process_exit = pipeline.poll_exit()
                if process_exit is not None:
                    _log_pipeline_exit(process_exit)
                    break
                if reload_event.is_set() and config_path is not None:
                    reload_event.clear()
                    LOG.info("SIGHUP received; reloading config from %s", config_path)
                    new_config = _load_reload_config(config_path, config)
                    _log_reload_changes(config, new_config)
                    if new_config.icecast != config.icecast:
                        pending_icecast_config = new_config
                        if _retry_pending_icecast_reload(
                            pipeline,
                            icecast_sources,
                            config,
                            pending_icecast_config,
                        ):
                            config = pipeline.apply_runtime_config(
                                pending_icecast_config
                            )
                            pending_icecast_config = None
                        else:
                            config = pipeline.apply_runtime_config(
                                AppConfig(
                                    csdr_server=new_config.csdr_server,
                                    station=new_config.station,
                                    icecast=config.icecast,
                                    audio=new_config.audio,
                                    fallback=new_config.fallback,
                                )
                            )
                            next_icecast_retry_at = (
                                time.monotonic() + RECONNECT_DELAY_SECONDS
                            )
                    else:
                        pending_icecast_config = None
                        config = pipeline.apply_runtime_config(new_config)
                    LOG.info("config reload applied")
                if (
                    pending_icecast_config is not None
                    and time.monotonic() >= next_icecast_retry_at
                ):
                    if _retry_pending_icecast_reload(
                        pipeline,
                        icecast_sources,
                        config,
                        pending_icecast_config,
                    ):
                        config = pipeline.apply_runtime_config(pending_icecast_config)
                        pending_icecast_config = None
                    else:
                        next_icecast_retry_at = (
                            time.monotonic() + RECONNECT_DELAY_SECONDS
                        )
                time.sleep(0.25)
        except Exception as exc:
            LOG.warning(
                "stream setup failed: %s; retrying in %s seconds",
                exc,
                RECONNECT_DELAY_SECONDS,
            )
            reconnect_delay = True
        finally:
            pipeline.stop()
            for icecast in icecast_sources:
                icecast.close()

        if reconnect_delay:
            if reload_event.wait(RECONNECT_DELAY_SECONDS) and config_path is not None:
                reload_event.clear()
                config = _load_reload_config(config_path, config)


def _load_reload_config(config_path: str | Path, current: AppConfig) -> AppConfig:
    LOG.debug("loading reload config from %s", config_path)
    try:
        raw = load_raw_config(config_path)
    except Exception as exc:
        LOG.error("config reload failed; keeping existing configuration: %s", exc)
        return current
    config, errors = merge_valid_reload_config(raw, current)
    for error in errors:
        LOG.error("config reload kept existing %s", error)
    LOG.debug(
        "reload config parsed: csdr=%s:%s frequency=%s icecast=%s audio=%r",
        config.csdr_server.host,
        config.csdr_server.port,
        config.station.frequency_hz,
        [_icecast_label(destination) for destination in config.icecast],
        config.audio,
    )
    return config


def _retry_pending_icecast_reload(
    pipeline: StreamPipeline,
    icecast_sources: list[IcecastSource],
    current: AppConfig,
    pending: AppConfig,
) -> bool:
    LOG.info(
        "attempting pending Icecast reload to %s",
        [_icecast_label(config) for config in pending.icecast],
    )
    if _apply_icecast_reload(
        pipeline,
        icecast_sources,
        current.icecast,
        pending.icecast,
        current.csdr_server.timeout,
    ):
        LOG.info("pending Icecast reload succeeded")
        return True
    LOG.warning(
        "pending Icecast reload failed; keeping current destinations and retrying "
        "in %s seconds",
        RECONNECT_DELAY_SECONDS,
    )
    return False


def _apply_icecast_reload(
    pipeline: StreamPipeline,
    icecast_sources: list[IcecastSource],
    current: tuple[IcecastConfig, ...],
    new: tuple[IcecastConfig, ...],
    timeout: float,
) -> bool:
    plan = _icecast_reload_plan(current, new)
    if not plan.has_changes:
        LOG.debug(
            "Icecast config reload found no destination changes; unchanged=%s",
            [_icecast_label(config) for config in plan.unchanged],
        )
        return True

    LOG.info(
        "Icecast config reload: keeping %s unchanged destination(s), replacing %s, "
        "removing %s, adding %s",
        len(plan.unchanged),
        len(plan.replacements),
        len(plan.pure_removes),
        len(plan.pure_adds),
    )
    LOG.debug(
        "Icecast reload plan: unchanged=%s replacements=%s pure_removes=%s "
        "pure_adds=%s",
        [_icecast_label(config) for config in plan.unchanged],
        [
            {
                "old": _icecast_label(old_config),
                "new": _icecast_label(new_config),
                "mode": (
                    "disconnect-before-connect"
                    if _icecast_identity(old_config) == _icecast_identity(new_config)
                    else "connect-before-disconnect"
                ),
            }
            for old_config, new_config in plan.replacements
        ],
        [_icecast_label(config) for config in plan.pure_removes],
        [_icecast_label(config) for config in plan.pure_adds],
    )
    additions: list[tuple[IcecastConfig, BinaryIO]] = []
    added_sources: list[IcecastSource] = []
    removed_sources: list[IcecastSource] = []
    preconnect_removes: list[IcecastConfig] = []
    preconnect_sources: list[IcecastSource] = []
    preconnect_outputs_removed = False
    try:
        for old_config, new_config in plan.replacements:
            if _icecast_identity(old_config) == _icecast_identity(new_config):
                LOG.info(
                    "Icecast reload will disconnect %s before reconnecting same mount",
                    _icecast_label(old_config),
                )
                preconnect_removes.append(old_config)
                continue
            LOG.info(
                "Icecast reload connecting replacement %s before dropping %s",
                _icecast_label(new_config),
                _icecast_label(old_config),
            )
            source, encoded_output = _connect_icecast(new_config, timeout)
            added_sources.append(source)
            additions.append((new_config, encoded_output))

        for config in plan.pure_adds:
            LOG.info("Icecast reload adding destination %s", _icecast_label(config))
            source, encoded_output = _connect_icecast(config, timeout)
            added_sources.append(source)
            additions.append((config, encoded_output))

        if preconnect_removes:
            preconnect_sources = _take_icecast_sources(
                icecast_sources,
                preconnect_removes,
            )
            pipeline.apply_icecast_outputs(
                _without_configs(current, preconnect_removes),
                preconnect_removes,
                [],
                stop_unused_encoders=False,
            )
            preconnect_outputs_removed = True
            for source in preconnect_sources:
                LOG.info(
                    "Icecast reload closing old same-mount destination %s",
                    _icecast_label(source.config),
                )
                source.close()

        for old_config, new_config in plan.replacements:
            if old_config not in preconnect_removes:
                continue
            LOG.info(
                "Icecast reload reconnecting replacement %s after dropping %s",
                _icecast_label(new_config),
                _icecast_label(old_config),
            )
            source, encoded_output = _connect_icecast(new_config, timeout)
            added_sources.append(source)
            additions.append((new_config, encoded_output))

        postconnect_removes = [
            old_config
            for old_config, new_config in plan.replacements
            if old_config not in preconnect_removes
        ]
        removed_sources = _take_icecast_sources(
            icecast_sources,
            [*plan.pure_removes, *postconnect_removes],
        )
        pipeline.apply_icecast_outputs(
            new,
            [*plan.pure_removes, *postconnect_removes, *preconnect_removes],
            additions,
        )
        for source in removed_sources:
            LOG.info(
                "Icecast reload closing old destination %s after replacement connected",
                _icecast_label(source.config),
            )
            source.close()
        icecast_sources.extend(added_sources)
        LOG.info(
            "Icecast reload applied: active destinations=%s",
            [_icecast_label(config) for config in new],
        )
        return True
    except Exception as exc:
        LOG.warning("Icecast config reload failed; keeping existing outputs: %s", exc)
        for source in added_sources:
            source.close()
        if preconnect_outputs_removed:
            _restore_changed_icecast_outputs(
                pipeline,
                icecast_sources,
                current,
                preconnect_removes,
                timeout,
            )
        else:
            icecast_sources.extend(removed_sources)
            icecast_sources.extend(preconnect_sources)
        return False


def _restore_changed_icecast_outputs(
    pipeline: StreamPipeline,
    icecast_sources: list[IcecastSource],
    current: tuple[IcecastConfig, ...],
    configs: list[IcecastConfig],
    timeout: float,
) -> None:
    additions: list[tuple[IcecastConfig, BinaryIO]] = []
    restored_sources: list[IcecastSource] = []
    try:
        for config in configs:
            source, encoded_output = _connect_icecast(config, timeout)
            restored_sources.append(source)
            additions.append((config, encoded_output))
        pipeline.apply_icecast_outputs(current, [], additions)
        icecast_sources.extend(restored_sources)
    except Exception as exc:
        LOG.error("failed to restore previous Icecast output after reload error: %s", exc)
        for source in restored_sources:
            source.close()


def _without_configs(
    configs: tuple[IcecastConfig, ...],
    remove_configs: list[IcecastConfig],
) -> tuple[IcecastConfig, ...]:
    remaining = list(configs)
    for remove_config in remove_configs:
        try:
            remaining.remove(remove_config)
        except ValueError:
            pass
    return tuple(remaining)


def _connect_icecast(
    config: IcecastConfig,
    timeout: float,
) -> tuple[IcecastSource, BinaryIO]:
    source = IcecastSource(
        config,
        config.content_type,
        timeout_seconds=timeout,
    )
    LOG.info(
        "connecting to Icecast at %s:%s%s",
        config.host,
        config.port,
        config.mount,
    )
    sink = source.connect()
    LOG.debug(
        "Icecast connection ready for %s; HTTP status=%s",
        _icecast_label(config),
        source.response_status_code,
    )
    return source, sink


def _diff_icecast_destinations(
    current: tuple[IcecastConfig, ...],
    new: tuple[IcecastConfig, ...],
) -> tuple[list[IcecastConfig], list[IcecastConfig]]:
    unmatched_current = list(current)
    add_configs: list[IcecastConfig] = []
    for new_config in new:
        try:
            index = unmatched_current.index(new_config)
        except ValueError:
            add_configs.append(new_config)
        else:
            del unmatched_current[index]
    return unmatched_current, add_configs


def _take_icecast_sources(
    icecast_sources: list[IcecastSource],
    configs: list[IcecastConfig],
) -> list[IcecastSource]:
    removed: list[IcecastSource] = []
    for config in configs:
        for source in list(icecast_sources):
            if source.config == config:
                icecast_sources.remove(source)
                removed.append(source)
                break
    return removed


def _icecast_reload_plan(
    current: tuple[IcecastConfig, ...],
    new: tuple[IcecastConfig, ...],
) -> IcecastReloadPlan:
    current_remaining = list(enumerate(current))
    new_remaining = list(enumerate(new))
    unchanged: list[IcecastConfig] = []
    replacement_pairs: list[tuple[IcecastConfig, IcecastConfig]] = []

    for current_index, current_config in list(current_remaining):
        for new_index, new_config in list(new_remaining):
            if current_config == new_config:
                unchanged.append(current_config)
                current_remaining.remove((current_index, current_config))
                new_remaining.remove((new_index, new_config))
                break

    for current_index, current_config in list(current_remaining):
        for new_index, new_config in list(new_remaining):
            if _icecast_identity(current_config) == _icecast_identity(new_config):
                replacement_pairs.append((current_config, new_config))
                current_remaining.remove((current_index, current_config))
                new_remaining.remove((new_index, new_config))
                break

    for current_index, current_config in list(current_remaining):
        if current_index >= len(new):
            continue
        new_config = new[current_index]
        indexed_new = (current_index, new_config)
        if indexed_new not in new_remaining:
            continue
        replacement_pairs.append((current_config, new_config))
        current_remaining.remove((current_index, current_config))
        new_remaining.remove(indexed_new)

    return IcecastReloadPlan(
        unchanged=tuple(unchanged),
        replacements=tuple(replacement_pairs),
        pure_removes=tuple(config for _, config in current_remaining),
        pure_adds=tuple(config for _, config in new_remaining),
    )


def _icecast_identity(config: IcecastConfig) -> tuple[str, int, str]:
    return config.host, config.port, config.mount


def _icecast_label(config: IcecastConfig) -> str:
    return (
        f"{config.format}@{config.sample_rate}Hz/{config.bitrate}kbps "
        f"{config.username}@{config.host}:{config.port}{config.mount}"
    )


def _log_reload_changes(current: AppConfig, new: AppConfig) -> None:
    if new == current:
        LOG.info("config reload parsed with no effective changes")
        return
    if new.csdr_server != current.csdr_server:
        LOG.info(
            "config reload changed csdr_server: %s:%s -> %s:%s",
            current.csdr_server.host,
            current.csdr_server.port,
            new.csdr_server.host,
            new.csdr_server.port,
        )
    if new.station != current.station:
        LOG.info(
            "config reload changed station frequency: %s Hz -> %s Hz",
            current.station.frequency_hz,
            new.station.frequency_hz,
        )
    if new.icecast != current.icecast:
        LOG.info(
            "config reload changed Icecast destinations: %s -> %s",
            [_icecast_label(config) for config in current.icecast],
            [_icecast_label(config) for config in new.icecast],
        )
    if new.audio != current.audio:
        LOG.info("config reload changed audio settings")
        LOG.debug("audio config changed from %r to %r", current.audio, new.audio)


def _log_pipeline_exit(process_exit: ProcessExit) -> None:
    LOG.warning(
        "%s exited with status %s; reconnecting Icecast in %s seconds",
        process_exit.stage,
        process_exit.returncode,
        RECONNECT_DELAY_SECONDS,
    )
