from __future__ import annotations

import logging
import signal
import threading
import time
from pathlib import Path

from typing import BinaryIO

from .config import AppConfig, IcecastConfig, load_raw_config, merge_valid_reload_config
from .icecast import IcecastSource
from .pipeline import ProcessExit, StreamPipeline


LOG = logging.getLogger(__name__)
RECONNECT_DELAY_SECONDS = 5


def run(config: AppConfig, config_path: str | Path | None = None) -> None:
    reload_event = threading.Event()
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
                    new_config = _load_reload_config(config_path, config)
                    if new_config.icecast != config.icecast:
                        if _apply_icecast_reload(
                            pipeline,
                            icecast_sources,
                            config.icecast,
                            new_config.icecast,
                            config.csdr_server.timeout,
                        ):
                            config = pipeline.apply_runtime_config(new_config)
                        else:
                            config = pipeline.apply_runtime_config(
                                AppConfig(
                                    csdr_server=new_config.csdr_server,
                                    station=new_config.station,
                                    icecast=config.icecast,
                                    audio=new_config.audio,
                                )
                            )
                    else:
                        config = pipeline.apply_runtime_config(new_config)
                    LOG.info("config reload applied")
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
    try:
        raw = load_raw_config(config_path)
    except Exception as exc:
        LOG.error("config reload failed; keeping existing configuration: %s", exc)
        return current
    config, errors = merge_valid_reload_config(raw, current)
    for error in errors:
        LOG.error("config reload kept existing %s", error)
    return config


def _apply_icecast_reload(
    pipeline: StreamPipeline,
    icecast_sources: list[IcecastSource],
    current: tuple[IcecastConfig, ...],
    new: tuple[IcecastConfig, ...],
    timeout: float,
) -> bool:
    remove_configs, add_configs = _diff_icecast_destinations(current, new)
    if not remove_configs and not add_configs:
        return True

    LOG.info(
        "Icecast config reload: keeping %s destination(s), removing %s, adding %s",
        len(current) - len(remove_configs),
        len(remove_configs),
        len(add_configs),
    )

    additions: list[tuple[IcecastConfig, BinaryIO]] = []
    added_sources: list[IcecastSource] = []
    changed_sources: list[IcecastSource] = []
    changed_outputs_removed = False
    changed_removes, changed_adds, pure_removes, pure_adds = (
        _split_changed_icecast_destinations(remove_configs, add_configs)
    )
    try:
        for config in pure_adds:
            source, encoded_output = _connect_icecast(config, timeout)
            added_sources.append(source)
            additions.append((config, encoded_output))

        if changed_removes:
            changed_sources = _take_icecast_sources(icecast_sources, changed_removes)
            pipeline.apply_icecast_outputs(
                _remove_exact_configs(current, changed_removes),
                changed_removes,
                [],
            )
            changed_outputs_removed = True
            for source in changed_sources:
                source.close()

        for config in changed_adds:
            source, encoded_output = _connect_icecast(config, timeout)
            added_sources.append(source)
            additions.append((config, encoded_output))

        removed_sources = _take_icecast_sources(icecast_sources, pure_removes)
        pipeline.apply_icecast_outputs(new, remove_configs, additions)
        for source in removed_sources:
            source.close()
        icecast_sources.extend(added_sources)
        return True
    except Exception as exc:
        LOG.warning("Icecast config reload failed; keeping existing outputs: %s", exc)
        for source in added_sources:
            source.close()
        if changed_outputs_removed:
            _restore_removed_icecast_outputs(
                pipeline,
                icecast_sources,
                current,
                changed_removes,
                timeout,
            )
        else:
            icecast_sources.extend(changed_sources)
        return False


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
    return source, source.connect()


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


def _restore_removed_icecast_outputs(
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


def _split_changed_icecast_destinations(
    remove_configs: list[IcecastConfig],
    add_configs: list[IcecastConfig],
) -> tuple[
    list[IcecastConfig],
    list[IcecastConfig],
    list[IcecastConfig],
    list[IcecastConfig],
]:
    changed_removes: list[IcecastConfig] = []
    changed_adds: list[IcecastConfig] = []
    pure_removes = list(remove_configs)
    pure_adds = list(add_configs)
    for remove_config in list(remove_configs):
        for add_config in list(pure_adds):
            if _icecast_identity(remove_config) == _icecast_identity(add_config):
                changed_removes.append(remove_config)
                changed_adds.append(add_config)
                pure_removes.remove(remove_config)
                pure_adds.remove(add_config)
                break
    return changed_removes, changed_adds, pure_removes, pure_adds


def _remove_exact_configs(
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


def _icecast_identity(config: IcecastConfig) -> tuple[str, int, str, str]:
    return config.host, config.port, config.mount, config.username


def _log_pipeline_exit(process_exit: ProcessExit) -> None:
    LOG.warning(
        "%s exited with status %s; reconnecting Icecast in %s seconds",
        process_exit.stage,
        process_exit.returncode,
        RECONNECT_DELAY_SECONDS,
    )
