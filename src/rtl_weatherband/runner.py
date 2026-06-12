from __future__ import annotations

import logging
import signal
import threading
import time
from pathlib import Path

from .config import AppConfig, load_raw_config, merge_valid_reload_config
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
        icecast = IcecastSource(
            config.icecast,
            config.icecast.content_type,
            timeout_seconds=config.csdr_server.timeout,
        )
        pipeline = StreamPipeline(
            config.audio,
            config.icecast,
            config.csdr_server,
            config.station.frequency_hz,
        )
        try:
            LOG.info(
                "connecting to Icecast at %s:%s%s",
                config.icecast.host,
                config.icecast.port,
                config.icecast.mount,
            )
            encoded_output = icecast.connect()
            pipeline.start(encoded_output)
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
                        LOG.info("Icecast configuration changed; reconnecting")
                        config = new_config
                        reconnect_delay = False
                        break
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


def _log_pipeline_exit(process_exit: ProcessExit) -> None:
    LOG.warning(
        "%s exited with status %s; reconnecting Icecast in %s seconds",
        process_exit.stage,
        process_exit.returncode,
        RECONNECT_DELAY_SECONDS,
    )
