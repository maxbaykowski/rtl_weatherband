from __future__ import annotations

import logging
import time

from .config import AppConfig
from .icecast import IcecastSource
from .pipeline import StreamPipeline


LOG = logging.getLogger(__name__)
RECONNECT_DELAY_SECONDS = 5


def run(config: AppConfig) -> None:
    while True:
        icecast = IcecastSource(
            config.icecast,
            config.audio.content_type,
            timeout_seconds=config.csdr_server.timeout_seconds,
        )
        pipeline = StreamPipeline(
            config.dsp,
            config.audio,
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
            process_exit = pipeline.wait()
            LOG.warning(
                "%s exited with status %s; reconnecting Icecast in %s seconds",
                process_exit.stage,
                process_exit.returncode,
                RECONNECT_DELAY_SECONDS,
            )
        except Exception as exc:
            LOG.warning(
                "stream setup failed: %s; retrying in %s seconds",
                exc,
                RECONNECT_DELAY_SECONDS,
            )
        finally:
            pipeline.stop()
            icecast.close()

        time.sleep(RECONNECT_DELAY_SECONDS)
