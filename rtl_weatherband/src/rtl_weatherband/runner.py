from __future__ import annotations

import logging
import time

from .config import AppConfig
from .csdr_server import IqStream, open_iq_stream
from .icecast import IcecastSource
from .pipeline import StreamPipeline


LOG = logging.getLogger(__name__)
RECONNECT_DELAY_SECONDS = 5


def run(config: AppConfig) -> None:
    while True:
        LOG.info(
            "connecting to csdr_server at %s:%s for %.6f MHz",
            config.csdr_server.host,
            config.csdr_server.listen_port,
            config.station.frequency_mhz,
        )
        try:
            iq_stream = open_iq_stream(config.csdr_server, config.station.frequency_hz)
        except Exception as exc:
            LOG.warning(
                "csdr_server connection failed: %s; retrying in %s seconds",
                exc,
                RECONNECT_DELAY_SECONDS,
            )
            time.sleep(RECONNECT_DELAY_SECONDS)
            continue

        try:
            _run_until_server_disconnect(config, iq_stream)
        finally:
            iq_stream.close()

        LOG.warning(
            "csdr_server connection dropped; retrying in %s seconds",
            RECONNECT_DELAY_SECONDS,
        )
        time.sleep(RECONNECT_DELAY_SECONDS)


def _run_until_server_disconnect(config: AppConfig, iq_stream: IqStream) -> None:
    icecast = IcecastSource(
        config.icecast,
        config.audio.content_type,
        timeout_seconds=config.csdr_server.timeout_seconds,
    )
    try:
        LOG.info(
            "connecting to Icecast at %s:%s%s",
            config.icecast.host,
            config.icecast.port,
            config.icecast.mount,
        )
        encoded_output = icecast.connect()
        while iq_stream.is_connected():
            pipeline = StreamPipeline(config.dsp, config.audio)
            try:
                pipeline.start(iq_stream.stream_socket, encoded_output)
                LOG.info("streaming started")
                process_exit = pipeline.wait()
                if not iq_stream.is_connected():
                    return
                LOG.warning(
                    "%s exited with status %s while csdr_server is still connected; "
                    "restarting local DSP pipeline",
                    process_exit.stage,
                    process_exit.returncode,
                )
            finally:
                pipeline.stop()
    finally:
        icecast.close()
