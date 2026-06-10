from __future__ import annotations

import logging

from .config import AppConfig
from .csdr_server import open_iq_stream
from .icecast import IcecastSource
from .pipeline import StreamPipeline


LOG = logging.getLogger(__name__)


def run(config: AppConfig) -> None:
    LOG.info(
        "connecting to csdr_server at %s:%s for %.6f MHz",
        config.csdr_server.host,
        config.csdr_server.listen_port,
        config.station.frequency_mhz,
    )
    iq_stream = open_iq_stream(config.csdr_server, config.station.frequency_hz)
    icecast = IcecastSource(
        config.icecast,
        config.audio.content_type,
        timeout_seconds=config.csdr_server.timeout_seconds,
    )
    pipeline = StreamPipeline(config.dsp, config.audio)
    try:
        LOG.info(
            "connecting to Icecast at %s:%s%s",
            config.icecast.host,
            config.icecast.port,
            config.icecast.mount,
        )
        encoded_output = icecast.connect()
        pipeline.start(iq_stream.stream_socket, encoded_output)
        LOG.info("streaming started")
        pipeline.wait()
    finally:
        pipeline.stop()
        icecast.close()
        iq_stream.close()

