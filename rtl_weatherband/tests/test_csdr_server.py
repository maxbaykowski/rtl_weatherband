from __future__ import annotations

import socket
import unittest

from rtl_weatherband.csdr_server import IqStream


class IqStreamTests(unittest.TestCase):
    def test_idle_open_sockets_are_connected(self) -> None:
        stream_client, stream_server = socket.socketpair()
        control_client, control_server = socket.socketpair()
        try:
            stream = IqStream(stream_client, control_client, {})
            self.assertTrue(stream.is_connected())
        finally:
            stream_client.close()
            stream_server.close()
            control_client.close()
            control_server.close()

    def test_closed_stream_socket_is_disconnected(self) -> None:
        stream_client, stream_server = socket.socketpair()
        control_client, control_server = socket.socketpair()
        try:
            stream_server.close()
            stream = IqStream(stream_client, control_client, {})
            self.assertFalse(stream.is_connected())
        finally:
            stream_client.close()
            control_client.close()
            control_server.close()


if __name__ == "__main__":
    unittest.main()

