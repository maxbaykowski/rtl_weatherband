from __future__ import annotations

import socket
import unittest

from rtl_weatherband.csdr_server import CsdrServerError, IqStream


class FakeControlSocket:
    def __init__(self, response: bytes) -> None:
        self.response = bytearray(response)
        self.sent = bytearray()

    def sendall(self, data: bytes) -> None:
        self.sent.extend(data)

    def recv(self, count: int, flags: int = 0) -> bytes:
        if not self.response:
            return b""
        chunk = bytes(self.response[:count])
        del self.response[:count]
        return chunk

    def close(self) -> None:
        pass


class FakeStreamSocket:
    def close(self) -> None:
        pass


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

    def test_retune_sends_control_command(self) -> None:
        control_socket = FakeControlSocket(
            b'{"status":"ok","command":"retune","frequency":162475000}\n'
        )
        stream = IqStream(FakeStreamSocket(), control_socket, {})

        response = stream.retune(162_475_000)

        self.assertEqual(
            control_socket.sent,
            b'{"command":"retune","frequency":162475000}\n',
        )
        self.assertEqual(response["status"], "ok")

    def test_retune_raises_on_error_response_without_command(self) -> None:
        control_socket = FakeControlSocket(
            b'{"status":"error","code":1,"error":"out of band"}\n'
        )
        stream = IqStream(FakeStreamSocket(), control_socket, {})

        with self.assertRaisesRegex(CsdrServerError, "out of band"):
            stream.retune(162_475_000)


if __name__ == "__main__":
    unittest.main()
