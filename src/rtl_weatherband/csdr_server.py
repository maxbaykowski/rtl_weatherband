from __future__ import annotations

import json
import socket
import uuid
from dataclasses import dataclass

from .config import CsdrServerConfig, IQ_SAMPLE_RATE


class CsdrServerError(RuntimeError):
    """Raised when csdr_server rejects or fails a stream setup."""


@dataclass
class IqStream:
    stream_socket: socket.socket
    control_socket: socket.socket
    handshake: dict[str, object]

    def is_connected(self) -> bool:
        return _socket_is_connected(self.control_socket) and _socket_is_connected(
            self.stream_socket
        )

    def close(self) -> None:
        for sock in (self.stream_socket, self.control_socket):
            try:
                sock.close()
            except OSError:
                pass

    def retune(self, frequency_hz: int) -> dict[str, object]:
        request = {"command": "retune", "frequency": frequency_hz}
        self.control_socket.sendall(
            json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
        )
        while True:
            response = _read_json_line(self.control_socket)
            if response.get("event") is not None:
                continue
            if response.get("status") == "error":
                raise CsdrServerError(str(response.get("error", response)))
            if response.get("command") != "retune":
                continue
            if response.get("status") != "ok":
                raise CsdrServerError(str(response.get("error", response)))
            return response


def _socket_is_connected(sock: socket.socket) -> bool:
    try:
        data = sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
    except BlockingIOError:
        return True
    except OSError:
        return False
    return data != b""


def open_iq_stream(config: CsdrServerConfig, frequency_hz: int) -> IqStream:
    token = uuid.uuid4().hex
    stream_sock = socket.create_connection(
        (config.host, config.port), timeout=config.timeout
    )
    control_sock: socket.socket | None = None
    try:
        stream_sock.sendall(token.encode("utf-8") + b"\n")
        control_sock = socket.create_connection(
            (config.host, config.control_port), timeout=config.timeout
        )
        request = {
            "stream_token": token,
            "frequency": frequency_hz,
            "mode": "iq",
            "sample_rate": IQ_SAMPLE_RATE,
            "format": "f32",
        }
        control_sock.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
        handshake = _read_json_line(control_sock)
        if handshake.get("status") != "ok":
            raise CsdrServerError(str(handshake.get("error", handshake)))
        if handshake.get("mode") != "iq" or handshake.get("format") != "f32":
            raise CsdrServerError(f"unexpected csdr_server handshake: {handshake}")
        stream_sock.settimeout(None)
        control_sock.settimeout(None)
        return IqStream(stream_sock, control_sock, handshake)
    except Exception:
        stream_sock.close()
        if control_sock is not None:
            control_sock.close()
        raise


def _read_json_line(sock: socket.socket) -> dict[str, object]:
    chunks: list[bytes] = []
    while True:
        chunk = sock.recv(1)
        if not chunk:
            raise CsdrServerError("csdr_server closed control socket before handshake")
        if chunk == b"\n":
            break
        chunks.append(chunk)
    try:
        message = json.loads(b"".join(chunks).decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise CsdrServerError(f"invalid csdr_server handshake JSON: {exc}") from exc
    if not isinstance(message, dict):
        raise CsdrServerError("csdr_server handshake must be a JSON object")
    return message
