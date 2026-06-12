from __future__ import annotations

import base64
import socket
import ssl
from dataclasses import dataclass
from typing import BinaryIO

from .config import IcecastConfig


class IcecastError(RuntimeError):
    """Raised when Icecast rejects the source connection."""


@dataclass
class IcecastSource:
    config: IcecastConfig
    content_type: str
    timeout_seconds: float = 10.0
    _socket: socket.socket | None = None
    _file: BinaryIO | None = None

    def connect(self) -> BinaryIO:
        raw_sock = socket.create_connection(
            (self.config.host, self.config.port), timeout=self.timeout_seconds
        )
        try:
            sock: socket.socket
            if self.config.tls:
                context = ssl.create_default_context()
                sock = context.wrap_socket(raw_sock, server_hostname=self.config.host)
            else:
                sock = raw_sock
            sock.sendall(self._request_headers())
            response = self._read_response(sock)
            if not (
                response.startswith("HTTP/1.0 200")
                or response.startswith("HTTP/1.1 200")
            ):
                raise IcecastError(f"Icecast source connection rejected: {response}")
            self._socket = sock
            self._file = sock.makefile("wb", buffering=0)
            return self._file
        except Exception:
            raw_sock.close()
            raise

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass

    def _request_headers(self) -> bytes:
        auth = base64.b64encode(
            f"{self.config.username}:{self.config.password}".encode("utf-8")
        ).decode("ascii")
        lines = [
            f"PUT {self.config.mount} HTTP/1.1",
            f"Host: {self.config.host}:{self.config.port}",
            f"Authorization: Basic {auth}",
            f"Content-Type: {self.content_type}",
            "User-Agent: rtl_weatherband/0.1.0",
            "Transfer-Encoding: identity",
            "Connection: close",
            f"Ice-Public: {1 if self.config.public else 0}",
        ]
        if self.config.name:
            lines.append(f"Ice-Name: {self.config.name}")
        if self.config.genre:
            lines.append(f"Ice-Genre: {self.config.genre}")
        if self.config.description:
            lines.append(f"Ice-Description: {self.config.description}")
        return ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8")

    @staticmethod
    def _read_response(sock: socket.socket) -> str:
        data = bytearray()
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(1)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > 8192:
                raise IcecastError("Icecast response headers exceeded 8192 bytes")
        first_line = bytes(data).splitlines()[0:1]
        if not first_line:
            raise IcecastError("Icecast closed the connection without a response")
        return first_line[0].decode("iso-8859-1", errors="replace")

