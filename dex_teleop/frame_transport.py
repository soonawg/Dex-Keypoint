"""Length-prefixed JPEG transport for frames captured on Windows."""

from __future__ import annotations

import socket
import struct

import cv2
import numpy as np


MAX_FRAME_BYTES = 8 * 1024 * 1024


def _read_exact(connection: socket.socket, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            if not chunks:
                return None
            raise ConnectionError("Camera sender disconnected mid-frame")
        chunks.extend(chunk)
    return bytes(chunks)


class CameraFrameServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        if not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, port))
        self._server.listen(1)
        self._client: socket.socket | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.getsockname()[:2]
        return str(host), int(port)

    def _accept_client(self) -> socket.socket:
        while self._client is None:
            connection, address = self._server.accept()
            self._client = connection
            print(f"Windows camera sender connected from {address[0]}:{address[1]}")
        return self._client

    def read(self) -> np.ndarray:
        while True:
            connection = self._accept_client()
            try:
                header = _read_exact(connection, 4)
                if header is None:
                    raise ConnectionError("Camera sender disconnected")
                frame_size = struct.unpack("!I", header)[0]
                if not 1 <= frame_size <= MAX_FRAME_BYTES:
                    raise ValueError(f"Invalid JPEG frame size: {frame_size}")
                payload = _read_exact(connection, frame_size)
                if payload is None:
                    raise ConnectionError("Camera sender disconnected mid-frame")
                frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Camera sender sent an invalid JPEG frame")
                return frame
            except ConnectionError as error:
                print(f"{error}; waiting for sender to reconnect")
                connection.close()
                self._client = None

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._server.close()
