import logging
import socket
import ssl
import struct
import sys
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HOST = "192.168.1.68"  # Server address – change as needed
PORTS = (8000, 8001)  # Must match the server ports
RECONNECT_DELAY = 2.0  # Seconds between reconnection attempts
SOCKET_TIMEOUT = 15.0  # Seconds for socket operations
MAX_FRAME_SIZE = 2 * 1024 * 1024  # 2 MiB – sanity check on incoming frames

# TLS configuration – enable when server uses TLS
TLS_ENABLED = False  # Set to True to use TLS
TLS_CERT: Optional[str] = None  # Path to CA bundle or server cert for verification

logger = logging.getLogger("client")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stderr,
)


class FrameBuffer:
    """Conserve uniquement la frame la plus récente."""

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()

    def put(self, frame: np.ndarray):
        with self._lock:
            self._frame = frame

    def get(self) -> np.ndarray | None:
        with self._lock:
            return self._frame


class FrameBufferPair:
    """Two FrameBuffers for stereo / dual-camera setups."""

    def __init__(self):
        self.frame1 = FrameBuffer()
        self.frame2 = FrameBuffer()


def recvall(sock: socket.socket, n: int) -> bytes:
    """Read *n* bytes from *sock*, handling short reads.

    Raises ``ConnectionError`` if the connection is broken before *n* bytes are read.
    """
    data = bytearray()
    while len(data) < n:
        try:
            packet = sock.recv(n - len(data))
        except socket.timeout as e:
            raise ConnectionError(f"Socket timed out: {e}")
        if not packet:
            raise ConnectionError("Socket connection broken")
        data.extend(packet)
    return bytes(data)


# ---------------------------------------------------------------------------
# Helper: create SSL context if TLS is enabled
# ---------------------------------------------------------------------------
def _create_ssl_context() -> Optional[ssl.SSLContext]:
    if not TLS_ENABLED:
        return None
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    if TLS_CERT:
        ctx.load_verify_locations(TLS_CERT)
    else:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ---------------------------------------------------------------------------
# Helper: establish a connection (with retry) and optional TLS wrapping
# ---------------------------------------------------------------------------
def connect_with_retry(port: int) -> socket.socket:
    """Attempt to connect to ``HOST:port``.

    Retries indefinitely until a connection is successful. If ``TLS_ENABLED``
    is ``True`` the socket is wrapped in an SSL layer before being returned.
    """
    ssl_ctx = _create_ssl_context()
    while True:
        try:
            raw_sock = socket.create_connection((HOST, port), timeout=SOCKET_TIMEOUT)
            raw_sock.settimeout(SOCKET_TIMEOUT)
            if ssl_ctx:
                try:
                    sock = ssl_ctx.wrap_socket(raw_sock, server_hostname=HOST)
                    logger.info("TLS connection established to %s:%s", HOST, port)
                except ssl.SSLError as e:
                    raw_sock.close()
                    logger.error("TLS handshake failed for %s:%s – %s", HOST, port, e)
                    time.sleep(RECONNECT_DELAY)
                    continue
            else:
                sock = raw_sock
            logger.info("Connected to %s:%s", HOST, port)
            return sock
        except (OSError, ConnectionRefusedError) as e:
            logger.warning(
                "Cannot connect to %s:%s – %s. Retrying in %.1fs...",
                HOST,
                port,
                e,
                RECONNECT_DELAY,
            )
            time.sleep(RECONNECT_DELAY)


def stream_reader(port, buffer: FrameBuffer, stop_event: threading.Event):
    """Thread dédié à la lecture d'un stream."""
    while not stop_event.is_set():
        sock = connect_with_retry(port)
        try:
            while not stop_event.is_set():
                raw_len = recvall(sock, 4)
                (msg_len,) = struct.unpack(">I", raw_len)
                if msg_len > MAX_FRAME_SIZE:
                    raise ValueError("Frame trop grande")
                jpeg_data = recvall(sock, msg_len)
                np_arr = np.frombuffer(jpeg_data, dtype=np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is not None:
                    buffer.put(img)  # toujours la plus récente
        except Exception as e:
            logger.error("Stream %s perdu: %s", port, e)
            sock.close()


def main():
    stop_event = threading.Event()
    buffers = [FrameBuffer() for _ in PORTS]

    # Un thread de lecture par caméra
    for port, buf in zip(PORTS, buffers):
        t = threading.Thread(
            target=stream_reader, args=(port, buf, stop_event), daemon=True
        )
        t.start()

    try:
        while True:
            frames = [buf.get() for buf in buffers]

            if any(f is None for f in frames):
                time.sleep(0.01)
                continue

            h_min = min(f.shape[0] for f in frames)
            resized = [
                cv2.resize(f, (int(f.shape[1] * h_min / f.shape[0]), h_min))
                for f in frames
            ]
            combined = np.hstack(resized)
            TARGET_W = 1440
            TARGET_H = int(combined.shape[0] * TARGET_W / combined.shape[1])
            combined = cv2.resize(combined, (TARGET_W, TARGET_H))
            cv2.imshow("Streams", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
