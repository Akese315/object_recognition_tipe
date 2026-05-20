#!/usr/bin/env python3
"""
client.py
----------

Connect to the video streams served by ``server.py`` on ports 8000 and 8001,
receive JPEG frames, decode them with OpenCV and display the two streams
side‑by‑side in a single window.

The client automatically retries connections if the server is not yet ready.
Press **q** (while the OpenCV window has focus) or **Ctrl‑C** to quit.

Usage
-----
    $ python client.py   # defaults to localhost, ports 8000/8001
"""

import logging
import socket
import ssl
import struct
import sys
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
SOCKET_TIMEOUT = 5.0  # Seconds for socket operations
MAX_FRAME_SIZE = 2 * 1024 * 1024  # 2 MiB – sanity check on incoming frames

# TLS configuration – enable when server uses TLS
TLS_ENABLED = False  # Set to True to use TLS
TLS_CERT: Optional[str] = None  # Path to CA bundle or server cert for verification

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("client")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stderr,
)


# ---------------------------------------------------------------------------
# Helper: receive exactly *n* bytes from a socket
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main loop – receive, decode, and display frames
# ---------------------------------------------------------------------------
def main() -> None:
    # Establish initial connections for both streams
    sockets: list[Optional[socket.socket]] = [connect_with_retry(p) for p in PORTS]

    try:
        while True:
            frames = []
            for idx, sock in enumerate(sockets):
                if sock is None:
                    # Socket was lost previously – try to reconnect now
                    sockets[idx] = connect_with_retry(PORTS[idx])
                    sock = sockets[idx]
                try:
                    # 1️⃣ read 4‑byte length
                    raw_len = recvall(sock, 4)
                    (msg_len,) = struct.unpack(">I", raw_len)
                    if msg_len > MAX_FRAME_SIZE:
                        raise ValueError(
                            f"Frame size {msg_len} exceeds limit of {MAX_FRAME_SIZE} bytes"
                        )

                    # 2️⃣ read the JPEG payload
                    jpeg_data = recvall(sock, msg_len)

                    # Decode JPEG to BGR image
                    np_arr = np.frombuffer(jpeg_data, dtype=np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError("Failed to decode JPEG frame")
                    frames.append(img)
                except (ConnectionError, socket.timeout, OSError, ValueError) as e:
                    logger.error("Stream %s lost: %s", idx, e)
                    # Cleanly close the broken socket before reconnecting
                    if sock:
                        try:
                            sock.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                        sock.close()
                    sockets[idx] = None  # Mark for reconnection on next loop iteration
                    break  # abort current display iteration; will retry next loop

            if len(frames) != 2:
                # Not enough frames to display – skip this iteration
                continue

            # Resize to same height if needed (optional)
            h_min = min(f.shape[0] for f in frames)
            resized = [
                cv2.resize(f, (int(f.shape[1] * h_min / f.shape[0]), h_min))
                for f in frames
            ]

            combined = np.hstack(resized)
            cv2.imshow("Jetson Streams (press 'q' to quit)", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("'q' pressed – exiting.")
                break
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt – exiting.")
    finally:
        for s in sockets:
            if s:
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                s.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
