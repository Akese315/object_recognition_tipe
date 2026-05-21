#!/usr/bin/env python3
"""
server.py
----------

Capture two video streams (camera IDs 0 and 1) with OpenCV and send each
stream over a separate TCP socket.

Protocol (per connection):
    1️⃣  Send a 4‑byte unsigned integer (big‑endian) indicating the JPEG
       payload size.
    2️⃣  Send the JPEG‑encoded frame bytes.

The server listens on ports 8000 (camera 0) and 8001 (camera 1).  A client
may connect to each port independently.  When a client disconnects the
server waits for the next client.

Press **Ctrl‑C** to stop the server – sockets are closed and cameras are
released cleanly.

Usage
-----
    $ python server.py
"""

import logging
import socket
import ssl
import struct
import sys
import threading
import time
from typing import Optional

import cv2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAMERA_IDS = (0, 1)  # IDs passed to cv2.VideoCapture
PORTS = (8000, 8001)  # Listening ports – one per camera
HOST = "0.0.0.0"  # Bind to all interfaces by default (configurable)
MAX_FRAME_SIZE = 2 * 1024 * 1024  # 2 MiB – safety limit for incoming frames
JPEG_QUALITY = 90  # JPEG compression quality (0‑100)
SOCKET_TIMEOUT = 5.0  # Seconds for socket operations

# TLS configuration – set TLS_ENABLED to True and provide certificate files
TLS_ENABLED = False  # Enable TLS for the data channel?
TLS_CERT: Optional[str] = None  # Path to server certificate (PEM)
TLS_KEY: Optional[str] = None  # Path to private key (PEM)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("server")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stderr,
)


# ---------------------------------------------------------------------------
# Worker that handles a single camera / socket pair
# ---------------------------------------------------------------------------
def _create_ssl_context() -> Optional[ssl.SSLContext]:
    """Create an SSL context if TLS is enabled.

    Returns ``None`` when TLS is disabled so calling code can skip wrapping.
    """
    if not TLS_ENABLED:
        return None
    if not TLS_CERT or not TLS_KEY:
        logger.error("TLS is enabled but certificate/key paths are not set.")
        return None
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(certfile=TLS_CERT, keyfile=TLS_KEY)
    return ctx


def gstreamer_pipeline(
    sensor_id: int,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):

    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, "
        f"height=(int){capture_height}, "
        f"format=(string)NV12, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, "
        f"width=(int){display_width}, "
        f"height=(int){display_height}, "
        f"format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )


def camera_worker(cam_id: int, port: int, stop_event: threading.Event) -> None:
    """Capture frames from ``cam_id`` and stream them on TCP ``port``.

    The function runs in its own thread and continues serving new clients
    until ``stop_event`` is set.
    """
    pipeline = gstreamer_pipeline(cam_id)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    logger.info("Using pipeline: %s", pipeline)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        logger.error("Cannot open camera %s", cam_id)
        return

    # -------------------------------------------------------------------
    # Prepare listening socket (optionally wrapped with TLS)
    # -------------------------------------------------------------------
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_sock.bind((HOST, port))
    except OSError as exc:
        logger.error("Failed to bind to %s:%s – %s", HOST, port, exc)
        cap.release()
        return
    server_sock.listen(1)
    server_sock.settimeout(1.0)  # allow periodic stop checks
    logger.info("Camera %s listening on %s:%s", cam_id, HOST, port)

    ssl_ctx = _create_ssl_context()

    conn: Optional[socket.socket] = None
    try:
        while not stop_event.is_set():
            # -----------------------------------------------------------
            # Accept a new client connection (with timeout to check stop_event)
            # -----------------------------------------------------------
            try:
                conn, addr = server_sock.accept()
                logger.info("Client %s connected to camera %s", addr, cam_id)
                # Apply TLS after the TCP handshake if requested
                if ssl_ctx:
                    try:
                        conn = ssl_ctx.wrap_socket(conn, server_side=True)
                        logger.info("TLS handshake successful for %s", addr)
                    except ssl.SSLError as e:
                        logger.error("TLS handshake failed for %s – %s", addr, e)
                        conn.close()
                        conn = None
                        continue
                conn.settimeout(SOCKET_TIMEOUT)
            except socket.timeout:
                continue  # loop again to check stop_event
            except OSError as exc:
                logger.error("Accept failed on port %s – %s", port, exc)
                break

            # -----------------------------------------------------------
            # Stream frames until the client disconnects or we are stopped
            # -----------------------------------------------------------
            try:
                while not stop_event.is_set():
                    # Initialize failure counter if not already defined
                    # (will be reset on successful read)
                    ret, frame = cap.read()
                    if not ret:
                        # Increment consecutive failure count
                        if "read_fail_count" not in locals():
                            read_fail_count = 0
                        read_fail_count += 1
                        logger.debug(
                            "Failed to read frame from camera %s (consecutive failures: %d)",
                            cam_id,
                            read_fail_count,
                        )
                        # Sleep briefly to reduce CPU usage when frames are unavailable
                        time.sleep(0.01)
                        # If failures exceed threshold, attempt to reinitialize the camera
                        if read_fail_count >= 30:
                            logger.warning(
                                "%d consecutive read failures – reinitializing camera %s",
                                read_fail_count,
                                cam_id,
                            )
                            cap.release()

                            time.sleep(1.0)

                            cap = cv2.VideoCapture(
                                gstreamer_pipeline(cam_id), cv2.CAP_GSTREAMER
                            )
                            if not cap.isOpened():
                                logger.error(
                                    "Failed to re-open camera %s after repeated failures",
                                    cam_id,
                                )
                            read_fail_count = 0
                        continue
                    # Reset failure counter on successful read
                    if "read_fail_count" in locals():
                        read_fail_count = 0

                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                    ok, encimg = cv2.imencode(".jpg", frame, encode_param)
                    if not ok:
                        logger.debug("JPEG encoding failed for camera %s", cam_id)
                        continue

                    data = encimg.tobytes()
                    if len(data) > MAX_FRAME_SIZE:
                        logger.error(
                            "Frame size %s exceeds maximum of %s bytes – closing connection",
                            len(data),
                            MAX_FRAME_SIZE,
                        )
                        break

                    length = struct.pack(">I", len(data))
                    try:
                        conn.sendall(length + data)
                    except (socket.error, BrokenPipeError) as e:
                        logger.warning("Send error to %s – %s", addr, e)
                        break
            except (socket.timeout, socket.error) as e:
                logger.warning("Connection to %s lost – %s", addr, e)
            finally:
                if conn:
                    try:
                        conn.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    conn.close()
                    conn = None
                    logger.info("Client %s disconnected from camera %s", addr, cam_id)
    finally:
        cap.release()
        server_sock.close()
        logger.info("Camera %s on port %s shut down.", cam_id, port)


# ---------------------------------------------------------------------------
# Main entry point – start two threads, one per camera/port pair
# ---------------------------------------------------------------------------
def main() -> None:
    stop_event = threading.Event()
    threads: list[threading.Thread] = []

    for cam_id, port in zip(CAMERA_IDS, PORTS):
        t = threading.Thread(
            target=camera_worker,
            args=(cam_id, port, stop_event),
            daemon=True,
            name=f"Cam{cam_id}-Port{port}",
        )
        t.start()
        threads.append(t)

    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received – shutting down...")
        stop_event.set()
        for t in threads:
            t.join()

    logger.info("Server terminated.")


if __name__ == "__main__":
    main()
