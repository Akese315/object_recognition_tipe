#!/usr/bin/env python3
"""
server.py
---------

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

import socket
import struct
import threading
import cv2
import sys

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CAMERA_IDS = (0, 1)          # IDs passed to cv2.VideoCapture
PORTS      = (8000, 8001)    # Listening ports – one per camera
HOST       = ""              # Bind to all interfaces
JPEG_QUALITY = 90            # JPEG compression quality (0‑100)

# --------------------------------------------------------------------------- #
# Worker that handles a single camera / socket pair
# --------------------------------------------------------------------------- #
def camera_worker(cam_id: int, port: int, stop_event: threading.Event) -> None:
    """Capture frames from ``cam_id`` and stream them on TCP ``port``."""
    # Initialise camera
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        sys.stderr.write(f"[ERROR] Cannot open camera {cam_id}\n")
        return

    # Initialise listening socket
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, port))
    server_sock.listen(1)
    sys.stdout.write(f"[INFO] Camera {cam_id} listening on port {port}\n")

    conn = None
    try:
        while not stop_event.is_set():
            # --------------------------------------------------------------- #
            # Accept a client (blocking). If a stop is requested, break out.
            # --------------------------------------------------------------- #
            try:
                server_sock.settimeout(1.0)          # allow periodic stop check
                conn, addr = server_sock.accept()
                sys.stdout.write(f"[INFO] Client {addr} connected to camera {cam_id}\n")
            except socket.timeout:
                continue  # loop again to check stop_event
            except OSError:
                break  # socket closed elsewhere

            # --------------------------------------------------------------- #
            # Send frames until the client disconnects or a stop is requested.
            # --------------------------------------------------------------- #
            try:
                while not stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    # Encode as JPEG
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                    ok, encimg = cv2.imencode('.jpg', frame, encode_param)
                    if not ok:
                        continue

                    data = encimg.tobytes()
                    length = struct.pack('>I', len(data))

                    # Send length + payload; use sendall to guarantee delivery
                    conn.sendall(length + data)
            except (socket.error, ConnectionResetError, BrokenPipeError):
                sys.stdout.write(f"[WARN] Client {addr} disconnected from camera {cam_id}\n")
            finally:
                if conn:
                    conn.close()
                    conn = None
    finally:
        # ------------------------------------------------------------------- #
        # Cleanup resources
        # ------------------------------------------------------------------- #
        cap.release()
        server_sock.close()
        sys.stdout.write(f"[INFO] Camera {cam_id} on port {port} shut down.\n")


# --------------------------------------------------------------------------- #
# Main entry point – start two threads, one per camera/port pair
# --------------------------------------------------------------------------- #
def main() -> None:
    stop_event = threading.Event()
    threads = []

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
        # Wait for KeyboardInterrupt
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        sys.stdout.write("\n[INFO] KeyboardInterrupt received – shutting down...\n")
        stop_event.set()
        # Give workers a moment to exit cleanly
        for t in threads:
            t.join()

    sys.stdout.write("[INFO] Server terminated.\n")


if __name__ == "__main__":
    main()
