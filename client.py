#!/usr/bin/env python3
"""
client.py
---------

Connect to the video streams served by ``server.py`` on ports 8000 and 8001,
receive JPEG frames, decode them with OpenCV and display the two streams
side‑by‑side in a single window.

The client automatically retries connections if the server is not yet ready.
Press **q** (while the OpenCV window has focus) or **Ctrl‑C** to quit.

Usage
-----
    $ python client.py   # defaults to localhost, ports 8000/8001
"""

import socket
import struct
import time
import cv2
import numpy as np
import sys

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
HOST = "127.0.0.1"      # Change to Jetson's IP address when remote
PORTS = (8000, 8001)    # Must match the server ports
RECONNECT_DELAY = 2.0  # Seconds between reconnection attempts
SOCKET_TIMEOUT = 5.0   # Seconds for socket operations

# --------------------------------------------------------------------------- #
# Helper: receive exactly *n* bytes from a socket
# --------------------------------------------------------------------------- #
def recvall(sock: socket.socket, n: int) -> bytes:
    """Read *n* bytes from *sock*, handling short reads."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Socket connection broken")
        data.extend(packet)
    return bytes(data)

# --------------------------------------------------------------------------- #
# Helper: establish a connection (with retry)
# --------------------------------------------------------------------------- #
def connect_with_retry(port: int) -> socket.socket:
    """Attempt to connect to HOST:port, retrying until successful."""
    while True:
        try:
            sock = socket.create_connection((HOST, port), timeout=SOCKET_TIMEOUT)
            sys.stdout.write(f"[INFO] Connected to {HOST}:{port}\n")
            return sock
        except (OSError, ConnectionRefusedError):
            sys.stdout.write(f"[WARN] Cannot connect to {HOST}:{port} – retrying in {RECONNECT_DELAY}s...\n")
            time.sleep(RECONNECT_DELAY)

# --------------------------------------------------------------------------- #
# Main loop – receive, decode, and display frames
# --------------------------------------------------------------------------- #
def main() -> None:
    # Establish connections for both streams
    sockets = [connect_with_retry(p) for p in PORTS]

    # Ensure sockets are closed on exit
    try:
        while True:
            frames = []
            for idx, sock in enumerate(sockets):
                try:
                    # 1️⃣ read 4‑byte length
                    raw_len = recvall(sock, 4)
                    (msg_len,) = struct.unpack('>I', raw_len)

                    # 2️⃣ read the JPEG payload
                    jpeg_data = recvall(sock, msg_len)

                    # Decode JPEG to BGR image
                    np_arr = np.frombuffer(jpeg_data, dtype=np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError("Failed to decode JPEG frame")
                    frames.append(img)
                except (ConnectionError, OSError, ValueError) as e:
                    sys.stderr.write(f"[ERROR] Stream {idx} lost: {e}\n")
                    # Try to reconnect this stream
                    sockets[idx] = connect_with_retry(PORTS[idx])
                    break  # abort current display loop; will retry next iteration

            if len(frames) != 2:
                # Not enough frames to display – skip this iteration
                continue

            # Resize to same height if needed (optional)
            h_min = min(f.shape[0] for f in frames)
            resized = [cv2.resize(f, (int(f.shape[1] * h_min / f.shape[0]), h_min))
                       for f in frames]

            # Concatenate side‑by‑side and show
            combined = np.hstack(resized)
            cv2.imshow("Jetson Streams (press 'q' to quit)", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.stdout.write("[INFO] 'q' pressed – exiting.\n")
                break
    except KeyboardInterrupt:
        sys.stdout.write("\n[INFO] KeyboardInterrupt – exiting.\n")
    finally:
        for s in sockets:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
