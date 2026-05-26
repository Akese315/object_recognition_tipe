#!/usr/bin/env python3

import logging
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
CAMERA_IDS = (0, 1)
PORTS = (8000, 8001)
HOST = "0.0.0.0"

MAX_FRAME_SIZE = 2 * 1024 * 1024
JPEG_QUALITY = 90
SOCKET_TIMEOUT = 5.0

logger = logging.getLogger("server")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
def gstreamer_pipeline(sensor_id: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode=3 ! "  # mode 3 = 1280x720
        f"video/x-raw(memory:NVMM), width=1280, height=720, framerate=60/1 ! "
        f"nvvidconv ! "
        f"video/x-raw(memory:NVMM), width=640, height=480 ! "  # scale explicite identique pour les 2
        f"nvvidconv ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


class Camera:
    def __init__(self, sensor_id):
        self.sensor_id = sensor_id
        self.cap = None
        self.open()

    def open(self):
        pipeline = gstreamer_pipeline(self.sensor_id)
        self.cap = cv2.VideoCapture(pipeline)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ← ICI
        if not self.cap.isOpened():
            logger.error("Camera %s open failed", self.sensor_id)

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            self.open()
            return False, None

        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        if self.cap:
            self.cap.release()


# ---------------------------------------------------------------------------
def camera_worker(cam_id: int, port: int, stop_event: threading.Event):

    cam = Camera(cam_id)

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, port))
    server_sock.listen(1)
    server_sock.settimeout(1.0)

    logger.info("Camera %s listening on %s", cam_id, port)

    conn = None

    try:
        while not stop_event.is_set():
            try:
                conn, addr = server_sock.accept()
                logger.info("Client %s connected cam %s", addr, cam_id)
                conn.settimeout(SOCKET_TIMEOUT)
            except socket.timeout:
                continue

            try:
                while not stop_event.is_set():
                    ret, frame = cam.read()

                    if not ret:
                        time.sleep(0.01)
                        continue

                    ok, enc = cv2.imencode(
                        ".jpg",
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                    )

                    if not ok:
                        continue

                    data = enc.tobytes()

                    if len(data) > MAX_FRAME_SIZE:
                        break

                    conn.sendall(struct.pack(">I", len(data)) + data)

            except (socket.error, BrokenPipeError):
                pass

            finally:
                if conn:
                    conn.close()
                    conn = None

    finally:
        cam.release()
        server_sock.close()
        logger.info("Camera %s stopped", cam_id)


# ---------------------------------------------------------------------------
def main():
    stop_event = threading.Event()
    threads = []

    for cam_id, port in zip(CAMERA_IDS, PORTS):
        t = threading.Thread(
            target=camera_worker,
            args=(cam_id, port, stop_event),
            daemon=True,
        )
        t.start()
        threads.append(t)

    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(0.5)

    except KeyboardInterrupt:
        logger.info("Stopping...")
        stop_event.set()
        for t in threads:
            t.join()

    logger.info("Server terminated")


if __name__ == "__main__":
    main()
