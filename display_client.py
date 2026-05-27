#!/usr/bin/env python3
"""
client_display.py — À lancer sur le PC.

Rôle :
  - Se connecte au Jetson sur un seul port (8000)
  - Reçoit à chaque paquet :
      * les bounding boxes JSON des deux caméras
      * l'image JPEG combinée déjà annotée
  - Affiche l'image et les informations de détection
  - Protocole : [4B len_json][json][4B len_jpeg][jpeg]
"""

import json
import logging
import socket
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

HOST = "192.168.1.68"  # Adresse IP du Jetson — à adapter
PORT = 8000
RECONNECT_DELAY = 2.0
SOCKET_TIMEOUT = 10.0
MAX_PACKET_SIZE = 8 * 1024 * 1024  # 8 MiB

logger = logging.getLogger("client_display")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stderr,
)


# ---------------------------------------------------------------------------
# Réception réseau
# ---------------------------------------------------------------------------


def recvall(sock: socket.socket, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        try:
            packet = sock.recv(n - len(data))
        except socket.timeout as e:
            raise ConnectionError(f"Timeout: {e}")
        if not packet:
            raise ConnectionError("Connexion fermée par le serveur")
        data.extend(packet)
    return bytes(data)


def recv_packet(sock: socket.socket) -> Tuple[dict, np.ndarray]:
    """Reçoit un paquet complet et retourne (boxes_dict, frame_bgr)."""
    # JSON
    raw_len = recvall(sock, 4)
    json_len = struct.unpack(">I", raw_len)[0]
    if json_len > MAX_PACKET_SIZE:
        raise ValueError(f"JSON trop grand: {json_len}")
    json_bytes = recvall(sock, json_len)
    boxes = json.loads(json_bytes.decode("utf-8"))

    # JPEG
    raw_len = recvall(sock, 4)
    jpeg_len = struct.unpack(">I", raw_len)[0]
    if jpeg_len > MAX_PACKET_SIZE:
        raise ValueError(f"JPEG trop grand: {jpeg_len}")
    jpeg_bytes = recvall(sock, jpeg_len)

    np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Impossible de décoder le JPEG")

    return boxes, frame


# ---------------------------------------------------------------------------
# Buffer partagé entre thread réseau et thread affichage
# ---------------------------------------------------------------------------


class DisplayBuffer:
    def __init__(self):
        self._data: Optional[Tuple[dict, np.ndarray]] = None
        self._lock = threading.Lock()

    def put(self, boxes: dict, frame: np.ndarray):
        with self._lock:
            self._data = (boxes, frame)

    def get(self) -> Optional[Tuple[dict, np.ndarray]]:
        with self._lock:
            return self._data


# ---------------------------------------------------------------------------
# Thread de réception
# ---------------------------------------------------------------------------


def receiver_thread(buf: DisplayBuffer, stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            sock = socket.create_connection((HOST, PORT), timeout=SOCKET_TIMEOUT)
            sock.settimeout(SOCKET_TIMEOUT)
            logger.info("Connecté à %s:%s", HOST, PORT)

            while not stop_event.is_set():
                boxes, frame = recv_packet(sock)
                buf.put(boxes, frame)

        except (ConnectionError, OSError, ValueError) as e:
            logger.warning(
                "Connexion perdue: %s. Reconnexion dans %.1fs...", e, RECONNECT_DELAY
            )
            try:
                sock.close()
            except Exception:
                pass
            time.sleep(RECONNECT_DELAY)


# ---------------------------------------------------------------------------
# Affichage des boxes sur la frame
# ---------------------------------------------------------------------------


def draw_boxes_info(frame: np.ndarray, boxes: dict) -> np.ndarray:
    """Affiche les infos de détection en overlay texte."""
    h, w = frame.shape[:2]

    # Caméra 1 (moitié gauche)
    for b in boxes.get("cam1", []):
        label = f"c1 obj={b['objectness']:.2f}"
        cv2.putText(
            frame,
            label,
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        break  # affiche seulement la première

    # Caméra 2 (moitié droite)
    for b in boxes.get("cam2", []):
        label = f"c2 obj={b['objectness']:.2f}"
        cv2.putText(
            frame,
            label,
            (w // 2 + 10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        break

    return frame


# ---------------------------------------------------------------------------
# Boucle principale d'affichage
# ---------------------------------------------------------------------------


def main():
    buf = DisplayBuffer()
    stop_event = threading.Event()
    last_frame_id = None

    t = threading.Thread(
        target=receiver_thread,
        args=(buf, stop_event),
        daemon=True,
    )
    t.start()

    logger.info("Affichage démarré. Appuie sur 'q' pour quitter.")
    prev_time = time.time()

    try:
        while True:
            result = buf.get()

            if result is None:
                # Pas encore de frame — affiche un écran noir
                blank = np.zeros((480, 1280, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    "En attente du serveur...",
                    (400, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("YOLO — Jetson Inference", blank)
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    break
                continue

            boxes, frame = result

            frame_id = id(frame)
            if frame_id == last_frame_id:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            last_frame_id = frame_id

            # FPS côté affichage
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # Overlay FPS + infos boxes
            frame = draw_boxes_info(frame, boxes)
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Nombre de détections
            n1 = len(boxes.get("cam1", []))
            n2 = len(boxes.get("cam2", []))
            cv2.putText(
                frame,
                f"Det: cam1={n1}  cam2={n2}",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("YOLO — Jetson Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        cv2.destroyAllWindows()
        logger.info("Client terminé")


if __name__ == "__main__":
    main()
