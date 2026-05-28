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

Raccourcis clavier :
  - Entrée : action personnalisée (on_enter_pressed)
  - q      : quitter
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import asyncio
import json
import logging
import math
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pynput import keyboard

from arm import Arm
from robot import Robot

try:
    from scservo_sdk import SerialException
except ImportError:
    SerialException = Exception  # Fallback si scservo_sdk est absent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HOST = "172.30.162.51"  # Adresse IP du Jetson — à adapter
PORT = 8000
RECONNECT_DELAY = 2.0  # secondes entre deux tentatives de reconnexion
SOCKET_TIMEOUT = 10.0  # timeout socket en secondes
MAX_PACKET_SIZE = 8 * 1024 * 1024  # 8 MiB par paquet

QUIT_KEYS = {ord("q"), ord("Q")}  # touches cv2 pour quitter

# ---------------------------------------------------------------------------
# Position 3D courante de l'objet détecté (mise à jour par la boucle principale)
# ---------------------------------------------------------------------------

x_reel: float = 0.0
y_reel: float = 0.0
z_reel: float = 0.0


@dataclass(frozen=True)
class Config:
    confidence_threshold: float = 0.2
    class_threshold: float = 0.9
    model_classes: List[str] = field(default_factory=lambda: ["face"])
    # FIX 1 : même résolution que inference_app.py
    capture_width: int = 640
    capture_height: int = 480
    input_size: Tuple[int, int] = (640, 480)

    baseline: float = 0.060
    camera_height: float = 0.1
    H_POV_deg: float = 73.0
    V_POV_deg: float = 50.0


CFG = Config()

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("client_display")

# ---------------------------------------------------------------------------
# Initialisation du robot
# ---------------------------------------------------------------------------


async def init_robot(rbt: Robot) -> None:
    await rbt.add_arm(
        servo_moteur_id=1,
        origin=180,
        axis="y",
        min_angle_limit=90,
        max_angle_limit=270,
        sens_rotation=-1.0,
        arm_len=np.array([0.035, 0.115, 0]),
    )

    await rbt.add_arm(
        servo_moteur_id=2,
        origin=180,
        axis="z",
        min_angle_limit=90,
        max_angle_limit=270,
        sens_rotation=-1.0,
        arm_len=np.array([0.03, 0.115, 0]),
    )
    await rbt.add_arm(
        servo_moteur_id=3,
        origin=90,
        axis="z",
        min_angle_limit=90,
        max_angle_limit=270,
        sens_rotation=-1.0,
        arm_len=np.array([0, 0.135, 0]),
    )

    await rbt.add_arm(
        servo_moteur_id=4,
        origin=180,
        axis="z",
        min_angle_limit=90,
        max_angle_limit=270,
        sens_rotation=-1.0,
        arm_len=np.array([0, 0.06, 0]),
    )
    await rbt.add_arm(
        servo_moteur_id=5,
        origin=45,
        axis="y",
        min_angle_limit=0,
        max_angle_limit=180,
        sens_rotation=-1.0,
        arm_len=np.array([0, 0.1, 0]),
    )

    await rbt.add_pince(
        servo_moteur_id=6,
        origin=180,
        axis="y",
        min_angle_limit=90,
        max_angle_limit=180,
        sens_rotation=-1.0,
        arm_len=np.array([0, 0.1, 0]),
    )


# ---------------------------------------------------------------------------
# Réception réseau
# ---------------------------------------------------------------------------


def recvall(sock: socket.socket, n: int) -> bytes:
    """Lit exactement n octets depuis le socket."""
    data = bytearray()
    while len(data) < n:
        try:
            chunk = sock.recv(n - len(data))
        except socket.timeout as e:
            raise ConnectionError(f"Timeout: {e}")
        if not chunk:
            raise ConnectionError("Connexion fermée par le serveur")
        data.extend(chunk)
    return bytes(data)


def recv_packet(sock: socket.socket) -> Tuple[dict, np.ndarray]:
    """
    Reçoit un paquet complet et retourne (boxes_dict, frame_bgr).

    Format du protocole :
        [4B big-endian : taille JSON][JSON bytes]
        [4B big-endian : taille JPEG][JPEG bytes]
    """
    # --- JSON ---
    json_len = struct.unpack(">I", recvall(sock, 4))[0]
    if json_len > MAX_PACKET_SIZE:
        raise ValueError(f"JSON trop grand : {json_len} octets")
    boxes: dict = json.loads(recvall(sock, json_len).decode("utf-8"))

    # --- JPEG ---
    jpeg_len = struct.unpack(">I", recvall(sock, 4))[0]
    if jpeg_len > MAX_PACKET_SIZE:
        raise ValueError(f"JPEG trop grand : {jpeg_len} octets")
    jpeg_bytes = recvall(sock, jpeg_len)

    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Impossible de décoder le JPEG reçu")

    return boxes, frame


# ---------------------------------------------------------------------------
# Buffer partagé thread réseau ↔ thread affichage
# ---------------------------------------------------------------------------


class DisplayBuffer:
    """Stocke le dernier paquet reçu de façon thread-safe."""

    def __init__(self):
        self._data: Optional[Tuple[dict, np.ndarray]] = None
        self._lock = threading.Lock()

    def put(self, boxes: dict, frame: np.ndarray) -> None:
        with self._lock:
            self._data = (boxes, frame)

    def get(self) -> Optional[Tuple[dict, np.ndarray]]:
        with self._lock:
            return self._data


# ---------------------------------------------------------------------------
# Thread de réception réseau
# ---------------------------------------------------------------------------


def receiver_thread(buf: DisplayBuffer, stop_event: threading.Event) -> None:
    """Se connecte au Jetson, lit les paquets et les pousse dans le buffer."""
    while not stop_event.is_set():
        try:
            sock = socket.create_connection((HOST, PORT), timeout=SOCKET_TIMEOUT)
            sock.settimeout(SOCKET_TIMEOUT)
            logger.info("Connecté à %s:%d", HOST, PORT)

            while not stop_event.is_set():
                buf.put(*recv_packet(sock))

        except (ConnectionError, OSError, ValueError) as exc:
            logger.warning(
                "Connexion perdue : %s — reconnexion dans %.1fs…", exc, RECONNECT_DELAY
            )
            try:
                sock.close()
            except Exception:
                pass
            time.sleep(RECONNECT_DELAY)


# ---------------------------------------------------------------------------
# Overlay d'informations sur la frame
# ---------------------------------------------------------------------------


def draw_overlay(frame: np.ndarray, boxes: dict, fps: float) -> np.ndarray:
    """
    Ajoute sur la frame :
      - FPS en haut à gauche
      - Score d'objectness de la première détection de chaque caméra
      - Position 3D courante (x_reel, y_reel, z_reel)
      - Nombre total de détections en bas
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    color_fps = (0, 255, 0)
    color_det = (0, 255, 255)
    color_pos = (255, 180, 0)
    color_info = (200, 200, 200)

    # FPS
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 50), font, 1.0, color_fps, 2, cv2.LINE_AA
    )

    # Première détection caméra 1
    if boxes.get("cam1"):
        obj = boxes["cam1"][0]["objectness"]
        cv2.putText(
            frame, f"cam1 obj={obj:.2f}", (10, 90), font, 0.6, color_det, 2, cv2.LINE_AA
        )

    # Première détection caméra 2 (moitié droite)
    if boxes.get("cam2"):
        obj = boxes["cam2"][0]["objectness"]
        cv2.putText(
            frame,
            f"cam2 obj={obj:.2f}",
            (w // 2 + 10, 90),
            font,
            0.6,
            color_det,
            2,
            cv2.LINE_AA,
        )

    # Position 3D
    cv2.putText(
        frame,
        f"X={x_reel * 100:.1f}cm  Y={y_reel * 100:.1f}cm  Z={z_reel * 100:.1f}cm",
        (10, 130),
        font,
        0.6,
        color_pos,
        2,
        cv2.LINE_AA,
    )

    # Compteur total
    n1, n2 = len(boxes.get("cam1", [])), len(boxes.get("cam2", []))
    cv2.putText(
        frame,
        f"Det: cam1={n1}  cam2={n2}",
        (10, h - 15),
        font,
        0.6,
        color_info,
        1,
        cv2.LINE_AA,
    )

    return frame


# ---------------------------------------------------------------------------
# Listener clavier (thread indépendant via pynput)
# — fiable quelle que soit la durée entre deux frames
# ---------------------------------------------------------------------------


def start_keyboard_listener(
    buf: DisplayBuffer, stop_event: threading.Event, robot: Robot
) -> keyboard.Listener:
    """
    Lance un listener pynput dans son propre thread.
    Entrée  → on_enter_pressed avec le dernier paquet disponible
    q / Q   → stop_event
    """

    def on_press(key: keyboard.Key) -> None:
        if key == keyboard.Key.enter:
            result = buf.get()
            if result is not None:
                boxes, frame = result
                on_enter_pressed(boxes, frame, robot)
        elif hasattr(key, "char") and key.char in ("q", "Q"):
            stop_event.set()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def on_enter_pressed(boxes: dict, frame: np.ndarray, robot: Robot) -> None:
    """Appelée chaque fois que l'utilisateur appuie sur Entrée."""
    print(
        f"Entrée pressée ! Position : X={x_reel * 100:.1f}cm  Y={y_reel * 100:.1f}cm  Z={z_reel * 100:.1f}cm"
    )

    robot.open_pince()
    time.sleep(1)
    z_robot = 0.58 - z_reel
    # angles, position = robot.IK.run_ccd(z_robot, y_reel, x_reel)
    angles, position = robot.IK.run_ccd(-0.15, 0.20, 0)
    print(
        f"calculated position X={position[0] * 100:.1f}cm  Y={position[1] * 100:.1f}cm  Z={position[2] * 100:.1f}cm"
    )
    angles_degrees = angles * 180 / (np.pi)
    robot.rotate_arms(angles_degrees)
    time.sleep(3)
    robot.fermer_pince()


# ---------------------------------------------------------------------------
# Calcul de profondeur stéréo
# ---------------------------------------------------------------------------


def compute_stereo_depth(
    x1: float, x2: float, y1: float, y2: float
) -> Optional[np.ndarray]:
    cx1, cy1 = x1 - 0.5, y1 - 0.5
    cx2, cy2 = x2 - 0.5, y2 - 0.5

    HPOV = math.radians(CFG.H_POV_deg)
    VPOV = math.radians(CFG.V_POV_deg)
    m = 2.0 * math.tan(HPOV / 2)
    k = 2.0 * math.tan(VPOV / 2)

    v1 = np.array([cx1 * m, k * cy1, 1.0])
    v2 = np.array([cx2 * m, k * cy2, 1.0])

    A = np.array([-CFG.baseline / 2, CFG.camera_height, 0.0])
    B = np.array([CFG.baseline / 2, CFG.camera_height, 0.0])

    d = A - B  # = [-baseline, 0, 0]

    denom = np.dot(v1, v2) ** 2 - np.dot(v1, v1) * np.dot(v2, v2)
    if abs(denom) < 1e-12:
        return None

    t_prime = abs(
        (np.dot(d, v2) * np.dot(v1, v1) - np.dot(d, v1) * np.dot(v1, v2)) / denom
    )
    t = abs((np.dot(d, v2) * np.dot(v1, v2) - np.dot(d, v1) * np.dot(v2, v2)) / denom)

    P1 = A + v1 * t
    P2 = B + v2 * t_prime

    return (P1 + P2) / 2.0


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------


async def main() -> None:
    global x_reel, y_reel, z_reel

    rbt = Robot("COM3")

    await init_robot(rbt)

    buf = DisplayBuffer()
    stop_event = threading.Event()

    # Thread réseau
    t = threading.Thread(target=receiver_thread, args=(buf, stop_event), daemon=True)
    t.start()

    # Listener clavier indépendant (pynput) — capte Entrée de façon fiable
    kb_listener = start_keyboard_listener(buf, stop_event, rbt)

    logger.info("Affichage démarré. Appuie sur Entrée pour l'action, 'q' pour quitter.")

    prev_time = time.time()
    last_frame_id = None

    try:
        while not stop_event.is_set():
            result = buf.get()

            # Pas encore de frame reçue → écran d'attente
            if result is None:
                blank = np.zeros((480, 1280, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    "En attente du serveur…",
                    (400, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("YOLO — Jetson Inference", blank)
                if cv2.waitKey(100) & 0xFF in QUIT_KEYS:
                    break
                continue

            boxes, frame = result

            # Ignorer si la frame n'a pas changé
            if id(frame) == last_frame_id:
                if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                    break
                continue
            last_frame_id = id(frame)

            # Calcul du FPS côté affichage
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # Calcul de la profondeur stéréo et mise à jour des globales
            boxes0 = boxes.get("cam1")
            boxes1 = boxes.get("cam2")

            if boxes0 and boxes1:
                p = compute_stereo_depth(
                    boxes0[0]["x_center"],
                    boxes1[0]["x_center"],
                    boxes0[0]["y_center"],
                    boxes1[0]["y_center"],
                )
                if p is not None:
                    x_reel, y_reel, z_reel = float(p[0]), float(p[1]), float(p[2])

            # Dessin de l'overlay et affichage
            frame = draw_overlay(frame, boxes, fps)
            cv2.imshow("YOLO — Jetson Inference", frame)

            # cv2.waitKey gère uniquement 'q' ; Entrée est gérée par pynput
            if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                break

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        kb_listener.stop()
        cv2.destroyAllWindows()
        logger.info("Client terminé.")


if __name__ == "__main__":
    asyncio.run(main())
