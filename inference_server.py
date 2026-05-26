#!/usr/bin/env python3
"""
server_inference.py — À lancer sur le Jetson Nano.

Rôle :
  - Capture les deux caméras IMX219 via GStreamer
  - Fait tourner le modèle YOLO localement (GPU Jetson)
  - Envoie à chaque client connecté :
      * les frames JPEG (caméra gauche + droite côte à côte)
      * les bounding boxes détectées (JSON)
  - Protocole par paquet :  [4B taille_json][JSON][4B taille_jpeg][JPEG]
"""

import json
import logging
import math
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import LightweightYOLO
from utils import calculate_area
from YOLO_loader import BoundingBox, CustomImage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "ball_models/cnn_yolo-light_reduc32-v1_fp32_2025-12-15 13-19-10/model.pth"

HOST = "0.0.0.0"
PORT = 8000  # port unique — on envoie tout sur une seule connexion

JPEG_QUALITY = 70
MAX_FRAME_SIZE = 4 * 1024 * 1024
SOCKET_TIMEOUT = 5.0

logger = logging.getLogger("server_inference")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@dataclass(frozen=True)
class Config:
    confidence_threshold: float = 0.3
    class_threshold: float = 0.9
    model_classes: List[str] = field(default_factory=lambda: ["face"])
    input_size: Tuple[int, int] = (640, 480)

    baseline: float = 0.060
    camera_height: float = 0.5
    H_POV_deg: float = 73.0
    V_POV_deg: float = 50.0


CFG = Config()


# ---------------------------------------------------------------------------
# GStreamer pipeline
# ---------------------------------------------------------------------------


def gstreamer_pipeline(sensor_id: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode=4 ! "
        f"video/x-raw(memory:NVMM), width=1280, height=720, framerate=59/1 ! "
        f"nvvidconv ! "
        f"video/x-raw(memory:NVMM), width=640, height=480 ! "
        f"nvvidconv ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


# ---------------------------------------------------------------------------
# Caméras
# ---------------------------------------------------------------------------


class Camera:
    def __init__(self, sensor_id: int):
        self.sensor_id = sensor_id
        self.cap = None
        self._open()

    def _open(self):
        self.cap = cv2.VideoCapture(gstreamer_pipeline(self.sensor_id))
        if not self.cap.isOpened():
            logger.error("Camera %s open failed", self.sensor_id)

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            self._open()
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------


def load_model() -> Tuple[LightweightYOLO, torch.device]:
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    logger.info("Chargement du modèle...")
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    anchors = state_dict["anchors"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = LightweightYOLO(
        num_classes=len(CFG.model_classes),
        base_kernel_num=32,
        conv_layer=5,
        divider=1,
        anchors=anchors,
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, device


# ---------------------------------------------------------------------------
# Inférence
# ---------------------------------------------------------------------------


def detect_boxes(
    input_tensor: torch.Tensor,
    model: LightweightYOLO,
) -> List[BoundingBox]:
    with torch.no_grad():
        output = model(input_tensor)
        output = model.predict(output)
        B, H, W, A, S = output.shape

        mask = output[0, ..., 0] > CFG.confidence_threshold
        valid_preds = output[0][mask]
        if valid_preds.numel() == 0:
            return []

        class_scores = valid_preds[:, 5:]
        max_scores, _ = torch.max(class_scores, dim=1)
        valid_preds = valid_preds[max_scores > CFG.class_threshold]

        boxes = [
            BoundingBox.from_tensor(pred.cpu(), len(CFG.model_classes), W, H)
            for pred in valid_preds
        ]
        boxes.sort(key=calculate_area, reverse=True)
        return boxes


def boxes_to_json(boxes1: List[BoundingBox], boxes2: List[BoundingBox]) -> bytes:
    """Sérialise les boîtes des deux caméras en JSON."""

    def box_dict(b: BoundingBox) -> dict:
        return {
            "x_center": float(b.x_center),
            "y_center": float(b.y_center),
            "width": float(b.width),
            "height": float(b.height),
            "objectness": float(b.objectness),
            "class_id": int(b.class_id),
        }

    payload = {
        "cam1": [box_dict(b) for b in boxes1],
        "cam2": [box_dict(b) for b in boxes2],
    }
    return json.dumps(payload).encode("utf-8")


# ---------------------------------------------------------------------------
# Profondeur stéréo
# ---------------------------------------------------------------------------


def compute_stereo_depth(x1, x2, y1, y2) -> Optional[np.ndarray]:
    HPOV = math.radians(CFG.H_POV_deg)
    VPOV = math.radians(CFG.V_POV_deg)
    m = 2.0 * math.tan(HPOV / 2)
    k = 2.0 * math.tan(VPOV / 2)

    v1 = np.array([x1 * m, k * y1, 1.0])
    v2 = np.array([x2 * m, k * y2, 1.0])
    A = np.array([-CFG.baseline / 2, CFG.camera_height, 1.0])
    B = np.array([CFG.baseline / 2, CFG.camera_height, 1.0])
    d = A - B

    denom = np.dot(v1, v2) ** 2 - np.dot(v1, v1) * np.dot(v2, v2)
    if abs(denom) < 1e-12:
        return None

    t_prime = (np.dot(d, v2) * np.dot(v1, v1) - np.dot(d, v1) * np.dot(v1, v2)) / denom
    t = (np.dot(d, v2) * np.dot(v1, v2) - np.dot(d, v1) * np.dot(v2, v2)) / denom
    return (A + B + v1 * t + v2 * t_prime) / 2.0


# ---------------------------------------------------------------------------
# Thread d'envoi vers un client
# ---------------------------------------------------------------------------


class ResultBuffer:
    """Stocke le dernier résultat (frame combinée + boxes JSON) prêt à envoyer."""

    def __init__(self):
        self._data: Optional[Tuple[bytes, bytes]] = None  # (json_bytes, jpeg_bytes)
        self._lock = threading.Lock()
        self._event = threading.Event()

    def put(self, json_bytes: bytes, jpeg_bytes: bytes):
        with self._lock:
            self._data = (json_bytes, jpeg_bytes)
        self._event.set()

    def get(self, timeout: float = 1.0) -> Optional[Tuple[bytes, bytes]]:
        self._event.wait(timeout)
        self._event.clear()
        with self._lock:
            return self._data


def client_sender(
    conn: socket.socket, result_buf: ResultBuffer, stop_event: threading.Event
):
    """Thread dédié à l'envoi vers un client connecté."""
    try:
        while not stop_event.is_set():
            result = result_buf.get(timeout=1.0)
            if result is None:
                continue

            json_bytes, jpeg_bytes = result

            # Protocole : [4B len_json][json][4B len_jpeg][jpeg]
            packet = (
                struct.pack(">I", len(json_bytes))
                + json_bytes
                + struct.pack(">I", len(jpeg_bytes))
                + jpeg_bytes
            )
            conn.sendall(packet)

    except (socket.error, BrokenPipeError, OSError):
        pass
    finally:
        conn.close()
        logger.info("Client déconnecté")


# ---------------------------------------------------------------------------
# Boucle principale d'inférence
# ---------------------------------------------------------------------------


def inference_loop(
    model: LightweightYOLO,
    device: torch.device,
    result_buf: ResultBuffer,
    stop_event: threading.Event,
):
    cam0 = Camera(0)
    cam1 = Camera(1)
    reduction = model.get_reduction_factor()

    logger.info("Inférence démarrée sur %s", device)

    try:
        while not stop_event.is_set():
            ret0, frame0 = cam0.read()
            ret1, frame1 = cam1.read()

            if not ret0 or not ret1:
                time.sleep(0.01)
                continue

            # ── Préparation tenseurs ──────────────────────────────
            def to_tensor(frame: np.ndarray) -> torch.Tensor:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

            t0 = to_tensor(frame0)
            t1 = to_tensor(frame1)

            inf0 = CustomImage.from_tensor(t0, CFG.input_size, reduction, [])
            inf1 = CustomImage.from_tensor(t1, CFG.input_size, reduction, [])

            inp0 = inf0.get_raw_tensor().unsqueeze(0).to(device)
            inp1 = inf1.get_raw_tensor().unsqueeze(0).to(device)

            # ── Inférence ────────────────────────────────────────
            boxes0 = detect_boxes(inp0, model)
            boxes1 = detect_boxes(inp1, model)

            # ── Profondeur stéréo ─────────────────────────────────
            if boxes0 and boxes1:
                p = compute_stereo_depth(
                    boxes0[0].x_center,
                    boxes1[0].x_center,
                    boxes0[0].y_center,
                    boxes1[0].y_center,
                )
                if p is not None:
                    logger.info("Profondeur: %.1f cm", p[2] * 100)

            # ── Construction image annotée ────────────────────────
            if boxes0:
                res0 = cv2.cvtColor(
                    np.array(inf0.get_image([boxes0[0]], objectness_strict=False)),
                    cv2.COLOR_RGB2BGR,
                )
            else:
                res0 = frame0

            if boxes1:
                res1 = cv2.cvtColor(
                    np.array(inf1.get_image([boxes1[0]], objectness_strict=False)),
                    cv2.COLOR_RGB2BGR,
                )
            else:
                res1 = frame1

            # ── Combinaison côte à côte ───────────────────────────
            h = min(res0.shape[0], res1.shape[0])
            combined = np.hstack(
                [
                    cv2.resize(res0, (int(res0.shape[1] * h / res0.shape[0]), h)),
                    cv2.resize(res1, (int(res1.shape[1] * h / res1.shape[0]), h)),
                ]
            )

            # ── Encodage JPEG ─────────────────────────────────────
            ok, enc = cv2.imencode(
                ".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if not ok:
                continue

            jpeg_bytes = enc.tobytes()
            json_bytes = boxes_to_json(boxes0, boxes1)

            result_buf.put(json_bytes, jpeg_bytes)

    finally:
        cam0.release()
        cam1.release()
        logger.info("Inférence arrêtée")


# ---------------------------------------------------------------------------
# Serveur TCP
# ---------------------------------------------------------------------------


def main():
    model, device = load_model()

    stop_event = threading.Event()
    result_buf = ResultBuffer()

    # Thread d'inférence (tourne en permanence)
    inf_thread = threading.Thread(
        target=inference_loop,
        args=(model, device, result_buf, stop_event),
        daemon=True,
    )
    inf_thread.start()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(1)
    server_sock.settimeout(1.0)
    logger.info("En attente d'un client sur le port %s...", PORT)

    try:
        while not stop_event.is_set():
            try:
                conn, addr = server_sock.accept()
                conn.settimeout(SOCKET_TIMEOUT)
                logger.info("Client connecté: %s", addr)
            except socket.timeout:
                continue

            # Un thread d'envoi par client
            sender_stop = threading.Event()
            t = threading.Thread(
                target=client_sender,
                args=(conn, result_buf, sender_stop),
                daemon=True,
            )
            t.start()
            t.join()  # attendre déconnexion avant d'accepter le suivant
            sender_stop.set()

    except KeyboardInterrupt:
        logger.info("Arrêt...")
    finally:
        stop_event.set()
        server_sock.close()
        inf_thread.join()
        logger.info("Serveur terminé")


if __name__ == "__main__":
    main()
