#!/usr/bin/env python3
# server_inference.py — À lancer sur le Jetson Nano.
#
# Reprend exactement server.py qui fonctionnait, en ajoutant :
#   - chargement du modèle YOLO
#   - inférence sur chaque frame
#   - envoi au client : [4B len_json][json][4B len_jpeg][jpeg]

# torch EN PREMIER pour éviter le conflit TLS avec GStreamer
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

from model import LightweightYOLO
from utils import calculate_area
from YOLO_loader import BoundingBox, CustomImage

# ---------------------------------------------------------------------------
# Configuration (identique à TIPE.py)
# ---------------------------------------------------------------------------

MODEL_PATH = "ball_models/cnn_yolo-light_reduc32-v1_fp32_2025-12-15 13-19-10/model.pth"

CAMERA_IDS = (0, 1)
PORT = 8000  # un seul port — on envoie json + jpeg ensemble
HOST = "0.0.0.0"

MAX_FRAME_SIZE = 4 * 1024 * 1024
JPEG_QUALITY = 70
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
# GStreamer pipeline — identique à server.py qui fonctionnait
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
# Camera — identique à server.py
# ---------------------------------------------------------------------------


class Camera:
    def __init__(self, sensor_id):
        self.sensor_id = sensor_id
        self.cap = None
        self.open()

    def open(self):
        pipeline = gstreamer_pipeline(self.sensor_id)
        self.cap = cv2.VideoCapture(pipeline)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            logger.error("Camera %s open failed", self.sensor_id)

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            self.open()
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()


# ---------------------------------------------------------------------------
# Modèle — identique à TIPE.py
# ---------------------------------------------------------------------------


def init_ai_model() -> Tuple[LightweightYOLO, torch.device]:
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    print("Chargement du modèle...")
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
# Inférence — identique à TIPE.py
# ---------------------------------------------------------------------------


def detect_boxes(
    input_tensor: torch.Tensor, model: LightweightYOLO
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

        boxes: List[BoundingBox] = [
            BoundingBox.from_tensor(pred.cpu(), len(CFG.model_classes), W, H)
            for pred in valid_preds
        ]
        boxes.sort(key=calculate_area, reverse=True)
        return boxes


def _prepare_input(image_tensor: torch.Tensor, reduction: int) -> CustomImage:
    return CustomImage.from_tensor(
        image_tensor=image_tensor,
        target_size=CFG.input_size,
        reduction_factor=reduction,
        bounding_boxes=[],
    )


# ---------------------------------------------------------------------------
# Profondeur stéréo — identique à TIPE.py
# ---------------------------------------------------------------------------


def compute_stereo_depth(
    x1: float, x2: float, y1: float, y2: float
) -> Optional[np.ndarray]:
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
# Sérialisation des boxes
# ---------------------------------------------------------------------------


def boxes_to_json(boxes1: List[BoundingBox], boxes2: List[BoundingBox]) -> bytes:
    def box_dict(b: BoundingBox) -> dict:
        return {
            "x_center": float(b.x_center),
            "y_center": float(b.y_center),
            "width": float(b.width),
            "height": float(b.height),
            "objectness": float(b.objectness),
            "class_id": int(b.class_id),
        }

    return json.dumps(
        {
            "cam1": [box_dict(b) for b in boxes1],
            "cam2": [box_dict(b) for b in boxes2],
        }
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Worker — reprend la structure exacte de camera_worker dans server.py
# ---------------------------------------------------------------------------


def camera_worker(
    model: LightweightYOLO, device: torch.device, stop_event: threading.Event
):
    cam0 = Camera(0)
    cam1 = Camera(1)
    reduction = model.get_reduction_factor()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(1)
    server_sock.settimeout(1.0)

    logger.info("En attente d'un client sur le port %s...", PORT)

    conn = None

    try:
        while not stop_event.is_set():
            # Attente connexion client
            try:
                conn, addr = server_sock.accept()
                logger.info("Client %s connecté", addr)
                conn.settimeout(SOCKET_TIMEOUT)
            except socket.timeout:
                continue

            try:
                while not stop_event.is_set():
                    ret0, frame0 = cam0.read()
                    ret1, frame1 = cam1.read()

                    if not ret0 or not ret1:
                        time.sleep(0.01)
                        continue

                    # ── Préparation tenseurs (comme TIPE.py) ──────────
                    frame0_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
                    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

                    raw_tensor0 = (
                        torch.tensor(frame0_rgb, dtype=torch.float32).permute(2, 0, 1)
                        / 255.0
                    )
                    raw_tensor1 = (
                        torch.tensor(frame1_rgb, dtype=torch.float32).permute(2, 0, 1)
                        / 255.0
                    )

                    inference0 = _prepare_input(raw_tensor0, reduction)
                    inference1 = _prepare_input(raw_tensor1, reduction)

                    input0 = inference0.get_raw_tensor().unsqueeze(0).to(device)
                    input1 = inference1.get_raw_tensor().unsqueeze(0).to(device)

                    # ── Inférence ─────────────────────────────────────
                    boxes0 = detect_boxes(input0, model)
                    boxes1 = detect_boxes(input1, model)

                    # ── Profondeur stéréo ─────────────────────────────
                    if boxes0 and boxes1:
                        p = compute_stereo_depth(
                            boxes0[0].x_center,
                            boxes1[0].x_center,
                            boxes0[0].y_center,
                            boxes1[0].y_center,
                        )
                        if p is not None:
                            logger.info("Profondeur: %.1f cm", p[2] * 100)

                    # ── Image annotée (comme TIPE.py) ─────────────────
                    if boxes0:
                        res0 = cv2.cvtColor(
                            np.array(
                                inference0.get_image(
                                    [boxes0[0]], objectness_strict=False
                                )
                            ),
                            cv2.COLOR_RGB2BGR,
                        )
                    else:
                        res0 = frame0

                    if boxes1:
                        res1 = cv2.cvtColor(
                            np.array(
                                inference1.get_image(
                                    [boxes1[0]], objectness_strict=False
                                )
                            ),
                            cv2.COLOR_RGB2BGR,
                        )
                    else:
                        res1 = frame1

                    # ── Combinaison côte à côte ───────────────────────
                    h = min(res0.shape[0], res1.shape[0])
                    combined = np.hstack(
                        [
                            cv2.resize(
                                res0, (int(res0.shape[1] * h / res0.shape[0]), h)
                            ),
                            cv2.resize(
                                res1, (int(res1.shape[1] * h / res1.shape[0]), h)
                            ),
                        ]
                    )

                    # ── Encodage JPEG ─────────────────────────────────
                    ok, enc = cv2.imencode(
                        ".jpg",
                        combined,
                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                    )
                    if not ok:
                        continue

                    jpeg_bytes = enc.tobytes()
                    json_bytes = boxes_to_json(boxes0, boxes1)

                    if len(jpeg_bytes) > MAX_FRAME_SIZE:
                        continue

                    # ── Envoi : [4B len_json][json][4B len_jpeg][jpeg] ─
                    packet = (
                        struct.pack(">I", len(json_bytes))
                        + json_bytes
                        + struct.pack(">I", len(jpeg_bytes))
                        + jpeg_bytes
                    )
                    conn.sendall(packet)

            except (socket.error, BrokenPipeError):
                pass

            finally:
                if conn:
                    conn.close()
                    conn = None

    finally:
        cam0.release()
        cam1.release()
        server_sock.close()
        logger.info("Serveur arrêté")


# ---------------------------------------------------------------------------
# Main — identique à server.py
# ---------------------------------------------------------------------------


def main():
    model, device = init_ai_model()
    stop_event = threading.Event()

    t = threading.Thread(
        target=camera_worker,
        args=(model, device, stop_event),
        daemon=True,
    )
    t.start()

    try:
        while t.is_alive():
            t.join(0.5)
    except KeyboardInterrupt:
        logger.info("Arrêt...")
        stop_event.set()
        t.join()

    logger.info("Terminé")


if __name__ == "__main__":
    main()
