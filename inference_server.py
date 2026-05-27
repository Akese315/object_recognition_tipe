#!/usr/bin/env python3
# inference_server.py — Jetson Nano, version optimisée

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
from torchvision import transforms

from model import LightweightYOLO
from utils import calculate_area
from YOLO_loader import BoundingBox, CustomImage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# MODEL_PATH = "face_models/cnn_yolo-light_reduc32-v1_fp32_2025-12-16 01-51-36/model.pth"
MODEL_PATH = "ball_models/cnn_yolo-light_reduc32-v1_fp32_2025-12-15 13-19-10/model.pth"

PORT = 8000
HOST = "0.0.0.0"
MAX_FRAME_SIZE = 4 * 1024 * 1024
JPEG_QUALITY = 70
SOCKET_TIMEOUT = 5.0

logger = logging.getLogger("inference_server")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


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
# FIX 2 : Pipeline GStreamer avec resize GPU intégré (nvvidconv)
#          + framerate explicite + drop=true
# ---------------------------------------------------------------------------


def gstreamer_pipeline(sensor_id: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        # Capture en résolution native raisonnable (pas Full HD)
        f"video/x-raw(memory:NVMM), width=1280, height=720, "
        f"framerate=30/1, format=NV12 ! "
        # Resize vers 640x480 sur le GPU Tegra (pas le CPU)
        f"nvvidconv ! "
        f"video/x-raw(memory:NVMM), width={CFG.capture_width}, height={CFG.capture_height} ! "
        f"nvvidconv ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        # max-buffers=1 drop=true : on ne garde que la frame la plus récente
        f"appsink max-buffers=1 drop=true sync=false"
    )


# ---------------------------------------------------------------------------
# Camera avec thread dédié pour ne jamais bloquer la boucle d'inférence
# ---------------------------------------------------------------------------


class Camera:
    """
    Thread de capture dédié : la dernière frame est toujours disponible
    immédiatement, sans attendre le prochain cap.read() qui peut bloquer.
    """

    def __init__(self, sensor_id: int):
        self.sensor_id = sensor_id
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ok = False
        self._stop = threading.Event()
        self.cap = self._open()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _open(self):
        pipeline = gstreamer_pipeline(self.sensor_id)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            logger.error("Camera %s open failed", self.sensor_id)
        return cap

    def _capture_loop(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            with self._lock:
                self._ok = ret
                if ret:
                    self._frame = frame
        self.cap.release()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ok, self._frame.copy()

    def release(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------


def init_ai_model() -> Tuple[LightweightYOLO, torch.device]:
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    logger.info("Chargement du modèle : %s", MODEL_PATH)
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

    # FIX 3 : pré-chauffer le modèle pour éviter la latence au 1er appel
    logger.info("Préchauffage du modèle...")
    dummy = torch.zeros(1, 3, CFG.capture_height, CFG.capture_width).to(device)
    with torch.no_grad():
        _ = model(dummy)
    logger.info("Modèle prêt.")

    return model, device


# ---------------------------------------------------------------------------
# FIX 4 : Conversion frame→tensor sans passer par PIL
# ---------------------------------------------------------------------------

_to_tensor = transforms.ToTensor()


def frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """BGR numpy → RGB float32 tensor [3, H, W], range [0,1], sans PIL."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # np → tensor directement, pas de détour par PIL.Image
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0)
    return tensor


# ---------------------------------------------------------------------------
# Inférence
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
# Profondeur stéréo
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

    # ✅ Z=0.0 et non 1.0
    A = np.array([-CFG.baseline / 2, CFG.camera_height, 0.0])
    B = np.array([CFG.baseline / 2, CFG.camera_height, 0.0])

    d = A - B  # = [-baseline, 0, 0]

    denom = np.dot(v1, v2) ** 2 - np.dot(v1, v1) * np.dot(v2, v2)
    if abs(denom) < 1e-12:
        return None

    t_prime = np.abs(np.dot(d, v2) * np.dot(v1, v1) - np.dot(d, v1) * np.dot(v1, v2)) / denom)
    t = np.abs((np.dot(d, v2) * np.dot(v1, v2) - np.dot(d, v1) * np.dot(v2, v2)) / denom)


    P1 = A + v1 * t
    P2 = B + v2 * t_prime

    print(f"P1 : x1 : {P1[0]},y1 : {P1[1]}")
    print(f"P2 : x2 : {P2[0]},y2 : {P2[1]}")

    return (P1 + P2) / 2.0


# ---------------------------------------------------------------------------
# Sérialisation
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
        {"cam1": [box_dict(b) for b in boxes1], "cam2": [box_dict(b) for b in boxes2]}
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Worker principal
# ---------------------------------------------------------------------------


def camera_worker(
    model: LightweightYOLO, device: torch.device, stop_event: threading.Event
):
    # FIX 5 : les cameras tournent dans leurs propres threads de capture
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
            try:
                conn, addr = server_sock.accept()
                logger.info("Client %s connecté", addr)
                conn.settimeout(SOCKET_TIMEOUT)
            except socket.timeout:
                continue

            fps_counter = 0
            fps_start = time.time()

            try:
                while not stop_event.is_set():
                    t0 = time.time()

                    ret0, frame0 = cam0.read()
                    ret1, frame1 = cam1.read()

                    if not ret0 or not ret1:
                        time.sleep(0.005)
                        continue

                    # FIX 4 : conversion rapide sans PIL
                    raw_tensor0 = frame_to_tensor(frame0)
                    raw_tensor1 = frame_to_tensor(frame1)

                    inference0 = _prepare_input(raw_tensor0, reduction)
                    inference1 = _prepare_input(raw_tensor1, reduction)

                    input0 = inference0.get_raw_tensor().unsqueeze(0).to(device)
                    input1 = inference1.get_raw_tensor().unsqueeze(0).to(device)

                    # FIX 6 : les deux inférences dans le même bloc no_grad
                    # pour minimiser les synchronisations CUDA
                    with torch.no_grad():
                        out0 = model(input0)
                        out1 = model(input1)
                        out0 = model.predict(out0)
                        out1 = model.predict(out1)

                    def _extract(out):
                        B, H, W, A, S = out.shape
                        mask = out[0, ..., 0] > CFG.confidence_threshold
                        vp = out[0][mask]
                        if vp.numel() == 0:
                            return []
                        ms, _ = torch.max(vp[:, 5:], dim=1)
                        vp = vp[ms > CFG.class_threshold]
                        boxes = [
                            BoundingBox.from_tensor(
                                p.cpu(), len(CFG.model_classes), W, H
                            )
                            for p in vp
                        ]
                        boxes.sort(key=calculate_area, reverse=True)
                        return boxes

                    boxes0 = _extract(out0)
                    boxes1 = _extract(out1)

                    # Profondeur stéréo
                    if boxes0 and boxes1:
                        p = compute_stereo_depth(
                            boxes0[0].x_center,
                            boxes1[0].x_center,
                            boxes0[0].y_center,
                            boxes1[0].y_center,
                        )
                        if p is not None:
                            logger.info("Profondeur: %.1f cm", p[2] * 100)

                    # Image annotée
                    res0 = (
                        cv2.cvtColor(
                            np.array(
                                inference0.get_image(
                                    [boxes0[0]], objectness_strict=False
                                )
                            ),
                            cv2.COLOR_RGB2BGR,
                        )
                        if boxes0
                        else frame0
                    )

                    res1 = (
                        cv2.cvtColor(
                            np.array(
                                inference1.get_image(
                                    [boxes1[0]], objectness_strict=False
                                )
                            ),
                            cv2.COLOR_RGB2BGR,
                        )
                        if boxes1
                        else frame1
                    )

                    # Combinaison côte à côte
                    # FIX 7 : les deux frames ont déjà la même taille (640x480)
                    # donc pas besoin de resize dynamique
                    combined = np.hstack([res0, res1])

                    # FPS overlay
                    fps_counter += 1
                    if time.time() - fps_start >= 2.0:
                        fps = fps_counter / (time.time() - fps_start)
                        logger.info("FPS serveur : %.1f", fps)
                        fps_counter = 0
                        fps_start = time.time()

                    # Encodage JPEG
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

                    packet = (
                        struct.pack(">I", len(json_bytes))
                        + json_bytes
                        + struct.pack(">I", len(jpeg_bytes))
                        + jpeg_bytes
                    )
                    conn.sendall(packet)

            except (socket.error, BrokenPipeError):
                logger.info("Client déconnecté")
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
# Main
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
