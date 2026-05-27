import asyncio
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import client as client_mod
from model import LightweightYOLO
from utils import calculate_area
from YOLO_loader import BoundingBox, CustomImage

# ---------------------------------------------------------------------------
# Configuration -------------------------------------------------------------
# ---------------------------------------------------------------------------

MODEL_PATH = "ball_models/cnn_yolo-light_reduc32-v1_fp32_2025-12-15 13-19-10/model.pth"


@dataclass(frozen=True)
class Config:
    """All immutable configuration for the inference pipeline."""

    # Model
    confidence_threshold: float = 0.3
    class_threshold: float = 0.9
    model_classes: List[str] = field(default_factory=lambda: ["face"])
    input_size: Tuple[int, int] = (640, 480)
    camera_angle_deg: float = 30.0

    # Camera (stereo / depth)
    baseline: float = 0.060  # baseline en mètre
    camera_height: float = 0.5  # hauteur de la caméra (m)
    H_POV_deg: float = 73.0  # champ horizontal vue
    V_POV_deg: float = 50.0  # champ vertical vue

    # Network
    ports: Tuple[int, int] = (8000, 8001)


CFG = Config()

# ---------------------------------------------------------------------------
# Helpers --------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _init_logging() -> None:
    """Initialise le logging basique pour le script."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


# ---------------------------------------------------------------------------
# Model loading -------------------------------------------------------------
# ---------------------------------------------------------------------------


def init_ai_model() -> tuple[LightweightYOLO, torch.device]:
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    print("Configuration terminée. Chargement du modèle...")
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    anchors = state_dict["anchors"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
# Inference ------------------------------------------------------------------
# ---------------------------------------------------------------------------


def detect_boxes(
    input_tensor: torch.Tensor,
    model: LightweightYOLO,
) -> List[BoundingBox]:
    with torch.no_grad():
        output = model(input_tensor)
        output = model.predict(output)  # ← AJOUTER CETTE LIGNE
        B, H, W, A, S = output.shape  # ← La shape change aussi !

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


# ---------------------------------------------------------------------------
# Stereo depth ---------------------------------------------------------------
# ---------------------------------------------------------------------------


def compute_stereo_depth(
    x1: float, x2: float, y1: float, y2: float
) -> Optional[np.ndarray]:
    """Intersect two rays to recover the 3-D point (stereo vision).

    Returns ``None`` when the rays do not produce a valid intersection.
    """
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

    p = (A + B + v1 * t + v2 * t_prime) / 2.0
    return p


# ---------------------------------------------------------------------------
# Display --------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _prepare_input(image_tensor: torch.Tensor, reduction: int):
    """Wrap a tensor in a ``CustomImage`` with letter-boxing."""
    return CustomImage.from_tensor(
        image_tensor=image_tensor,
        target_size=CFG.input_size,
        reduction_factor=reduction,
        bounding_boxes=[],
    )


def _draw_fps_overlay(frame: np.ndarray, fps: float) -> None:
    cv2.putText(
        img=frame,
        text=f"FPS: {fps:.2f}",
        org=(10, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Main loop ------------------------------------------------------------------
# ---------------------------------------------------------------------------


async def run_loop(model: LightweightYOLO, device: torch.device) -> None:
    model.eval()
    stop_event = threading.Event()
    buffers = client_mod.FrameBufferPair()

    # Thread de lecture par caméra
    for port, buf in zip(CFG.ports, (buffers.frame1, buffers.frame2)):
        t = threading.Thread(
            target=client_mod.stream_reader, args=(port, buf, stop_event), daemon=True
        )
        t.start()

    # Laisser le temps aux threads de recevoir les premières frames
    await asyncio.sleep(2.0)  # ← ajouter ceci

    reduction = model.get_reduction_factor()
    consecutive_missing = 0
    max_consecutive_missing = 30  # ~30s si 1 fps

    print("Inférence en temps réel démarrée. Appuie sur 'q' pour quitter.")

    while True:
        t_start = time.time()

        frame1 = buffers.frame1.get()
        frame2 = buffers.frame2.get()

        if frame1 is not None:
            print(f"Frame size: {frame1.shape}")

        if frame1 is None or frame2 is None:
            consecutive_missing += 1
            gray = np.zeros((480, 1280, 3), dtype=np.uint8)
            _draw_fps_overlay(gray, 0.0)
            cv2.imshow("YOLO Real-time Detection", gray)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if consecutive_missing >= max_consecutive_missing:
                logging.info("No frames received for too long. Quitting.")
                break  # ← manquait
            continue

        consecutive_missing = 0

        frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        raw_tensor1 = torch.from_numpy(frame1_rgb).permute(2, 0, 1).float() / 255.0
        raw_tensor2 = torch.from_numpy(frame2_rgb).permute(2, 0, 1).float() / 255.0

        inference1 = _prepare_input(raw_tensor1, reduction)
        inference2 = _prepare_input(raw_tensor2, reduction)

        input1 = inference1.get_raw_tensor().unsqueeze(0).to(device)
        input2 = inference2.get_raw_tensor().unsqueeze(0).to(device)

        # Inférence en parallèle (les deux threads tournent vraiment en même temps)
        #

        boxes1 = detect_boxes(input1, model)
        boxes2 = detect_boxes(input2, model)

        print(boxes1)
        # ── Caméra 1 ──────────────────────────────────────────────
        if boxes1:
            image_pil1 = inference1.get_image(
                predicted_bb_boxes=[boxes1[0]],
                objectness_strict=False,
            )
            res_bgr1 = cv2.cvtColor(np.array(image_pil1), cv2.COLOR_RGB2BGR)
        else:
            res_bgr1 = frame1.copy()

        # ── Caméra 2 ──────────────────────────────────────────────
        if boxes2:
            image_pil2 = inference2.get_image(
                predicted_bb_boxes=[boxes2[0]],
                objectness_strict=False,
            )
            res_bgr2 = cv2.cvtColor(np.array(image_pil2), cv2.COLOR_RGB2BGR)
        else:
            res_bgr2 = frame2.copy()

        # ── Profondeur stéréo ──────────────────────────────────────
        if boxes1 and boxes2:
            point_reel = compute_stereo_depth(
                boxes1[0].x_center,
                boxes2[0].x_center,
                boxes1[0].y_center,
                boxes2[0].y_center,
            )
            if point_reel is not None:
                print(
                    f"Point réel: {point_reel}, profondeur: {point_reel[2] * 100:.1f} cm"
                )

        # ── Affichage côte à côte (comme client.py) ───────────────
        h = min(res_bgr1.shape[0], res_bgr2.shape[0])
        if res_bgr1.shape[0] != h:
            res_bgr1 = cv2.resize(
                res_bgr1, (int(res_bgr1.shape[1] * h / res_bgr1.shape[0]), h)
            )
        if res_bgr2.shape[0] != h:
            res_bgr2 = cv2.resize(
                res_bgr2, (int(res_bgr2.shape[1] * h / res_bgr2.shape[0]), h)
            )

        combined = np.hstack([res_bgr1, res_bgr2])

        fps = 1.0 / max(time.time() - t_start, 1e-6)
        _draw_fps_overlay(combined, fps)

        cv2.imshow("YOLO Real-time Detection", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stop_event.set()


def cleanup() -> None:
    """Fermer proprement les fenêtres OpenCV."""
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point ----------------------------------------------------------------
# ---------------------------------------------------------------------------


async def _run() -> None:
    _init_logging()
    model, device = init_ai_model()

    try:
        await run_loop(model, device)
    finally:
        cleanup()


if __name__ == "__main__":
    asyncio.run(_run())
