import asyncio
import math
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import robot
from model import LightweightYOLO

# Camera calibration constants (adjust as needed)
CAMERA_ANGLE = math.radians(30)  # example angle
CAMERA_HEIGHT = 0.5  # meters
from utils import calculate_area, get_x_reel, get_y_reel, get_z_reel
from YOLO_loader import BoundingBox, CustomImage

MODEL_PATH = os.path.join(
    "face_models",
    "cnn_yolo-light_reduc32-v1_fp32_2025-12-16 01-51-36",
    "model.pth",
)
# Verify model path exists
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
# MODEL_PATH = "ball_models/cnn_yolo-light_reduc32-v1_fp32_2025-12-15 13-19-10/model.pth"
CONF_THRESHOLD = 0.3
CLASS_THRESHOLD = 0.9

CLASSES = ["face"]

N_CLASS = len(CLASSES)
COLORS = [(0, 255, 0), (255, 0, 0)]  # Vert pour card, bleu pour screen


"""async def init_robot(rbt: robot.Robot) -> None:
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
        origin=0,
        axis="y",
        min_angle_limit=0,
        max_angle_limit=180,
        sens_rotation=-1.0,
        arm_len=np.array([0, 0.1, 0]),
    )
"""


async def main() -> None:
    """Initialize robot, model, and start inference loop.

    The robot is instantiated locally and passed to the helper
    functions that require it.
    """
    # Create robot instance and configure arms
    # rbt = robot.Robot("COM3")
    # await init_robot(rbt)

    # Compute initial joint angles
    # angles, _ = rbt.IK.run_ccd(0.2, 0.0, 0.0)
    # angles *= 180 / np.pi  # Convert radians to degrees
    # await rbt._rotate_arms_async(angles, 1000, 128)

    # Load AI model
    model, device = init_ai_model()
    await run_loop(model, device)


def init_ai_model():
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    anchors = state_dict["anchors"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Configuration terminée. Chargement du modèle...")
    model = LightweightYOLO(
        num_classes=N_CLASS,
        base_kernel_num=32,
        conv_layer=5,
        divider=1,
        anchors=anchors,
    )
    model.load_state_dict(state_dict)
    model.to(device)

    return model, device


async def moving_arm_async(rbt: robot.Robot, x: float, y: float, z: float) -> None:
    """Run ``moving_arm`` in a background thread.

    Parameters
    ----------
    rbt:
        Instance of :class:`robot.Robot` controlling the arm.
    x, y, z:
        Target coordinates in real‑world space.
    """
    await asyncio.to_thread(moving_arm, rbt, x, y, z)


def moving_arm(rbt: robot.Robot, x: float, y: float, z: float) -> None:
    angles, calculated_position = rbt.IK.run_ccd(0.05, y / 10, x / 10)
    if np.array_equal(angles, np.zeros(5)):
        print(
            f"x:{x}, y:{y}\n Position pas atteignable ",
            end="\r",
            flush=True,
        )
    else:
        print(
            f"x:{x}, y:{y}\n",
            end="\r",
            flush=True,
        )
    rbt.rotate(angles, 1000, 128)


async def run_loop(model: LightweightYOLO, device: torch.device) -> None:
    model.eval()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    if not cap.isOpened():
        raise RuntimeError("Erreur : Impossible d'ouvrir la caméra.")

    print("Inférence en temps réel démarrée. Appuie sur 'q' pour quitter.")
    reduction = model.get_reduction_factor()
    transform = transforms.Compose([transforms.ToTensor()])
    while True:
        ret, frame = cap.read()
        process_time_start = time.time()

        if not ret:
            break

        display_frame = frame.copy()
        orig_h, orig_w = frame.shape[:2]

        # Préparation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_tensor = transforms.ToTensor()(Image.fromarray(frame_rgb))

        inference_image = CustomImage.from_tensor(
            image_tensor=raw_tensor,
            target_size=(640, 480),
            reduction_factor=reduction,
            bounding_boxes=[],
        )
        input_tensor = inference_image.get_raw_tensor().unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            output = model.predict(output)
            # print(f"Output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}")

            B, H, W, A, S = output.shape
            pred_boxes = []

            mask = output[0, ..., 0] > CONF_THRESHOLD

            valid_preds = output[0][mask]
            class_scores = valid_preds[:, 5:]
            max_class_scores, _ = torch.max(class_scores, dim=1)
            valid_preds = valid_preds[max_class_scores > CLASS_THRESHOLD]

            pred_boxes = [
                BoundingBox.from_tensor(pred.cpu(), N_CLASS, W, H)
                for pred in valid_preds
            ]
            pred_boxes.sort(key=lambda box: calculate_area(box), reverse=True)

            res_rgb = frame_rgb
            if len(pred_boxes) != 0:
                HPOV = math.radians(47.1)
                VPOV = math.radians(36.2)
                main_box = pred_boxes[0]
                res_pil = inference_image.get_image(
                    predicted_bb_boxes=[main_box], objectness_strict=False
                )
                x = main_box.x_center
                y = main_box.y_center

                # Compute real-world coordinates (angle and height should be calibrated)
                z = get_z_reel(y, CAMERA_ANGLE, HPOV, VPOV, CAMERA_HEIGHT)
                x_reel = get_x_reel(x, z, HPOV)
                y_reel = get_y_reel(y, z, VPOV)
                width = main_box.width
                height = main_box.height

                # asyncio.create_task(moving_arm_async(rbt, x_reel, y_reel, z))

            # Convertir en format OpenCV pour l'afficher (RGB -> BGR)
            res_np = np.array(res_pil)
            res_bgr = cv2.cvtColor(res_np, cv2.COLOR_RGB2BGR)
            process_time_end = time.time()
            process_time_interval = process_time_end - process_time_start
            fps_text = f"FPS: {1.0 / process_time_interval:.2f}"

            cv2.putText(
                img=res_bgr,
                text=fps_text,
                org=(10, 50),  # Position (x, y) en pixels (coin haut gauche)
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,  # Taille de la police
                color=(0, 255, 0),  # Couleur (B, G, R) -> Ici Vert
                thickness=2,  # Épaisseur du trait
                lineType=cv2.LINE_AA,  # Anti-aliasing pour un texte plus net
            )

            cv2.imshow("YOLO Real-time Detection", res_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources after loop ends
    cap.release()
    cv2.destroyAllWindows()


# Entry point
if __name__ == "__main__":
    asyncio.run(main())
