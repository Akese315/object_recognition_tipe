import asyncio
import json
import math
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from websockets.asyncio.server import broadcast, serve

import robot
from model import LightweightYOLO
from utils import calculate_area, find_objects, get_x_reel, get_y_reel, get_z_reel
from YOLO_loader import BoundingBox, CustomImage

rbt = robot.Robot("COM3")

async def init_robot():
    await rbt.add_arm(1, 180, "y", 90, 270, np.array([0.035, 0, 0]))
    await rbt.add_arm(2, 180, "z", 90, 270, np.array([0, 0.09, 0]))
    await rbt.add_arm(3, 90, "z", 90, 270, np.array([0, 0.115, 0]))
    await rbt.add_arm(4, 180, "z", 90, 270, np.array([0, 0.135, 0]))
    await rbt.add_arm(5, 0, "y", 0, 180, np.array([0, 0.15, 0]))

USERS = set()


async def handler(websocket):
    global USERS
    USERS.add(websocket)
    try:
        async for message in websocket:
            event = json.loads(message)
            if event["event_type"] == "move":
                print(f"move to coodinates : ${event['position']}")
    finally:
        USERS.remove(websocket)


async def broadcast_loop():
    while True:
        x = float(await asyncio.to_thread(input, "x: "))
        y = float(await asyncio.to_thread(input, "y: "))
        z = float(await asyncio.to_thread(input, "z: "))
        calculated_angles = rbt.calculate_angles_CDD(x, y, z, 20)
        calculated_angles = [math.degrees(angle) for angle in calculated_angles]
        event = json.dumps({"event_type": "joint_angles", "value": calculated_angles})
        await broadcast(USERS, event)


async def main():
    async with serve(handler, "localhost", 5678):
        print("listening")
        await asyncio.gather(broadcast_loop())


if __name__ == "__main__":
    asyncio.run(main())


# center_x, center_y = 0.20, 0.20
# radius = 0.05
# steps = 100

# for i in range(steps + 1):
#     angle = 2 * math.pi * i / steps
#     x = center_x + radius * math.cos(angle)
#     y = center_y + radius * math.sin(angle)
#     print(math.degrees(angle))
#     robot.move_to_2arms(x, y, 0)
"""
#MODEL_PATH = 'face_models/cnn_yolo-light_reduc32-v1_fp32_2025-12-16 01-51-36/model.pth'
MODEL_PATH = 'ball_models/cnn_yolo-light_reduc32-v1_fp32_2025-12-15 13-19-10/model.pth'

CONF_THRESHOLD = 0.3
CLASS_THESHOLD = 0.9


CLASSES = ["face"]

N_CLASS = len(CLASSES)
COLORS = [(0, 255, 0), (255, 0, 0)]  # Vert pour card, bleu pour screen



transform = transforms.Compose([
    transforms.ToTensor()
])

state_dict = torch.load(MODEL_PATH, map_location='cpu')
anchors = state_dict["anchors"]

print("Configuration terminée.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LightweightYOLO(num_classes=N_CLASS, base_kernel_num=32, conv_layer=5, divider=1,anchors=anchors)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
    exit()

print("Inférence en temps réel démarrée. Appuie sur 'q' pour quitter.")
reduction = model.get_reduction_factor()

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
        target_size=(640,480),
        reduction_factor=reduction,
        bounding_boxes=[]
    )
    input_tensor = inference_image.get_raw_tensor().unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        output = model.predict(output)
        #print(f"Output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}")

        B, H, W, A, S = output.shape
        pred_boxes = []

        torch.max(output[0, ..., 0], dim=2)

        mask = output[0, ..., 0] > CONF_THRESHOLD


        valid_preds = output[0][mask]
        class_scores = valid_preds[:, 5:]
        max_class_scores, _ = torch.max(class_scores, dim=1)
        valid_preds = valid_preds[max_class_scores > CLASS_THESHOLD]

        pred_boxes = [BoundingBox.from_tensor(pred.cpu(), N_CLASS, W, H) for pred in valid_preds]
        pred_boxes.sort(key=lambda box: calculate_area(box), reverse=True)

        res_pil = frame_rgb
        if len(pred_boxes) != 0:
            HPOV = math.radians(47.1)
            VPOV = math.radians(36.2)
            main_box = pred_boxes[0]
            res_pil = inference_image.get_image(predicted_bb_boxes=[main_box], objectness_strict=False)
            x = main_box.x_center
            y = main_box.y_center

            z = get_z_reel(y, camera_angle, HPOV, VPOV, camera_height)
            x_reel = get_x_reel(x, z, camera_angle, HPOV)
            y_reel = get_y_reel(y, z, camera_angle, VPOV)
            width = main_box.width
            height = main_box.height

            #robot.move_to_2arms(x, y, 0)


        # Convertir en format OpenCV pour l'afficher (RGB -> BGR)
        res_np = np.array(res_pil)
        res_bgr = cv2.cvtColor(res_np, cv2.COLOR_RGB2BGR)
        process_time_end = time.time()
        process_time_interval = process_time_end - process_time_start
        fps_text = f"FPS: {1.0 / process_time_interval:.2f}"

        cv2.putText(
            img=res_bgr,
            text=fps_text,
            org=(10, 50),             # Position (x, y) en pixels (coin haut gauche)
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,              # Taille de la police
            color=(0, 255, 0),        # Couleur (B, G, R) -> Ici Vert
            thickness=2,              # Épaisseur du trait
            lineType=cv2.LINE_AA      # Anti-aliasing pour un texte plus net
        )

        cv2.imshow('YOLO Real-time Detection', res_bgr)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
"""
