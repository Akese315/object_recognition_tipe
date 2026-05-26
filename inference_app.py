import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import LightweightYOLO
from utils import find_objects
from YOLO_loader import BoundingBox, CustomImage

# ================= CONFIGURATION =================
# MODEL_PATH = 'face_models/cnn_yolo-light_reduc32-v1_fp32_2025-12-16 01-51-36/model.pth'
MODEL_PATH = "ball_models/cnn_yolo-light_reduc32-v1_fp32_2025-12-15 13-19-10/model.pth"

CONF_THRESHOLD = 0.3
CLASS_THESHOLD = 0.9


CLASSES = ["face"]

N_CLASS = len(CLASSES)
COLORS = [(0, 255, 0), (255, 0, 0)]  # Vert pour card, bleu pour screen


# Transformations
transform = transforms.Compose([transforms.ToTensor()])

state_dict = torch.load(MODEL_PATH, map_location="cpu")
anchors = state_dict["anchors"]

print("Configuration terminée.")
# ================= CHARGEMENT MODÈLE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightweightYOLO(
    num_classes=N_CLASS, base_kernel_num=32, conv_layer=5, divider=1, anchors=anchors
)
model.load_state_dict(state_dict)
model.to(device)
model.eval()


# ================= BOUCLE WEBCAM =================
cap = cv2.VideoCapture(0)
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
        target_size=(640, 480),
        reduction_factor=reduction,
        bounding_boxes=[],  # Pas de vérité terrain en inférence
    )
    input_tensor = inference_image.get_raw_tensor().unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        output = model.predict(output)
        # print(f"Output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}")

        B, H, W, A, S = output.shape
        pred_boxes = []

        torch.max(output[0, ..., 0], dim=2)

        mask = output[0, ..., 0] > CONF_THRESHOLD

        valid_preds = output[0][mask]
        class_scores = valid_preds[:, 5:]
        max_class_scores, _ = torch.max(class_scores, dim=1)
        valid_preds = valid_preds[max_class_scores > CLASS_THESHOLD]

        pred_boxes = [
            BoundingBox.from_tensor(pred.cpu(), N_CLASS, W, H) for pred in valid_preds
        ]

        print(f"Détections brutes : {len(pred_boxes)}")

        res_pil = inference_image.get_image(
            predicted_bb_boxes=pred_boxes, objectness_strict=False
        )

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


cap.release()
cv2.destroyAllWindows()
