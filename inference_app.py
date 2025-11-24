import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from YOLO_loader import BoundingBox
from model import LightweightYOLO

# ================= CONFIGURATION =================
MODEL_PATH = 'model/version6.pth'
IMG_SIZE = 416
N_CELL = 52
N_ANCHORS = 3
N_CLASS = 3
CONF_THRESHOLD = 0.9
CLASS_THESHOLD = 0.3
IOU_THRESHOLD = 0.4

CLASSES = ['card', 'face']
COLORS = [(0, 255, 0), (255, 0, 0)]  # Vert pour card, bleu pour screen


# Transformations
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ================= CHARGEMENT MODÈLE =================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LightweightYOLO(num_classes=N_CLASS, num_anchors=N_ANCHORS, conv_layer=3, divider=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


# ================= BOUCLE WEBCAM =================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
    exit()

print("Inférence en temps réel démarrée. Appuie sur 'q' pour quitter.")
reduction = model.get_reduction_factor()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    orig_h, orig_w = frame.shape[:2]

    # Préparation
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)  # [B, H, W, A, 5+C]

        

        # Parcourir la grille et les ancres pour récupérer toutes les prédictions
        B,H, W, A, S = output.shape
        pred_boxes = []
        for i in range(H):
            for j in range(W):
                for a in range(A):
                    pred = output[0, i, j, a]
                    
                    if pred[0].item() > CONF_THRESHOLD:
                        # Construire un objet BoundingBox depuis le tenseur
                        box = BoundingBox.from_tensor(pred.cpu(), N_CLASS, i, j,  W, H)
                        if box.class_id_prob >= CLASS_THESHOLD:
                          # Limiter le nombre de boîtes pour éviter les débordements
                            pred_boxes.append(box)
        
        # Appliquer la suppression non maximale

        
        # Dessiner toutes les bounding boxes retrouvées
        for box in pred_boxes:

            i_cell, j_cell = box.get_cell_position(int(W//reduction), int(H//reduction))
            den = box.get_denormalized_tensor((orig_w, orig_h), i_cell, j_cell,  int(W//reduction), int(H//reduction))
            # den: [obj, x_center, y_center, width, height, ...]
            x_center = den[1]
            y_center = den[2]
            w_box = den[3]
            h_box = den[4]

            x1 = int(x_center - w_box / 2)
            y1 = int(y_center - h_box / 2)
            x2 = int(x_center + w_box / 2)
            y2 = int(y_center + h_box / 2)

            label = CLASSES[box.class_id] if box.class_id < len(CLASSES) else str(box.class_id)
            color = COLORS[box.class_id] if box.class_id < len(COLORS) else (0, 255, 0)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"{label} {box.class_id_prob:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('YOLO Real-time Detection', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break