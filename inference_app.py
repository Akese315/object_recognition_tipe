import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from YOLO_loader import BoundingBox, CustomImage
from model import LightweightYOLO
from utils import find_objects

# ================= CONFIGURATION =================
MODEL_PATH = 'ball_models/cnn_yolo-light_reduc8-v1_fp32_2025-12-04 12-48-42/model.pth'
N_ANCHORS = 1

CONF_THRESHOLD = 0.2
CLASS_THESHOLD = 0.8
IOU_THRESHOLD = 0.4

CLASSES = ["ball"]

N_CLASS = len(CLASSES)
COLORS = [(0, 255, 0), (255, 0, 0)]  # Vert pour card, bleu pour screen


# Transformations
transform = transforms.Compose([
    
    transforms.ToTensor()
])

print("Configuration terminée.")
# ================= CHARGEMENT MODÈLE =================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LightweightYOLO(num_classes=N_CLASS, num_anchors=N_ANCHORS, base_kernel_num=16, conv_layer=3, divider=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
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
        bounding_boxes=[] # Pas de vérité terrain en inférence
    )
    input_tensor = inference_image.get_raw_tensor().unsqueeze(0).to(device)
    with torch.no_grad():        
        output = model(input_tensor)
        output = model.predict(output)
        #print(f"Output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}")
        
        B, H, W, A, S = output.shape
        pred_boxes = []
        
        for i in range(H):
            for j in range(W):
                for a in range(A):
                    pred = output[0, i, j, a]
                    if pred[0].item() > CONF_THRESHOLD:
                        box = BoundingBox.from_tensor(pred.cpu(), N_CLASS, i, j, W, H)
                        pred_boxes.append(box)
                        

        print(f"Détections brutes : {len(pred_boxes)}")
        
        res_pil = inference_image.get_image(predicted_bb_boxes=pred_boxes, objectness_strict=False)
        
        # Convertir en format OpenCV pour l'afficher (RGB -> BGR)
        res_np = np.array(res_pil)
        res_bgr = cv2.cvtColor(res_np, cv2.COLOR_RGB2BGR)

        cv2.imshow('YOLO Real-time Detection', res_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()