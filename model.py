import torchvision.models
import torch.nn as nn
import torch
from utils import batch_IoU

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False) # Bias False car BatchNorm suit
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) # inplace économise de la mémoire

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class YOLOHead(nn.Module):
    def __init__(self, num_classes, num_anchors, kernel_size):
        super(YOLOHead, self).__init__()   
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.kernel_size = kernel_size
        out_channels = num_anchors * (5 + num_classes)

        self.detector = nn.Conv2d(self.kernel_size, out_channels, kernel_size=1)

    def forward(self, x):
        
        B, _, H, W = x.shape

        pred = self.detector(x)
        pred = pred.permute(0, 2, 3, 1).contiguous()
        pred = pred.view(B, H, W, self.num_anchors, 5 + self.num_classes)

        # Extraction des composantes
        tobj = pred[..., 0]
        tx = pred[..., 1]
        ty = pred[..., 2]
        tw = pred[..., 3]
        th = pred[..., 4]
        tcls = pred[..., 5:]        

        bw = torch.sigmoid(tw) 
        bh = torch.sigmoid(th) 

        bx = torch.sigmoid(tx)
        by = torch.sigmoid(ty)

        # On garde les logits pour l'objectness et les classes (pour BCEWithLogitsLoss)
        obj_score = tobj
        cls_prob = tcls

        # Reconstitution du tenseur de sortie
        pred_boxes = torch.stack([obj_score, bx, by, bw, bh], dim=-1)
        pred_final = torch.cat([pred_boxes, cls_prob], dim=-1)

        return pred_final  # [B, H, W, A, 5+C]


    
class LightweightYOLO(nn.Module):
    def __init__(self, num_classes=20, num_anchors=3, conv_layer=3,base_kernel_num=32, divider=1):
        super(LightweightYOLO, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.reduction_factor = (2**conv_layer) * divider
        self.base_kernel_num = base_kernel_num

        in_channels = 3
        layers = []
        for i in range(conv_layer):
            out_channels = self.base_kernel_num * (2 ** i)
            layers.append(ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            
        self.backbone = nn.Sequential(*layers)
        self.head = YOLOHead(self.num_classes, self.num_anchors, out_channels)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        # Redimensionnement dynamique silencieux si nécessaire
        if H % self.reduction_factor != 0 or W % self.reduction_factor != 0:
            new_H = (H // self.reduction_factor + 1) * self.reduction_factor
            new_W = (W // self.reduction_factor + 1) * self.reduction_factor
            # On évite le print ici pour ne pas spammer les logs
            x = nn.functional.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)

        features = self.backbone(x)
        return self.head(features)
        
    def predict(self, x):
        B, H, W, A, _ = x.shape

        tobj = x[..., 0]
        tx = x[..., 1]
        ty = x[..., 2]
        bw = x[..., 3]
        bh = x[..., 4]
        tcls = x[..., 5:]

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing='ij')

        grid_x = grid_x.view(1, H, W, 1).float()
        grid_y = grid_y.view(1, H, W, 1).float()

        
        bx = (tx + grid_x) / W
        by = (ty + grid_y) / H
         
        
        obj_score = torch.sigmoid(tobj)
        cls_prob = torch.sigmoid(tcls)

        pred_boxes = torch.stack([obj_score, bx, by, bw, bh], dim=-1)
        pred_final = torch.cat([pred_boxes, cls_prob], dim=-1)
        return pred_final

    def get_reduction_factor(self):
        return self.reduction_factor

class YoloLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5, lambda_obj=1.0, smooth_factor=0.0, IoU_loss=True):
        super(YoloLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.IoU_loss = IoU_loss
        self.smooth_factor = smooth_factor
        
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, preds, targets):
        # preds: [B, H, W, A, 5+C] (local sigmoid)
        # targets: [B, H, W, A, 5+C] (global 0-1)
        
        device = preds.device 
        B, H, W, A, S = preds.shape
        
        obj_mask = targets[..., 0] == 1 
        noobj_mask = targets[..., 0] == 0

        # --- 1. Préparation des Cibles Locales (pour la régression) ---
        # On clone pour ne pas casser 'targets' qui sert à l'IoU plus bas
        t_local = targets.clone()
        
        # Formule : (Global * TailleGrille) - IndexCellule
        # Cela donne une valeur entre 0 et 1
        t_local[..., 1] = t_local[..., 1] * W - torch.floor(t_local[..., 1] * W)
        t_local[..., 2] = t_local[..., 2] * H - torch.floor(t_local[..., 2] * H)

        # --- 2. Loss Coordonnées ---
        # On compare des valeurs 0-1 (preds) avec des valeurs 0-1 (t_local)
        loss_coord = self.mse(preds[..., 1:3][obj_mask], t_local[..., 1:3][obj_mask]) \
                   + self.mse(preds[..., 3:5][obj_mask], t_local[..., 3:5][obj_mask])
        
        # --- 3. Calcul de l'IoU Score (Besoin de Global) ---
        with torch.no_grad():
            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
            grid_x = grid_x.view(1, H, W, 1).float()
            grid_y = grid_y.view(1, H, W, 1).float()

            # On reconstruit les prédictions globales pour l'IoU
            preds_global = preds.clone()
            preds_global[..., 1] = (preds[..., 1] + grid_x) / W
            preds_global[..., 2] = (preds[..., 2] + grid_y) / H
            
            # targets est toujours global ici (car on a utilisé t_local plus haut)
            iou_scores = batch_IoU(preds_global, targets).detach().clamp(0, 1)

        # --- 4. Loss Objectness ---
        target_obj = torch.zeros_like(preds[..., 0], device=device)

        if self.IoU_loss:
            target_obj[obj_mask] = iou_scores[obj_mask] # On vise l'IoU réelle
        else:
            target_obj[obj_mask] = 1.0 

        loss_obj = self.bce(preds[..., 0][obj_mask], target_obj[obj_mask])
        loss_noobj = self.bce(preds[..., 0][noobj_mask], target_obj[noobj_mask])
        
        # --- 5. Loss Classes ---
        loss_class = torch.tensor(0.0, device=device)
        if preds.size(-1) > 5 and obj_mask.sum() > 0:
            # Gestion du label smoothing manuel
            t_class = targets[..., 5:][obj_mask]
            if self.smooth_factor > 0:
                n_c = preds.size(-1) - 5
                t_class = t_class * (1 - self.smooth_factor) + (self.smooth_factor / n_c)
            
            loss_class = self.bce(preds[..., 5:][obj_mask], t_class)

        # --- Normalisation ---
        num_objects = obj_mask.sum().float() + 1e-6

        total = (
            self.lambda_coord * loss_coord +
            self.lambda_obj * loss_obj +
            self.lambda_noobj * loss_noobj +
            loss_class
        ) / num_objects

        avg_iou = iou_scores[obj_mask].mean().item() if obj_mask.sum() > 0 else 0.0

        return total, {
            "coord": loss_coord.item() / num_objects,
            "obj": loss_obj.item() / num_objects,
            "noobj": loss_noobj.item() / num_objects,
            "class": loss_class.item() / num_objects,
            "iou": avg_iou
        }