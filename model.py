import torchvision.models
import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False) # Bias False car BatchNorm suit
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) # inplace économise de la mémoire

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class YOLOHead(nn.Module):
    def __init__(self, num_classes, num_anchors,kernel_size):
        super(YOLOHead, self).__init__()   
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.kernel_size = kernel_size
        out_channels = num_anchors * (5 + num_classes)

        self.detector = nn.Conv2d(self.kernel_size, out_channels, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.shape
        
        pred = self.detector(x)                       # [B, A*(5+C), H, W]
        pred = pred.permute(0, 2, 3, 1).contiguous()  # [B, H, W, A*(5+C)]
        pred = pred.view(B, H, W, self.num_anchors, 5 + self.num_classes)  # [B, H, W, A, 5+C]

        # Split predictions
        tobj = pred[..., 0]
        tx = pred[..., 1]
        ty = pred[..., 2]
        tw = pred[..., 3]
        th = pred[..., 4]
        tcls = pred[..., 5:]

        # Activations
        bx = torch.sigmoid(tx) 
        by = torch.sigmoid(ty) 
        bw = torch.sigmoid(tw) 
        bh = torch.sigmoid(th) 

        obj_score = tobj
        cls_prob = tcls

        # Concat final [objectness, bx, by, bw, bh, class_probs]
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
         
        tobj = x[..., 0]
        tx = x[..., 1]
        ty = x[..., 2]
        tw = x[..., 3]
        th = x[..., 4]
        tcls = x[..., 5:]
        obj_score = torch.sigmoid(tobj)
        cls_prob = torch.sigmoid(tcls)

        pred_boxes = torch.stack([obj_score, tx, ty, tw, th], dim=-1)
        pred_final = torch.cat([pred_boxes, cls_prob], dim=-1)
        return pred_final  # [B, H, W, A, 5+C]
    
    def get_reduction_factor(self):
        return self.reduction_factor

class YoloLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5, lambda_obj=1.0, smooth_factor=0.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCEWithLogitsLoss(reduction='sum') # Changé en sum pour être cohérent avec MSE
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.smooth_factor = smooth_factor

    def forward(self, preds, targets):
        # preds, targets = [B, H, W, A, 5+C]

        B,H, W, A, S = preds.shape
        n_class = S - 5 # Nombre de classes 5 est pour (obj, x, y, w, h)
        
        obj_mask = targets[..., 0] == 1 
        noobj_mask = targets[..., 0] == 0

        # 1. Loss Coordonnées (uniquement si objet présent)
        # Attention: s'assurer que targets contient les valeurs relatives à la cellule (0-1)
        loss_coord = self.mse(preds[..., 1:3][obj_mask], targets[..., 1:3][obj_mask]) \
                   + self.mse(preds[..., 3:5][obj_mask], targets[..., 3:5][obj_mask])
        


        class_smooth = 1-self.smooth_factor

        if n_class > 1:
            no_class_smooth = self.smooth_factor/(n_class-1)
        else:
            no_class_smooth = 0

        # 2. Loss Objectness
        loss_obj = self.bce(preds[..., 0][obj_mask], targets[..., 0][obj_mask])
        loss_noobj = self.bce(preds[..., 0][noobj_mask], targets[..., 0][noobj_mask])
        
        # 3. Loss Classes
        loss_class = 0.0
        if preds.size(-1) > 5:
            t_class =  targets[..., 5:][obj_mask]

            t_class_smoothed = t_class =  targets[..., 5:][obj_mask] * class_smooth + (1-t_class)*no_class_smooth

            loss_class = self.bce(preds[..., 5:][obj_mask], t_class_smoothed)

        # On divise par le nombre d'objets réels dans le batch, pas la taille du batch.
        # On ajoute 1e-6 pour éviter la division par zéro si batch vide.
        num_objects = obj_mask.sum().float() + 1e-6

        total = (
            self.lambda_coord * loss_coord +
            self.lambda_obj * loss_obj +
            self.lambda_noobj * loss_noobj +
            loss_class
        ) / num_objects

        return total