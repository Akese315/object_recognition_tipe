import math 
import torch
import numpy as np
import scipy
import numpy as np
from typing import List, Optional
from YOLO_loader import BoundingBox
import torch
import numpy as np
import math





def IoU(gt: BoundingBox, pred: BoundingBox) -> float:
    x1 = max(gt.x_center - gt.width/2, pred.x_center - pred.width/2)
    y1 = max(gt.y_center - gt.height/2, pred.y_center - pred.height/2)
    x2 = min(gt.x_center + gt.width/2, pred.x_center + pred.width/2)
    y2 = min(gt.y_center + gt.height/2, pred.y_center + pred.height/2)

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_gt = gt.width * gt.height
    area_pred = pred.width * pred.height
    union = area_gt + area_pred - inter
    return inter / union if union > 0 else 0.0

def batch_IoU(pred_boxes : torch.tensor , true_boxes: torch.tensor):
    
    x_center_pred_tensor = pred_boxes[...,1]
    x_center_ground_truth_tensor = true_boxes[...,1]

    y_center_pred_tensor = pred_boxes[...,2]
    y_center_ground_truth_tensor = true_boxes[...,2]

    width_pred_tensor = pred_boxes[...,3]
    width_ground_truth_tensor = true_boxes[...,3]

    height_pred_tensor = pred_boxes[...,4]
    height_ground_truth_tensor = true_boxes[...,4]

    x_1_pred_tensor = x_center_pred_tensor - width_pred_tensor/2
    x_1_ground_truth_tensor = x_center_ground_truth_tensor - width_ground_truth_tensor/2

    y_1_pred_tensor = y_center_pred_tensor - height_pred_tensor/2
    y_1_ground_truth_tensor = y_center_ground_truth_tensor - height_ground_truth_tensor/2


    x_2_pred_tensor = x_center_pred_tensor + width_pred_tensor/2    
    x_2_ground_truth_tensor = x_center_ground_truth_tensor + width_ground_truth_tensor/2

    y_2_pred_tensor = y_center_pred_tensor + height_pred_tensor/2
    y_2_ground_truth_tensor = y_center_ground_truth_tensor + height_ground_truth_tensor/2

    x_1_inter = torch.max(x_1_ground_truth_tensor,x_1_pred_tensor)
    x_2_inter = torch.min(x_2_pred_tensor,x_2_ground_truth_tensor)

    y_1_inter =torch.max(y_1_pred_tensor,y_1_ground_truth_tensor)
    y_2_inter =torch.min(y_2_pred_tensor,y_2_ground_truth_tensor)

    area_inter = (x_2_inter -x_1_inter).clamp(min=0) * (y_2_inter-y_1_inter).clamp(min=0)
    area_ground_truth = (width_ground_truth_tensor) * (height_ground_truth_tensor)
    area_prediction = (width_pred_tensor) * (height_pred_tensor)

    return area_inter/(area_prediction+area_ground_truth-area_inter + 1e-6)


def compute_norm_euclidian(ref):
    return math.sqrt((ref**2).sum())

def standardize_color(tensor):
    N, C, X, Y = tensor.shape
    if C != 3:
        raise ValueError("Dataset doit avoir 3 canaux (RGB)")
    
    tensor = tensor.float()

    red_mean = mean(tensor[0])
    green_mean = mean(tensor[1])
    blue_mean = mean(tensor[2])
    
    red_ecart = ecart_type(tensor[0],red_mean)
    green_ecart = ecart_type(tensor[1],green_mean)
    blue_ecart = ecart_type(tensor[2],blue_mean)    

    return ([red_mean,green_mean,blue_mean],[red_ecart,green_ecart,blue_ecart])

def standardize_tensor(tensor,mean,std):

    tensor = tensor.float()
    channel = tensor.shape[0]

    if len(mean) != len(std) or len(mean) != channel:
        raise ValueError("shape or size is not correct")
    

    standardized = torch.empty_like(tensor) 

    for i in range(channel):
        standardized[i] = (tensor[i]-mean[i])/(std[i]+ 1e-8)
    return standardized

def ecart_type(tensor, moyenne):
    return torch.sqrt(torch.mean(torch.pow(tensor-moyenne,2)))


def mean(tensor):
    flatten_tensor = tensor.flatten()
    return torch.mean(flatten_tensor)


def find_objects(tensor, threshold=1.0):
    mask = tensor[...,0] >=threshold
    coordinates = torch.nonzero(mask, as_tuple=False)
    return coordinates  

