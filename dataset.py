from YOLO_loader import get_images_file_name,get_classes,get_label
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from YOLO_loader import CustomImage,BoundingBox
import torch.nn as nn
from torchvision.transforms import v2
import albumentations as Albu
from PIL import Image as PILImage
from torchvision import transforms
from tqdm.auto import tqdm


class CardRecognitionDataset(Dataset):
    def __init__(self, directory, is_dark_and_white=False, n_cells = 3,target_size : Tuple[int] = (416,416), bounding_boxes_ratio =[]):  
        
        self.transform = Albu.Compose([
            # --- 1. Transformations Géométriques de base ---
            Albu.HorizontalFlip(p=0.5),
            Albu.VerticalFlip(p=0.2),
            
            Albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=45, border_mode=0, p=0.7),

            # --- 2. Distorsions ---
            Albu.OneOf([
                Albu.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1),
                Albu.GridDistortion(num_steps=5, distort_limit=0.05, p=1),
                Albu.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1),
            ], p=0.3),

            # --- 3. Qualité et Texture ---
            Albu.OneOf([
                Albu.ImageCompression(quality_lower=85, quality_upper=95, p=1),
                Albu.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1), # CORRECTION ICI (pas d'espace)
                Albu.ToGray(p=1),
            ], p=0.2),

            # --- 4. Couleur et Luminosité ---
            Albu.OneOf([
                Albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                Albu.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
                Albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            ], p=0.5),

            # --- 5. Flou et Netteté ---
            Albu.OneOf([
                Albu.GaussianBlur(blur_limit=(3, 7), p=1),
                Albu.MotionBlur(blur_limit=3, p=1),
                Albu.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            ], p=0.3),

            # --- 6. Conditions Climatiques ---
            Albu.OneOf([
                Albu.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1),
                Albu.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=1),
                Albu.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, alpha_coef=0.1, p=1),
            ], p=0.1),

            # --- 7. Regularization (Trous) ---
            Albu.CoarseDropout(
                max_holes=8, 
                max_height=32, 
                max_width=32, 
                min_holes=1, 
                min_height=8, 
                min_width=8, 
                fill_value=0, 
                p=0.5
            ),

        ], bbox_params=Albu.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
        if len(directory) == 0:
            raise ValueError("Le dossier est vide !")
        
        if len(bounding_boxes_ratio) == 0:
            raise ValueError("Pas de bounding box ratio !")
        
        classes = get_classes(directory)
        file_names = get_images_file_name(directory)
        N = len(file_names)
        A = len(bounding_boxes_ratio)
        H = target_size[1]
        W = target_size[0]  
        C = len(classes) 
        S = n_cells
        if is_dark_and_white:
            print("dark and white")
        self.features = torch.zeros((N,3,H, W), dtype=torch.float32)
        self.labels = torch.zeros((N,S,S,A,5+C), dtype=torch.float32)
        
        self.custom_images : list[CustomImage] = []
        self.length = N
        self.n_classes = C
        self.n_cells = S
        self.bboxes_ratio = bounding_boxes_ratio


        for i in tqdm(range(N), desc="Traitement des labels", leave=True):
            file_name = file_names[i]
            label = get_label(directory,file_name.replace(".jpg",".txt"),C)
            bounding_boxes = label.get_bounding_boxes()
            file_path = directory+"/images/"+file_name
            image_pil = PILImage.open(file_path).convert("RGB")
            image_tensor = transforms.ToTensor()(image_pil)
            image = CustomImage(image_tensor,bounding_boxes,target_size)
            
            for bounding_box in bounding_boxes:
                bb_box_index = bounding_box.get_bounding_box_index(bounding_boxes_ratio)
                i_cell,j_cell =  bounding_box.get_cell_position(S)
                label = torch.tensor(bounding_box.get_tensor(), dtype=torch.float32)
                print("i cell : ",i_cell, "j cell :",j_cell)

                self.labels[int(i), int(i_cell), int(j_cell), int(bb_box_index)] = label

            self.features[i] = image.get_raw_tensor()
            self.custom_images.append(image)
       
        self.features_pil = self.features.clone()
        self.mean = self.features.mean(dim=[0, 2, 3]).tolist()  # [3]
        self.std  = self.features.std(dim=[0, 2, 3]).tolist()   # [3]

        print(f"Mean: {self.mean}")
        print(f"Std:  {self.std}")

        for img in self.custom_images:
            img.set_std_mean(std=self.std, mean=self.mean)
            '''self.features[i] = img.get_normalized_image()
            img.show_image()'''

    def get_mean(self):
        return self.mean
    
    def get_std(self):
        return self.std
    
    def get_classes(self):
        return self.n_classes

    def __len__(self):
        return self.length

    def __getitem__(self, index) -> tuple[CustomImage,torch.Tensor]:

        image:CustomImage = self.custom_images[index]
        image_tensor = image.get_raw_tensor()

        image_np = np.transpose(image_tensor.numpy(), (1, 2, 0)) 
        bb_boxes = image.get_bounding_boxes()
        label = self.labels[index].clone()
        
        bb_boxes_yolo_format = []
        bb_boxes_classes = []
        for box in bb_boxes:
            coordinates = box.get_coordinate()
            box_np = np.array([coordinates.x_center,coordinates.y_center,coordinates.width,coordinates.height])
            bb_boxes_classes.append(box.class_id)
            bb_boxes_yolo_format.append(box_np)
        bb_boxes_yolo_format = np.array(bb_boxes_yolo_format)
        
        augmented = self.transform(image=image_np, bboxes=bb_boxes_yolo_format, class_labels=bb_boxes_classes)

        aug_image = image_tensor = transforms.ToTensor()(augmented['image'])
        aug_boxes = augmented['bboxes']


        #reconstruit le tenseur [Objectness, x_center,y_center, width,height,c1,...,cn]

        if len(aug_boxes) != len(bb_boxes):
            print(len(aug_boxes),"/",len(bb_boxes))


        boxes = []
        for i in range(len(aug_boxes)):
            box = bb_boxes[i]
            anchor_index = box.get_bounding_box_index(self.bboxes_ratio)
            x_center,y_center,width,height =torch.tensor(aug_boxes[i])
            box = BoundingBox(True,x_center,y_center,width,height,box.class_id,box.num_classes)
            new_coordinates = box.get_cell_position(self.n_cells)
            anchor_index = box.get_bounding_box_index(self.bboxes_ratio)
            label[new_coordinates[0],new_coordinates[1],anchor_index,1:5] = torch.tensor(aug_boxes[i])
            boxes.append(box)


        image = CustomImage(aug_image,boxes,image.target_size,image.grid_division,image.mean, image.std)

        return image, label

def custom_collate_fn(batch):
    custom_images = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch], dim=0)
    images = torch.stack([img.get_raw_tensor() for img in custom_images])  # [B, C, H, W]

    return custom_images, images, labels