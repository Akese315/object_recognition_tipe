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
from torch.utils.data import Sampler
import random

class CardRecognitionDataset(Dataset):
    def __init__(self, directory, is_dark_and_white=False,reduction_factor=8, bounding_boxes_ratio =[], max_width =800, max_height=800):  
        self.mean = [0.485, 0.456, 0.406] # mean ImageNet values
        self.std = [0.229, 0.224, 0.225] # standard ImageNet values

        self.transform = Albu.Compose([
            # --- 1. transformations géométriques de base ---
            Albu.HorizontalFlip(p=0.5),
            Albu.VerticalFlip(p=0.2),
            
            Albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=45, border_mode=0, p=0.7),

            # --- 2. distorsions ---
            Albu.OneOf([
                Albu.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1),
                Albu.GridDistortion(num_steps=5, distort_limit=0.05, p=1),
                Albu.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1),
            ], p=0.3),

            # --- 7. regularization (trous) ---
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

            # --- 3. qualité et texture ---
            Albu.OneOf([
                Albu.ImageCompression(quality_lower=85, quality_upper=95, p=1),
                Albu.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1), # CORRECTION ICI (pas d'espace)
                Albu.ToGray(p=1),
            ], p=0.2),

            # --- 4. couleur et luminosité ---
            Albu.OneOf([
                Albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                Albu.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
                Albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            ], p=0.5),

            # --- 5. flou et netteté ---
            Albu.OneOf([
                Albu.GaussianBlur(blur_limit=(3, 7), p=1),
                Albu.MotionBlur(blur_limit=3, p=1),
                Albu.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            ], p=0.3),

            # --- 6. conditions climatiques ---
            Albu.OneOf([
                Albu.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1),
                Albu.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=1),
                Albu.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, alpha_coef=0.1, p=1),
            ], p=0.1),

           

        ], bbox_params=Albu.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.0))
        if len(directory) == 0:
            raise ValueError("Le dossier est vide !")
        
        if len(bounding_boxes_ratio) == 0:
            raise ValueError("Pas de bounding box ratio !")
        
        classes = get_classes(directory)
        file_names = get_images_file_name(directory)
        N = len(file_names)
        A = len(bounding_boxes_ratio)
        
        C = len(classes) 
        if is_dark_and_white:
            print("dark and white")
        
        
        self.custom_images : list[CustomImage] = []
        self.length = N
        self.n_classes = C
        self.bboxes_ratio = bounding_boxes_ratio
        self.reduction_factor = reduction_factor
        self.image_size_groups = {}
        self.max_width = max_width
        self.max_height = max_height

        for idx in tqdm(range(N), desc="Traitement des labels", leave=True):
            file_name = file_names[idx]
            label_file_name = file_name.split(".")[0] + ".txt"
            label = get_label(directory,label_file_name,C)
            bounding_boxes = label.get_bounding_boxes()
            file_path = directory+"/images/"+file_name
            image_pil = PILImage.open(file_path).convert("RGB")
            target_size = (int(image_pil.size[0]//self.reduction_factor -1)*self.reduction_factor,
                            int(image_pil.size[1]//self.reduction_factor -1)*self.reduction_factor)
             # ajuste à la grille la plus proche en dessous
            if image_pil.size[0] > self.max_width or image_pil.size[1] > self.max_height:
                target_size = (int(self.max_width//self.reduction_factor -1)*self.reduction_factor,
                            int(self.max_height//self.reduction_factor -1)*self.reduction_factor)
            #si la taille est supérieure à la max_width ou max_height autorisé, alors on set la target size à la max_size
            image = CustomImage(file_name=file_path,bounding_boxes=bounding_boxes,target_size=target_size,reduction_factor=self.reduction_factor)
            if target_size not in self.image_size_groups:
                self.image_size_groups[target_size] = []
            self.image_size_groups[target_size].append(idx)
            self.custom_images.append(image)
       
     

        print(f"Mean: {self.mean}")
        print(f"Std:  {self.std}")

    def get_mean(self):
        return self.mean
    
    def get_std(self):
        return self.std
    
    def get_classes(self):
        return self.n_classes
    
    def get_image_size_groups(self)->dict:
        return self.image_size_groups

    def __len__(self):
        return self.length

    def transform_image(self,image:CustomImage)->torch.Tensor:
        image_tensor = image.get_raw_tensor()
        grid_division_x, grid_division_y = image.get_grid_division()

        image_np = np.transpose(image_tensor.numpy(), (1, 2, 0)) 
        
        bb_boxes = image.get_bounding_boxes()
        grid_division_x, grid_division_y = image.get_grid_division()
        label = torch.zeros(grid_division_y, grid_division_x, len(self.bboxes_ratio), 5 + self.n_classes)
        
        bb_boxes_yolo_format = []
        bb_boxes_classes = []
        for box in bb_boxes:
            coordinates = box.get_coordinate()
            
            # --- ÉTAPE 1 : Récupérer les valeurs brutes ---
            xc, yc, w, h = coordinates.x_center, coordinates.y_center, coordinates.width, coordinates.height
            
            # --- ÉTAPE 2 : Convertir en coins pour gérer les dépassements ---
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            
            # --- ÉTAPE 3 : "Clip" (Couper ce qui dépasse) ---
            x1 = np.clip(x1, 0.0, 1.0)
            y1 = np.clip(y1, 0.0, 1.0)
            x2 = np.clip(x2, 0.0, 1.0)
            y2 = np.clip(y2, 0.0, 1.0)
            
            # --- ÉTAPE 4 : Recalculer le format YOLO avec les valeurs corrigées ---
            w_new = x2 - x1
            h_new = y2 - y1
            xc_new = x1 + w_new / 2
            yc_new = y1 + h_new / 2
            
            # (Si une boîte était entièrement hors de l'image, w_new serait 0)
            if w_new > 0 and h_new > 0:
                box_np = np.array([xc_new, yc_new, w_new, h_new])
                bb_boxes_classes.append(box.class_id)
                bb_boxes_yolo_format.append(box_np)

        bb_boxes_yolo_format = np.array(bb_boxes_yolo_format)

        
        
        augmented = self.transform(image=image_np, bboxes=bb_boxes_yolo_format, class_labels=bb_boxes_classes)

        aug_image = image_tensor = transforms.ToTensor()(augmented['image']) # back to tensor [C,H,W]
        aug_boxes = augmented['bboxes']
        #reconstruit le tenseur [Objectness, x_center,y_center, width,height,c1,...,cn]

        '''if len(aug_boxes) != len(bb_boxes):
            print(len(aug_boxes),"/",len(bb_boxes))'''


        boxes = []
        for i in range(len(aug_boxes)):
            box = bb_boxes[i]
            anchor_index = box.get_bounding_box_index(self.bboxes_ratio)
            x_center,y_center,width,height =torch.tensor(aug_boxes[i])
            box = BoundingBox(True,x_center,y_center,width,height,box.class_id,box.num_classes)
            new_coordinates = box.get_cell_position(grid_division_x, grid_division_y)
            anchor_index = box.get_bounding_box_index(self.bboxes_ratio)
            #need to set objectness to 1 and class probabilities to one-hot encoding
            label[new_coordinates[0],new_coordinates[1],anchor_index,0] = 1.0 # objectness
            label[new_coordinates[0],new_coordinates[1],anchor_index,1:5] = torch.tensor(aug_boxes[i])
            label[new_coordinates[0],new_coordinates[1],anchor_index,5 + box.class_id] = 1.0 # class probability one-hot
            boxes.append(box)

        '''new_image = CustomImage.from_tensor(image_tensor=aug_image,file_name=image.file_name,bounding_boxes=boxes,
                                       target_size=image.target_size,reduction_factor=self.reduction_factor)'''
        return aug_image,label


    def __getitem__(self, index) -> tuple[CustomImage,torch.Tensor]:
        image:CustomImage = self.custom_images[index]
        return self.transform_image(image)

class CustomBatchSampler(Sampler):
    def __init__(self, dataset: CardRecognitionDataset, batch_size: int,groups:Optional[dict]=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batches = []
        cleaned_groups = {k: v for k, v in groups.items() if v}
        self.image_size_group = cleaned_groups

        self._create_batches()

    def _create_batches(self):
        # create batches renvoie uniquement des tableaux d'indices

        self.batches = []
        # Pour chaque groupe de taille (ex: 1024x768)
        for size, indices in self.image_size_group.items():
            # On mélange les indices dans ce groupe
            random.shuffle(indices)
            
            # Si pas assez d'images pour faire un batch complet ?
            # Option choisie : On complète avec des doublons (Data Augmentation fera le reste)
            count = len(indices)
            if count < self.batch_size:
                # On duplique les indices existants jusqu'à remplir le batch
                # Ex: indices=[1], batch=4 -> [1, 1, 1, 1]
                extended_indices = indices * (self.batch_size // count + 1)
                indices = extended_indices[:self.batch_size]
            
            # Création des chunks de taille batch_size
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                
                # Si le dernier morceau est trop petit, on le jette ou on le complète
                # Ici on le complète (drop_last=False logic)
                if len(batch) < self.batch_size and len(batch) > 0:
                    needed = self.batch_size - len(batch)
                    # On complète avec des images aléatoires DU MÊME GROUPE
                    extras = random.choices(indices, k=needed)
                    batch.extend(extras)
                
                if len(batch) == self.batch_size:
                    self.batches.append(batch) 

    def __iter__(self):
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

def custom_collate_fn(batch) -> Tuple[List[CustomImage], torch.Tensor]:
    images = torch.stack([item[0] for item in batch],dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)

    return images, labels