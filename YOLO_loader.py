import numpy as np
import math
from typing import Tuple, Optional
import torch
from torchvision import transforms
import cv2
import io
from pathlib import Path
from IPython.display import Image, display
from PIL import Image as PILImage

class Coordinates:
    def __init__(self, x_center, y_center, width, height):
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return f"Coords({self.x_center:.2f}, {self.y_center:.2f}, {self.width:.2f}, {self.height:.2f})"

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "Coordinates":
        return cls(tensor[1].item(), tensor[2].item(), tensor[3].item(), tensor[4].item())


class BoundingBox:
    def __init__(
        self,
        is_detected: bool,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        class_id: int,
        num_classes: int,
        class_id_prob: float = 1.0,
    ):
        self.is_detected = bool(is_detected)
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.class_id = class_id
        self.class_id_prob = class_id_prob
        self.num_classes = num_classes

        self.x_center_cell = 0
        self.y_center_cell = 0



        # One-hot
        self.class_tensor = np.zeros(num_classes, dtype=np.float32)
        self.class_tensor[class_id] = 1.0

    def get_cell_position(self, grid_division:int):
        
        
        j_cell = math.floor(self.x_center * grid_division)  # colonne
        i_cell = math.floor(self.y_center *grid_division) # ligne
        if j_cell == 53:
            print("j cell 53, :", self.x_center)
        return (i_cell,j_cell)
    
    def get_coordinate(self) -> Coordinates:
        """Retourne un objet Coordinates avec les valeurs actuelles (après resize/padding)"""
        return Coordinates(self.x_center, self.y_center, self.width, self.height)
    
    def get_ratio(self):
        """Retourne le ratio de la bounding box"""
        if self.width <= 0 or self.height <= 0:
            return [0.0, 0.0]
        ratio = self.height / self.width if self.width > self.height else self.width / self.height
        ratio_vec = [1.0, ratio] if ratio <= 1 else [ratio, 1.0]
        return ratio_vec

    def _update_ratio_index(self, aspect_ratios: list[float]) -> int:
        """Recalcule dynamiquement le ratio_index à partir des valeurs actuelles"""
        ratio_vec = self.get_ratio()
        diffs = np.linalg.norm(np.array(aspect_ratios) - np.array(ratio_vec), axis=1)
        return int(np.argmin(diffs))
        
    def get_bounding_box_index(self, aspect_ratios: list[float]) -> int:
        """Retourne l'index du rapport d'aspect le plus proche (dynamique)"""
        return self._update_ratio_index(aspect_ratios)

    def get_tensor(self) -> np.ndarray:
        """Retourne le tenseur de la bounding box:\n
            0x00: objectness\n
            0x01: x_center (relative à la cellule)\n
            0x02: y_center (relative à la cellule)\n
            0x03: width (relative à l'image)\n
            0x04: height (relative à l'image)\n
            0x05: class0\n
            0x06: class1\n
            ..."""
        
        return np.concatenate([
            np.array([float(self.is_detected), self.x_center_cell, self.y_center_cell, self.width, self.height], dtype=np.float32),
            self.class_tensor
        ])

    def get_denormalized_tensor(self, target_size: Tuple[int, int], i_cell:int,j_cell:int, grid_division) -> np.ndarray:
        t = self.get_tensor().copy()
        x_scale, y_scale = target_size
        t[1] = (j_cell + t[1])/grid_division * x_scale
        t[2] = (i_cell + t[2])/grid_division * y_scale
        t[3] *= x_scale
        t[4] *= y_scale
        return t

    @classmethod
    def from_tensor(
        cls, tensor: np.ndarray, num_classes: int, i_cell:int,j_cell:int, grid_division
    ) -> "BoundingBox":
        
        """Create a Boundinx from a tensor with the following structure:\n
            0x00: objectness\n
            0x01: x_center (relative à la cellule)\n
            0x02: y_center (relative à la cellule)\n
            0x03: width (relative à l'image)\n
            0x04: height (relative à l'image)\n
            0x05: class0\n
            0x06: class1\n
            ..."""
        
        np_tensor = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
        is_detected = bool(np_tensor[0])

        x_center_cell, y_center_cell, w, h = np_tensor[1:5]

        x_center = (j_cell + x_center_cell)/grid_division
        y_center = (i_cell + y_center_cell)/grid_division
        class_prob = np.max(np_tensor[5:5+num_classes])
        class_id = int(np.argmax(np_tensor[5:5+num_classes]))
        return cls(is_detected,x_center,y_center, w, h, class_id, num_classes,class_prob)

    @classmethod
    def from_file(cls,array:np.ndarray,num_classes:int)-> "BoundingBox":
        class_id = int(array[0])
        x, y, w, h = array[1:5]
        is_detected = True
        return cls(is_detected, x, y, w, h, class_id, num_classes)

    def __repr__(self):
        return f"Box(det={self.is_detected}, c=({self.x_center:.5f},{self.y_center:.5f}), size={self.width:.5f}x{self.height:.5f}, class={self.class_id})"


class Label:
    def __init__(self,file_name:str,num_classes:int):
        self.num_classes = num_classes
        self.bb_boxes = self.read_data(file_name)

    def read_data(self,file_name:str):
        boxes = []
        with open(file_name, "r") as f:
            for line in f:
                boxes.append(BoundingBox.from_file(np.array(list(map(float, line.split()))),self.num_classes))
        return boxes

    def get_bounding_boxes(self) -> list[BoundingBox]:
        return self.bb_boxes
    

class   CustomImage:
    def __init__(
        self,
        image_tensor: torch.Tensor,
        bounding_boxes: list[BoundingBox],
        target_size: Tuple[int, int] = (500, 500),
        grid_division = 52,
        mean: Optional[list[float]] = None,
        std: Optional[list[float]] = None
        ):

        self.target_size = target_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        self.grid_division = grid_division

        # Chargement
        
        self._image_tensor = image_tensor

        self._original_size = (self._image_tensor.size(1),self._image_tensor.size(2)) # (H,W)
        self._bb_boxes = bounding_boxes
        # Pré-traitement
        self._scale = 1.0
        self._pad = (0, 0)  # (left, top)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        # Appliquer letterbox
        self._apply_letterbox()

    def _apply_letterbox(self):
        w, h = self._original_size
        r = min(self.target_size[0] / w, self.target_size[1] / h)
        new_w, new_h = int(round(w * r)), int(round(h * r))

        # Resize 
        self.resize_image(self.target_size)
        # Padding centré
        pad_w = self.target_size[0] - new_w
        pad_h = self.target_size[1] - new_h
        left = pad_w // 2
        top = pad_h // 2
        padding = (left,top)

        self.apply_padding(padding,self.target_size)
        # Mise à jour
        self._scale = r
        self._pad = (left, top)

        #making the center of the objective relative to the cell
        
        for box in self._bb_boxes:
            coordinates = box.get_cell_position(self.grid_division)
            box.x_center_cell = (box.x_center*self.grid_division) - coordinates[1]
            box.y_center_cell = (box.y_center*self.grid_division) - coordinates[0]

    def set_std_mean(self, std: list[float], mean: list[float]):
        self._std = std
        self._mean = mean
        self.normalize = transforms.Normalize(mean=self._mean, std=self._std)

    def resize_image(self, target_size : Tuple[int,int]) -> torch.Tensor:
        """Resize the image to the target size"""
        # torch.Tensor (3, H, W) → numpy (H, W, 3)
        image_np = (self._image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image_pil = PILImage.fromarray(image_np)
        resized = image_pil.resize((target_size[0], target_size[1]), PILImage.Resampling.LANCZOS)

        # Reconvertir en tensor (3, H, W)
        self._image_tensor = self.to_tensor(resized)

    def apply_padding(self,padding:Tuple[int,int],target_size:Tuple[int,int]) -> torch.Tensor:
        """Apply a padding to the image"""
        if padding == (0, 0):
            return
        # convertir le tensor → PIL avant le collage
        image_np = (self._image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image_pil = PILImage.fromarray(image_np)

        img_padded = PILImage.new("RGB", target_size, (114, 114, 114))
        left, top = padding
        img_padded.paste(image_pil, (left, top))

        # reconvertir PIL → Tensor
        self._image_tensor = self.to_tensor(img_padded)

    def get_raw_tensor(self) -> torch.Tensor:
        return self._image_tensor
    
    def get_bounding_boxes(self) -> list[BoundingBox]:
        return self._bb_boxes

    def get_bounding_boxes_tensors(self) -> torch.Tensor:
        tensors = [box.get_normalized_tensor(self.target_size[0]) for box in self._bb_boxes]
        return torch.from_numpy(np.stack(tensors)) if tensors else torch.zeros((0, 5 + self._bb_boxes[0].num_classes))

    def show_image(self, predicted_bb_boxes: list[BoundingBox] = None):
        image = self.get_image(predicted_bb_boxes)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        display(Image(data=buf.getvalue()))

    def get_image(self, predicted_bb_boxes: list[BoundingBox] = None)->PILImage:
        img = (self._image_tensor.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        img = np.ascontiguousarray(img)


        # GT (bleu)
        for box in self._bb_boxes:
            cell_position = box.get_cell_position(self.grid_division)
            coordinates = Coordinates.from_tensor(box.get_denormalized_tensor(self.target_size,cell_position[0],cell_position[1],self.grid_division))
            x1 = int(coordinates.x_center - coordinates.width / 2)
            y1 = int(coordinates.y_center - coordinates.height / 2)
            x2 = int(coordinates.x_center + coordinates.width / 2)
            y2 = int(coordinates.y_center + coordinates.height / 2)


            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"GT {box.class_id} p={box.class_id_prob:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Prédictions (vert)
        if predicted_bb_boxes is not None:
            for box in predicted_bb_boxes:
                cell_position = box.get_cell_position(self.grid_division)
                coordinates = Coordinates.from_tensor(box.get_denormalized_tensor(self.target_size,cell_position[0],cell_position[1],self.grid_division))
                x1 = int(coordinates.x_center - coordinates.width / 2)
                y1 = int(coordinates.y_center - coordinates.height / 2)
                x2 = int(coordinates.x_center + coordinates.width / 2)
                y2 = int(coordinates.y_center + coordinates.height / 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"Pred {box.class_id} p={box.class_id_prob:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        image = PILImage.fromarray(img)
        return image

    def __repr__(self):
        return f"CustomImage({self.file_name}, boxes={len(self._bb_boxes)}, size={self._original_size} → {self.target_size})"


def get_classes(directory:str):
    file_name = directory+"/classes.txt"
    classes = []
    with open(file_name, "r") as f:
            for line in f:
                classes.append(line.strip())
    return classes

def get_images_file_name(directory:str):
    folder_str = directory+"/images/"
    folder = Path(folder_str)
    files = sorted([f.name for f in folder.iterdir() if f.is_file()])
    return files
            
def get_labels(directory:str) -> list[Label]:
    classes = get_classes(directory)
    folder_str = directory+"/labels/"
    folder = Path(folder_str)
    files = sorted([f for f in folder.iterdir() if f.is_file()])
    return [Label(file,len(classes)) for file in files]

def get_label(directory:str, file_name : str,num_classes :int):
    file = directory+"/labels/"+file_name
    return Label(file,num_classes)