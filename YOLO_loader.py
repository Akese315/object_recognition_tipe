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
        objectness: float,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        class_id: int,
        num_classes: int,
        class_id_prob: float = 1.0,
    ):
        self.objectness = objectness
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

    def get_cell_position(self, grid_division_x:int, grid_division_y:int):
        j_cell = math.floor(self.x_center * grid_division_x)  # colonne
        i_cell = math.floor(self.y_center *grid_division_y) # ligne
        if self.x_center < 0.0 or self.x_center> 1.0:
            print("Warning: x_center or y_center > 1.0",self.x_center,self.y_center)
        return (i_cell,j_cell)
    
    def get_objectness(self) -> float:
        """Retourne la confiance d'objet (objectness)"""
        return float(self.objectness)
    
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
            np.array([float(self.objectness), self.x_center_cell, self.y_center_cell, self.width, self.height], dtype=np.float32),
            self.class_tensor
        ])

    def get_denormalized_tensor(self, target_size: Tuple[int, int], i_cell:int,j_cell:int, grid_division_x:int, grid_division_y:int) -> np.ndarray:
        t = self.get_tensor().copy()
        x_scale, y_scale = target_size
        t[1] = (j_cell + t[1])/grid_division_x * x_scale
        t[2] = (i_cell + t[2])/grid_division_y * y_scale
        t[3] *= x_scale
        t[4] *= y_scale
        return t

    @classmethod
    def from_tensor(
        cls, tensor: np.ndarray, num_classes: int, i_cell:int,j_cell:int, grid_division_x:int, grid_division_y:int
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
        objectness = np_tensor[0]

        x_center_cell, y_center_cell, w, h = np_tensor[1:5]
        x_center = (j_cell + x_center_cell)/grid_division_x
        y_center = (i_cell + y_center_cell)/grid_division_y
        class_prob = np.max(np_tensor[5:5+num_classes])
        class_id = int(np.argmax(np_tensor[5:5+num_classes]))
        return cls(objectness,x_center,y_center, w, h, class_id, num_classes,class_prob)

    @classmethod
    def from_file(cls,array:np.ndarray,num_classes:int)-> "BoundingBox":
        class_id = int(array[0])
        x, y, w, h = array[1:5]
        objectness = 1.0
        return cls(objectness, x, y, w, h, class_id, num_classes)

    def __repr__(self):
        return f"Box(det={self.objectness}, c=({self.x_center:.5f},{self.y_center:.5f}), size={self.width:.5f}x{self.height:.5f}, class={self.class_id})"


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
    

class CustomImage:
    def __init__(
        self,
        file_name: str,
        bounding_boxes: list[BoundingBox],
        target_size: Tuple[int, int] = (500, 500),
        reduction_factor: int = 8,
        ):

        self.target_size = target_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.reduction_factor = reduction_factor
        self.grid_division_x = target_size[0] // reduction_factor
        self.grid_division_y = target_size[1] // reduction_factor
        self.file_name = file_name
        self._image_tensor: Optional[torch.Tensor] = None

        # Chargement
        
        self.to_tensor = transforms.ToTensor()
        #self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        
        self._bb_boxes = bounding_boxes
        # Pré-traitement
        self._scale = 1.0
        self._pad = (0, 0)  # (left, top)
        

    def load_image(self) -> torch.Tensor:
        # Charger l'image depuis le fichier

        image_pil = PILImage.open(self.file_name).convert("RGB")

        image_tensor = self.to_tensor(image_pil)  # [C, H, W]
        self._original_size = (image_tensor.size(2),image_tensor.size(1)) # (W,H)
        image_tensor= self._apply_letterbox(image_tensor=image_tensor)
        if image_tensor is None:
            raise Exception("Image_tensor is none : file_name :",self)
        return image_tensor

    def _apply_letterbox(self,image_tensor):
        

        image_tensor = self.resize_image(self.target_size,image_tensor=image_tensor)
        if image_tensor is None:
            raise Exception("Image_tensor is none after resize : file_name :",self)
        
        for box in self._bb_boxes:
            coordinates = box.get_cell_position(self.grid_division_x, self.grid_division_y)
            box.x_center_cell = (box.x_center*self.grid_division_x) - coordinates[1]
            box.y_center_cell = (box.y_center*self.grid_division_y) - coordinates[0]
        
        return image_tensor

    def set_std_mean(self, std: list[float], mean: list[float]):
        self._std = std
        self._mean = mean
        self.normalize = transforms.Normalize(mean=self._mean, std=self._std)

    def resize_image(self, target_size : Tuple[int,int],image_tensor: torch.Tensor) -> torch.Tensor:
        """Resize the image to the target size"""
        if image_tensor is None:
            raise Exception("Image tensor is None. Load the image first.")


        # torch.Tensor (3, H, W) → numpy (H, W, 3)
        image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image_pil = PILImage.fromarray(image_np)
        resized = image_pil.resize((target_size[0], target_size[1]), PILImage.Resampling.LANCZOS)

        # Reconvertir en tensor (3, H, W)
        image_tensor = self.to_tensor(resized)
        return image_tensor

    def apply_padding(self,padding:Tuple[int,int],target_size:Tuple[int,int],image_tensor: torch.Tensor) -> torch.Tensor:
        """Apply a padding to the image"""
        if image_tensor is None:
            raise Exception("Image tensor is None. Load the image first.")
        if padding == (0, 0):
            return
        # convertir le tensor → PIL avant le collage
        image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image_pil = PILImage.fromarray(image_np)

        img_padded = PILImage.new("RGB", target_size, (114, 114, 114))
        left, top = padding
        img_padded.paste(image_pil, (left, top))

        # reconvertir PIL → Tensor
        image_tensor = self.to_tensor(img_padded)
        return image_tensor

    def get_raw_tensor(self) -> torch.Tensor:
        if self._image_tensor is not None:
            return self._image_tensor
        else:
            return self.load_image()
    
    def get_bounding_boxes(self) -> list[BoundingBox]:
        return self._bb_boxes

    def get_bounding_boxes_tensors(self) -> torch.Tensor:
        tensors = [box.get_normalized_tensor(self.target_size[0]) for box in self._bb_boxes]
        return torch.from_numpy(np.stack(tensors)) if tensors else torch.zeros((0, 5 + self._bb_boxes[0].num_classes))

    def show_image(self, predicted_bb_boxes: list[BoundingBox] = None):
        if predicted_bb_boxes is None:
            raise Exception("No predicted bounding boxes provided.")
        image = self.get_image(predicted_bb_boxes)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        display(Image(data=buf.getvalue()))

    def get_image(self, predicted_bb_boxes: list[BoundingBox] = None, objectness_strict: bool = True)->PILImage:
        image_tensor = self.get_raw_tensor()

        img = (image_tensor.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        img = np.ascontiguousarray(img)
    

        # GT (bleu)
        for box in self._bb_boxes:
            cell_position = box.get_cell_position(self.grid_division_x, self.grid_division_y)
            coordinates = Coordinates.from_tensor(box.get_denormalized_tensor(self.target_size,cell_position[0],cell_position[1],self.grid_division_x, self.grid_division_y))
            x1 = int(coordinates.x_center - coordinates.width / 2)
            y1 = int(coordinates.y_center - coordinates.height / 2)
            x2 = int(coordinates.x_center + coordinates.width / 2)
            y2 = int(coordinates.y_center + coordinates.height / 2)


            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"GT {box.class_id} p={box.class_id_prob:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if predicted_bb_boxes is not None:
            if objectness_strict:
                objectnesses = []
                for box in predicted_bb_boxes:
                    objectnesses.append(box.get_objectness())
                max_objectness_index = np.argmax(np.array(objectnesses))
                selected_box = predicted_bb_boxes[max_objectness_index]
                cell_position = selected_box.get_cell_position(self.grid_division_x, self.grid_division_y)
                coordinates = Coordinates.from_tensor(selected_box.get_denormalized_tensor(self.target_size,cell_position[0],cell_position[1],self.grid_division_x, self.grid_division_y))
                x1 = int(coordinates.x_center - coordinates.width / 2)
                y1 = int(coordinates.y_center - coordinates.height / 2)
                x2 = int(coordinates.x_center + coordinates.width / 2)
                y2 = int(coordinates.y_center + coordinates.height / 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"Pred {selected_box.class_id} p={selected_box.class_id_prob:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Prédictions (vert)
            else:
                for box in predicted_bb_boxes:
                    cell_position = box.get_cell_position(self.grid_division_x, self.grid_division_y)
                    coordinates = Coordinates.from_tensor(box.get_denormalized_tensor(self.target_size,cell_position[0],cell_position[1],self.grid_division_x, self.grid_division_y))
                    x1 = int(coordinates.x_center - coordinates.width / 2)
                    y1 = int(coordinates.y_center - coordinates.height / 2)
                    x2 = int(coordinates.x_center + coordinates.width / 2)
                    y2 = int(coordinates.y_center + coordinates.height / 2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"Pred {box.class_id} p={box.class_id_prob:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            image = PILImage.fromarray(img)
        return image
    
    def get_grid_division(self) -> Tuple[int,int]:
        return (self.grid_division_x, self.grid_division_y)
    
    def from_file_and_label(cls, directory: str, file_name: str, num_classes: int, target_size: Tuple[int, int], reduction_factor: int) -> 'CustomImage':
        """Constructeur alternatif à partir d'un nom de fichier et d'un dossier."""
        
        file_path = directory + "/images/" + file_name
        label = get_label(directory, file_name.replace(".jpg", ".txt"), num_classes)
        bounding_boxes = label.get_bounding_boxes()
        
        return cls(
            file_path=file_path,
            bounding_boxes=bounding_boxes,
            target_size=target_size,
            reduction_factor=reduction_factor,
            file_name=file_name
        )
    
    @classmethod
    def from_tensor(
        cls, 
        image_tensor: torch.Tensor, 
        target_size: Tuple[int, int], 
        reduction_factor: int, 
        bounding_boxes: list[BoundingBox] = [], 
        file_name: Optional[str] = None,
       
    ) -> 'CustomImage':
        """
        Constructeur pour une image déjà chargée comme tenseur (e.g., pour l'inférence/prédiction).
        Ce constructeur ne nécessite pas de chargement lazy.
        """
        
        # 1. Créer une instance avec un chemin factice (pour satisfaire l'__init__ qui attend file_path)
        
        # L'image de prédiction n'a pas de Ground Truth, donc bounding_boxes est vide
        image = cls(
            file_name=file_name,
            bounding_boxes=bounding_boxes, 
            target_size=target_size,
            reduction_factor=reduction_factor,
        )
        
        # 2. Remplacer le tenseur interne et la taille originale
        image._image_tensor = image_tensor
        # Le tenseur est en [C, H, W]. On utilise [2] pour W et [1] pour H.
        image._original_size = (image_tensor.size(2), image_tensor.size(1)) 
        # 3. Appliquer le prétraitement (redimensionnement et padding) au tenseur préchargé
        image._apply_letterbox(image_tensor) 
        
        return image

    def __repr__(self):
        if self.file_name:
            return f"CustomImage(file='{self.file_name}', boxes={len(self._bb_boxes)}, size={self._original_size} → {self.target_size})"
        else:
            return f"CustomImage(boxes={len(self._bb_boxes)}, size={self._original_size} → {self.target_size})"


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