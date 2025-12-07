import torch
import os
import datetime
from typing import Optional
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import *
from YOLO_loader import BoundingBox, CustomImage
from utils import find_objects
import math
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import io
from PIL import Image
import time

def run_one_epoch(loader, model, loss_fn, optimizer, scheduler, device, N_CLASS, reduction_factor, centers, N_ANCHORS, train=True, show=False):
    model.train(train)
    model.to(device)
    
    running_loss = 0.0
    # On ajoute "iou" aux composants suivis
    running_components = {"coord": 0.0, "obj": 0.0, "noobj": 0.0, "class": 0.0, "iou": 0.0}
    
    preds, targets = [], []
    start_time = time.time()

    loop = tqdm(loader, desc="Train" if train else "Val", leave=True)

    for images, labels in loop:
        inputs = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(inputs)
            
            loss, components = loss_fn(outputs, labels)

            if train:
                loss.backward()
                
                # Gradient Clipping (Conseillé pour YOLO pour éviter les explosions)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                
                optimizer.step()
                if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()

            # --- Visualisation (Optionnel) ---
            if show:
                try:
                    copied_labels = labels.detach().cpu()
                    copied_outputs = model.predict(outputs).detach().cpu()
                    copied_inputs = inputs.detach().cpu()
                    
                    B_batch, _, _, _, _ = copied_outputs.shape
                    H, W = copied_inputs.shape[2], copied_inputs.shape[3]
                    vis_images = []
                    
                    for i in range(min(B_batch, 4)):
                        input_boxes = []
                        indices = find_objects(copied_labels[i])
                        _, grid_div_y, grid_div_x, _, _ = copied_outputs.shape
                        
                        for indice in indices:
                            input_boxes.append(BoundingBox.from_tensor(
                                copied_labels[i, indice[0], indice[1], indice[2], :],
                                N_CLASS, indice[0], indice[1], grid_div_x, grid_div_y
                            ))

                        custom_img = CustomImage.from_tensor(
                            image_tensor=inputs[i],
                            bounding_boxes=input_boxes,
                            target_size=(W, H),
                            reduction_factor=reduction_factor
                        )
                        
                        output_bb_boxes = []
                        
                        for bb_box in custom_img.get_bounding_boxes():
                            position = bb_box.get_cell_position(grid_div_x, grid_div_y)
                            x, y = int(position[0]), int(position[1])
                            
                            output_bb_boxes.append([
                                BoundingBox.from_tensor(
                                    copied_outputs[i, x, y, a, :],
                                    N_CLASS, x, y, grid_div_x, grid_div_y
                                ) for a in range(N_ANCHORS)
                            ])
                        
                        output_bb_boxes = np.array(output_bb_boxes)
                        vis_images.append(custom_img.get_image(output_bb_boxes))

                    if len(vis_images) > 0:
                        vis_images_np = np.array(vis_images)
                        grid_size = math.ceil(math.sqrt(len(vis_images)))
                        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
                        if isinstance(axes, np.ndarray):
                            axes = axes.flatten()
                        else:
                            axes = [axes]
                            
                        for i, ax in enumerate(axes):
                            if i < len(vis_images):
                                ax.imshow(vis_images_np[i])
                            ax.axis('off')
                        plt.tight_layout()
                        plt.show()
                        
                except Exception as e:
                    print(f"Erreur visualisation : {e}")

        running_loss += loss.item()
        
        for k, v in components.items():
            if k in running_components:
                running_components[k] += v

        loop.set_postfix(loss=loss.item(), iou=components.get('iou', 0.0))

        if not train:
            preds.append(outputs.detach().cpu())
            targets.append(labels.detach().cpu())

    # Fin de l'époque
    epoch_loss = running_loss / len(loader)
    epoch_components = {k: v / len(loader) for k, v in running_components.items()}

    # Enregistrement historique
    if train:
        if not hasattr(model, 'loss_history'):
            # On s'assure que toutes les clés existent
            model.loss_history = {k: [] for k in running_components.keys()}
            model.loss_history["total"] = []

        model.loss_history["total"].append(epoch_loss)
        
        for k, v in epoch_components.items():
            # Initialisation lazy si une nouvelle clé (comme 'iou') apparaît
            if k not in model.loss_history:
                model.loss_history[k] = []
            model.loss_history[k].append(v)

    return epoch_loss


def get_plot_loss(show=True, model:Optional[nn.Module] = None) -> Image:
    if not hasattr(model, 'loss_history'):
        model.loss_history = {"total": [], "coord": [], "obj": [], "noobj": [], "class": []}  
    
    clear_output(wait=True)
    plt.figure(figsize=(12, 8))
    
    # Total Loss
    plt.subplot(2, 3, 1)
    plt.plot(model.loss_history["total"], label="Total Loss")
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Component Losses
    plot_idx = 2
    for k in ["coord", "obj", "noobj", "class"]:
        if k in model.loss_history:
            plt.subplot(2, 3, plot_idx)
            plt.plot(model.loss_history[k], label=f"{k} Loss", color=f"C{plot_idx}")
            plt.title(f"{k} Loss")
            plt.xlabel("Epoch")
            plt.legend()
            plot_idx += 1
            
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    if show:
        plt.show() 
    return img


class TrainSettings:
    def __init__(self, num_classes: int, num_epochs: int, batch_size: int,
                 reduction_factor: int, centers: list[float], N_ANCHORS: int, 
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 loss_fn: Optional[torch.nn.Module] = None):
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.reduction_factor = reduction_factor
        self.centers = centers
        self.N_ANCHORS = N_ANCHORS
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def create_experiment_folder(self, model, base_dir="experiments"):
        # Get reduction factor from model or settings
        if hasattr(model, 'get_reduction_factor'):
            rf = model.get_reduction_factor()
        else:
            rf = self.reduction_factor

        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        folder_name = f"cnn_yolo-light_reduc{rf}-v1_fp32_{date_str}"
        
        self.model_dir = os.path.join(base_dir, folder_name)
        
        # Handle duplicate folder names by appending a counter
        if os.path.exists(self.model_dir):
            counter = 1
            while True:
                new_name = f"{folder_name}_{counter}"
                new_dir = os.path.join(base_dir, new_name)
                if not os.path.exists(new_dir):
                    experiment_dir = new_dir
                    break
                counter += 1
                
        os.makedirs(self.model_dir, exist_ok=True)

        # Save model weights
        model_save_path = os.path.join(self.model_dir, "model.pth")
        torch.save(model.state_dict(), model_save_path)

        summary_path = os.path.join(self.model_dir, "summary.md")
        
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"# Experiment Summary - {os.path.basename(self.model_dir)}\n\n")
            
            f.write("## Model Architecture\n")
            f.write("```\n")
            f.write(str(model))
            f.write("\n```\n\n")
            
            f.write("## Hyperparameters\n")
            f.write(f"- **Num Classes**: {self.num_classes}\n")
            f.write(f"- **Num Epochs**: {self.num_epochs}\n")
            f.write(f"- **Batch Size**: {self.batch_size}\n")
            f.write(f"- **Learning Rate**: (See Optimizer details)\n")
            f.write(f"- **Reduction Factor**: {self.reduction_factor}\n")
            f.write(f"- **Centers**: {self.centers}\n")
            f.write(f"- **N_ANCHORS**: {self.N_ANCHORS}\n")
            
            f.write("\n## Training Components\n")
            if self.loss_fn:
                f.write(f"- **Loss Function**: {self.loss_fn}\n")
                f.write(f"- **Lambda Coord**: {self.loss_fn.lambda_coord}\n")
                f.write(f"- **Lambda Noobj**: {self.loss_fn.lambda_noobj}\n")
                f.write(f"- **Lambda Obj**: {self.loss_fn.lambda_obj}\n")
                f.write(f"- **Smooth Factor**: {self.loss_fn.smooth_factor}\n")
            else:
                f.write("- **Loss Function**: None\n")
                
            if self.optimizer:
                f.write(f"- **Optimizer**: {self.optimizer}\n")
            else:
                f.write("- **Optimizer**: None\n")
                
            if self.scheduler:
                f.write(f"- **Scheduler**: {self.scheduler}\n")
            else:
                f.write("- **Scheduler**: None\n")

            loss_history = model.loss_history
            last_loss = loss_history["total"][-1]
            f.write(f"- **Last Loss**: {last_loss}\n")
            last_coord = loss_history["coord"][-1]
            f.write(f"- **Last Coord**: {last_coord}\n")
            last_obj = loss_history["obj"][-1]
            f.write(f"- **Last Obj**: {last_obj}\n")
            last_noobj = loss_history["noobj"][-1]
            f.write(f"- **Last Noobj**: {last_noobj}\n")
            last_class = loss_history["class"][-1]
            f.write(f"- **Last Class**: {last_class}\n")


    def add_loss_history(self, model:Optional[nn.Module] = None):
        if model is None:
            raise ValueError("Model must be provided")
        
        image = get_plot_loss(model=model)
        image.save(os.path.join(self.model_dir, "loss_history.png"))
