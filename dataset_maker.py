import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from pathlib import Path
import os
import glob
from pathlib import Path
import csv

# === Paramètres ===
CLASS_NAME = "card"   # tu peux changer la classe ici

DATASET_DIRECTORY = Path("dataset/")
IMAGES_DIRECTORY = Path("images/")
boxes = []
paths = []
image_paths = []
running = False

def get_last_dataset():
    
    dataset_id = 1
    files = [f for f in DATASET_DIRECTORY.iterdir() if f.is_file()]
    if len(files) == 0:
        return dataset_id
    for file in files :
        name =file.absolute().name
        id = int(name[-5:-4])
        if id > dataset_id:
            dataset_id = id
    return dataset_id +1


DATASET_DIRECTORY = Path("dataset/")

def save_to_csv():
    global boxes, image_paths, CLASS_NAME
    
    if not boxes:
        print("Aucune box ou image non chargée.")
        return
    
    # Crée le dossier s'il n'existe pas
    DATASET_DIRECTORY.mkdir(parents=True, exist_ok=True)
    
    output = DATASET_DIRECTORY / f"dataset_{get_last_dataset()}.csv"
    
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # en-tête CSV
        writer.writerow(["file_name", "class", "x", "y", "w", "h"])
        
        # lignes
        for (x, y, w, h, file_name) in boxes:
            writer.writerow([file_name, CLASS_NAME, x, y, w, h])
    
    print(f"Annotations sauvegardées dans {output}")



def on_mouse_press(event):
    global start_x, start_y, running
    start_x, start_y = event.x, event.y
    running = True
    draw_loop()

def on_motion(event):
    global x_now, y_now
    x_now, y_now = event.x, event.y

def draw_loop():
    global x_now, y_now
    if running:
        next()
        canvas.create_rectangle(start_x, start_y, x_now, y_now, outline="red")
        root.after(100, draw_loop) 
        

def on_mouse_release(event):
    global start_x, start_y, running,image_paths,boxes
    end_x, end_y = event.x, event.y
    x, y = min(start_x, end_x), min(start_y, end_y)
    w, h = abs(end_x - start_x), abs(end_y - start_y)

    # Sauvegarde les coordonnées
    if len(image_paths) == 0: return
    boxes.append((x, y, w, h, image_paths[0]))
    if len(image_paths) ==0 : return
    print(f"Box ajoutée : {x},{y},{w},{h}", image_paths[0])
    running = False
    image_paths.pop(0)
    next()

def load_images():
    global image_paths
    dossier = filedialog.askdirectory(title="Choisir un dossier contenant des PNG")

    # Récupère tous les fichiers .png du dossier
    image_paths = [os.path.join(dossier, f)
                for f in os.listdir(dossier)
                if f.lower().endswith(".png")]
    image_paths.sort()
    print(image_paths)
    next()

def next():
    global image_paths, img, tk_img
    if not image_paths: return
    img = Image.open(image_paths[0])
    tk_img = ImageTk.PhotoImage(img)
    canvas.config(width=tk_img.width(), height=tk_img.height())
    canvas.create_image(0, 0, anchor="nw", image=tk_img)


# === Interface ===
root = tk.Tk()
root.title("Annotation Tool")
root.geometry("900x600")

btn_frame = tk.Frame(root)
btn_frame.pack(side="top", fill="x")

btn_load = tk.Button(btn_frame, text="Charger Image", command=load_images)
btn_load.pack(side="left")

btn_save = tk.Button(btn_frame, text="Sauver en SVG", command=save_to_csv)
btn_save.pack(side="left")

canvas = tk.Canvas(root, cursor="cross")
canvas.pack(fill="both", expand=True)

canvas.bind("<ButtonPress-1>", on_mouse_press)
canvas.bind("<ButtonRelease-1>", on_mouse_release)
canvas.bind("<Motion>", on_motion)

root.mainloop()
