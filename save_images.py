import cv2
from pathlib import Path
import os
cap = cv2.VideoCapture(0)  # 0 = caméra par défaut



IMAGES_DIRECTORY = Path("images/")
image_id = 1

def get_last_dataset():
    
    dataset_id = 1
    files = [f for f in IMAGES_DIRECTORY.iterdir() if f.is_file()]
    if len(files) == 0:
        return dataset_id
    for file in files :
        name =file.absolute().name
        id = int(name[-5:-4])
        if id > dataset_id:
            dataset_id = id
    return dataset_id+1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (500, 500))
    cv2.imshow("Camera", resized)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        
        print("images/capture_"+str(image_id)+".png")
        cv2.imwrite("images/capture_"+str(image_id)+".png", resized)   
        image_id+=1

    if key == ord("d"):
        files = [f for f in IMAGES_DIRECTORY.iterdir() if f.is_file()]
        for file in files:
            file.unlink()
        images_id = 1

    if key == ord('q'):
        break

    
cap.release()
cv2.destroyAllWindows()