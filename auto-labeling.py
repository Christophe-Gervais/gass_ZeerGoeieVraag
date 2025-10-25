from ultralytics import YOLO
import cv2
import os
from glob import glob

# Load your trained YOLO model for bottle detection
model = YOLO("runs/detect/gasbottle_yolo11m_final/weights/best.pt")

# Folder with extra OK images
images_folder = "extra_ok_images"
images = glob(os.path.join(images_folder, "*.[jp][pn]g"))  # jpg or png

dataset_dir = "bottle_dataset"
images_dir = os.path.join(dataset_dir, "images", "train", "OK")
labels_dir = os.path.join(dataset_dir, "labels", "train", "OK")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

cls_id = 0  # OK class

for img_path in images:
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # Run YOLO detection on the image
    results = model.predict(img, conf=0.25, save=False)
    
    # Save image to dataset
    new_image_path = os.path.join(images_dir, f"{base_name}.jpg")
    cv2.imwrite(new_image_path, img)

    # Save YOLO labels for each detected bottle
    label_path = os.path.join(labels_dir, f"{base_name}.txt")
    with open(label_path, "w") as f:
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

print(f"Processed {len(images)} OK images and saved labels.")
