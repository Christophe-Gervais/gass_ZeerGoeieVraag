from ultralytics import YOLO
import cv2
import os
import datetime
import random
import numpy as np

model = YOLO("runs/best.pt")

VALIDATION_SPLIT = 0.2
SKIP_FRAMES = 100

# Kies hier welke video je wilt labelen
VIDEO_PATH = 'videos/14_55_back_left_cropped.mp4'
VIDEO_IDX = 0  # Index voor in de filename

cap = cv2.VideoCapture(VIDEO_PATH)

dataset_dir = 'test/single'
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

classes = {"OK": 0, "NOK": 1}

def save_frame(frame, results, label_type):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    train_images_dir = os.path.join(images_dir, 'train')
    train_labels_dir = os.path.join(labels_dir, 'train')
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)

    image_path = os.path.join(train_images_dir, f"video{VIDEO_IDX}_{timestamp}.jpg")
    label_path = os.path.join(train_labels_dir, f"video{VIDEO_IDX}_{timestamp}.txt")
    cv2.imwrite(image_path, frame)

    with open(label_path, 'w') as f:
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = classes[label_type]
                x1, y1, x2, y2 = box.xyxy[0]
                h, w, _ = frame.shape
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

def move_some_samples_to_validation():
    train_images_dir = os.path.join(images_dir, 'train')
    train_labels_dir = os.path.join(labels_dir, 'train')
    val_images_dir = os.path.join(images_dir, 'val')
    val_labels_dir = os.path.join(labels_dir, 'val')
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    all_images = [f for f in os.listdir(train_images_dir) if f.endswith('.jpg')]
    if len(all_images) == 0:
        print("No images to split into validation set")
        return
        
    num_val_samples = int(VALIDATION_SPLIT * len(all_images))
    val_samples = random.sample(all_images, num_val_samples)
    
    for img_name in val_samples:
        base_name = os.path.splitext(img_name)[0]
        label_name = base_name + '.txt'
        
        src_img = os.path.join(train_images_dir, img_name)
        dst_img = os.path.join(val_images_dir, img_name)
        src_lbl = os.path.join(train_labels_dir, label_name)
        dst_lbl = os.path.join(val_labels_dir, label_name)
        
        if os.path.exists(src_img):
            os.rename(src_img, dst_img)
        if os.path.exists(src_lbl):
            os.rename(src_lbl, dst_lbl)
    
    print(f"Moved {num_val_samples} samples to validation set")

def run():
    DISPLAY_WIDTH = 960

    print(f"\n=== Controls ===")
    print("o = Save as OK")
    print("n = Save as NOK")
    print("f = Fast forward (skip 100 frames)")
    print("SPACE = Skip 15 frames")
    print("p = Pause")
    print("q = Quit")
    print(f"\nLabeling video: {VIDEO_PATH}\n")

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break

        # Predict and annotate
        results = model.predict(frame, conf=0.25, save=False)
        annotated_frame = frame.copy()
        for result in results:
            annotated_frame = result.plot()
        
        # Resize to display width while keeping aspect ratio
        h, w = annotated_frame.shape[:2]
        new_height = int(DISPLAY_WIDTH * h / w)
        annotated_frame = cv2.resize(annotated_frame, (DISPLAY_WIDTH, new_height))
        
        # Add video info to frame
        video_name = os.path.basename(VIDEO_PATH)
        cv2.putText(annotated_frame, f"Video: {video_name}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Video Labeling", annotated_frame)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('o'), ord('n')]:
            label_type = "OK" if key == ord('o') else "NOK"
            save_frame(frame, results, label_type)
            print(f"Saved frame as {label_type}")
        elif key == ord('p'):
            cv2.waitKey(-1)
        elif key == ord('f'):
            for _ in range(SKIP_FRAMES):
                cap.read()
        elif key == 32:  # Spacebar
            skip_count = 15
            for _ in range(skip_count):
                cap.read()

run()
move_some_samples_to_validation()
cap.release()
cv2.destroyAllWindows()