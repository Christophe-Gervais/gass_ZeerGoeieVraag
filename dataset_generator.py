from ultralytics import YOLO
import cv2
import os
import datetime
import random

model = YOLO("runs/detect/gasbottle_yolo11m_final/weights/best.pt")

VALIDATION_SPLIT = 0.2
SKIP_FRAMES = 5  

video_path = 'videos/14_55_top_cropped.mp4'
cap = cv2.VideoCapture(video_path)

dataset_dir = 'bottle_dataset'
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

classes = {"OK": 0, "NOK": 1}  # your new classification labels

def save_frame(frame, results, label_type):
    """Save frame + YOLO label file under OK/NOK"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    train_images_dir = os.path.join(images_dir, 'train', label_type)
    train_labels_dir = os.path.join(labels_dir, 'train', label_type)
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)

    image_path = os.path.join(train_images_dir, f"{timestamp}.jpg")
    label_path = os.path.join(train_labels_dir, f"{timestamp}.txt")
    cv2.imwrite(image_path, frame)

    # Save YOLO labels (using original detection for bounding boxes)
    with open(label_path, 'w') as f:
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # assign the new class based on OK/NOK
                cls_id = classes[label_type]
                x1, y1, x2, y2 = box.xyxy[0]
                img_height, img_width, _ = frame.shape
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

def move_some_samples_to_validation():
    for label_type in classes.keys():
        train_images_dir = os.path.join(images_dir, 'train', label_type)
        train_labels_dir = os.path.join(labels_dir, 'train', label_type)
        val_images_dir = os.path.join(images_dir, 'val', label_type)
        val_labels_dir = os.path.join(labels_dir, 'val', label_type)
        os.makedirs(val_images_dir, exist_ok=True)
        os.makedirs(val_labels_dir, exist_ok=True)

        all_images = os.listdir(train_images_dir)
        num_val_samples = int(VALIDATION_SPLIT * len(all_images))
        val_samples = random.sample(all_images, num_val_samples)
        for img_name in val_samples:
            base_name = os.path.splitext(img_name)[0]
            label_name = base_name + '.txt'
            os.rename(os.path.join(train_images_dir, img_name),
                      os.path.join(val_images_dir, img_name))
            if os.path.exists(os.path.join(train_labels_dir, label_name)):
                os.rename(os.path.join(train_labels_dir, label_name),
                          os.path.join(val_labels_dir, label_name))

def run():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(frame, conf=0.25, save=False)
        annotated_frame = frame.copy()
        for result in results:
            annotated_frame = result.plot()
        destination_height = 600
        aspect_ratio = annotated_frame.shape[1] / annotated_frame.shape[0]
        destination_width = int(destination_height * aspect_ratio)
        annotated_frame = cv2.resize(annotated_frame, (destination_width, destination_height))
        cv2.imshow("Inference Result", annotated_frame)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            save_frame(frame, results, "OK")
            print("Saved as OK")
        elif key == ord('n'):
            save_frame(frame, results, "NOK")
            print("Saved as NOK")
        elif key == ord('p'):
            cv2.waitKey(-1)

run()
move_some_samples_to_validation()
cap.release()
cv2.destroyAllWindows()
