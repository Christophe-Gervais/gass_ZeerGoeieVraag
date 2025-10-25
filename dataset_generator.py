from ultralytics import YOLO
import cv2
import os
import datetime
import random
import numpy as np

model = YOLO("runs/detect/gasbottle_yolo11m_final/weights/best.pt")

VALIDATION_SPLIT = 0.2
SKIP_FRAMES = 5

video_paths = [
    'videos/14_55_back_right_cropped.mp4',
    'videos/14_55_top_cropped.mp4',
    'videos/14_55_back_left_cropped.mp4',
    'videos/14_55_top_cropped.mp4'
]

caps = [cv2.VideoCapture(p) for p in video_paths]

dataset_dir = 'bottle_dataset'
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

classes = {"OK": 0, "NOK": 1}

def save_frame(frame, results, label_type, video_idx):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    train_images_dir = os.path.join(images_dir, 'train', label_type)
    train_labels_dir = os.path.join(labels_dir, 'train', label_type)
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)

    image_path = os.path.join(train_images_dir, f"video{video_idx}_{timestamp}.jpg")
    label_path = os.path.join(train_labels_dir, f"video{video_idx}_{timestamp}.txt")
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
    DISPLAY_WIDTH = 480  # width per video

    while True:
        frames = []
        results_list = []

        # Read frame from each video
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((int(DISPLAY_WIDTH * 3/4), DISPLAY_WIDTH, 3), dtype=np.uint8)  # placeholder black
            frames.append(frame)

        # Predict and annotate each frame
        annotated_frames = []
        for i, frame in enumerate(frames):
            results = model.predict(frame, conf=0.25, save=False)
            results_list.append(results)
            annotated_frame = frame.copy()
            for result in results:
                annotated_frame = result.plot()
            
            # Resize to 480px width while keeping aspect ratio
            h, w = annotated_frame.shape[:2]
            new_height = int(DISPLAY_WIDTH * h / w)
            annotated_frame = cv2.resize(annotated_frame, (DISPLAY_WIDTH, new_height))
            annotated_frames.append(annotated_frame)

        # Make sure all frames in a row have same height by padding if necessary
        top_row = np.hstack([cv2.copyMakeBorder(f, 0, max(0, max(f.shape[0] for f in annotated_frames[:2])-f.shape[0]), 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0]) for f in annotated_frames[:2]])
        bottom_row = np.hstack([cv2.copyMakeBorder(f, 0, max(0, max(f.shape[0] for f in annotated_frames[2:])-f.shape[0]), 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0]) for f in annotated_frames[2:]])
        combined = np.vstack([top_row, bottom_row])

        cv2.imshow("Multi-Video Inference", combined)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('o'), ord('n')]:
            label_type = "OK" if key == ord('o') else "NOK"
            for i, frame in enumerate(frames):
                save_frame(frame, results_list[i], label_type, i)
            print(f"Saved all videos as {label_type}")
        elif key == ord('p'):
            cv2.waitKey(-1)
        elif key == ord('f'):
            for _ in range(SKIP_FRAMES):
                for cap in caps:
                    cap.read()

run()
move_some_samples_to_validation()
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
