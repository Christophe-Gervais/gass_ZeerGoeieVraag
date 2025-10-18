from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train15/weights/best.pt")

results = model.predict(source="videos/14_55/14_55_front_cropped.mp4", conf=0.25, imgsz=160)

for result in results:
    # print(result.boxes)  # Boxes object for bbox outputs
    # print(result.masks)  # Masks object for segm outputs
    # print(result.keypoints)  # Keypoints object for keypoint outputs
    frame = result.orig_img