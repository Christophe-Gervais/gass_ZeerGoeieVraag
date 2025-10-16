from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np

# https://docs.ultralytics.com/tasks/classify/#predict

BATCH_SIZE = 5
IMAGE_SIZE = 320
PREVIEW_IMAGE_SIZE = 1080

SKIP_FRAMES = 5
MAX_FRAMES = 100

SAVE_VIDEO = False

FPS = 30

if __name__ == '__main__':
    model = YOLO("runs/detect/train11/weights/best.pt")
    
    cameras = [
        {'name': 'front', 'video_path': 'videos/14_55/14_55_front_cropped.mp4'},
        {'name': 'back_left', 'video_path': 'videos/14_55/14_55_back_left_cropped.mp4'},
        {'name': 'back_right', 'video_path': 'videos/14_55/14_55_back_right_cropped.mp4'},
        {'name': 'back_right', 'video_path': 'videos/14_55/14_55_top_cropped.mp4'}
    ]
    
    video_path = 'videos/14_55/14_55_front_cropped.mp4'
    input_path = Path(video_path)
    output_path = f'runs/detect/track/{input_path.stem}_tracked.mp4'
    
    frame_count = 0
    processed_count = 0
    tracked_objects = {}
    
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    aspect_ratio = width / height
    adjusted_width = int(IMAGE_SIZE * aspect_ratio)
    adjusted_height = IMAGE_SIZE
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (adjusted_width, adjusted_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret or processed_count >= MAX_FRAMES:
            break
            
        frame_count += 1
        if frame_count % (SKIP_FRAMES + 1) != 0:
            continue
        
        small_frame = cv2.resize(frame, (adjusted_width, adjusted_height))
        
        # We can preivew on a larger scale if wanted, I'm not gonna implement that yet
        # preview_frame = cv2.resize(frame, (PREVIEW_IMAGE_SIZE, int(PREVIEW_IMAGE_SIZE / aspect_ratio)))
    
        # results = model.track(frame, conf=0.25, save=True, imgsz=IMAGE_SIZE, batch=BATCH_SIZE, device=0)
        results = model.track(small_frame, conf=0.25, persist=True, save=SAVE_VIDEO, device=0)
        
        for result in results:
            print(result.boxes)
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu()
                
                print("Boxes:", boxes)
                print("Track IDs:", track_ids)
                print("Confidences:", confidences)
                
                print("\n--- BOX SIZES ---")
                for i, box in enumerate(boxes):
                    x_center, y_center, box_width, box_height = box
                    box_area = box_width * box_height
                    print(f"Box {i} (ID: {track_ids[i]}): {box_width:.1f}x{box_height:.1f} pixels, Area: {box_area:.1f} px²")
                
                annotated_frame = result.plot()
                out.write(annotated_frame)
                
                cv2.imshow('Live Tracking Preview', annotated_frame)
        
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.waitKey(-1)
                
        processed_count += 1
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()