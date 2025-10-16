from ultralytics import YOLO
import cv2

# https://docs.ultralytics.com/tasks/classify/#predict

BATCH_SIZE = 5
IMAGE_SIZE = 320

SKIP_FRAMES = 5
MAX_FRAMES = 100

SAVE_VIDEO = False

if __name__ == '__main__':
    model = YOLO("runs/detect/train11/weights/best.pt")
    
    cameras = [
        {'name': 'front', 'video_path': 'videos/14_55/14_55_front_cropped.mp4'},
        {'name': 'back_left', 'video_path': 'videos/14_55/14_55_back_left_cropped.mp4'},
        {'name': 'back_right', 'video_path': 'videos/14_55/14_55_back_right_cropped.mp4'},
        {'name': 'back_right', 'video_path': 'videos/14_55/14_55_top_cropped.mp4'}
    ]
    
    video_path = 'videos/14_55/14_55_front_cropped.mp4'
    
    frame_count = 0
    processed_count = 0
    tracked_objects = {}
    
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret or processed_count >= MAX_FRAMES:
            break
            
        frame_count += 1
        if frame_count % (SKIP_FRAMES + 1) != 0:
            continue
    
        # results = model.track(frame, conf=0.25, save=True, imgsz=IMAGE_SIZE, batch=BATCH_SIZE, device=0)
        results = model.track(frame, conf=0.25, persist=True, save=SAVE_VIDEO, imgsz=IMAGE_SIZE, device=0)
        
        for result in results:
            print(result.boxes)
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu()
                
                print("Boxes:", boxes)
                print("Track IDs:", track_ids)
                print("Confidences:", confidences)
                
        processed_count += 1