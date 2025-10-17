from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# https://docs.ultralytics.com/tasks/classify/#predict

BATCH_SIZE = 5
IMAGE_SIZE = 320
PREVIEW_IMAGE_SIZE = 1080

SKIP_FRAMES = 5
MAX_FRAMES = 1000

SAVE_VIDEO = False

WIDTH_CHANGE_THRESHOLD = 40.0  # pixels
LAST_WIDTH_COUNT = 3

FPS = 30

class Bottle:
    index: int
    x: float
    y: float
    in_view: bool = True
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

class Camera:
    name: str
    video_path: str
    output_path: str
    cap: cv2.VideoCapture
    out: cv2.VideoWriter
    width: int
    height: int
    adjusted_width: int
    adjusted_height: int
    aspect_ratio: float
    processed_frame_count: int = 0
    def __init__(self, name: str, video_path: str):
        self.name = name
        self.video_path = video_path
        input_path = Path(video_path)
        self.output_path = f'runs/detect/track/{input_path.stem}_tracked.mp4'
        
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.aspect_ratio = self.width / self.height
        self.adjusted_width = int(IMAGE_SIZE * self.aspect_ratio)
        self.adjusted_height = IMAGE_SIZE
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, FPS, (self.adjusted_width, self.adjusted_height))
        
    def get_frame(self):
        return self.cap.read()
    
    def is_open(self):
        return self.cap.isOpened()
    
    def release(self):
        self.cap.release()
        self.out.release()
    
    def finish_frame(self):
        self.processed_frame_count += 1

if __name__ == '__main__':
    model = YOLO("runs/detect/train11/weights/best.pt")
    
    cameras: list[Camera] = [
        Camera('front', 'videos/14_55/14_55_front_cropped.mp4'),
        
        Camera('back_left', 'videos/14_55/14_55_back_left_cropped.mp4'),
        # Camera('back_right', 'videos/14_55/14_55_back_right_cropped.mp4'),
        # Camera('back_right', 'videos/14_55/14_55_top_cropped.mp4')
    ]
    
    # enumerate the cameras
    

    # video_path = camera.video_path
    # input_path = Path(video_path)
    # output_path = f'runs/detect/track/{input_path.stem}_tracked.mp4'
    
    # frame_count = 0
    # processed_count = 0
    # tracked_objects = {}
    
    # first_bottle_widths = []
    
    
    # cap = cv2.VideoCapture(video_path)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # aspect_ratio = width / height
    # adjusted_width = int(IMAGE_SIZE * aspect_ratio)
    # adjusted_height = IMAGE_SIZE
    
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_path, fourcc, FPS, (adjusted_width, adjusted_height))
    
    bottles: list[Bottle] = []
    track_ids = []
    
    def register_bottle(x, y):
        bottle = Bottle(x, y)
        bottle.index = len(bottles) + 1
        bottles.append(bottle)
    
    bottle_was_entering = False
    
    while True:
        for i, camera in enumerate(cameras):
            print(f"Camera {i}: {camera.name} - {camera.video_path}")
            
            if not camera.is_open():
                break
            
            ret, frame = camera.get_frame()
            
            if not ret or camera.processed_frame_count >= MAX_FRAMES:
                break
                
            if camera.processed_frame_count % (SKIP_FRAMES + 1) != 0:
                continue
            
            small_frame = cv2.resize(frame, (camera.adjusted_width, camera.adjusted_height))
            
            # We can preivew on a larger scale if wanted, I'm not gonna implement that yet
            # preview_frame = cv2.resize(frame, (PREVIEW_IMAGE_SIZE, int(PREVIEW_IMAGE_SIZE / aspect_ratio)))
        
            results = model.track(small_frame, conf=0.25, persist=True, device=0)
            
            for result in results:
                # print(result.boxes)
                annotated_frame = result.plot()
                camera.out.write(annotated_frame)
                    
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                    confidences = result.boxes.conf.cpu()
                    
                    print("Boxes:", boxes)
                    print("Track IDs:", track_ids)
                    print("Confidences:", confidences)
                    
                    
                    annotated_frame = result.plot()
                    camera.out.write(annotated_frame)
                            
                
                # Displqy the frame
                cv2.imshow('Live Tracking Preview - ' + camera.name, annotated_frame)
        
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.waitKey(-1)
                    
                    
                    
            # processed_count += 1
            camera.finish_frame()
            
    for camera in cameras:
        camera.release()
        
    cv2.destroyAllWindows()