from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# https://docs.ultralytics.com/tasks/classify/#predict

BATCH_SIZE = 5
IMAGE_SIZE = 320
PREVIEW_IMAGE_SIZE = 1080

SKIP_FRAMES = 1
MAX_FRAMES = 1000

SAVE_VIDEO = False

TEMPORAL_CUTOFF_THRESHOLD = 10  # Amount of frames a bottle needs to be seen to be considered tracked
LAST_WIDTH_COUNT = 3

INPUT_VIDEO_FPS = 60
# FPS = 30

MODEL_PATH   = "runs/detect/train11/weights/best.pt"

class Bottle:
    index: int = 0
    x: float
    y: float
    in_view: bool = True
    yolo_id: int
    
    def __init__(self, x: float, y: float, yolo_id: int):
        self.x = x
        self.y = y
        self.yolo_id = yolo_id

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
    frame_count: int = 0
    processed_frame_count: int = 0
    model: YOLO
    temporary_bottles: dict[int, Bottle] = {}
    bottles: dict[int, Bottle] = {}
    track_ids_seen: dict[int, int] = {}  # Count of frames each track ID has been seen in
    bottle_index_counter: int = 0
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
        self.out = cv2.VideoWriter(self.output_path, fourcc, INPUT_VIDEO_FPS / SKIP_FRAMES if SKIP_FRAMES > 0 else 1, (self.adjusted_width, self.adjusted_height))
        
        self.model = YOLO(MODEL_PATH)
        
    def get_frame(self):
        self.frame_count += 1
        return self.cap.read()
    
    def process_frame(self, frame):
        small_frame = cv2.resize(frame, (camera.adjusted_width, camera.adjusted_height))
        results = self.model.track(small_frame, conf=0.25, persist=True, device=0)
        return results
    
    def is_open(self):
        return self.cap.isOpened()
    
    def release(self):
        self.cap.release()
        self.out.release()
    
    def finish_frame(self):
        self.processed_frame_count += 1
        
    def register_bottle(self, x, y, track_id):
        if track_id in self.bottles:
            self.bottles[track_id].x = x
            self.bottles[track_id].y = y
            return False
        if track_id in self.temporary_bottles:
            self.track_ids_seen[track_id] += 1
            self.temporary_bottles[track_id].x = x
            self.temporary_bottles[track_id].y = y
            if self.track_ids_seen[track_id] >= TEMPORAL_CUTOFF_THRESHOLD:
                # bottle_index_counter = len(self.bottles) + 1
                print("Bottle tracked with ID:", track_id)
                bottle = self.temporary_bottles[track_id]
                self.bottle_index_counter += 1
                bottle.index = self.bottle_index_counter#len(self.bottles) + 1
                self.bottles[track_id] = bottle
                del self.temporary_bottles[track_id]
            return False
        bottle = Bottle(x, y, track_id)
        self.temporary_bottles[track_id] = bottle
        self.track_ids_seen[track_id] = 1

if __name__ == '__main__':
    
    
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
    
    # bottles: dict[int, Bottle] = {}
    track_ids = []
    
    # def register_bottle(x, y, yolo_id):
    #     if yolo_id in bottles:
    #         bottles[yolo_id].x = x
    #         bottles[yolo_id].y = y
    #         return False
    #     bottle = Bottle(x, y, yolo_id)
    #     bottle.index = len(bottles) + 1
    #     bottles[yolo_id] = bottle
        
    
    bottle_was_entering = False
    
    while True:
        
        frames = []
        
        for camera_index, camera in enumerate(cameras):
            print(f"Camera {camera_index}: {camera.name} - {camera.video_path} Processed: {camera.processed_frame_count}")
            
            ret, frame = camera.get_frame()
            
            if camera.processed_frame_count >= MAX_FRAMES or not camera.is_open():
                break
                
            if camera.frame_count % (SKIP_FRAMES + 1) != 0:
                continue
            
            
            results = camera.process_frame(frame)
            
            if not ret:
                break
            
            # small_frame = cv2.resize(frame, (camera.adjusted_width, camera.adjusted_height))
            
            # We can preivew on a larger scale if wanted, I'm not gonna implement that yet
            # preview_frame = cv2.resize(frame, (PREVIEW_IMAGE_SIZE, int(PREVIEW_IMAGE_SIZE / aspect_ratio)))
        
            # results = model.track(small_frame, conf=0.25, persist=True, device=0)
            
            for result in results:
                # print(result.boxes)
                annotated_frame = result.plot()
                frames.append(annotated_frame)
                # camera.out.write(annotated_frame)
                    
                if result.boxes is not None:
                    boxes = result.boxes.xywh.cpu()
                    track_ids = None
                    if result.boxes.id is not None:
                        track_ids = result.boxes.id.cpu().numpy().astype(int)
                    confidences = result.boxes.conf.cpu()
                    
                    # print("Boxes:", boxes)
                    
                    for box_index, box in enumerate(boxes):
                        x_center, y_center, box_width, box_height = box
                        if track_ids is not None:
                            track_id = track_ids[box_index]
                            print("Camera ID:", camera_index,"Track ID: ", track_id)
                            # register_bottle(x_center, y_center, track_id)
                            camera.register_bottle(x_center, y_center, track_id)
                            
                            if track_id in camera.bottles:
                                print("This bottle was already seen")
                                bottle = camera.bottles[track_id]
                                cv2.putText(annotated_frame, 'ID: ' + str(bottle.index), (int(x_center), int(y_center)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        # print("Confidences:", confidences)
                    
                    
                    annotated_frame = result.plot()
                    camera.out.write(annotated_frame)
                    
                    
                    
            # processed_count += 1
            camera.finish_frame()
                            
                            
        
        
        # Displqy the frame
        if not len(frames) > 0:
            continue
        
        combined_frame = np.hstack(frames)
        
        cv2.imshow('Live Tracking Preview - ' + camera.name, combined_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(-1)
                    
                    
            
    for camera in cameras:
        camera.release()
        
    cv2.destroyAllWindows()