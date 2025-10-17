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

TEMPORAL_CUTOFF_THRESHOLD = 10  # Amount of frames a bottle needs to be seen to be considered tracked
LAST_WIDTH_COUNT = 3

INPUT_VIDEO_FPS = 60
# FPS = 30

MODEL_PATH   = "runs/detect/train11/weights/best.pt"

VERBOSE = False

class Bottle:
    index: int = -1
    x: float
    y: float
    in_view: bool = True
    yolo_id: int
    
    def __init__(self, x: float, y: float, yolo_id: int):
        self.x = x
        self.y = y
        self.yolo_id = yolo_id

class CameraTracker:
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
    bottles: dict[int, Bottle]
    track_ids_seen: dict[int, int] = {}  # Count of frames each track ID has been seen in
    bottle_index_counter: int = 0
    # start_delay: float = 0 # seconds
    capture_fps: float
    def __init__(self, name: str, video_path: str, start_delay: int = 0, start_index: int = 0):
        self.name = name
        self.video_path = video_path
        input_path = Path(video_path)
        self.output_path = f'runs/detect/track/{input_path.stem}_tracked.mp4'
        
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.capture_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        
        
        self.aspect_ratio = self.width / self.height
        self.adjusted_width = int(IMAGE_SIZE * self.aspect_ratio)
        self.adjusted_height = IMAGE_SIZE
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.capture_fps / SKIP_FRAMES if SKIP_FRAMES > 0 else 1, (self.adjusted_width, self.adjusted_height))
        
        self.model = YOLO(MODEL_PATH)
        self.bottles = {}
        self.bottle_index_counter = start_index
        # self.start_delay = start_delay
        
        frames_to_skip = int(start_delay * self.capture_fps)
        if frames_to_skip > 0:
            print(f"Camera {self.name}: Skipping first {frames_to_skip} frames for start delay of {start_delay} seconds.")
        for _ in range(frames_to_skip):
            ret, frame = self.cap.read()
            if not ret:
                break
        
    def get_frame(self):
        self.frame_count += 1
        return self.cap.read()

    def should_process_frame(self):
        return SKIP_FRAMES > 0 and self.frame_count % (SKIP_FRAMES + 1) == 0
    
    def process_frame(self, frame):
        small_frame = cv2.resize(frame, (camera.adjusted_width, camera.adjusted_height))
        results = self.model.track(small_frame, conf=0.25, persist=True, device=0, verbose=VERBOSE)
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
            if self.track_ids_seen[track_id] >= TEMPORAL_CUTOFF_THRESHOLD / SKIP_FRAMES:
                bottle = self.temporary_bottles[track_id]
                self.bottle_index_counter += 1
                bottle.index = self.bottle_index_counter#len(self.bottles) + 1
                self.bottles[track_id] = bottle
                print("Bottle assigned index:", bottle.index, "Track ID:", track_id)
                # print(self.bottles)
                del self.temporary_bottles[track_id]
            return False
        bottle = Bottle(x, y, track_id)
        self.temporary_bottles[track_id] = bottle
        self.track_ids_seen[track_id] = 1

if __name__ == '__main__':
    
    
    cameras: list[CameraTracker] = [
        CameraTracker('Front', 'videos/14_55/14_55_front_cropped.mp4', start_delay=0),
        
        CameraTracker('Back Left', 'videos/14_55/14_55_back_left_cropped.mp4', start_delay=2),
        CameraTracker('Back Right', 'videos/14_55/14_55_back_right_cropped.mp4', start_delay=1, start_index=-1),
        CameraTracker('Top', 'videos/14_55/14_55_top_cropped.mp4', start_delay=4)
    ]
    
    track_ids = []
    
    bottle_was_entering = False
    
    while True:
        
        frames = []
        
        for camera_index, camera in enumerate(cameras):
            # print(f"Camera {camera_index}: {camera.name} - {camera.video_path} Processed: {camera.processed_frame_count}")
            
            ret, frame = camera.get_frame()
            
            if camera.processed_frame_count >= MAX_FRAMES or not camera.is_open():
                break
                
            if not camera.should_process_frame():
                continue
            
            
            results = camera.process_frame(frame)
            
            if not ret:
                break
            
            for result in results:
                annotated_frame = result.plot()
                    
                if result.boxes is not None:
                    boxes = result.boxes.xywh.cpu()
                    track_ids = None
                    if result.boxes.id is not None:
                        track_ids = result.boxes.id.cpu().numpy().astype(int)
                    confidences = result.boxes.conf.cpu()
                    
                    # print("Boxes:", boxes)
                    
                    annotated_frame = result.plot()
                    
                    cv2.putText(annotated_frame, 'Camera: ' + camera.name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    for box_index, box in enumerate(boxes):
                        x_center, y_center, box_width, box_height = box
                        if track_ids is not None:
                            track_id = track_ids[box_index]
                            
                            camera.register_bottle(x_center, y_center, track_id)
                            
                            if track_id in camera.bottles:
                                # print("This bottle was already seen")
                                bottle = camera.bottles[track_id]
                                cv2.putText(annotated_frame, 'ID: ' + str(bottle.index), (int(x_center), int(y_center)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    frames.append(annotated_frame)
                    
                    if SAVE_VIDEO:
                        camera.out.write(annotated_frame)
                        
            camera.finish_frame()
                            
                            
        
        
        # Displqy the frame
        if not len(frames) > 0:
            continue
        
        def split_array(arr, max_length):
            return [arr[i:i + max_length] for i in range(0, len(arr), max_length)] if arr else []
        
        frame_rows = split_array(frames, 2)
        row_frames = []
        for row in frame_rows:
            while len(row) < 2:
                row.append(np.zeros_like(row[0]))
            row_frame = np.hstack(row)
            row_frames.append(row_frame)
        
        combined_frame = np.vstack(row_frames)
        
        cv2.imshow('Live Tracking Preview - ' + camera.name, combined_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(-1)
                    
                    
            
    for camera in cameras:
        camera.release()
        
    cv2.destroyAllWindows()