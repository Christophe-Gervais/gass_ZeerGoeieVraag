from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# https://docs.ultralytics.com/tasks/classify/#predict

BATCH_SIZE = 5
IMAGE_SIZE = 320


SKIP_FRAMES = 0 # Skip this many frames between each processing step

MAX_FRAMES = 1000 # The amount of frames to process before quitting

SAVE_VIDEO = False

LAST_WIDTH_COUNT = 3

INPUT_VIDEO_FPS = 60

MODEL_PATH = "runs/detect/train11/weights/best.pt"


TEMPORAL_CUTOFF_THRESHOLD = 20  # Amount of frames a bottle needs to be seen to be considered tracked.
BOTTLE_DISAGREEMENT_TOLERANCE = 30  # Amount of frames the cameras can disagree before correction is applied.
SEQUENTIAL_CORRECTION_THRESHOLD = 3 # If a tracker has to be corrected this many times in a row, it's permanently steered back on track.
ENFORCE_INCREMENTAL_CORRECTION = False # Make sure the corrected index is unique.

EXTRA_CAMERA_DELAY = 0  # Delay in seconds

# Preview options
PREVIEW_IMAGE_SIZE = 400
PREVIEW_WINDOW_NAME = "Live Tracking Preview"

# Logging parameters
VERBOSE_YOLO = False
VERBOSE_LOGS = True

def log(message: str):
    if not VERBOSE_LOGS: return
    print(message)

class Bottle:
    index: int = -1
    x: float
    y: float
    in_view: bool = True
    yolo_id: int
    was_corrected: bool = False
    
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
    bottles: dict[int, Bottle]
    track_ids_seen: dict[int, int] = {}  # Count of frames each track ID has been seen in
    bottle_index_counter: int = 0
    # start_delay: float = 0 # seconds
    capture_fps: float
    frames_since_last_registration: int = 0
    last_registered_bottle: Bottle = None
    last_registered_bottle_track_id: int = -1
    sequential_correction_count: int = 0
    def __init__(self, name: str, video_path: str, start_delay: int = 0, start_index: int = 0):
        self.name = name
        self.video_path = video_path
        input_path = Path(video_path)
        
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.capture_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        
        
        self.aspect_ratio = self.width / self.height
        self.adjusted_width = int(IMAGE_SIZE * self.aspect_ratio)
        self.adjusted_height = IMAGE_SIZE
        
        if SAVE_VIDEO:
            self.output_path = f'runs/detect/track/{input_path.stem}_tracked.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.output_path, fourcc, self.capture_fps / SKIP_FRAMES if SKIP_FRAMES > 0 else 1, (self.adjusted_width, self.adjusted_height))
        
        self.bottles = {}
        self.bottle_index_counter = start_index
        
        start_delay += EXTRA_CAMERA_DELAY
        frames_to_skip = int(start_delay * self.capture_fps)
        if frames_to_skip > 0:
            print(f"Camera {self.name}: Skipping first {frames_to_skip} frames for start delay of {start_delay} seconds.")
            for _ in range(frames_to_skip):
                ret, frame = self.cap.read()
                if not ret:
                    break
            print("Finished skipping frames.")
        
        self.frame_count = 0
        self.processed_frame_count = 0
        
        print("Finished camera setup. Loading model...")
        self.model = YOLO(MODEL_PATH)
        print("Finished loading model!")
        
    def get_frame(self):
        self.frame_count += 1
        return self.cap.read()

    def should_process_frame(self):
        return SKIP_FRAMES > 0 and self.frame_count % (SKIP_FRAMES + 1) == 0
    
    def process_frame(self, frame):
        small_frame = cv2.resize(frame, (self.adjusted_width, self.adjusted_height))
        results = self.model.track(small_frame, conf=0.25, persist=True, device=0, verbose=VERBOSE_YOLO)
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
                bottle.index = self.bottle_index_counter
                self.bottles[track_id] = bottle
                print("Bottle assigned index:", bottle.index, "Track ID:", track_id, "Camera:", self.name)
                del self.temporary_bottles[track_id]
                self.last_registered_bottle = bottle
                self.last_registered_bottle_track_id = track_id
                return True
            return False
        bottle = Bottle(x, y, track_id)
        self.temporary_bottles[track_id] = bottle
        self.track_ids_seen[track_id] = 1
        
        return False

    def register_correction(self, corrected_index):
        self.sequential_correction_count += 1
        if self.sequential_correction_count > SEQUENTIAL_CORRECTION_THRESHOLD:
            self.bottle_index_counter = corrected_index
            print(f"I, Camera {self.name}, was wrong {self.sequential_correction_count} in a row. I really thought I was right but I guess I wasn't. As punishment I will correct myself, remember that the correct index from now on is {corrected_index} and I will try my best to never do this again. I'm so sorry.")


    

class BottleTracker:
    cameras: list[Camera]
    track_ids: list[int]
    bottle_was_entering: bool = False
    window_was_open: bool = False
    
    camera_disagreement_counts: dict[int, int] = {}
    last_corrected_index: int = -1
    
    def __init__(self, cameras: list[Camera]):
        self.cameras = cameras
        self.track_ids = []

    # def render_box(self, frame, box):
        
    
    def run(self):
        while True:
            
            frames = []
            
            for camera_index, camera in enumerate(self.cameras):
                # print(f"Camera {camera_index}: {camera.name} - {camera.video_path} Processed: {camera.processed_frame_count}")
                # w, h = camera.adjusted_width, camera.adjusted_height
                ret, frame = camera.get_frame()
                
                if camera.processed_frame_count >= MAX_FRAMES or not camera.is_open():
                    print(f"Done. {camera.processed_frame_count} frames processed.")
                    break
                    
                if not camera.should_process_frame():
                    continue
                
                
                results = camera.process_frame(frame)
                
                output_frame_width = int(PREVIEW_IMAGE_SIZE * camera.aspect_ratio)
                output_frame_height = PREVIEW_IMAGE_SIZE
                output_frame = cv2.resize(frame, (output_frame_width, output_frame_height))
                x_scale = output_frame_width / camera.adjusted_width
                y_scale = output_frame_height / camera.adjusted_height
                
                if not ret:
                    break
                
                for result in results:
                    # annotated_frame = result.plot()
                        
                    if result.boxes is not None:
                        boxes = result.boxes.xywh.cpu()
                        self.track_ids = None
                        if result.boxes.id is not None:
                            self.track_ids = result.boxes.id.cpu().numpy().astype(int)
                        confidences = result.boxes.conf.cpu()
                        
                        # print("Boxes:", boxes)
                        
                        # annotated_frame = result.plot()
                        
                        cv2.putText(output_frame, 'Camera: ' + camera.name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        for box_index, box in enumerate(boxes):
                            x_center, y_center, box_width, box_height = box
                            
                            if self.track_ids is not None:
                                track_id = self.track_ids[box_index]
                                
                                if camera.register_bottle(x_center, y_center, track_id):
                                    print("Bottle got accepted as being new.")
                                
                                # Render box on frame
                                scaled_x_center = int(x_center * x_scale)
                                scaled_y_center = int(y_center * y_scale)
                                scaled_box_width = int(box_width * x_scale)
                                scaled_box_height = int(box_height * y_scale)
                                
                                x1 = int(scaled_x_center - scaled_box_width / 2)
                                y1 = int(scaled_y_center - scaled_box_height / 2)
                                x2 = int(scaled_x_center + scaled_box_width / 2)
                                y2 = int(scaled_y_center + scaled_box_height / 2)
                                
                                # Ensure coordinates are within frame bounds
                                x1 = max(0, min(x1, output_frame_width - 1))
                                y1 = max(0, min(y1, output_frame_height - 1))
                                x2 = max(0, min(x2, output_frame_width - 1))
                                y2 = max(0, min(y2, output_frame_height - 1))
                                
                                thickness = 2
                                color = (255, 0, 0)
                                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)
                                
                                def draw_bottle_id(color):
                                    cv2.putText(output_frame, 'Bottle ' + str(bottle.index), (int(x1 + 10), int(y1 + 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                                    
                                
                                if track_id in camera.temporary_bottles:
                                    bottle = camera.temporary_bottles[track_id]
                                    draw_bottle_id((0, 0, 255))
                                if track_id in camera.bottles:
                                    bottle = camera.bottles[track_id]
                                    draw_bottle_id((255, 255, 0) if bottle.was_corrected else (0, 255, 0))
                        
                        frames.append(output_frame)
                        
                        if SAVE_VIDEO:
                            camera.out.write(output_frame)
                            
                camera.finish_frame()
                                
                                
            
            
            # Check if all cameras agree with each other on the last bottle index
            last_bottle_indices = set()
            for camera in self.cameras:
                if camera.last_registered_bottle is not None:
                    last_bottle_indices.add(camera.last_registered_bottle.index)
            
            if len(last_bottle_indices) > 1:
                # print("Warning: Cameras disagree on last registered bottle indices:", last_bottle_indices, "Camera number:", camera_index)
                self.camera_disagreement_counts[camera_index] = self.camera_disagreement_counts.get(camera_index, 0) + 1
                
                if self.camera_disagreement_counts[camera_index] >= BOTTLE_DISAGREEMENT_TOLERANCE:
                    self.correct_index_disagreements()
                
            
            # Display the frame
            if not len(frames) > 0:
                continue
            
            frame_rows = self.split_array(frames, 2)
            row_frames = []
            for row in frame_rows:
                while len(row) < 2:
                    row.append(np.zeros_like(row[0]))
                row_frame = np.hstack(row)
                row_frames.append(row_frame)
            
            combined_frame = np.vstack(row_frames)
            
            if self.window_was_open and self.is_window_closed(PREVIEW_WINDOW_NAME):
                print("Window closed, exiting.")
                break
                
            cv2.imshow(PREVIEW_WINDOW_NAME, combined_frame)
            

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(-1)
                
            self.window_was_open = True
            
        print("Finished processing.")
        self.release()
        
    def correct_index_disagreements(self):
        # Find the most common last registered bottle index among cameras
        print("Correcting index disagreements among cameras.")
        last_indices = []
        for camera in self.cameras:
            if camera.last_registered_bottle is not None:
                last_indices.append(camera.last_registered_bottle.index)
        print("Last registered bottle indices from cameras:", last_indices)
        if not last_indices:
            return
        
        index_counts = Counter(last_indices)
        most_common_index, most_common_count = index_counts.most_common(1)[0]
        
        print(f"Most common last registered bottle index is {most_common_index} seen {most_common_count} times.")
        
        if ENFORCE_INCREMENTAL_CORRECTION and not most_common_index > self.last_corrected_index:
            print("Warning: I can't correct to an index that was used before. Incrementing might skip an index but ensures all indexes link to one bottle.")
            most_common_index = self.last_corrected_index + 1
        
        if most_common_count > len(self.cameras) / 2:
            print("Majority agreement found, correcting outcast.")
            self.last_corrected_index = most_common_index
            for camera in self.cameras:
                if camera.last_registered_bottle is not None:
                    print(f"Correcting camera {camera.name} from index {camera.last_registered_bottle.index} to {most_common_index}")
                    if camera.last_registered_bottle.index != most_common_index:
                        camera.last_registered_bottle.index = most_common_index
                        camera.last_registered_bottle.was_corrected = True
                        camera.register_correction(most_common_index)
                        
        
        # reset disagreement counts
        self.camera_disagreement_counts = {}

    # Boring functions

    def split_array(self, arr, max_length):
        return [arr[i:i + max_length] for i in range(0, len(arr), max_length)] if arr else []
            
    def is_window_closed(self, window_name):
        try: return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
        except: return True

    def release(self):
        for camera in self.cameras: camera.release()
        cv2.destroyAllWindows()


def main():
    cameras: list[Camera] = [
        Camera('Front', 'videos/14_55/14_55_front_cropped.mp4', start_delay=0),
        
        Camera('Back Left', 'videos/14_55/14_55_back_left_cropped.mp4', start_delay=2),
        Camera('Back Right', 'videos/14_55/14_55_back_right_cropped.mp4', start_delay=1, start_index=-1),
        Camera('Top', 'videos/14_55/14_55_top_cropped.mp4', start_delay=4)
    ]
    
    bottle_tracker = BottleTracker(cameras)
    bottle_tracker.run()

if __name__ == '__main__':
    main()