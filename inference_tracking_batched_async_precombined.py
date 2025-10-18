from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
from collections import Counter
import threading
import queue
from time import time, sleep
import concurrent.futures


# Input options
MODEL_PATH = "runs/detect/train26/weights/best.pt"
INPUT_VIDEO_FPS = 60
EXTRA_CAMERA_DELAY = 1  # Delay in seconds
MAX_FRAMES = 1000000 # The amount of frames to process before quitting

# Algorithm options
IMAGE_SIZE = 160
BATCH_SIZE = 70
SKIP_FRAMES = 8 # Skip this many frames between each processing step
TEMPORAL_CUTOFF_THRESHOLD = 20  # Amount of frames a bottle needs to be seen to be considered tracked.
BOTTLE_DISAGREEMENT_TOLERANCE = 15  # Amount of frames the cameras can disagree before correction is applied.
SEQUENTIAL_CORRECTION_THRESHOLD = 3 # If a tracker has to be corrected this many times in a row, it's permanently steered back on track.
ENFORCE_INCREMENTAL_CORRECTION = False # Make sure the corrected index is unique.
EXTRA_CORRECTION = False # Allow correcting half the feed if one half disagrees with itself.

# Preview options
PREVIEW_IMAGE_SIZE = 640
SAVE_VIDEO = False
PREVIEW_WINDOW_NAME = "Live Tracking Preview"
DISPLAY_FRAMERATE = 12
MAX_QUEUE_SIZE = 1000 # The limit for the queue size, set to -1 to disable limit (but beware you might run out of memory then!)
QUEUE_SIZE_CHECK_INTERVAL = 0.1 # Amount of seconds to wait when queue is full
RENDER_SKIPPED_FRAMES = False # Whether to render skipped frames in between processed frames
SKIPPED_IMAGE_SIZE = 200
EASE_DISPLAY_SPEED = True
INCREASE_SPEED_AT = 150 # Speed up frame pacing when enough frames are queued
SPEED_MULTIPLIER = 0.8

# Logging options
VERBOSE_YOLO = False # Show YOLO debug info
VERBOSE_LOGS = True # Show general info
VERBOSE_BLAB = False # Show detailed debug info
VERBOSE_DBUG = True # Show debug info

def log(*values: object, **kwargs):
    if not VERBOSE_LOGS: return
    print(*values, *kwargs)
    
def blabber(*values: object, **kwargs):
    if not VERBOSE_BLAB: return
    print(*values, *kwargs)

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
    
    
    # frames: list[cv2.typing.MatLike]
    # result_queue: list
    frame_index: int
    stack_rect: tuple[int, int, int, int]  # x, y, w, h
    # last_results = None
    
    def get_allowed_frame_skip(self):
        return SKIP_FRAMES if SKIP_FRAMES > 0 else 1
    
    def skip_frames(self, frames_to_skip: int, collect_skipped: bool = False):
        frames = []
        if frames_to_skip > 0:
            for _ in range(frames_to_skip):
                ret, frame = self.cap.read()
                if not ret:
                    return frames
                if collect_skipped:
                    output_frame_width = int(SKIPPED_IMAGE_SIZE * self.aspect_ratio)
                    output_frame_height = SKIPPED_IMAGE_SIZE
                    output_frame = cv2.resize(frame, (output_frame_width, output_frame_height))
                    frames.append(output_frame)
        return frames
    
    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def get_inference_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.resize(frame, (self.inference_width // 2, self.inference_height // 2))
        return frame
    
    def __init__(self, name: str, video_path: str, start_skip: int = 0, start_index: int = 0):
        self.name = name
        self.video_path = video_path
        input_path = Path(video_path)
        
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.capture_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.last_results = None
        
        self.aspect_ratio = self.width / self.height
        self.adjusted_width = int(IMAGE_SIZE * self.aspect_ratio)
        self.adjusted_height = IMAGE_SIZE
        
        if SAVE_VIDEO:
            self.output_path = f'runs/detect/track/{input_path.stem}_tracked.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.output_path, fourcc, self.capture_fps / self.get_allowed_frame_skip(), (self.adjusted_width, self.adjusted_height))
        
        self.bottles = {}
        self.bottle_index_counter = start_index
        
        start_skip += EXTRA_CAMERA_DELAY
        frames_to_skip = int(start_skip * self.capture_fps)
        log(f"Camera {self.name}: Skipping first {frames_to_skip} frames for start delay of {start_skip} seconds.")
        self.skip_frames(frames_to_skip)
        
        self.processed_frame_count = 0
        
        blabber("Finished camera setup. Loading model...")
        # self.model = YOLO(MODEL_PATH)
        blabber("Finished loading model!")
        
        
        self.frame_index = 0
        self.running = False
        self.producer_thread = None
        self.last_output_frame: cv2.typing.MatLike = None
        
        self.stack_rect = (0, 0, self.width, self.height)
        
        self.disagreement_count = 0
        
        self.inference_width = int(IMAGE_SIZE * self.aspect_ratio)
        self.inference_height = IMAGE_SIZE
        
        # As
    def get_ready_frames_count(self):
        return self.frame_queue.qsize()
    
    def _image_processing_worker(self):
        blabber("Starting batch preprocessing.")
        while not self.preprocess_frames(BATCH_SIZE):
            blabber(f"Processed a batch of {BATCH_SIZE} images")
            pass
    
    def start_preprocessing(self):
        self.running = True
        self.producer_thread = threading.Thread(target=self._image_processing_worker)
        self.producer_thread.daemon = True
        self.producer_thread.start()
        print("Background producer started")
    
    def stop_preprocessing(self):
        self.running = False
        if self.producer_thread:
            self.producer_thread.join(timeout=10)
        print("Background producer stopped")    
    
    

    def is_open(self):
        return self.cap.isOpened()
    
    def release(self):
        self.cap.release()
        if SAVE_VIDEO: self.out.release()
        self.stop_preprocessing()
    
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
            if self.track_ids_seen[track_id] >= TEMPORAL_CUTOFF_THRESHOLD / self.get_allowed_frame_skip():
                bottle = self.temporary_bottles[track_id]
                self.bottle_index_counter += 1
                bottle.index = self.bottle_index_counter
                self.bottles[track_id] = bottle
                blabber("Bottle assigned index:", bottle.index, "Track ID:", track_id, "Camera:", self.name)
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
        if self.sequential_correction_count > TEMPORAL_CUTOFF_THRESHOLD / self.get_allowed_frame_skip():
            self.bottle_index_counter = corrected_index
            log(f"I, Camera {self.name}, was wrong {self.sequential_correction_count} in a row. I really thought I was right but I guess I wasn't. As punishment I will correct myself, remember that the correct index from now on is {corrected_index} and I will try my best to never do this again. I'm so sorry.")


    

class BottleTracker:
    cameras: list[Camera]
    track_ids: list[int]
    bottle_was_entering: bool = False
    window_was_open: bool = False
    
    camera_disagreement_counts: dict[int, int] = {}
    last_corrected_index: int = -1
    
    last_frame_time = time()
    
    def __init__(self, cameras: list[Camera]):
        self.cameras = cameras
        self.track_ids = []
        
        self.frame_queue: queue.Queue[cv2.typing.MatLike] = queue.Queue()
        self.results_queue = queue.Queue()
        self.batch_queue: queue.Queue[list[cv2.typing.MatLike]] = queue.Queue()
        self.last_results = None
        
        self.aspect_ratio = cameras[0].aspect_ratio
        self.inference_width = int(IMAGE_SIZE * self.aspect_ratio)
        self.inference_height = IMAGE_SIZE
        
        self.model = YOLO(MODEL_PATH)
        
        self.calculate_camera_stack_rects()
        
        self.batch_thread = None
        self.inference_thread = None
        self.combined_frame = np.zeros((self.inference_height, self.inference_width, 3), dtype=np.uint8)
        # self.camera_stack_coordinates = []
    
    def ready_frames_count(self):
        return self.frame_queue.qsize()
    
    def _inference_processing_worker(self):
        blabber("Starting inference preprocessing.")
        while not self.preprocess_frames(BATCH_SIZE):
            blabber(f"Processed a batch of {BATCH_SIZE} images")
            pass
        
    def _batch_processing_worker(self):
        blabber("Starting batch preprocessing.")
        while not self.pregenerate_batch_frames(BATCH_SIZE):
            blabber(f"Prepared a batch of {BATCH_SIZE} images")
            pass
    
    def start_preprocessing(self):
        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_processing_worker)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
        self.batch_thread = threading.Thread(target=self._batch_processing_worker)
        self.batch_thread.daemon = True
        self.batch_thread.start()
        print("Background producer started")
    
    
    def stop_preprocessing(self):
        self.running = False
        if self.inference_thread:
            self.inference_thread.join(timeout=10)
        if self.batch_thread:
            self.batch_thread.join(timeout=10)
        print("Background producer stopped")  
    
    def get_allowed_frame_skip(self):
        return SKIP_FRAMES if SKIP_FRAMES > 0 else 1
    
    def skip_frames(self, frames_to_skip: int, collect_skipped: bool = False):
        frames = []
        if frames_to_skip > 0:
                
            if collect_skipped:
                for _ in range(frames_to_skip):
                    frame = self.get_combined_frame()
                    if frame is None:
                        return frames
                    output_frame_width = int(SKIPPED_IMAGE_SIZE * self.aspect_ratio)
                    output_frame_height = SKIPPED_IMAGE_SIZE
                    output_frame = cv2.resize(frame, (output_frame_width, output_frame_height))
                    frames.append(output_frame)
            else:
                for camera in self.cameras:
                    for _ in range(frames_to_skip):
                        if not camera.cap.grab():
                            return frames
        return frames
    
    def pregenerate_batch_frames(self, num_frames: int):
        frames = []
        for _ in range(num_frames):
            self.skip_frames(self.get_allowed_frame_skip() - 1, collect_skipped=RENDER_SKIPPED_FRAMES)
            frame = self.get_combined_frame()
            if frame is None:
                break
            frames.append(frame)
        log("Batch generated")
        self.batch_queue.put(frames)
    
    def preprocess_frames(self, num_frames: int):
        
        # Limit the queue size
        if MAX_QUEUE_SIZE > 0:
            while self.frame_queue.qsize() > MAX_QUEUE_SIZE - BATCH_SIZE:
                sleep(QUEUE_SIZE_CHECK_INTERVAL)
                
        log(f"Queue freed up.")
        
        inference_frames = self.batch_queue.get()
        output_frames = inference_frames
        skipped_frameses = []
        finished = False
        log(f"Running inference on batch of {len(inference_frames)} frames.")
        if len(inference_frames) == 0:
            return True
        resultses = self.model.track(inference_frames, conf=0.25, persist=True, device=0, verbose=VERBOSE_YOLO)
        # self.frame_count += num_frames
        for i, results in enumerate(resultses):
            if RENDER_SKIPPED_FRAMES and self.last_results is not None:
                for skipped_frame in skipped_frameses[i]:
                    self.results_queue.put(self.last_results)
            self.results_queue.put(results)
            self.last_results = results
        for i, frame in enumerate(output_frames):
            if RENDER_SKIPPED_FRAMES:
                for skipped_frame in skipped_frameses[i]:
                    self.frame_queue.put(skipped_frame)
            self.frame_queue.put(frame)
        return finished
    
    def calculate_camera_stack_rects(self):
        for camera_index, camera in enumerate(self.cameras):
            width = self.inference_width // 2
            height = self.inference_height // 2
            x = (camera_index % 2) * width
            y = (camera_index // 2) * height
            w = width
            h = height
            camera.stack_rect = (x, y, w, h)
            log(f"Camera {camera.name} stack rect: {camera.stack_rect}")
    
    def get_combined_frame(self):
        try:
            log("Getting combined")
            
            mm
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_camera = {executor.submit(camera.get_inference_frame): camera 
                                for camera in self.cameras}
                frames = []
                for future in concurrent.futures.as_completed(future_to_camera):
                    frame = future.result()
                    if frame is not None:
                        frames.append(frame)
                
                return self._combine_pre_resized_frames(frames)
        except queue.Empty:
            return None

    def _combine_pre_resized_frames(self, frames):
        if not frames:
            return None
        
        # Create combined frame
        
        
        target_height = self.inference_height // 2
        target_width = self.inference_width // 2
        
        # Fill frames directly (frames are already resized)
        for i, frame in enumerate(frames):
            if i >= 4:
                break
            row = i // 2
            col = i % 2
            y = row * target_height
            x = col * target_width
            self.combined_frame[y:y+target_height, x:x+target_width] = frame
        
        return self.combined_frame
    
    def draw_rect_on_frame(self, frame, x_center, y_center, box_width, box_height, scale, bottle: Bottle = None, id_color=(255, 0, 0)):
        
        output_frame_width = frame.shape[1]
        output_frame_height = frame.shape[0]
        
        scaled_x_center = int(x_center * scale)
        scaled_y_center = int(y_center * scale)
        scaled_box_width = int(box_width * scale)
        scaled_box_height = int(box_height * scale)
        
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
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        if bottle is not None:
            cv2.putText(frame, 'Bottle ' + str(bottle.index), (int(x1 + 10), int(y1 + 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, id_color, 2)
        
        return x1, y1, y2, y2
    
    def wait_if_needed(self):
        now = time()
        time_passed = now - self.last_frame_time
        min_time_passed = 1 / DISPLAY_FRAMERATE
        
        if EASE_DISPLAY_SPEED:
            if self.ready_frames_count() > INCREASE_SPEED_AT:
                min_time_passed *= SPEED_MULTIPLIER
        
        if time_passed < min_time_passed:
            sleep_time = min_time_passed - time_passed
            if sleep_time > 0:
                queued_frames = self.ready_frames_count()
                blabber(f"I'm being rate limited. {queued_frames} frames are already prepared. Sleeping for {sleep_time} seconds...")
                sleep(sleep_time)
        
        self.last_frame_time = time()
    
    def get_frame(self):
        try:
            frame = self.frame_queue.get()
            results = self.results_queue.get()
            # log(f"Camera {self.name}: Retrieved frame {self.frame_queue.qsize()} items remaining.")
            return frame, results
        except queue.Empty:
            return None
    
    def solve_disagreements(self):
        last_bottle_indices = set()
        for camera in self.cameras:
            if camera.last_registered_bottle is not None:
                last_bottle_indices.add(camera.last_registered_bottle.index)
        
        for camera in self.cameras:
            if len(last_bottle_indices) > 1:
                # log("Warning: Cameras disagree on last registered bottle indices:", last_bottle_indices, "Camera number:", camera_index)
                camera.disagreement_count += 1
                
                if camera.disagreement_count >= BOTTLE_DISAGREEMENT_TOLERANCE:
                    self.correct_index_disagreements()
                    camera.disagreement_count = 0
    
    def run(self):
        self.start_preprocessing()
        while True:
            combined_frame, results = self.get_frame()
            
            output_frame_width = int(PREVIEW_IMAGE_SIZE * self.aspect_ratio)
            output_frame_height = PREVIEW_IMAGE_SIZE
            output_frame = cv2.resize(combined_frame, (output_frame_width, output_frame_height))

            scale = output_frame_height / self.inference_height
            
            for result in results:
                # Get results inside the camera stack frame rect
                if result.boxes is not None:
                    for camera in self.cameras:
                        x, y, w, h = camera.stack_rect
                    
                        boxes = result.boxes.xywh.cpu()
                        track_ids = None
                        if result.boxes.id is not None:
                            track_ids = result.boxes.id.cpu().numpy().astype(int)
                        
                        for box_index, box in enumerate(boxes):
                            x_center, y_center, box_width, box_height = box
                            
                            blabber("Processing box at:", x_center, y_center, "for camera", camera.name)
                            
                            # Check if the box is inside the camera's stack rect
                            abs_x_center = x_center
                            abs_y_center = y_center
                            
                            blabber("Absolute center:", abs_x_center, abs_y_center, "Camera rect:", camera.stack_rect)
                            
                            if x <= abs_x_center <= x + w and y <= abs_y_center <= y + h:
                                relative_x = (abs_x_center - x) / w
                                relative_y = (abs_y_center - y) / h
                                
                                if track_ids is None:
                                    continue
                                
                                track_id = track_ids[box_index]
                                
                                if camera.register_bottle(relative_x, relative_y, track_id):
                                    blabber("Bottle got accepted as being new.")
                                # Render box on frame
                                # if camera.name == "Top":
                                # self.draw_rect_on_frame(output_frame, abs_x_center, abs_y_center, box_width, box_height, scale)
                                
                                bottle = None
                                bottle_id_color = (0, 0, 255)
                                if track_id in camera.temporary_bottles:
                                    bottle = camera.temporary_bottles[track_id]
                                    bottle_id_color = (0, 0, 255)
                                if track_id in camera.bottles:
                                    bottle = camera.bottles[track_id]
                                    bottle_id_color = (255, 255, 0) if bottle.was_corrected else (0, 255, 0)
                                
                                self.draw_rect_on_frame(output_frame, abs_x_center, abs_y_center, box_width, box_height, scale, bottle, bottle_id_color)
                                
                                
                                # def draw_bottle_id(color):
                                    
                                
                if VERBOSE_DBUG:
                    cv2.putText(output_frame, f'Inference queue: {self.frame_queue.qsize()} batch queue: {self.batch_queue.qsize()}', (10, output_frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
           
                cv2.imshow("Inference Result", output_frame)
            
            self.solve_disagreements()
            
            self.wait_if_needed()
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p') or key == ord(' '):
                cv2.waitKey(-1)
                
            self.window_was_open = True
            
        log("Finished processing.")
        self.release()
        
    def correct_index_disagreements(self):
        # Find the most common last registered bottle index among cameras
        log("Correcting index disagreements among cameras.")
        last_indices = []
        for camera in self.cameras:
            if camera.last_registered_bottle is not None:
                last_indices.append(camera.last_registered_bottle.index)
        log("Last registered bottle indices from cameras:", last_indices)
        if not last_indices:
            return
        
        index_counts = Counter(last_indices)
        most_common_index, most_common_count = index_counts.most_common(1)[0]
        
        log(f"Most common last registered bottle index is {most_common_index} seen {most_common_count} times.")
        
        if ENFORCE_INCREMENTAL_CORRECTION and not most_common_index > self.last_corrected_index:
            log("Warning: I can't correct to an index that was used before. Incrementing might skip an index but ensures all indexes link to one bottle.")
            most_common_index = self.last_corrected_index + 1
        
        if most_common_count > len(self.cameras) / 2 or (EXTRA_CORRECTION and most_common_count == len(self.cameras) / 2):
            log("Majority agreement found, correcting outcast.")
            self.last_corrected_index = most_common_index
            for camera in self.cameras:
                if camera.last_registered_bottle is not None:
                    log(f"Correcting camera {camera.name} from index {camera.last_registered_bottle.index} to {most_common_index}")
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
        self.stop_preprocessing()
        for camera in self.cameras: camera.release()
        cv2.destroyAllWindows()


def main():
    cameras: list[Camera] = [
        Camera('Top', 'videos/14_55/14_55_top_cropped.mp4', start_skip=3),
        Camera('Front', 'videos/14_55/14_55_front_cropped.mp4', start_skip=0),
        
        Camera('Back Left', 'videos/14_55/14_55_back_left_cropped.mp4', start_skip=2),
        Camera('Back Right', 'videos/14_55/14_55_back_right_cropped.mp4', start_skip=1, start_index=-1),
    ]
    
    bottle_tracker = BottleTracker(cameras)
    
    log("Created cameras. Initiating tracking...")
    bottle_tracker.run()

if __name__ == '__main__':
    main()