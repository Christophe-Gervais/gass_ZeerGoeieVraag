from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
from collections import Counter
import threading
import queue
from time import time, sleep, perf_counter
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import math
import matplotlib.pyplot as plt


# Input options
MODEL_PATH = "runs/detect/train26/weights/best.pt"
INPUT_VIDEO_FPS = 60
EXTRA_CAMERA_DELAY = 1  # Delay in seconds
MAX_FRAMES = 1000000 # The amount of frames to process before quitting

# Algorithm options
IMAGE_SIZE = 320
BATCH_SIZE = 7
FRAMES_TO_SKIP = 1 # Skip this many frames between each processing step, -1 to disable.
TEMPORAL_CUTOFF_THRESHOLD = 40  # Amount of frames a bottle needs to be seen to be considered tracked.
PRECOMBINE = False

# Correction algorithm options
BOTTLE_CORRECTION_START_OFFSET = 20 # Amount of frames to wait before allowing the correction algorithm to kick in.
BOTTLE_DISAGREEMENT_TOLERANCE = 5  # Amount of frames the cameras can disagree before correction is applied.
SEQUENTIAL_CORRECTION_THRESHOLD = 2 # If a tracker has to be corrected this many times in a row, it's permanently steered back on track.
ENFORCE_INCREMENTAL_CORRECTION = False # Make sure the corrected index is unique.
EXTRA_CORRECTION = False # Allow correcting half the feed if one half disagrees with itself.
LOWER_DISPUTE_CORRECTION = True

# Counting algorithm options
COUNT_BY_BOTTLE_SIZE = True # Use the bottle size to count bottles.
SIZE_STREAK_THRESHOLD = 3 # How many times in a row the bottle has to increase in size for it to be considered a new bottle.
SIZE_INCREASE_THRESHOLD = 0 # How much the bottle has to increase in size to be considered entering the frame.
MOVEMENT_THRESHOLD = 10 # How much the bottle has to be moved for the size change to be registered. This is to prevent problems when the belt is stopped.

# Size change algorithm options
SIZE_CHANGE_THRESHOLD = 2.5 # How much the value has to change for it to be considered a . Beyond which value is the bottle considered to be entering of exiting the frame.
FRAME_CHANGE_COUNT = 3 # How many frames to compare the size change on

YOLO_CONF = 0.8

# Preview options
PREVIEW_IMAGE_SIZE = 320
SAVE_VIDEO = False
PREVIEW_WINDOW_NAME = "Live Tracking Preview"
DISPLAY_FRAMERATE = 2 # Display fps
MAX_QUEUE_SIZE = 20 # The limit for the queue size, set to -1 to disable limit (but beware you might run out of memory then!)
QUEUE_SIZE_CHECK_INTERVAL = 0.1 # Amount of seconds to wait when queue is full
RENDER_SKIPPED_FRAMES = False # Whether to render skipped frames in between processed frames
SKIPPED_IMAGE_SIZE = 200
MAXIMIZE_DISPLAY_SPEED = True # Speed up display when enough frames are queued
INCREASE_SPEED_AT = 150 # If more than this many frames are queued, increase display speed
SPEED_MULTIPLIER = 0.2 # How much to speed up when easing display speed (lower is faster)

# Logging options
VERBOSE_YOLO = False # Show YOLO debug info
VERBOSE_LOGS = True # Show general info
VERBOSE_BLAB = False # Show detailed debug info
VERBOSE_DBUG = True # Show debug info
VERBOSE_PERF = False
VERBOSE_PLOT = False

def main():
    cameras: list[Camera] = [
        # Camera('Top', 'videos/14_55/14_55_top_cropped.mp4', start_skip=3),
        Camera('Front', 'videos/14_55/14_55_front_cropped.mp4', start_skip=0),
        
        # Camera('Back Left', 'videos/14_55/14_55_back_left_cropped.mp4', start_skip=2),
        # Camera('Back Right', 'videos/14_55/14_55_back_right_cropped.mp4', start_skip=1, start_index=-1),
    ]
    
    bottle_tracker = BottleTracker(cameras)
    
    log("Created cameras. Initiating tracking...")
    if PRECOMBINE:
        bottle_tracker.run()
    else:
        # bottle_tracker.run_without_precombined()
        bottle_tracker.run(False)

def log(*values: object, **kwargs):
    if not VERBOSE_LOGS: return
    print(*values, *kwargs)
    
def blabber(*values: object, **kwargs):
    if not VERBOSE_BLAB: return
    print(*values, *kwargs)

def get_frame_skip_divider():
    return FRAMES_TO_SKIP if FRAMES_TO_SKIP > 0 else 1

class PerformanceMeter:
    def __init__(self):
        self.start_time = perf_counter()
    
    def elapsed(self):
        return perf_counter() - self.start_time
    
    def log_elapsed(self, message: str):
        if VERBOSE_PERF:
            print(message, f"took {self.elapsed()} seconds.")
        
class BottleState(Enum):
    UNKNOWN = 0
    ENTERING = 1
    IN_FRAME = 2
    EXITING = 3
    OUT_OF_FRAME = 4

class Vector:
    x: float
    y: float

class Plotter:
    def __init__(self, yolo_id):
        self._plot_initialized = False
        self._fig = None
        self._ax1 = None
        self._ax2 = None
        self._ax3 = None
        self._size_line = None
        self._dist_line = None
        self._dsize_line = None
        
        self._fig, (self._ax1, self._ax2, self._ax3) = plt.subplots(3, 1, figsize=(6, 6))
        self._fig.suptitle(f"Bottle YOLO ID: {yolo_id} (Live)")
        
        self._ax1.set_ylabel("Size")
        self._ax1.grid(True)
        (self._size_line,) = self._ax1.plot([], [], 'b-o', label='Size')
        self._ax1.legend()

        self._ax2.set_xlabel("Frame")
        self._ax2.set_ylabel("Move Dist")
        self._ax2.grid(True)
        (self._dist_line,) = self._ax2.plot([], [], 'r-o', label='Movement')
        self._ax2.legend()
        
        self._ax3.set_ylabel("Size Change over x")
        self._ax3.grid(True)
        (self._dsize_line,) = self._ax3.plot([], [], 'b-o', label='Size Change')
        self._ax3.legend()

        self._plot_initialized = True
        self._fig.canvas.draw()
        plt.show(block=False)
    
    def plot(self, size_hist: list[tuple[float, float, float]]):
        frames = list(range(len(size_hist)))
        sizes = [s for s, d, ds in size_hist]
        dists = [d for s, d, ds in size_hist]
        dsizes = [ds for s, d, ds in size_hist]

        self._size_line.set_data(frames, sizes)
        self._dist_line.set_data(frames, dists)
        self._dsize_line.set_data(frames, dsizes)

        self._ax1.relim()
        self._ax1.autoscale_view()
        self._ax2.relim()
        self._ax2.autoscale_view()
        self._ax3.relim()
        self._ax3.autoscale_view()

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
    
    def release(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None

class Bottle:
    index: int = -1
    x: float
    y: float
    prev_x: float
    prev_y: float
    in_view: bool = True
    yolo_id: int
    was_corrected: bool = False
    width: float
    is_ok: bool
    times_seen: int
    state: BottleState
    bottle_size_dist_history: list[tuple[float, float]]
    last_size_change: float
    plotter: Plotter
    
    def __init__(self, x: float, y: float, yolo_id: int, is_ok = True):
        self.times_seen = 0
        self.update(x, y, is_ok)
        self.prev_x = 0
        self.prev_y = 0
        # self.x = x
        # self.y = y
        self.yolo_id = yolo_id
        self.bottle_size_dist_history = list()
        # self.is_ok = is_ok
        self.state = BottleState.UNKNOWN
        
    
        self.plotter = Plotter(self.yolo_id) if VERBOSE_PLOT else None
    
    def update(self, x: float, y: float, is_ok: bool | None):
        self.x = x
        self.y = y
        if is_ok is not None:
            self.is_ok = is_ok
        self.times_seen += 1
    
    def calculate_size_change_from_x_number_of_frames(self, x) -> float:
        if len(self.bottle_size_dist_history) < x:
            return 0.0
        
        history = self.bottle_size_dist_history[-x:]
        
        total_change = 0.0
        for i in range(1, len(history)):
            total_change += history[i][0] - history[i - 1][0]

        return total_change / (len(history) - 1)
    
    def calculate_state_change(self) -> BottleState:
        # size_streak = 0
        last_size = 0
        size_change = 0
        # for size_dist in self.bottle_size_dist_history:
        #     size = size_dist[0]
        #     dist = size_dist[1]
        #     if size <= last_size + SIZE_INCREASE_THRESHOLD:
        #         log("This one didn't grow enough.")
        #         break
        #     last_size = size
            # size_streak += 1
        # print("Size streak:", size_streak, " YOLO ID: ", self.yolo_id)
        # if size_streak > SIZE_STREAK_THRESHOLD / get_frame_skip_divider():
        #     # self.promote_bottle(track_id)
        #     # # bottle.bottle_size_history.clear()
        #     # return True
        #     return BottleState.ENTERING
        # elif size_streak < SIZE_STREAK_THRESHOLD / get_frame_skip_divider():
            # self.promote_bottle(track_id)
            # # bottle.bottle_size_history.clear()
            # return True
        blabber("Last size change is self.last_size_change", self.last_size_change)
            
        if self.last_size_change == 0.0:
            return BottleState.UNKNOWN
        if self.last_size_change > SIZE_CHANGE_THRESHOLD:
            return BottleState.ENTERING
        elif self.last_size_change < -SIZE_CHANGE_THRESHOLD:
            return BottleState.EXITING
        else:
            return BottleState.IN_FRAME
    
    def register_state_change(self, size) -> BottleState:
        distance = math.dist((self.x, self.y), (self.prev_x, self.prev_y))
        dsize = self.calculate_size_change_from_x_number_of_frames(FRAME_CHANGE_COUNT)
        self.last_size_change = dsize
        self.bottle_size_dist_history.append((size, distance, dsize))
        
        
        
        self.state = self.calculate_state_change()
        blabber("Made me think it should be", self.state)
        if self.state is BottleState.EXITING and self.plotter is not None:
            self.plotter.release()
            self.plotter = None
        
        if self.plotter is not None:
            self.plotter.plot(self.bottle_size_dist_history)
        
        self.prev_x = self.x
        self.prev_y = self.y
        return self.state

class Batch:
    def __init__(self, frames, is_precombined = True):
        self.frames: list[cv2.typing.MatLike] = frames
        self.is_precombined = is_precombined

class FrameGenerator:
    def __init__(self):
        self.results_queue = queue.Queue()
        self.frame_queue: queue.Queue[cv2.typing.MatLike] = queue.Queue()
        self.model = YOLO(MODEL_PATH)
        pass
    def get_frame(self):
        try:
            return self.frame_queue.get(), self.results_queue.get()
        except queue.Empty:
            return None
    
    def draw_rect_on_frame(self, frame, x_center, y_center, box_width, box_height, scale, bottle: Bottle = None):
        
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
        bounding_color = (0, 0, 255) # Red
        if bottle.state is BottleState.IN_FRAME:
            bounding_color = (0, 255, 0) # Green
        elif bottle.state is BottleState.ENTERING:
            bounding_color = (0, 255, 255) # Yellow
        elif bottle.state is BottleState.EXITING:
            bounding_color = (255, 0, 255) # Purple
            
        bottle_id_color = (0, 0, 255)
        if bottle is not None:
            if bottle.was_corrected:
                bottle_id_color = (255, 255, 0)  # Geel voor gecorrigeerde bottles
            elif bottle.is_ok:
                bottle_id_color = (0, 255, 0)  # Groen voor OK
            else:
                bottle_id_color = (0, 0, 255)  # Rood voor NOK
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), bounding_color, thickness)
        
        if bottle is not None:
            # cv2.putText(frame, 'Bottle ' + str(bottle.index), (int(x1 + 10), int(y1 + 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, id_color, 2)
            status_text = "OK" if bottle.is_ok else "NOK"
            label = f'Bottle {bottle.index} ({status_text})'
            # Changed font scale from 1 to 0.4 and thickness from 2 to 1 for smaller text
            cv2.putText(frame, label, (int(x1 + 5), int(y1 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bottle_id_color, 2)
            cv2.putText(frame, 'YOLO ID: ' + str(bottle.yolo_id), (int(x1 + 10), int(y1 + 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return x1, y1, y2, y2

class Camera(FrameGenerator):
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
    last_bottle_index: int = 0
    # start_delay: float = 0 # seconds
    capture_fps: float
    frames_since_last_registration: int = 0
    last_registered_bottle: Bottle = None
    sequential_correction_count: int = 0
    
    
    frame_index: int
    stack_rect: tuple[int, int, int, int]  # x, y, w, h
    
    def __init__(self, name: str, video_path: str, start_skip: int = 0, start_index: int = 0):
        super().__init__()
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
            self.out = cv2.VideoWriter(self.output_path, fourcc, self.capture_fps / FRAMES_TO_SKIP, (self.adjusted_width, self.adjusted_height))
        
        self.bottles = {}
        self.last_bottle_index = start_index
        
        start_skip += EXTRA_CAMERA_DELAY
        frames_to_skip = int(start_skip * self.capture_fps)
        log(f"Camera {self.name}: Skipping first {frames_to_skip} frames for start delay of {start_skip} seconds.")
        self.skip_frames(frames_to_skip)
        
        self.processed_frame_count = 0
        
        blabber("Finished camera setup. Loading model...")
        # self.model = YOLO(MODEL_PATH)
        blabber("Finished loading model!")
        
        
        # self.results_queue = queue.Queue()
        # self.frame_queue: queue.Queue[cv2.typing.MatLike] = queue.Queue()
        self.frame_index = 0
        self.running = False
        self.producer_thread = None
        self.last_output_frame: cv2.typing.MatLike = None
        
        
        self.stack_rect = (0, 0, self.width, self.height)
        
        self.disagreement_count = 0
        
        self.inference_width = int(IMAGE_SIZE * self.aspect_ratio)
        self.inference_height = IMAGE_SIZE
    
    def preprocess_frames(self, num_frames: int):
        
        # Limit the queue size
        if MAX_QUEUE_SIZE > 0:
            while self.frame_queue.qsize() > MAX_QUEUE_SIZE - BATCH_SIZE:
                sleep(QUEUE_SIZE_CHECK_INTERVAL)
        
        inference_frames = []
        output_frames = []
        skipped_frameses = []
        finished = False
        for _ in range(num_frames):
            skipped_frames = self.skip_frames(get_frame_skip_divider() - 1, collect_skipped=RENDER_SKIPPED_FRAMES)
            skipped_frameses.append(skipped_frames)
            ret, frame = self.cap.read()
            
            if not ret:
                log("No more frames to read from video.")
                finished = False
                break
            # log(self.adjusted_width, self.adjusted_height)
            inference_frame = cv2.resize(frame, (self.adjusted_width, self.adjusted_height))
            
            output_frame_width = int(PREVIEW_IMAGE_SIZE * self.aspect_ratio)
            output_frame_height = PREVIEW_IMAGE_SIZE
            output_frame = cv2.resize(frame, (output_frame_width, output_frame_height))
            
            inference_frames.append(inference_frame)
            output_frames.append(output_frame)
        log(f"Camera {self.name}: Running inference on batch of {len(inference_frames)} frames.")
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
    
    def render_frame(self):
        frame, results = self.get_frame()
        
        output_frame_width = int(PREVIEW_IMAGE_SIZE * self.aspect_ratio)
        output_frame_height = PREVIEW_IMAGE_SIZE
        
        frame = cv2.resize(frame, (output_frame_width, output_frame_height))
        # output_frame = cv2.resize(frame, (output_frame_width, output_frame_height))
        x_scale = output_frame_width / self.adjusted_width
        y_scale = output_frame_height / self.adjusted_height
        
        blabber(f"I got {len(results)} results? Yes bru")
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xywh.cpu()
                self.track_ids = None
                if result.boxes.id is not None:
                    self.track_ids = result.boxes.id.cpu().numpy().astype(int)
                
                cv2.putText(frame, 'Camera: ' + self.name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, 'Correction count: ' + str(self.sequential_correction_count), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                
                for box_index, box in enumerate(boxes):
                    x_center, y_center, box_width, box_height = box
                    
                    if self.track_ids is not None:
                        track_id = self.track_ids[box_index]
                        
                        if self.register_bottle(x_center, y_center, box_width, box_height, track_id):
                            log("Bottle got accepted as being new.")
                        
                        bottle = None
                        if track_id in self.temporary_bottles:
                            bottle = self.temporary_bottles[track_id]
                        if track_id in self.bottles:
                            bottle = self.bottles[track_id]
                        
                        self.draw_rect_on_frame(frame, x_center, y_center, box_width, box_height, x_scale, bottle)
                
        
        if VERBOSE_DBUG:
            cv2.putText(frame, f'Queue size: {self.frame_queue.qsize()}', (10, output_frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)        
        
        self.last_output_frame = frame
        
        if SAVE_VIDEO:
            self.out.write(frame)
            
        self.finish_frame()
        return self.last_output_frame
    
    def get_inference_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.resize(frame, (self.inference_width // 2, self.inference_height // 2))
        return frame
        
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
        
    def promote_bottle(self, track_id):
        bottle = self.temporary_bottles[track_id]
        self.last_bottle_index += 1
        bottle.index = self.last_bottle_index
        self.bottles[track_id] = bottle
        blabber("Bottle assigned index:", bottle.index, "Track ID:", track_id, "Camera:", self.name)
        del self.temporary_bottles[track_id]
        self.last_registered_bottle = bottle
        
    def register_bottle(self, x, y, width, height, track_id, is_ok = True) -> bool:
        if track_id in self.bottles:
            self.bottles[track_id].update(x, y, is_ok)
            return False 
        
        if track_id in self.temporary_bottles:
            bottle = self.temporary_bottles[track_id]
            bottle.update(x, y, is_ok)
            if COUNT_BY_BOTTLE_SIZE:
                # bottle.bottle_size_history.append(width)
                # size_streak = 0
                # last_size = 0
                # for size in bottle.bottle_size_history:
                #     if size <= last_size + SIZE_INCREASE_THRESHOLD:
                #         log("This one didn't grow enough.")
                #         break
                #     last_size = size
                #     size_streak += 1
                # print("Size streak:", size_streak, " YOLO ID: ", bottle.yolo_id)
                # if size_streak > SIZE_STREAK_THRESHOLD / get_frame_skip_divider():
                #     self.promote_bottle(track_id)
                state = bottle.register_state_change(width)
                    # bottle.bottle_size_history.clear()
                if state is BottleState.ENTERING:
                    return True
            else:
                if self.track_ids_seen[track_id] >= TEMPORAL_CUTOFF_THRESHOLD / get_frame_skip_divider():
                    self.promote_bottle(track_id)
                    return True

            return False
        bottle = Bottle(x, y, track_id, is_ok)
        self.temporary_bottles[track_id] = bottle
        self.track_ids_seen[track_id] = 1
        
        return False

    def register_correction(self, corrected_index):
        self.sequential_correction_count += 1
        if self.sequential_correction_count > TEMPORAL_CUTOFF_THRESHOLD / get_frame_skip_divider():
            self.last_bottle_index = corrected_index
            log(f"I, Camera {self.name}, was wrong {self.sequential_correction_count} in a row. I really thought I was right but I guess I wasn't. As punishment I will correct myself, remember that the correct index from now on is {corrected_index} and I will try my best to never do this again. I'm so sorry.")

class TrackingAlgorithm(Enum):
    TEMPORAL = 0
    SIZE = 1
    SIZE_CHANGE = 2

class BottleTracker(FrameGenerator):
    cameras: list[Camera]
    track_ids: list[int]
    bottle_was_entering: bool = False
    window_was_open: bool = False
    
    camera_disagreement_counts: dict[int, int] = {}
    last_corrected_index: int = -1
    
    last_frame_time = time()
    
    algorithm: TrackingAlgorithm
    
    def __init__(self, cameras: list[Camera], algorithm: TrackingAlgorithm = TrackingAlgorithm.SIZE):
        super().__init__()
        self.cameras = cameras
        self.track_ids = []
        
        # self.frame_queue: queue.Queue[cv2.typing.MatLike] = queue.Queue()
        # self.results_queue = queue.Queue()
        self.batch_queue: queue.Queue[Batch] = queue.Queue()
        self.last_results = None
        
        self.aspect_ratio = cameras[0].aspect_ratio
        self.inference_width = int(IMAGE_SIZE * self.aspect_ratio)
        self.inference_height = IMAGE_SIZE
        
        self.algorithm = algorithm
        
        self.calculate_camera_stack_rects()
        
        self.batch_thread = None
        self.inference_thread = None
        self.combined_frame = np.zeros((self.inference_height, self.inference_width, 3), dtype=np.uint8)
        # self.camera_stack_coordinates = []
    
    def ready_frames_count(self):
        return self.frame_queue.qsize()
    
    def _inference_processing_worker(self):
        blabber("Starting inference preprocessing.")
        while not self.preprocess_frames():
            blabber(f"Processed a batch of {BATCH_SIZE} images")
            pass
        
    def _batch_processing_worker(self):
        blabber("Starting batch preprocessing.")
        while self.pregenerate_batch_frames(BATCH_SIZE):
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
    
    def skip_frames(self, frames_to_skip: int, collect_skipped: bool = False):
        frames = []
        if frames_to_skip > 0:
                
            if collect_skipped:
                for _ in range(frames_to_skip):
                    frame = self.get_combined_frame_parallel()
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
        
        self.wait_if_queue_full(self.batch_queue)
        
        more = True
        frames: list[cv2.typing.MatLike] = []
        meter = PerformanceMeter()
        for _ in range(num_frames):
            meter = PerformanceMeter()
            self.skip_frames(FRAMES_TO_SKIP - 1, collect_skipped=RENDER_SKIPPED_FRAMES)
            meter = PerformanceMeter()
            combined_frame = self.get_combined_frame()
            meter.log_elapsed("Combining frames")
            if combined_frame is None:
                more = False
                break
            frames.append(combined_frame)
        
        meter.log_elapsed("Batch generation")
        batch = Batch(frames)
        self.batch_queue.put(batch)
        return more
    
    def wait_if_queue_full(self, queue: queue.Queue):
        if MAX_QUEUE_SIZE > 0:
            while queue.qsize() > MAX_QUEUE_SIZE - BATCH_SIZE:
                sleep(QUEUE_SIZE_CHECK_INTERVAL)
    
    def preprocess_frames(self):
        
        # Limit the queue size
        self.wait_if_queue_full(self.frame_queue)
                
        log(f"Queue freed up.")
        batch = self.batch_queue.get()
        
        inference_frames = batch.frames
        output_frames = inference_frames
        skipped_frameses = []
        finished = False
        if len(inference_frames) == 0:
            return True
        log(f"Running inference on batch of {len(inference_frames)} frames.")
        meter = PerformanceMeter()
        resultses = self.model.track(inference_frames, conf=YOLO_CONF, persist=True, device=0, verbose=VERBOSE_YOLO)
        elapsed_time = meter.elapsed()
        ips = BATCH_SIZE / elapsed_time
        meter.log_elapsed(f"Finished inferencing on {BATCH_SIZE} images ({ips} images/s).")
        # log(f"Finished inferencing. Took {meter.elapsed()} seconds for {BATCH_SIZE} images ({ips} images/s).")
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
    
    def get_combined_frame(self, parallel = False):
        try:
            # log("Getting combined")
            
            target_height = self.inference_height // 2
            target_width = self.inference_width // 2
            
            combined_frame = np.zeros((self.inference_height, self.inference_width, 3), dtype=np.uint8)
            
            if not parallel:
                # Sequential but optimized reading
                for i, camera in enumerate(self.cameras):
                    frame = camera.get_inference_frame()
                    if frame is not None:
                        # Resize immediately to target size
                        frame = cv2.resize(frame, (self.inference_width // 2, self.inference_height // 2))
                        # frames.append(frame)
                        if i >= 4:
                            break
                        row = i // 2
                        col = i % 2
                        y = row * target_height
                        x = col * target_width
                        combined_frame[y:y+target_height, x:x+target_width] = frame
                
                    
                    # return self._combine_pre_resized_frames(frames)
                return combined_frame
            else:
                frames = [None] * 4
                with ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_idx = {
                        executor.submit(self.cameras[i].get_inference_frame): i 
                        for i in range(min(4, len(self.cameras)))
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            frame = future.result()
                            if frame is not None:
                                resized_frame = cv2.resize(frame, (target_width, target_height))
                                row = idx // 2
                                col = idx % 2
                                y = row * target_height
                                x = col * target_width
                                self.combined_frame[y:y+target_height, x:x+target_width] = resized_frame
                        except Exception as e:
                            log(f"Camera {idx} failed: {e}")
                
                return self.combined_frame
        except queue.Empty:
            return None
    
    def get_combined_frame_parallel(self):
        try:
            target_height = self.inference_height // 2
            target_width = self.inference_width // 2
            
            # Get frames in parallel
            frames = [None] * 4
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_idx = {
                    executor.submit(self.cameras[i].get_inference_frame): i 
                    for i in range(min(4, len(self.cameras)))
                }
                
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        frame = future.result()
                        if frame is not None:
                            resized_frame = cv2.resize(frame, (target_width, target_height))
                            row = idx // 2
                            col = idx % 2
                            y = row * target_height
                            x = col * target_width
                            self.combined_frame[y:y+target_height, x:x+target_width] = resized_frame
                    except Exception as e:
                        log(f"Camera {idx} failed: {e}")
            
            return self.combined_frame
        except Exception as e:
            log(f"Combined frame error: {e}")
            return None
    
    
    
    def wait_if_needed(self):
        now = time()
        time_passed = now - self.last_frame_time
        min_time_passed = 1 / DISPLAY_FRAMERATE
        
        if MAXIMIZE_DISPLAY_SPEED:
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
            return self.frame_queue.get(), self.results_queue.get()
        except queue.Empty:
            return None
    
    
    
    def run(self, precombined = True):
        if precombined:
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
                            class_ids = None
                            
                            if result.boxes.id is not None:
                                track_ids = result.boxes.id.cpu().numpy().astype(int)
                                
                            
                            # Fetch the class labels op (0 = OK, 1 = NOK)
                            if result.boxes.cls is not None:
                                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                            
                            for box_index, box in enumerate(boxes):
                                x_center, y_center, box_width, box_height = box
                                
                                blabber("Absolute center:", x_center, y_center, "for camera", camera.name, "Camera rect:", camera.stack_rect)
                                
                                # Check if the box is inside the camera's stack rect
                                if x <= x_center <= x + w and y <= y_center <= y + h:
                                    relative_x = (x_center - x) / w
                                    relative_y = (y_center - y) / h
                                    
                                    if track_ids is None:
                                        continue
                                    
                                    track_id = track_ids[box_index]
                                    
                                    is_ok = True
                                    if class_ids is not None:
                                        is_ok = (class_ids[box_index] == 0)
                                    
                                    if camera.register_bottle(relative_x, relative_y, box_width, box_height, track_id, is_ok):
                                        blabber("Bottle got accepted as being new.")
                                    # Render box on frame
                                    # if camera.name == "Top":
                                    # self.draw_rect_on_frame(output_frame, abs_x_center, abs_y_center, box_width, box_height, scale)
                                    
                                    bottle = None
                                    bottle_id_color = (0, 0, 255)
                                    if track_id in camera.temporary_bottles:
                                        bottle = camera.temporary_bottles[track_id]
                                        bottle_id_color = (0, 255, 255)
                                    if track_id in camera.bottles:
                                        bottle = camera.bottles[track_id]
                                        # bottle_id_color = (255, 255, 0) if bottle.was_corrected else (0, 255, 0)
                                        # Groen voor OK bottles, Rood voor NOK bottles
                                        if bottle.was_corrected:
                                            bottle_id_color = (255, 255, 0)  # Geel voor gecorrigeerde bottles
                                        elif bottle.is_ok:
                                            bottle_id_color = (0, 255, 0)  # Groen voor OK
                                        else:
                                            bottle_id_color = (0, 0, 255)  # Rood voor NOK
                                    
                                    self.draw_rect_on_frame(output_frame, x_center, y_center, box_width, box_height, scale, bottle)
                                    
                    if VERBOSE_DBUG:
                        cv2.putText(output_frame, f'Inference queue: {self.frame_queue.qsize()} batch queue: {self.batch_queue.qsize()}', (10, output_frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
                    cv2.imshow("Inference Result", output_frame)
                
                self.solve_disagreements()
                
                self.wait_if_needed()
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.release()
                    break
                elif key == ord('p') or key == ord(' '):
                    cv2.waitKey(-1)
                    
                self.window_was_open = True

        else:
            for camera in self.cameras:
                    camera.start_preprocessing()
            while True:
                frames = []
                
                for camera_index, camera in enumerate(self.cameras):
                    frame, results = camera.get_frame()
                    
                    if camera.processed_frame_count >= MAX_FRAMES or not camera.is_open():
                        log(f"Done. {camera.processed_frame_count} frames processed.")
                        break
                        
                    frames.append(camera.render_frame())
                                    
                # Check if all cameras agree with each other on the last bottle index
                last_bottle_indices = set()
                for camera in self.cameras:
                    if camera.last_registered_bottle is not None:
                        last_bottle_indices.add(camera.last_registered_bottle.index)
                
                if len(last_bottle_indices) > 1:
                    # log("Warning: Cameras disagree on last registered bottle indices:", last_bottle_indices, "Camera number:", camera_index)
                    self.camera_disagreement_counts[camera_index] = self.camera_disagreement_counts.get(camera_index, 0) + 1
                    
                    if self.camera_disagreement_counts[camera_index] >= BOTTLE_DISAGREEMENT_TOLERANCE:
                        self.correct_index_disagreements()
                    
                
                # If the last frame was displayed recently, wait
                now = time()
                time_passed = now - self.last_frame_time
                min_time_passed = 1 / DISPLAY_FRAMERATE
                if time_passed < min_time_passed:
                    sleep_time = min_time_passed - time_passed
                    if sleep_time > 0:
                        queued_frames = camera.get_ready_frames_count()
                        blabber(f"I'm being rate limited. {queued_frames} frames are already prepared. Sleeping for {sleep_time} seconds...")
                        sleep(sleep_time)
                
                self.last_frame_time = time()
                
                if self.window_was_open and self.is_window_closed(PREVIEW_WINDOW_NAME):
                    log("Window closed, exiting.")
                    break
                
                # Display the frame
                try:
                    camera_count = len(self.cameras)
                    frame_rows = self.split_array(frames, 2)
                    row_frames = []
                    for row in frame_rows:
                        while len(row) < 2:
                            row.append(np.zeros_like(row[0]))
                        row_frame = np.hstack(row)
                        row_frames.append(row_frame)
                
                    combined_frame = np.vstack(row_frames)
                    cv2.imshow(PREVIEW_WINDOW_NAME, combined_frame)
                except:
                    log("Error combining frames for preview:")
                    # combined_frame = np.hstack(frames)
                
                    
                

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.waitKey(-1)
                    
                self.window_was_open = True
                
        log("Finished processing.")
        self.release()
    
    # def run_without_precombined(self):
        # frames: dict[Camera, cv2.typing.MatLike] = []
        
        
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
        
    def correct_index_disagreements(self):
        # Find the most common last registered bottle index among cameras
        log("Correcting index disagreements among cameras.")
        last_indices = []
        for camera in self.cameras:
            if camera.last_registered_bottle is not None:
                if camera.last_registered_bottle.times_seen < BOTTLE_CORRECTION_START_OFFSET:
                    return
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
        
        if most_common_count > len(self.cameras) / 2 or (EXTRA_CORRECTION and most_common_count == len(self.cameras) / 2) or (LOWER_DISPUTE_CORRECTION and most_common_count > 1 and all(most_common_index <= x for x in index_counts)):
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

if __name__ == '__main__':
    main()