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
    
    first_bottle_widths = []
    
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    aspect_ratio = width / height
    adjusted_width = int(IMAGE_SIZE * aspect_ratio)
    adjusted_height = IMAGE_SIZE
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (adjusted_width, adjusted_height))
    
    bottles: list[Bottle] = []
    track_ids = []
    
    def register_bottle(x, y):
        bottle = Bottle(x, y)
        bottle.index = len(bottles) + 1
        bottles.append(bottle)
    
    bottle_was_entering = False
    
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
        # results = model.track(small_frame, conf=0.25, persist=True, save=SAVE_VIDEO, device=0)
        results = model(small_frame, conf=0.25, device=0)
        
        for result in results:
            # print(result.boxes)
            annotated_frame = result.plot()
            out.write(annotated_frame)
                
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xywh.cpu()
                confidences = result.boxes.conf.cpu()
                
                print("Boxes:", boxes)
                print("Confidences:", confidences)
                
                
                
                print("\n--- BOX SIZES ---")
                for i, box in enumerate(boxes):
                    x_center, y_center, box_width, box_height = box
                    box_area = box_width * box_height
                    print(f"Box {i} (ID: {len(bottles)}): {box_width:.1f}x{box_height:.1f} pixels, Area: {box_area:.1f} px²")
                    cv2.putText(annotated_frame, 'ID: ' + str(len(bottles)), (int(x_center), int(y_center)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # first_bottle_widths.append(float(box_width))k
                
                first_bottle_widths.append(float(boxes[0][2]))
            
                last_widths = first_bottle_widths[-LAST_WIDTH_COUNT:]
                print("Last two box widths:", last_widths)
                if len(last_widths) == LAST_WIDTH_COUNT:
                    change = 0
                    for i in range(0, LAST_WIDTH_COUNT - 1):
                        print(i)
                        change += last_widths[1 + i] - last_widths[0 + i]
                    
                    if change > WIDTH_CHANGE_THRESHOLD:
                        if not bottle_was_entering:
                            print("Bottle is entering.")
                            cv2.putText(annotated_frame, 'New bottle entering', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            bottle_was_entering = True
                            register_bottle()
                    else:
                        bottle_was_entering = False
                
                        
            
            # Displqy the frame
            cv2.imshow('Live Tracking Preview', annotated_frame)
    
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(-1)
                
                
                
        processed_count += 1
        
    # Plot box widths
    plt.figure(figsize=(10, 6))
    plt.plot(first_bottle_widths, marker='o', linestyle='-')
    plt.title('Box Widths Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Box Width (pixels)')
    plt.grid(True)
    # plt.savefig('box_widths_over_time.png')
    # plt.show()
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()