from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque

model = YOLO("runs/detect/gasbottle_yolo11m_final/weights/best.pt")

video_paths = [
    "videos/Front.mp4",
    "videos/BackL.mp4",
    "videos/BackR.mp4",
]

caps = [cv2.VideoCapture(vp) for vp in video_paths]

# Config
CHECK_EVERY_N_FRAMES = 1
FRAME_OFFSET = 0
TOLERANCE_FRAMES = 2
DISTANCE_THRESHOLD = 50  # pixel afstand voor matching centroids

camera_names = ["Front", "BackL", "BackR"]
frame_count = 0
global_bottle_id = 0

# Active bottles: each has global_id, last_seen_frame, camera local_ids
active_bottles = []

# ---------------- Helper Functions ---------------- #

def get_boxes_and_centroids(results):
    """
    Returns a list of dicts per box: {local_id, bbox, centroid}
    """
    boxes_info = []
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return boxes_info
    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        local_id = int(results[0].boxes.id[i].item())
        boxes_info.append({"local_id": local_id, "bbox": (x1, y1, x2, y2), "centroid": (cx, cy)})
    return boxes_info

def match_candidates(front_boxes, backl_boxes, backr_boxes):
    """
    Simple centroid matching: returns a list of matched triplets [(Front_box, BackL_box, BackR_box)]
    """
    matched = []
    used_bl = set()
    used_br = set()
    for fb in front_boxes:
        best_bl = None
        best_br = None
        # match Front -> BackL
        for i, bl in enumerate(backl_boxes):
            if i in used_bl:
                continue
            dist = np.linalg.norm(np.array(fb["centroid"]) - np.array(bl["centroid"]))
            if dist < DISTANCE_THRESHOLD:
                best_bl = (i, bl)
                break
        # match Front -> BackR
        for i, br in enumerate(backr_boxes):
            if i in used_br:
                continue
            dist = np.linalg.norm(np.array(fb["centroid"]) - np.array(br["centroid"]))
            if dist < DISTANCE_THRESHOLD:
                best_br = (i, br)
                break
        if best_bl and best_br:
            matched.append((fb, best_bl[1], best_br[1]))
            used_bl.add(best_bl[0])
            used_br.add(best_br[0])
    return matched

def assign_global_ids(matched_triplets, frame_count):
    global global_bottle_id, active_bottles
    for f, bl, br in matched_triplets:
        # Check if this combination matches any existing active bottle
        found = False
        for bottle in active_bottles:
            ids_set = {bottle["cam_ids"]["Front"], bottle["cam_ids"]["BackL"], bottle["cam_ids"]["BackR"]}
            current_set = {f["local_id"], bl["local_id"], br["local_id"]}
            if ids_set & current_set:  # any overlap → same bottle
                # update local IDs (persistent)
                bottle["cam_ids"]["Front"].add(f["local_id"])
                bottle["cam_ids"]["BackL"].add(bl["local_id"])
                bottle["cam_ids"]["BackR"].add(br["local_id"])
                bottle["last_seen_frame"] = frame_count
                found = True
                break
        if not found:
            # New bottle
            new_bottle = {
                "global_id": global_bottle_id,
                "cam_ids": {
                    "Front": set([f["local_id"]]),
                    "BackL": set([bl["local_id"]]),
                    "BackR": set([br["local_id"]])
                },
                "last_seen_frame": frame_count
            }
            active_bottles.append(new_bottle)
            print(f"✅ New bottle #{global_bottle_id} assigned: Front {f['local_id']}, BackL {bl['local_id']}, BackR {br['local_id']}")
            global_bottle_id += 1

def draw_boxes(frame, boxes, cam):
    annotated = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        # check if any active bottle
        label = "?"
        color = (0, 165, 255)
        for bottle in active_bottles:
            if box["local_id"] in bottle["cam_ids"][cam]:
                label = f"#{bottle['global_id']}"
                color = (0, 255, 0)
                break
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cv2.rectangle(annotated, (x1, y1 - h - 5), (x1 + w + 5, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
    return annotated

# ---------------- Main Loop ---------------- #

# Skip initial frames if needed
for _ in range(FRAME_OFFSET):
    for cap in caps:
        cap.read()

while True:
    frames = []
    all_results = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    if len(frames) != 3:
        break
    frame_count += 1

    # Get tracking results
    for frame in frames:
        results = model.track(
            source=frame,
            conf=0.8,
            device="cuda:0",
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )
        all_results.append(results)

    # Extract boxes + centroids
    front_boxes = get_boxes_and_centroids(all_results[0])
    backl_boxes = get_boxes_and_centroids(all_results[1])
    backr_boxes = get_boxes_and_centroids(all_results[2])

    # Match boxes across cameras
    matched_triplets = match_candidates(front_boxes, backl_boxes, backr_boxes)

    # Assign or update global IDs
    assign_global_ids(matched_triplets, frame_count)

    # Remove bottles not seen for a while
    active_bottles = [b for b in active_bottles if frame_count - b["last_seen_frame"] <= TOLERANCE_FRAMES]

    # Draw boxes
    annotated_frames = [
        draw_boxes(frames[0], front_boxes, "Front"),
        draw_boxes(frames[1], backl_boxes, "BackL"),
        draw_boxes(frames[2], backr_boxes, "BackR")
    ]


    # Camera labels
    for i, cam in enumerate(camera_names):
        cv2.putText(annotated_frames[i], cam, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

    # Stack frames
    target_height = 480
    annotated_frames = [cv2.resize(f, (int(f.shape[1]*target_height/f.shape[0]), target_height)) for f in annotated_frames]
    grid = np.hstack(annotated_frames)
    grid = cv2.resize(grid, (1600, 540))

    cv2.imshow("Loopband Multi-Camera Tracking", grid)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
