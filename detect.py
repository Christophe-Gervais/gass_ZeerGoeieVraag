from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/detect/gasbottle_yolo11m_final/weights/best.pt")

video_paths = [
    "videos/14_55_front_cropped.mp4",
    "videos/14_55_top_cropped.mp4",
    "videos/14_55_back_right_cropped.mp4",
    "videos/14_55_back_left_cropped.mp4",
]

caps = [cv2.VideoCapture(vp) for vp in video_paths]

while True:
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(
            source=frame,
            conf=0.05,
            device="cuda:0",
            tracker="botsort.yaml",
            save=False,
        )

        annotated_frame = results[0].plot(
            line_width=5, 
            labels=True,    
            conf=False        
        )
        frames.append(annotated_frame)

    if len(frames) != 4:
        break

    top_row = np.hstack((frames[0], frames[1]))
    bottom_row = np.hstack((frames[2], frames[3]))
    grid = np.vstack((top_row, bottom_row))

    grid = cv2.resize(grid, (1600, 900))

    cv2.imshow("Gass Bottle Multi-View", grid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
