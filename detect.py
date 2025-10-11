from ultralytics import YOLO

model = YOLO("runs/detect/gasbottle_yolo11n_test2/weights/best.pt")

input_video = "videos/14_55_front_cropped.mp4"
output_video = "runs/detect&trace/gasbottle_yolo11_video_result.mp4"

results = model.predict(
    source=input_video,   
    show=True,            
    save=True,           
    conf=0.8,            
    device="cuda:0",     
    tracker="botsort.yaml",
    show_labels=False,   
    show_conf=False,      
)
