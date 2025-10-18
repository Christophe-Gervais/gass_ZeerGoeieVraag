from ultralytics import YOLO

# https://docs.ultralytics.com/tasks/classify/#train

if __name__ == '__main__':
    model = YOLO("yolo11m.pt")
    
    results = model.train(data="inference_generated_dataset.yaml", epochs=30, imgsz=320, save=True, workers=2, batch=16, device=0)  # train the model
    print(results)