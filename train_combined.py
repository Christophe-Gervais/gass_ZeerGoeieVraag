from ultralytics import YOLO

# https://docs.ultralytics.com/tasks/classify/#train

if __name__ == '__main__':
    model = YOLO("yolo11m.pt")
    
    results = model.train(data="combined_dataset.yaml", epochs=200, imgsz=320, save=True, workers=2, batch=32, device=0)  # train the model
    print(results)