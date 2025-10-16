from ultralytics import YOLO

# https://docs.ultralytics.com/tasks/classify/#train

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")
    
    results = model.train(data="dataset.yaml", epochs=10, imgsz=320, save=True, workers=2, batch=16, device=0)  # train the model
    print(results)