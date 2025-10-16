from ultralytics import YOLO

# https://docs.ultralytics.com/tasks/classify/#train

if __name__ == '__main__':
    # Load a model
    # model = YOLO("yolo11n-cls.yaml")  # build a new model from YAML
    model = YOLO("yolo11n.pt")  # load a pretrained model (https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt)
    # model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="dataset.yaml", epochs=10, imgsz=320, save=True, workers=2, batch=16, device=0)  # train the model
    print(results)