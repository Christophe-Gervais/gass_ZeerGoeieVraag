from ultralytics import YOLO

# https://docs.ultralytics.com/tasks/classify/#train

# Load a model
# model = YOLO("yolo11n-cls.yaml")  # build a new model from YAML
model = YOLO("yolo11n-cls.pt")  # load a pretrained model (https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt)
# model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="mnist160", epochs=100, imgsz=64)
print(results)