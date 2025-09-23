from ultralytics import YOLO

# https://docs.ultralytics.com/tasks/classify/#predict

# Load a model
model = YOLO("yolo11n-cls.pt") # Replace with trained model path

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")