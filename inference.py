from ultralytics import YOLO

# https://docs.ultralytics.com/tasks/classify/#predict

if __name__ == '__main__':
  # Load a model
  model = YOLO("runs/detect/train11/weights/best.pt") # Replace with trained model path

  # Predict with the model
  results = model(source='videos/14_55/14_55_front_cropped.mp4', conf=0.25, save=True, imgsz=320, batch=100, device=0)  # predict on an image