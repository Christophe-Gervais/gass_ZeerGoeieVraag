from ultralytics import YOLO

def main():
    model = YOLO("yolo11m.pt")

    model.train(
        data="dataset_ok_nok.yaml",
        epochs=10,
        imgsz=640,
        batch=16,
        name="gasbottle_ok_nok",
        workers=20,
        device="cuda:0",
    )

    metrics = model.val(data="dataset_ok_nok.yaml")
    print("Validation metrics:", metrics)

if __name__ == "__main__":
    main()