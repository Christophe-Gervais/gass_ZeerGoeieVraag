from ultralytics import YOLO

def main():
    model = YOLO("runs/best.pt")

    model.train(
        data="dataset_ok_nok.yaml",
        epochs=25,
        imgsz=640,
        batch=16,
        name="gasbottle_ok_nok_final_try2",
        workers=20,
        device="cuda:0",
        resume=False,
    )

    metrics = model.val(data="dataset_ok_nok.yaml")
    print("Validation metrics:", metrics)

if __name__ == "__main__":
    main()