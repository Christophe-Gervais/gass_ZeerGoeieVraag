from ultralytics import YOLO

# https://docs.ultralytics.com/tasks/classify/#train


def main():
    model = YOLO("yolo11x_finetuned_bottles_on_site_v3.pt")

    #Dit traint het model
    model.train(
        data="gasbottle.yaml",
        epochs=5,
        imgsz=640,
        batch=16,
        name="gasbottle_yolo11x_finetuned_v3",
        workers=20,
        device="cuda:0",
    )

    # Dit valideert het model
    metrics = model.val(data="gasbottle.yaml")
    print("Validation metrics:", metrics)

if __name__ == "__main__":
    main()