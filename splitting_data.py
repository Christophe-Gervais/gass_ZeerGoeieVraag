import os
import random
import shutil

# Dataset base folder
dataset_dir = "bottle_dataset"
images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")

VALIDATION_SPLIT = 0.2  # fraction of images to move to validation

# Classes
classes = ["OK", "NOK"]

for cls in classes:
    train_images_dir = os.path.join(images_dir, "train", cls)
    train_labels_dir = os.path.join(labels_dir, "train", cls)

    val_images_dir = os.path.join(images_dir, "val", cls)
    val_labels_dir = os.path.join(labels_dir, "val", cls)

    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Get all images in training folder
    all_images = [f for f in os.listdir(train_images_dir) if f.lower().endswith(".jpg")]
    num_val = int(len(all_images) * VALIDATION_SPLIT)
    val_samples = random.sample(all_images, num_val)

    for img_name in val_samples:
        # Move image
        src_img = os.path.join(train_images_dir, img_name)
        dst_img = os.path.join(val_images_dir, img_name)
        shutil.move(src_img, dst_img)

        # Move corresponding label
        label_name = os.path.splitext(img_name)[0] + ".txt"
        src_label = os.path.join(train_labels_dir, label_name)
        dst_label = os.path.join(val_labels_dir, label_name)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)

    print(f"Moved {len(val_samples)} images from {cls} training to validation.")

print("Validation split completed.")
