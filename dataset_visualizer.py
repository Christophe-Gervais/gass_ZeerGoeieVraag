import cv2
import yaml

DATASET_FILE = "dataset.yaml"


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


config = read_yaml(DATASET_FILE)
print(config['path'])

dataset_path: str = config['path']
train_path: str = config['train']
val_path: str = config['val']
names = config['names']

def visualize_dataset_sample(image_path, label_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    lines = []
    with open(label_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        img_height, img_width, _ = image.shape
        
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, names[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    #calculate width from height
    destination_height = 600
    aspect_ratio = image.shape[1] / image.shape[0]
    destination_width = int(destination_height * aspect_ratio)
    image = cv2.resize(image, (destination_width, destination_height))
    # image = cv2.resize(image, (800, 600))
    cv2.imshow("Dataset Sample", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

# Iterate through a few samples in the training set
import os

train_images = os.listdir(dataset_path + train_path)
for i, img_name in enumerate(train_images[:3]):  # Visualize first 5 images
    img_path = os.path.join(dataset_path, train_path, img_name)
    # Get matching label file
    
    label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
    # repalace only the last occurrence
    label_path = os.path.join(dataset_path, train_path.replace('images', 'labels'), label_name) 
    
    print(f"Visualizing: {img_path}")
    visualize_dataset_sample(img_path, label_path)