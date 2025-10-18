import cv2
import yaml

DATASET_FILE = "dataset.yaml"


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


config = read_yaml(DATASET_FILE)
print(config)