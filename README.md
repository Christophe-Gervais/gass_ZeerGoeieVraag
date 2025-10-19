# Wat is dit? Zeer goeie vraag!
*AI Applications – WS2 | Group: ZeerGoeieVraag | Date: 19/10/2025*

---

## 🚀 Overview
This **Gass Bottle Project** aims to automatically **detect and track gas bottles** using computer vision.  
We use the **YOLOv11** object detection model to recognize bottles in images and videos.  
Later in the project, we plan to connect multiple camera views and extract printed text (e.g., weight and expiry date) using **OCR**. Further updates will come when we finish certain milestones.

---

## 🧩 Project Structure

- 🧠 `train.py` — Train the YOLO model on labeled data  
- 🎥 `detect.py` — Detect and track bottles in video  
- 🧠 `botsort.yaml` — Configuration file for BoT-SORT tracker    
- 📦 `requirements.txt` — Project dependencies
- 📦 `minio_download.py` Run this script to download the required folders
    - 📁 `videos/` — Folder for input videos  
    - 📁 `images/` — Folder for input images  
    - 📁 `labels/` — Folder for input labels for images   
    - 📁 `runs/` — Auto-generated YOLO output  

---

## 📦Installation

### Install the dependencies

```bash
pip install -r requirements.txt
```

### Download the data

```sh
python minio_download.py
```


## 🧪 How to Train
1. Go to train.py
2. If you have cuda, dont do anything.
3. If you don't, delete line 17. And remove the "," after line 16
4. Change workers on line 16 for each system. My system can do 20 workers, yours can maybe do 10 or less.
5. After all this, you can run train.py


## 🎥 How to Detect and Track Bottles
1. Go to detect.py
2. Change video path if needed
3. Run detect.py

---

## 🧠 Model Training

| Model    | Data Used                     | Precision | Recall | Notes              |
|----------|-------------------------------|-----------|--------|--------------------|
| YOLO11n  | Initial dataset (poor labels) | 0.77      | 0.70   | Early baseline     |
| YOLO11n  | MinIO dataset (correct labels)| 0.94      | 1.00   | Huge improvement   |
| YOLO11m  | MinIO dataset (correct labels)| 0.96      | 1.00   | Final model        |


---

## 🔧 Next Steps

🔁 Multi-camera synchronization: connecting 4 video feeds to track the same bottle across angles

🔤 Text recognition (OCR): extracting printed details (e.g., weight, expiration date)