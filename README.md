# 🔥 Gas Bottle Detection & Classification System
*AI Applications – FINAL | Group: ZeerGoeieVraag | Date: 8 December 2025*

---

## 🚀 Project Overview

Dit project is een **end-to-end AI systeem** voor het automatisch detecteren, tracken, en classificeren van gasflessen op een productielijn. Het systeem gebruikt:

- **YOLOv11** voor object detectie en tracking
- **Multi-camera synchronisatie** voor 360° tracking
- **OCR (EasyOCR)** voor het uitlezen van tarra gewicht en productiejaar
- **Classificatie model** voor OK/NOK beslissingen
- **Performance analytics** voor het evalueren van systeem accuraatheid

---

## 📦 Quick Start Guide

### 1️⃣ Installatie

```bash
# Clone de repository
git clone <repository-url>
cd gass_ZeerGoeieVraag

# Installeer dependencies
pip install -r requirements.txt
```

### 2️⃣ Maak benodigde folders aan

```bash
# Maak de video folder aan (voor input videos)
mkdir videos

# Maak de bottle_dataset folder aan (voor training data)
mkdir bottle_dataset
mkdir bottle_dataset/images
mkdir bottle_dataset/labels

# Maak de csv folder aan (voor groundtruth en resultaten)
mkdir csv
```

### 3️⃣ Download data van MinIO

```bash
# Download videos en images van MinIO server
python minio_download.py
```

### 4️⃣ Run het systeem

#### Voor Detection, Tracking & Classification:
```bash
python inference.py
```

Dit zal:
- Bottles detecteren met YOLO
- Bottles tracken across frames
- Classificatie uitvoeren (OK/NOK)
- Resultaten opslaan in CSV

#### Voor OCR (Tarra & Jaar uitlezen):
Open de notebook en run alle cellen:
```bash
jupyter notebook inference_tracking_batched_async_ocr.ipynb
```

### 5️⃣ Performance Evaluatie

```bash
# Bekijk de volledige performance metrics:
python results_with_OCR.py

# Of zonder OCR data:
python results_without_OCR.py
```

Dit geeft je:
- Push performance (hoeveel bottles worden correct geweigerd)
- OCR accuracy (tarra & jaar)
- Classification metrics
- Dangerous fills detectie

---

## 📂 Complete File Structure

```
gass_ZeerGoeieVraag/
│
├── 📋 CONFIGURATION FILES
│   ├── requirements.txt              # Python dependencies
│   ├── dataset.yaml                  # ⭐ YOLO dataset config voor train.py
│   └── dataset_ok_nok.yaml          # ⭐ OK/NOK classificatie config voor train_nok_ok.py
│
├── 🎓 MAIN TRAINING SCRIPTS
│   ├── train.py                     # ⭐ MAIN: Train YOLO bottle detection
│   ├── train_nok_ok.py             # ⭐ MAIN: Train OK/NOK classificatie
│   ├── train_combined.py            # Extra: Train op combined dataset
│   └── train_generated.py           # Extra: Train op gegenereerde data
│
├── 🔍 INFERENCE SCRIPTS
│   ├── inference.py                 # ⭐ MAIN: Detection + Tracking + Classification
│   ├── inference_tracking_batched_async_ocr.ipynb  # ⭐ MAIN: OCR notebook
│   │
│   └── 📦 OUDE VERSIES (voor referentie)
│       ├── inference_tracking.py
│       ├── inference_tracking_batched.py
│       ├── inference_tracking_batched_async.py
│       └── inference_tracking_batchAsync_OK-NOK.py
│
├── 📊 ANALYSIS & RESULTS
│   ├── results_with_OCR.py          # ⭐ MAIN: Performance metrics MET OCR
│   ├── merge_csv_ocr_classification.py  # ⭐ MAIN: Merge OCR + classificatie CSV's
│   └── filtering_ocr_results.ipynb  # ⭐ MAIN: Filter OCR output
│
├── 🏗️ DATASET TOOLS
│   ├── splitting_data.py            # ⭐ MAIN: Split train/validation data
│   ├── dataset_generator.py         # ⭐ MAIN: Screenshots uit video's voor training
│   ├── dataset_generator_ok-nok.py  # ⭐ MAIN: Screenshots voor OK/NOK training
│   ├── dataset_visualizer.py        # ⭐ MAIN: Visualiseer dataset
│   ├── minio_download.py            # ⭐ MAIN: Download videos/images van MinIO
│   └── auto-labeling.py             # Extra: Auto-label nieuwe images
│
├── 📁 DATA DIRECTORIES (⚠️ MOET JE ZELF AANMAKEN)
│   ├── videos/                      # ⚠️ AANMAKEN: Plaats hier input video's
│   ├── csv/                         # ⚠️ AANMAKEN: CSV resultaten & groundtruth
│   │   ├── groundtruth_*.csv       # Groundtruth labels (voor results)
│   │   ├── bottle_ocr_*.csv        # OCR resultaten
│   │   ├── bottle_classification_*.csv  # Classificatie resultaten
│   │   └── merge_results_*.csv     # Merged data
│   │
│   ├── bottle_dataset/              # ⚠️ AANMAKEN: Training dataset
│   │   ├── images/                  # Images voor training
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/                  # Labels voor training
│   │       ├── train/
│   │       └── val/
│   │
│   ├── runs/                        # ✅ AUTO: Training outputs (auto aangemaakt)
│   │   └── detect/                  # Detection model runs
│   │
│   ├── crop_cache/                  # ✅ AUTO: Gecropte bottle images
│   │   └── debug_ocr/              # OCR debug visualisaties
│   │
│   ├── extra_ok_images/             # Extra OK samples (optioneel)
│   └── extra_nok_images/            # Extra NOK samples (optioneel)
│
└── 📖 DOCUMENTATION
    ├── README.md                    # Dit bestand
    └── strategy.md                  # Project strategie

```

---

## 🎯 Belangrijkste Scripts Uitgelegd

### 🔴 Inference & Detection

#### `inference.py` ⭐ (MAIN INFERENCE SCRIPT)
**Wat doet het?**
- Detecteert bottles in video met YOLO
- Tracked elke bottle across frames  
- Classificeert als OK/NOK
- Slaat resultaten op in CSV

**Gebruik:**
```bash
python inference.py
```

**Output:**
- `csv/bottle_classifications.csv` - OK/NOK beslissingen per bottle

---

#### `inference_tracking_batched_async_ocr.ipynb` ⭐ (MAIN OCR SCRIPT)
**Wat doet het?**
Jupyter notebook voor OCR op video's:
- Leest tarra gewicht van bottles
- Leest productiejaar van bottles
- Slaat OCR resultaten op in CSV

**Gebruik:**
```bash
jupyter notebook inference_tracking_batched_async_ocr.ipynb
# Run alle cellen
```

**Output:**
- `csv/bottle_ocr_results.csv` - OCR data (tarra & jaar)
- `crop_cache/debug_ocr/` - Debug visualisaties

---

### 🔵 Training

#### `train.py` ⭐ (MAIN DETECTION TRAINING)
Train YOLO model voor bottle detection.

**Gebruik:**
```bash
python train.py
```

**Configuratie:** Gebruikt `dataset.yaml`  
**Parameters:**
- `epochs`: 100 (aantal training epochs)
- `imgsz`: 320 (image size)
- `batch`: 16 (batch size)
- `workers`: 2 (pas aan voor jouw CPU)
- `device`: 0 (GPU) of `cpu`

---

#### `train_nok_ok.py` ⭐ (MAIN CLASSIFICATION TRAINING)
Train classificatie model voor OK vs NOK bottles.

**Gebruik:**
```bash
python train_nok_ok.py
```

**Configuratie:** Gebruikt `dataset_ok_nok.yaml`

---

### 🟢 Results & Performance

#### `results_with_OCR.py` ⭐ (PERFORMANCE MET OCR)
Volledige performance analyse van het systeem.

**⚠️ Vereist EERST:**
OCR resultaten filteren + mergen met classification
```bash
jupyter notebook filtering_ocr_results.ipynb
```

```bash
# Merge OCR + classificatie CSV's:
python merge_csv_ocr_classification.py
```

**Gebruik:**
```bash
python results_with_OCR.py
```

**Vereist in csv/ folder:**
- `groundtruth_*.csv` (groundtruth labels)
- `merge_results_*.csv` (merged OCR + classificatie - **MOET je eerst maken!**)

**Output metrics:**
1. **Push Performance** - Bottles die correct geweigerd worden
2. **Push Performance Enhanced** - Inclusief unreadable tarra
3. **Classification Performance** - OK/NOK accuracy
4. **Dangerous Fills** - Bottles met >500g overfill
5. **Tarra Reading** - OCR accuracy voor gewicht
6. **Year Reading** - OCR accuracy voor jaar

---

### 🟠 Dataset Generation

#### `dataset_generator.py` ⭐
Maak training dataset uit video's (voor detection).

**Gebruik:**
```bash
python dataset_generator.py
```

**Wat doet het?**
- Leest video frame-by-frame
- Detecteert bottles met YOLO
- Slaat screenshots + labels op in YOLO formaat

---

#### `dataset_generator_ok-nok.py` ⭐
## 🧪 Complete Workflows

### 🎬 Workflow 1: Volledige Inference Pipeline

```bash
# 1. Zorg dat je videos hebt in de videos/ folder
mkdir videos
# Plaats je video's in videos/

# 2. Run detection + tracking + classification
python inference.py
# Output: csv/bottle_classifications.csv

# 3. Run OCR in de notebook
jupyter notebook inference_tracking_batched_async_ocr.ipynb
# Run alle cellen
# Output: csv/bottle_ocr_results.csv

# 4. Filter OCR resultaten
jupyter notebook filtering_ocr_results.ipynb
# Output: csv/bottle_ocr_filtered_results.csv

# 5. ⚠️ BELANGRIJK: Merge OCR + classificatie resultaten
python merge_csv_ocr_classification.py
# Input: csv/bottle_classifications.csv + csv/bottle_ocr_filtered_results.csv
# Output: csv/merge_results.csv

# 6. Analyseer performance (VEREIST merge_results.csv!)
python results_with_OCR.py
```

---

### 🎓 Workflow 2: Train een Nieuw Detection Model

```bash
# 1. Maak benodigde folders
mkdir bottle_dataset
mkdir bottle_dataset/images
mkdir bottle_dataset/labels

# 2. Genereer dataset uit video's
python dataset_generator.py
# Screenshots worden opgeslagen in bottle_dataset/

# 3. Visualiseer de dataset (check of alles goed is)
python dataset_visualizer.py

# 4. Split in training/validation sets
python splitting_data.py
# 80% training, 20% validation

# 5. Train het YOLO model
python train.py
# Configuratie: dataset.yaml
# Output: runs/detect/train*/weights/best.pt

# 6. Test het nieuwe model
python inference.py
# Update het model path in inference.py naar je nieuwe model
```

---

### 🏷️ Workflow 3: Train een OK/NOK Classificatie Model

```bash
# 1. Genereer OK/NOK dataset
python dataset_generator_ok-nok.py

# 2. Split data
python splitting_data.py

# 3. Train classificatie model
python train_nok_ok.py
# Configuratie: dataset_ok_nok.yaml
# Output: runs/classify/train*/weights/best.pt
```

### 🔧 Utilities

#### `minio_download.py` ⭐
Download videos en images van MinIO server.

**Gebruik:**
```bash
python minio_download.py
```

### Voor het trainen van een nieuw model:

```bash
# 1. Genereer dataset uit video's
python dataset_generator.py

# 2. (Optioneel) Auto-label extra images
python auto-labeling.py

# 3. Split data in train/val
python splitting_data.py

# 4. Train het model
python train.py

# 5. Test het nieuwe model
python inference_tracking.py
```

---

## 📊 Model Performance

| Model                  | Task            | Precision | Recall | F1    | Notes                |
|------------------------|-----------------|-----------|--------|-------|----------------------|
| YOLO11n               | Detection       | 0.94      | 1.00   | 0.97  | Snelste model        |
| YOLO11m               | Detection       | 0.96      | 1.00   | 0.98  | Beste balans         |
| YOLO11x finetuned     | Detection       | 0.98      | 1.00   | 0.99  | Meest accurate       |
| Classification model   | OK/NOK         | 0.92      | 0.87   | 0.89  | Met OCR features     |

---

## 📝 CSV Output Formats

### `bottle_ocr_results.csv`
```csv
bottle_id,timestamp,tarra,year
1,1764765522.0,10.8,2036.0
2,1764765524.9,10.8,2036.0
```

### `bottle_classifications.csv`
```csv
bottle_id,pushed_by_ai,classification,timestamp
1,True,NOK,1764765522.0
2,False,OK,1764765524.9
```

### `merge_results.csv`
```csv
bottle_id,pushed_by_ai,classification,timestamp,tarra,year
1,True,NOK,1764765522.0,10.8,2036.0
```

### `groundtruth.csv`
```csv
id,classification,tarra,year,primagaz_status
1,NOK,10.8,2036,NOK
2,OK,10.8,2036,OK
```

---