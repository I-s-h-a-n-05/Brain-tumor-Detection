# NeuroScan — AI-Assisted Brain Tumour Classification

> End-to-end diagnostic intelligence platform for brain MRI analysis using deep learning and explainable AI.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![Flask](https://img.shields.io/badge/Flask-2.3-green)
![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-91.50%25-brightgreen)
![Val Accuracy](https://img.shields.io/badge/Val%20Accuracy-97.06%25-brightgreen)

---

## Overview

NeuroScan is a full-stack medical AI platform that classifies brain MRI scans into four categories — **Glioma**, **Meningioma**, **Pituitary Adenoma**, and **No Tumour** — using a fine-tuned EfficientNetB0 model trained on 5,712 MRI scans.

The system generates clinical-grade PDF radiology reports, visualises model attention using Grad-CAM, maintains a persistent scan history via SQLite, and provides an AI-powered neuro-oncology assistant powered by Groq LLaMA 3.3.

---

## Results

| Metric | Score |
|---|---|
| Validation Accuracy | **97.06%** |
| Test Accuracy | **91.50%** |
| Macro F1 Score | **0.9129** |
| Glioma Recall (v3) | **98.1%** (was 76.2% before focal loss) |

### Per-Class Metrics (Test Set — 1,600 samples)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Glioma | 93.5% | 79.0% | 85.6% |
| Meningioma | 89.5% | 87.3% | 88.4% |
| No Tumour | 93.5% | 100.0% | 96.6% |
| Pituitary | 89.9% | 99.8% | 94.6% |

---

## Features

- **MRI Classification** — 4-class brain tumour detection from uploaded MRI scans
- **Grad-CAM Explainability** — Gradient-weighted Class Activation Maps highlight the regions the model focused on
- **Clinical PDF Reports** — Radiology-style downloadable reports with findings, scan images, Grad-CAM heatmap, and clinical recommendations
- **AI Chat Assistant** — Neuro-oncology specialist powered by Groq LLaMA 3.3 70B
- **Persistent Scan History** — SQLite database stores all scan results with thumbnails and timestamps
- **Light / Dark Mode** — Theme toggle with localStorage persistence

---

## Model Architecture

```
Input (224×224×3 RGB)
    → EfficientNetB0 backbone (frozen, ImageNet pretrained)
    → Global Average Pooling  (1280-dim feature vector)
    → BatchNorm
    → Dense(256, ReLU) → Dropout(0.5)
    → Dense(128, ReLU) → Dropout(0.4)
    → Softmax(4 classes)
```

**Three-stage training:**

| Stage | Description | Val Accuracy |
|---|---|---|
| Base training | Frozen backbone, custom head, 30 epochs | 94.6% |
| Fine-tune v2 | Top 10 layers unfrozen, LR 5e-6 | 96.96% |
| Fine-tune v3 | Focal Loss (γ=2) + Glioma class weight 1.6× + 1.5× oversampling, top 20 layers | **97.06%** |

Focal loss in v3 fixed Glioma recall from **76.2% → 98.1%** on the validation set (most clinically significant improvement — Glioma is the most dangerous class to miss).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | TensorFlow 2.13 / Keras, EfficientNetB0 |
| Backend | Python 3.11, Flask 2.3 |
| Explainability | Grad-CAM (custom implementation) |
| PDF Generation | ReportLab |
| AI Chat | Groq API, LLaMA 3.3 70B Versatile |
| Database | SQLite (Python stdlib) |
| Frontend | Vanilla JS, CSS custom properties, no frameworks |

---

## Installation

### Prerequisites
- Python 3.11
- `brain_tumor_model_v3.h5` model file (not in repo — see note below)

### Steps

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/neuroscan.git
cd neuroscan

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place model file in project root
# brain_tumor_model_v3.h5 → C:\neuroscan\brain_tumor_model_v3.h5

# 5. (Optional) Set Groq API key for chat feature
set GROQ_API_KEY=your_key_here

# 6. Run
python app.py
# → http://localhost:5000
```

### Model file
The model was trained on the [Brain Tumour MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle, Masoud Nickparvar). To retrain from scratch, run `train.py` → `finetune.py` → `finetune_v3.py` in sequence.

---

## Project Structure

```
neuroscan/
├── app.py              # Flask app — all API routes
├── model.py            # EfficientNetB0 architecture
├── train.py            # Initial training
├── finetune.py         # Fine-tuning v2
├── finetune_v3.py      # Fine-tuning v3 (focal loss + class weights)
├── evaluate.py         # Test set evaluation + confusion matrix
├── gradcam.py          # Grad-CAM XAI implementation
├── report.py           # Clinical PDF report generator (ReportLab)
├── db.py               # SQLite history manager
├── requirements.txt
├── templates/
│   └── index.html      # Single-page frontend (vanilla JS)
└── README.md
```

---

## Dataset

**Brain Tumour MRI Dataset** — Masoud Nickparvar (Kaggle)
- 5,712 training images · 1,600 test images
- Classes: Glioma, Meningioma, No Tumour, Pituitary
- 80/20 train-validation split during training

---

## Medical Disclaimer

This project is developed for **academic and research purposes only**. NeuroScan is not a certified medical device and must not be used for clinical diagnosis or treatment decisions. Always consult a qualified neurosurgeon, radiologist, or oncologist.

---

## License

MIT — see [LICENSE](LICENSE) for details.