---
title: NeuroScan
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: AI-assisted brain tumour classification from MRI scans
---

# NeuroScan — AI-Assisted Brain Tumour Classification

End-to-end diagnostic intelligence platform for brain MRI analysis.

- **4 classes**: Glioma, Meningioma, No Tumour, Pituitary
- **97.06% validation accuracy** — EfficientNetB0 + Focal Loss fine-tuning
- **Grad-CAM** explainability heatmaps
- **Clinical PDF reports** — radiology-style downloadable reports
- **AI chat assistant** — Groq LLaMA 3.3 70B neuro-oncology specialist
- **SQLite scan history**

> ⚠️ For academic/research use only. Not a certified medical device.

## Setup

Set `GROQ_API_KEY` in Space secrets for the chat feature to work:
Settings → Variables and secrets → New secret → `GROQ_API_KEY`