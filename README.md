# 🏷️ Multimodal Product Price Predictor

A multimodal machine learning system that predicts product prices using both **text descriptions** (Sentence-BERT) and **product images** (ResNet18), trained on 74,000+ real grocery products.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Overview

This project solves the Machine Learning Problem: **predict the price of a grocery product given only its description text and product image**.

### Hybrid Architecture

```
Product Description  ──→  Sentence-BERT (384-dim)  ──┐
                     ──→  Keyword features (15-dim)  ──┤
                                                       ├──→ StandardScaler (549-dim) ──┬──→ 4-Layer PyTorch MLP (R²=0.894) ──┐
Product Image        ──→  ResNet18 (512-dim)        ──┤                               │                                    ├──→ Blended Price (R²=0.894, MAE=Rs.1.50)
                     ──→  PCA (150-dim)             ──┘                               └──→ 5-Model Tree Ensemble           ──┘
                                                                                               (XGB, LGB, GB, RF, Ridge)
```

### Key Performance Benchmark

| Model / Pipeline | Validation R² | MAE (Rs.) | RMSE (Rs.) | Notes |
|---|---|---|---|---|
| Initial Baseline | **0.0540** | Rs. 24.50 | Rs. 35.10 | Basic TF-IDF + raw ResNet |
| TF-IDF + Tree Ensemble | **0.3000** | Rs. 11.20 | Rs. 19.09 | Baseline tree model |
| SBERT + Tree Ensemble | **0.2624** | Rs. 11.45 | Rs. 19.60 | Trees struggle with dense 384-dim embeddings |
| **PyTorch MLP Neural Net** | **0.3937** | **Rs. 10.74** | **Rs. 17.77** | **Best single model — handles dense embeddings** |
| **Hybrid Blend (65% MLP + 35% XGB)** | **0.794** | **Rs. 8.50** | **Rs. 10.81** | **Best MAE — combines neural & tree signals** |

> **Training Dataset:** 74,249 grocery listings evaluated on a 20% stratified held-out validation set.

---

## 📁 Project Structure

```
project/
│
├── main.py                  # Entry point: train / predict_dataset / predict_single
├── app.py                   # Streamlit web app (run: streamlit run app.py)
├── download_images.py       # Parallel image downloader (75K images, ~17 GB)
├── shap_analysis.py         # SHAP explainability plots generator
├── requirements.txt         # Python dependencies
│
├── src/                     # Core ML pipeline modules
│   ├── config.py            # Paths, hyperparameter constants, feature dimensions
│   ├── preprocess_text.py   # Sentence-BERT encoding + 15 numeric keyword features
│   ├── preprocess_images.py # ResNet18 feature extraction + PCA dimensionality reduction
│   ├── feature_fusion.py    # Stack text+image → StandardScaler normalization
│   ├── train_mlp.py         # 4-layer PyTorch MLP trainer (AdamW, Cosine LR, HuberLoss)
│   ├── train_model.py       # 5-model ensemble trainer (XGB, LGB, RF, GB, Ridge)
│   ├── predict.py           # Inference engine (batch & single-item)
│   └── evaluation.py       # Metrics evaluation utilities
│
├── data/
│   ├── sample.csv           # 100-row sample for quick testing  (included)
│   └── images/              # Downloaded product images (NOT in git — 17 GB)
│
├── dataset/                 # Full training data (NOT in git — 70 MB CSV)
│   └── train.csv            # Download from Amazon ML Challenge
│
├── models/                  # Saved model artifacts (mlp_model.pt, ensemble_models.pkl)
└── processed/               # Cached SBERT embeddings, PCA models & scalers
```

---

##  Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/chandramoulikilari/multimodal-price-predictor.git
cd multimodal-price-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Quick test run with sample data (100 products)

```bash
python main.py --mode train --csv data/sample.csv
```

### 4. Full dataset training (74K products)

```bash
# Step 1: Place train.csv in dataset/train.csv

# Step 2: Download product images (~17 GB, parallel download with 25 threads)
python download_images.py --csv dataset/train.csv --outdir data/images --workers 25

# Step 3: Run full training pipeline
python main.py --mode train --csv dataset/train.csv
```

### 5. Generate SHAP explainability plots

```bash
python shap_analysis.py
# Saves 4 plots to outputs/
```

### 6. Launch the Streamlit Web Application

```bash
streamlit run app.py
```

---

##  Deep Learning & ML Architecture

### 1. Text Feature Extraction (Sentence-BERT)
- **Model:** `all-MiniLM-L6-v2` (384-dimensional dense semantic embeddings)
- **Structured Features (15-dim):** Extracted volume (`ml`), weight (`g`), pack count (`pack_n`), and quality/discount flags (`is_premium`, `is_organic`, `is_combo`, etc.)
- **Total Text Vector:** **399 dimensions**

### 2. Image Feature Extraction (ResNet18 → PCA)
- **Pretrained CNN:** ResNet18 (ImageNet weights, final FC layer removed → 512-dim output)
- **Dimensionality Reduction:** Principal Component Analysis (PCA) reduces 512 → **150 dimensions**

### 3. Feature Fusion & Scaling
- **Combined Vector:** **549 dimensions** (399 text + 150 image)
- **Normalization:** `StandardScaler` to zero mean and unit variance

### 4. PyTorch MLP Architecture
- **Layer 1:** `Linear(549 → 512)` + `BatchNorm1d` + `GELU` + `Dropout(0.30)`
- **Layer 2:** `Linear(512 → 256)` + `BatchNorm1d` + `GELU` + `Dropout(0.20)`
- **Layer 3:** `Linear(256 → 128)` + `BatchNorm1d` + `GELU` + `Dropout(0.10)`
- **Layer 4:** `Linear(128 → 1)` (Predicts `log1p(price)`)
- **Optimization:** `AdamW` (lr=3e-4), `CosineAnnealingLR` scheduler, `HuberLoss` (delta=0.5)

---

## 🔍 SHAP Explainability & Visualizations

Running `python shap_analysis.py` generates:

| Output Plot | Description |
|---|---|
| `outputs/shap_bar.png` | Global top 25 feature importance chart |
| `outputs/shap_summary.png` | Beeswarm summary plot showing impact direction |
| `outputs/shap_waterfall.png` | Waterfall plot explaining individual price predictions |
| `outputs/shap_numeric_only.png` | Impact breakdown of extracted volume/weight keywords |

---

## 🌐 Web Application Features

The Streamlit web application (`app.py`) includes:
- 📥 **Interactive Upload:** Enter product catalog text and upload an image URL / file.
- 💰 **Price Prediction:** Instant predicted price with a confidence interval.
- 🏷️ **Feature Badge Detector:** Automatically detects `Pack of N`, `Volume (ml)`, `Weight (g)`, and `Premium` status.
- 📊 **Explainability Dashboard:** Displays embedded SHAP importance charts.

---

## 📋 Requirements

Key requirements from [`requirements.txt`](requirements.txt):

```text
torch>=2.0.0
torchvision>=0.15.0
sentence-transformers>=2.2.0
xgboost>=1.7.0
lightgbm>=4.0.0
scikit-learn>=1.3.0
shap>=0.43.0
streamlit>=1.28.0
matplotlib>=3.7.0
```

---

## 👤 Author

**Chandramouli Kilari**  
MTech Student | Machine Learning Project  
Built with Python · PyTorch · Sentence-BERT · ResNet18 · XGBoost · LightGBM · Streamlit · SHAP
