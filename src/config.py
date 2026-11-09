"""
src/config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration file for the Multimodal Price Predictor.

All directory paths, model hyperparameter constants, and feature dimensions
are defined here. Importing this module also auto-creates required directories
so the rest of the pipeline never has to worry about missing folders.
"""

import os

# ── Directory structure ───────────────────────────────────────────────────────
BASE_DIR      = os.getcwd()
DATA_DIR      = os.path.join(BASE_DIR, "data")        # raw data & images
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")   # cached features & scalers
MODEL_DIR     = os.path.join(BASE_DIR, "models")       # saved trained models

for d in [DATA_DIR, PROCESSED_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Text feature config (Sentence-BERT) ──────────────────────────────────────
# all-MiniLM-L6-v2 outputs 384-dim embeddings; 15 numeric features appended
SBERT_MODEL_NAME  = "all-MiniLM-L6-v2"
SBERT_DIMS        = 384
NUMERIC_FEAT_DIMS = 15
TEXT_FEAT_DIMS    = SBERT_DIMS + NUMERIC_FEAT_DIMS   # = 399

# ── Image feature config (ResNet18 → PCA) ────────────────────────────────────
IMAGE_PCA_COMPONENTS = 150   # ResNet18 (512-dim) reduced to 150 via PCA

# ── Total fused feature dimensions ───────────────────────────────────────────
FUSED_DIMS = TEXT_FEAT_DIMS + IMAGE_PCA_COMPONENTS   # = 549

# ── Legacy config (kept for backward compatibility) ───────────────────────────
TFIDF_MAX_FEATURES = 10000
SVD_COMPONENTS     = 300
