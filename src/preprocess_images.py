"""
src/preprocess_images.py
─────────────────────────────────────────────────────────────────────────────
Extracts and reduces visual features from product images.

Pipeline:
  1. Load each image from data/images/<id>.jpg
  2. Pass through ResNet18 (pretrained on ImageNet) — removes final FC layer
     to get a 512-dim feature vector per image
  3. Fit PCA during training to reduce 512 → 150 dims (saves image_pca.pkl)
  4. At inference: load saved PCA and transform

Missing or corrupt images are silently skipped; the caller receives indices
of valid rows so it can align with other features.

Key functions:
  extract_raw_features(image_paths)  → (features, valid_indices)
  fit_pca(features)                  → reduced_features  [train only]
  transform_pca(features)            → reduced_features  [inference]
"""

import os
import torch
import joblib
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import models, transforms

from src.config import IMAGE_PCA_COMPONENTS, PROCESSED_DIR


def extract_raw_features(image_paths):
    """
    Extract ResNet18 CNN features from a list of image paths.

    Returns:
        feats         (np.ndarray) shape [N_valid, 512]
        valid_indices (list[int])  indices into image_paths that succeeded

    Rows whose images are missing or corrupt are silently skipped;
    the caller must use valid_indices to filter other arrays (text, labels).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval().to(device)

    prep = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    feats         = []
    valid_indices = []
    missing       = 0

    for i, p in enumerate(image_paths):
        if not os.path.exists(p):
            missing += 1
            continue
        try:
            img = Image.open(p).convert("RGB")
            x   = prep(img).unsqueeze(0).to(device)
            with torch.no_grad():
                f = model(x).cpu().numpy().flatten()
            feats.append(f)
            valid_indices.append(i)
        except Exception as e:
            print(f"Warning: could not load {p}: {e}")
            missing += 1

    print(f"Image extraction: {len(feats)} loaded, {missing} skipped (missing/corrupt)")
    return np.array(feats), valid_indices


def fit_pca(feats):
    """Fit PCA on extracted ResNet features and save model to disk."""
    n_comp = min(IMAGE_PCA_COMPONENTS, feats.shape[0], feats.shape[1])
    pca     = PCA(n_components=n_comp, random_state=42)
    reduced = pca.fit_transform(feats)
    joblib.dump(pca, f"{PROCESSED_DIR}/image_pca.pkl")
    print(f"PCA fitted and saved. Reduced to {reduced.shape}")
    return reduced


def transform_pca(feats):
    """Transform extracted ResNet features using saved PCA model."""
    pca = joblib.load(f"{PROCESSED_DIR}/image_pca.pkl")
    return pca.transform(feats)
