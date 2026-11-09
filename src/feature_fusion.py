"""
src/feature_fusion.py
─────────────────────────────────────────────────────────────────────────────
Combines text and image feature vectors into a single fused representation.

Two functions are provided:
  fuse_train()  — fits a StandardScaler on training data, saves it to disk
  fuse_infer()  — loads the saved scaler and transforms inference data

Scaling is critical so that 384-dim SBERT embeddings and 150-dim image PCA
features contribute equally (neither dominates the other by magnitude).

Output: numpy array of shape [N, 549]  (399 text + 150 image dims, scaled)
"""

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from src.config import PROCESSED_DIR


def fuse_train(text_feats, img_feats):
    """
    Horizontally stack text and image features, fit-transform with StandardScaler.
    Saves the fitted scaler to processed/scaler.pkl for use at inference time.

    Args:
        text_feats : np.ndarray  [N, 399]  — SBERT(384) + numeric(15)
        img_feats  : np.ndarray  [N, 150]  — PCA-reduced ResNet18 features

    Returns:
        np.ndarray [N, 549] — zero-mean, unit-variance fused features
    """
    X  = np.hstack([text_feats, img_feats])
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    joblib.dump(sc, f"{PROCESSED_DIR}/scaler.pkl")
    return Xs


def fuse_infer(text_feats, img_feats):
    """
    Horizontally stack text and image features, transform with the saved scaler.
    Must be called AFTER training (scaler.pkl must exist).

    Args:
        text_feats : np.ndarray  [N, 399]
        img_feats  : np.ndarray  [N, 150]

    Returns:
        np.ndarray [N, 549] — scaled fused features
    """
    X  = np.hstack([text_feats, img_feats])
    sc = joblib.load(f"{PROCESSED_DIR}/scaler.pkl")
    return sc.transform(X)
