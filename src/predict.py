"""
src/predict.py
─────────────────────────────────────────────────────────────────────────────
Inference module for Multimodal Product Price Predictor.

Provides functions:
  predict_dataset(csv_path, out_path)  — Batch inference on CSV dataset
  predict_single(description, image_path) — Single product price inference
"""

import os
import joblib
import numpy as np
import pandas as pd

from src import preprocess_text, preprocess_images, feature_fusion
from src.config import MODEL_DIR


def predict_dataset(csv_path, out_path="outputs/predictions.csv"):
    os.makedirs("outputs", exist_ok=True)
    models = joblib.load(f"{MODEL_DIR}/ensemble_models.pkl")
    df = pd.read_csv(csv_path)
    texts = df["catalog_content"].astype(str).tolist()
    imgs = df["image_path"].tolist()

    tfeat = preprocess_text.preprocess_text_infer(texts)
    if imgs[0].endswith(".jpg") or imgs[0].endswith(".png") or imgs[0].endswith(".jpeg"):
        raw, valid = preprocess_images.extract_raw_features(imgs)
        if raw.shape[0] == 0:
            raise ValueError("No valid images found in test dataset.")
        pca_model = joblib.load("processed/image_pca.pkl")
        if raw.shape[1] != pca_model.components_.shape[1]:
            raw = raw[:, :pca_model.components_.shape[1]]
        imgf = preprocess_images.transform_pca(raw)
    else:
        imgf = np.zeros((len(texts), 30))

    X = feature_fusion.fuse_infer(tfeat, imgf)

    weights = {"xgb": 0.38, "lgb": 0.34, "gb": 0.18, "ridge": 0.06, "rf": 0.04}
    preds = np.sum([m.predict(X) * weights[n] for n, m in models.items() if n in weights], axis=0)
    final_prices = np.maximum(0.01, np.expm1(preds))  # revert log transform and bound > 0

    df["predicted_price"] = final_prices
    df.to_csv(out_path, index=False)
    print(f"✅ Saved predictions to {out_path}")
    print("🔍 Prediction features:", X.shape, "Mean:", np.mean(X), "Std:", np.std(X))

    return df


def predict_single(description, image_path):
    models = joblib.load(f"{MODEL_DIR}/ensemble_models.pkl")
    tfeat = preprocess_text.preprocess_text_infer([description])
    raw, valid = preprocess_images.extract_raw_features([image_path])
    if len(valid) == 0:
        print("Warning: Could not load image from provided path. Using zero image features.")
        imgf = np.zeros((1, joblib.load("processed/image_pca.pkl").n_components_))
    else:
        imgf = preprocess_images.transform_pca(raw)

    X = feature_fusion.fuse_infer(tfeat, imgf)
    weights = {"xgb": 0.38, "lgb": 0.34, "gb": 0.18, "ridge": 0.06, "rf": 0.04}
    final_log = sum(models[n].predict(X)[0] * w for n, w in weights.items() if n in models)
    final_price = max(0.01, float(np.expm1(final_log)))  # convert from log space & bound > 0
    print(f"💰 Predicted Price: Rs.{final_price:.2f}")
    return final_price
