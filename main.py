"""
main.py
─────────────────────────────────────────────────────────────────────────────
Entry point for the Multimodal Amazon Product Price Predictor.

USAGE
─────
  Train on full dataset:
    python main.py --mode train --csv dataset/train.csv

  Train on small sample (quick test):
    python main.py --mode train --csv data/sample.csv

  Generate predictions on a test CSV:
    python main.py --mode predict_dataset --csv dataset/test.csv

TRAINING PIPELINE
─────────────────
  Step 1  Extract image features       — ResNet18 → 512-dim per image
  Step 2  Encode text descriptions     — Sentence-BERT → 384-dim + 15 numeric
  Step 3  Reduce image features        — PCA 512 → 150 dims
  Step 4  Fuse text + image features   — StandardScaler → 549-dim vector
  Step 5  Train ensemble               — XGBoost, LightGBM, RF, GB, Ridge
  Step 6  Evaluate on held-out 20%     — R², MAE, RMSE, MAPE
  Step 7  Save scatter plot            — outputs/true_vs_predicted.png

NOTES
─────
  • Prices are log1p-transformed before training and expm1-reversed for metrics
  • Top 1% price outliers are clipped to keep training stable
  • Stratified split by price decile ensures balanced validation
  • SBERT embeddings are cached to processed/sbert_train_cache.npy — rerun
    is fast (skips re-encoding) unless the dataset changes
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit

from src import preprocess_text, preprocess_images, feature_fusion, train_model, predict
from src.train_mlp import train_mlp, predict_mlp
from src.config import DATA_DIR, PROCESSED_DIR


def main():
    parser = argparse.ArgumentParser(description="Multimodal Amazon Product Price Predictor")
    parser.add_argument("--mode", default="train", choices=["train", "predict_dataset", "predict_single"], required=True)
    parser.add_argument("--csv", help="path to CSV file (overrides default dataset)")
    parser.add_argument("--desc", help="description for single prediction")
    parser.add_argument("--img", help="image path for single prediction")
    args = parser.parse_args()

    if args.mode == "train":
        # Support --csv override; default to data/sample.csv or data/dataset_with_images.csv
        if args.csv:
            csv_file = args.csv
        elif os.path.exists(os.path.join(DATA_DIR, "sample.csv")):
            csv_file = os.path.join(DATA_DIR, "sample.csv")
        else:
            csv_file = os.path.join(DATA_DIR, "dataset_with_images.csv")

        print(f"Loading data from: {csv_file}")
        df = pd.read_csv(csv_file)

        # Normalise ID column name (sample_id or id -> img_id)
        for col in ["sample_id", "id"]:
            if col in df.columns and "img_id" not in df.columns:
                df = df.rename(columns={col: "img_id"})

        # Build relative image path from img_id (images saved as data/images/<id>.jpg)
        df["image_path"] = df["img_id"].apply(lambda x: f"images/{x}.jpg")

        # Keep only rows with text and price
        df = df.dropna(subset=["catalog_content", "price"])
        print(f"Rows after dropping NaNs: {len(df)}")

        # Build absolute image paths
        imgs = [os.path.join(DATA_DIR, str(p)) for p in df["image_path"]]

        # ── Step 1: Extract image features (returns only valid rows) ──────────
        print("\nStep 1: Extracting image features (ResNet18)...")
        img_raw, valid_indices = preprocess_images.extract_raw_features(imgs)

        if len(valid_indices) == 0:
            print("No valid images found. Cannot train. Run download_images.py first.")
            return

        # Filter DataFrame to only rows with valid images
        df_valid = df.iloc[valid_indices].reset_index(drop=True)
        print(f"Valid rows with images: {len(df_valid)} / {len(df)}")

        # ── Clip extreme price outliers (top 1%) ──────────────────────────────
        price_cap   = df_valid["price"].quantile(0.99)
        before_clip = len(df_valid)
        clip_mask   = df_valid["price"].values <= price_cap
        df_valid    = df_valid[clip_mask].reset_index(drop=True)
        img_raw     = img_raw[clip_mask]
        print(f"After outlier clipping (cap=Rs.{price_cap:.2f}): {len(df_valid)} rows "
              f"(removed {before_clip - len(df_valid)} extreme outliers)")

        # ── Step 2: Text features ─────────────────────────────────────────────
        print("\nStep 2: Text preprocessing (Sentence-BERT)...")
        text_feat = preprocess_text.preprocess_text_train(df_valid["catalog_content"])

        # ── Step 3: PCA on image features ─────────────────────────────────────
        print("\nStep 3: Reducing image features (PCA)...")
        img_feat = preprocess_images.fit_pca(img_raw)
        # Save PCA-reduced image features so MLP can be retrained without re-extracting
        np.save(os.path.join(PROCESSED_DIR, "img_pca_features.npy"), img_feat)

        # ── Step 4: Fuse features ─────────────────────────────────────────────
        print("\nStep 4: Fusing text + image features...")
        X = feature_fusion.fuse_train(text_feat, img_feat)

        y = np.log1p(df_valid["price"].values)   # log-transform prices
        print(f"Training features shape: {X.shape}  |  Mean: {np.mean(X):.4f}  Std: {np.std(X):.4f}")

        # Stratified split: bin prices into deciles so every price range
        # is proportionally represented in both train and val sets
        price_bins = pd.qcut(df_valid["price"], q=10, labels=False, duplicates="drop")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(sss.split(X, price_bins))
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        print(f"Train: {len(X_train)} samples  |  Val: {len(X_val)} samples (stratified by price bucket)")

        # Save train split for standalone MLP retraining
        np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
        np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
        np.save(os.path.join(PROCESSED_DIR, "X_val.npy"),   X_val)
        np.save(os.path.join(PROCESSED_DIR, "y_val.npy"),   y_val)

        # ── Step 5: Train ensemble ────────────────────────────────────────────
        print("\nStep 5: Training ensemble models...")
        trained_models = train_model.train_ensemble(X_train, y_train)
        # Ensemble weights based on per-model R2 from full 74K-row run:
        # xgb=0.338, lgb=0.327, gb=0.283, ridge=0.241, rf=0.200
        weights = {"xgb": 0.38, "lgb": 0.34, "gb": 0.18, "ridge": 0.06, "rf": 0.04}

        y_pred = np.sum(
            [trained_models[n].predict(X_val) * w for n, w in weights.items()],
            axis=0
        )
        y_val_real  = np.expm1(y_val)
        y_pred_real = np.expm1(y_pred)

        # ── Step 6: Per-model diagnostics ─────────────────────────────────────
        print("\nPer-model R2 on validation set:")
        for name, m in trained_models.items():
            r2_i = r2_score(y_val_real, np.expm1(m.predict(X_val)))
            print(f"  {name:8s}  R2={r2_i:.4f}")

        # Save validation sample for SHAP analysis (max 3000 rows)
        shap_n = min(3000, len(X_val))
        np.save(os.path.join(PROCESSED_DIR, "X_val_shap.npy"), X_val[:shap_n])
        np.save(os.path.join(PROCESSED_DIR, "y_val_shap.npy"), y_val_real[:shap_n])
        print(f"Saved {shap_n} validation samples for SHAP analysis")

        # ── Step 5b: Train MLP neural network ────────────────────────────────
        print("\nStep 5b: Training MLP neural network (SBERT + image → price)...")
        mlp_model  = train_mlp(X_train, y_train, X_val, y_val, epochs=150, patience=20)
        mlp_pred   = predict_mlp(mlp_model, X_val)           # log-price
        mlp_real   = np.expm1(mlp_pred)                      # Rs.

        mlp_r2   = r2_score(y_val_real, mlp_real)
        mlp_mae  = mean_absolute_error(y_val_real, mlp_real)
        mlp_rmse = np.sqrt(mean_squared_error(y_val_real, mlp_real))
        print(f"  MLP alone  R2={mlp_r2:.4f}  MAE=Rs.{mlp_mae:.2f}  RMSE=Rs.{mlp_rmse:.2f}")

        # Blended: 65% MLP + 35% XGBoost (best tree model)
        xgb_log = trained_models["xgb"].predict(X_val)
        y_pred_blend = np.expm1(0.65 * mlp_pred + 0.35 * xgb_log)
        blend_r2   = r2_score(y_val_real, y_pred_blend)
        blend_mae  = mean_absolute_error(y_val_real, y_pred_blend)
        blend_rmse = np.sqrt(mean_squared_error(y_val_real, y_pred_blend))
        print(f"  Blend(65%MLP+35%XGB) R2={blend_r2:.4f}  MAE=Rs.{blend_mae:.2f}  RMSE=Rs.{blend_rmse:.2f}")

        # ── Step 7: Ensemble metrics ──────────────────────────────────────────
        r2   = r2_score(y_val_real, y_pred_real)
        mae  = mean_absolute_error(y_val_real, y_pred_real)
        rmse = np.sqrt(mean_squared_error(y_val_real, y_pred_real))
        mape = np.mean(np.abs((y_val_real - y_pred_real) / (y_val_real + 1e-8))) * 100

        # Use best of ensemble vs blend for final scatter plot
        best_pred_real  = y_pred_blend if blend_r2 > r2 else y_pred_real
        best_r2_overall = max(blend_r2, r2)

        print("\n" + "="*60)
        print("FINAL RESULTS (held-out 20% validation set)")
        print("="*60)
        print(f"  Tree Ensemble only   R2={r2:.4f}  MAE=Rs.{mae:.2f}  RMSE=Rs.{rmse:.2f}  MAPE={mape:.1f}%")
        print(f"  MLP only             R2={mlp_r2:.4f}  MAE=Rs.{mlp_mae:.2f}  RMSE=Rs.{mlp_rmse:.2f}")
        print(f"  Blend(65%+35%)       R2={blend_r2:.4f}  MAE=Rs.{blend_mae:.2f}  RMSE=Rs.{blend_rmse:.2f}")
        print("="*60)

        # Scatter plot of best predictor
        final_mape = np.mean(np.abs((y_val_real - best_pred_real) / (y_val_real + 1e-8))) * 100
        os.makedirs("outputs", exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_val_real, best_pred_real, alpha=0.3, s=10)
        plt.plot([y_val_real.min(), y_val_real.max()],
                 [y_val_real.min(), y_val_real.max()],
                 "r--", linewidth=1, label="Perfect prediction")
        plt.xlabel("True Price (Rs.)")
        plt.ylabel("Predicted Price (Rs.)")
        plt.title(f"True vs Predicted  |  R2={best_r2_overall:.3f}  MAPE={final_mape:.1f}%")
        plt.legend()
        plt.tight_layout()
        plt.savefig("outputs/true_vs_predicted.png", dpi=150)
        plt.close()
        print("Plot saved to outputs/true_vs_predicted.png")

    elif args.mode == "predict_dataset":
        if not args.csv:
            print("Provide --csv for predict_dataset mode.")
        else:
            predict.predict_dataset(args.csv)

    elif args.mode == "predict_single":
        if not args.desc or not args.img:
            print("Provide --desc and --img for predict_single mode.")
        else:
            predict.predict_single(args.desc, args.img)


if __name__ == "__main__":
    main()
