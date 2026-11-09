# src/evaluation.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from predict import load_pipeline, process_text, extract_image_features, fuse_features
from config import PROCESSED_DIR, IMAGE_DIR

def evaluate_model(test_csv_path):
    """
    Evaluate the trained ensemble model on a dataset.
    Prints R², MAE, RMSE, and MAPE.
    """

    if not os.path.exists(test_csv_path):
        print(f"❌ Test CSV not found at: {test_csv_path}")
        return

    # Load trained components
    print("🔹 Loading trained pipeline and model...")
    vectorizer, svd, pca, scaler, model = load_pipeline()

    # Load dataset
    df = pd.read_csv(test_csv_path)
    if 'price' not in df.columns:
        print("❌ Test CSV must contain 'price' column for evaluation.")
        return

    descriptions = df['catalog_content'].astype(str).tolist()
    image_files = df['image_path'].astype(str).tolist()
    y_true = df['price'].values

    # --- Text features ---
    print("🔤 Extracting text features...")
    text_features = process_text(vectorizer, svd, descriptions)

    # --- Image features ---
    print("🖼️ Extracting image features...")
    image_features = extract_image_features(pca, image_files)

    # --- Combine features ---
    print("🔗 Fusing features...")
    X = fuse_features(scaler, text_features, image_features)

    # --- Predict ---
    print("💡 Generating predictions...")
    preds = model.predict(X)

    # --- Optional: reverse log transform if applied during training ---
    # Uncomment if you trained on np.log1p(price)
    # y_true = np.expm1(y_true)
    # preds = np.expm1(preds)

    # --- Metrics ---
    r2 = r2_score(y_true, preds)
    mae = mean_absolute_error(y_true, preds)
    rmse = mean_squared_error(y_true, preds, squared=False)
    mape = np.mean(np.abs((y_true - preds) / (y_true + 1e-8))) * 100

    print("\n📊 MODEL EVALUATION RESULTS")
    print(f"R² Score  : {r2:.4f}")
    print(f"MAE       : ₹{mae:,.2f}")
    print(f"RMSE      : ₹{rmse:,.2f}")
    print(f"MAPE      : {mape:.2f}%")

    # --- Save predictions ---
    df["predicted_price"] = preds
    out_path = os.path.join("outputs", "evaluation_results.csv")
    os.makedirs("outputs", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n✅ Detailed predictions saved to {out_path}")

    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}
