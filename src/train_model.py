"""
src/train_model.py
─────────────────────────────────────────────────────────────────────────────
Trains a 5-model weighted ensemble for product price prediction.

Models trained (all on log-transformed prices):
  1. XGBoost      — gradient boosted trees, best performer (R²≈0.34)
  2. LightGBM     — fast gradient boosting (R²≈0.33)
  3. GradientBoost— scikit-learn GB, slower but stable (R²≈0.28)
  4. RandomForest — bagging ensemble (R²≈0.20)
  5. Ridge        — linear baseline (R²≈0.24)

All 5 models are saved together in models/ensemble_models.pkl as a dict.
Inference weights are applied in main.py and src/predict.py.
"""

import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from src.config import MODEL_DIR


def train_ensemble(X, y):
    """
    Train all 5 ensemble models on the provided features and log-price targets.

    Args:
        X : np.ndarray [N, 549] — scaled fused features (text + image)
        y : np.ndarray [N]      — log1p-transformed prices

    Returns:
        dict  {model_name: fitted_model}  — also saved to models/ensemble_models.pkl
    """
    models = {
        # XGBoost: best performer; regularised to avoid overfitting rare high prices
        "xgb": XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            min_child_weight=10,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        ),

        # LightGBM: faster alternative to XGBoost; handles large datasets well
        "lgb": LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=30,  # prevents "no positive gain" warnings
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        ),

        # RandomForest: bagging ensemble; good for variance reduction
        "rf": RandomForestRegressor(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),

        # GradientBoosting: sklearn's slower but stable boosting implementation
        "gb": GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
        ),

        # Ridge: linear baseline; surprisingly competitive on log-price targets
        "ridge": Ridge(alpha=1.0),
    }

    for name, m in models.items():
        print(f"Training {name}...")
        m.fit(X, y)

    joblib.dump(models, f"{MODEL_DIR}/ensemble_models.pkl")
    print("Ensemble models saved.")
    return models
