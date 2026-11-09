"""
shap_analysis.py
─────────────────
Generate SHAP explainability plots for the trained XGBoost model.
Run AFTER training:  python shap_analysis.py

Outputs (saved to outputs/):
  1. shap_summary.png       — global feature importance (beeswarm)
  2. shap_bar.png           — mean |SHAP| bar chart (top 25 features)
  3. shap_waterfall.png     — single-prediction explanation
  4. shap_numeric_only.png  — importance of just the 15 keyword features
"""

import os, sys
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Resolve project root ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.config import MODEL_DIR, PROCESSED_DIR

os.makedirs("outputs", exist_ok=True)


# ── Load model and data ───────────────────────────────────────────────────────
print("Loading XGBoost model...")
models = joblib.load(os.path.join(MODEL_DIR, "ensemble_models.pkl"))
xgb_model = models["xgb"]

print("Loading validation data (saved during training)...")
X_val = np.load(os.path.join(PROCESSED_DIR, "X_val_shap.npy"))
y_val = np.load(os.path.join(PROCESSED_DIR, "y_val_shap.npy"))

# Load feature names
try:
    text_feat_names = joblib.load(os.path.join(PROCESSED_DIR, "text_feature_names.pkl"))
    n_img  = X_val.shape[1] - len(text_feat_names)
    img_names  = [f"img_pca_{i}" for i in range(n_img)]
    feat_names = img_names + text_feat_names   # image PCA first, then text
    # Actually feature fusion stacks text first, then image — check config
    # Default: text features (399) + image features (150) = 549
    n_text = len(text_feat_names)
    n_img  = X_val.shape[1] - n_text
    feat_names = text_feat_names + [f"img_pca_{i}" for i in range(n_img)]
except FileNotFoundError:
    feat_names = [f"feature_{i}" for i in range(X_val.shape[1])]

print(f"Validation set: {X_val.shape[0]} samples × {X_val.shape[1]} features")


# ── Compute SHAP values ───────────────────────────────────────────────────────
print("Computing SHAP values (TreeExplainer — fast for XGBoost)...")
import shap, builtins

orig_float = builtins.float
def _safe_float(val):
    if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
        val = val[1:-1]
    return orig_float(val)

builtins.float = _safe_float
try:
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_val)
finally:
    builtins.float = orig_float

mean_abs_shap = np.abs(shap_values).mean(axis=0)
top25_idx     = np.argsort(mean_abs_shap)[::-1][:25]
top25_names   = [feat_names[i] if i < len(feat_names) else f"f{i}" for i in top25_idx]


# ── Plot 1: Bar chart — top 25 features ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#E63946" if "sbert" not in n else "#457B9D" for n in top25_names]
bars = ax.barh(range(25), mean_abs_shap[top25_idx][::-1], color=colors[::-1])
ax.set_yticks(range(25))
ax.set_yticklabels(top25_names[::-1], fontsize=10)
ax.set_xlabel("Mean |SHAP value| (impact on log-price prediction)", fontsize=11)
ax.set_title("SHAP Feature Importance — Top 25 Features\n"
             "(Red = Numeric/keyword  |  Blue = Sentence-BERT)",
             fontsize=13, fontweight="bold")
ax.axvline(0, color="black", linewidth=0.5)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#E63946", label="Numeric / keyword features"),
                   Patch(facecolor="#457B9D", label="Sentence-BERT text features")]
ax.legend(handles=legend_elements, loc="lower right")

plt.tight_layout()
plt.savefig("outputs/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/shap_bar.png")


# ── Plot 2: SHAP Summary (beeswarm) — top 20 ─────────────────────────────────
print("Generating SHAP beeswarm plot...")
shap_df_vals  = shap_values[:, top25_idx[:20]]
shap_df_feats = X_val[:, top25_idx[:20]]
names_20 = top25_names[:20]

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    shap_df_vals,
    shap_df_feats,
    feature_names=names_20,
    show=False,
    plot_size=None,
    max_display=20,
)
plt.title("SHAP Summary — Feature Impact on Price Prediction", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/shap_summary.png")


# ── Plot 3: Waterfall — single prediction explanation ─────────────────────────
print("Generating waterfall plot for a sample prediction...")
# Pick a mid-price product (median of val set)
median_idx = np.argsort(y_val)[len(y_val)//2]

fig, ax = plt.subplots(figsize=(10, 6))
top10_idx_local = np.argsort(np.abs(shap_values[median_idx]))[::-1][:10]
vals  = shap_values[median_idx][top10_idx_local]
names = [feat_names[i] if i < len(feat_names) else f"f{i}" for i in top10_idx_local]
bar_colors = ["#E63946" if v > 0 else "#1D3557" for v in vals]
ax.barh(names[::-1], vals[::-1], color=bar_colors[::-1])
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("SHAP value (positive = increases predicted price)", fontsize=11)
ax.set_title(
    f"Single Prediction Explanation\n"
    f"True price: Rs.{y_val[median_idx]:.2f}  |  "
    f"Predicted: Rs.{np.expm1(xgb_model.predict(X_val[[median_idx]])[0]):.2f}",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("outputs/shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/shap_waterfall.png")


# ── Plot 4: Numeric feature importance only ───────────────────────────────────
from src.preprocess_text import NUMERIC_FEATURE_NAMES

n_text_total = len(text_feat_names) if 'text_feat_names' in dir() else 399
n_sbert      = n_text_total - 15
numeric_indices = list(range(n_sbert, n_sbert + 15))
numeric_indices = [i for i in numeric_indices if i < X_val.shape[1]]

if numeric_indices:
    numeric_shap = mean_abs_shap[numeric_indices]
    numeric_names = NUMERIC_FEATURE_NAMES[:len(numeric_indices)]

    fig, ax = plt.subplots(figsize=(9, 6))
    sorted_idx = np.argsort(numeric_shap)
    ax.barh(
        [numeric_names[i] for i in sorted_idx],
        numeric_shap[sorted_idx],
        color="#E63946",
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("SHAP Importance of Keyword / Numeric Features\n"
                 "(pack size, volume, weight, quality flags)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/shap_numeric_only.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: outputs/shap_numeric_only.png")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("SHAP Analysis Complete")
print("="*55)
print("Top 10 most influential features:")
for rank, idx in enumerate(top25_idx[:10], 1):
    name = feat_names[idx] if idx < len(feat_names) else f"f{idx}"
    ftype = "Keyword" if any(kw in name for kw in NUMERIC_FEATURE_NAMES) else "SBERT"
    print(f"  {rank:2d}. {name:30s}  [{ftype}]  mean|SHAP|={mean_abs_shap[idx]:.5f}")

print(f"\nAll plots saved to: outputs/")
print("   shap_bar.png          — global feature importance")
print("   shap_summary.png      — beeswarm (impact distribution)")
print("   shap_waterfall.png    — single prediction breakdown")
print("   shap_numeric_only.png — keyword feature importance")
