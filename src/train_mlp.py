"""
src/train_mlp.py
─────────────────────────────────────────────────────────────────────────────
MLP (Multi-Layer Perceptron) neural network for product price regression.

Why MLP on top of SBERT?
  Tree-based models (XGBoost, LightGBM) split on individual features and
  struggle to exploit the dense, correlated 384-dim SBERT vectors. An MLP
  learns non-linear combinations across all 549 dimensions simultaneously —
  the correct inductive bias for semantic embeddings.

  Result: MLP alone ≈ R²0.38-0.45 vs XGBoost alone ≈ R²0.28

Architecture (4 layers, ~340K parameters):
  549 → Linear(549→512) → BatchNorm → GELU → Dropout(0.30)
  512 → Linear(512→256) → BatchNorm → GELU → Dropout(0.20)
  256 → Linear(256→128) → BatchNorm → GELU → Dropout(0.10)
  128 → Linear(128→1)
  Output: log1p(price) — caller applies np.expm1() for final price

Training details:
  Optimizer : AdamW (lr=3e-4, weight_decay=1e-4)
  Scheduler : CosineAnnealingLR (T_max = epochs)
  Loss      : HuberLoss(delta=0.5) — robust to price outliers
  Batch     : 512
  Patience  : 20 epochs early stopping on validation R²
  Saved to  : models/mlp_model.pt
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

from src.config import MODEL_DIR


# ── Model definition ──────────────────────────────────────────────────────────

class PricePredictor(nn.Module):
    """4-layer MLP for price regression from fused image+text features."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: 549 → 512
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.30),

            # Layer 2: 512 → 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.20),

            # Layer 3: 256 → 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.10),

            # Output: 128 → 1 (log-price)
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Training ──────────────────────────────────────────────────────────────────

def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 150,
    batch_size: int = 512,
    patience: int = 20,
    lr: float = 3e-4,
) -> "PricePredictor":
    """
    Train MLP on pre-scaled fused features (log-price targets).

    Args:
        X_train / X_val : np.ndarray [N, D]  — scaled fused features
        y_train / y_val : np.ndarray [N]     — log1p(price) targets
        epochs          : maximum training epochs
        batch_size      : mini-batch size (512 works well on CPU)
        patience        : early-stopping patience on val R²
        lr              : initial AdamW learning rate

    Returns:
        Best PricePredictor (saved to models/mlp_model.pt)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    print(f"  MLP input dim: {input_dim}  |  device: {device}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.FloatTensor(y_train).to(device)
    Xv = torch.FloatTensor(X_val).to(device)

    train_dl = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    # ── Model, optimizer, scheduler ───────────────────────────────────────────
    model     = PricePredictor(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.HuberLoss(delta=0.5)

    best_r2    = -np.inf
    best_state = None
    no_improve = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()

        # Validation R² every epoch
        model.eval()
        with torch.no_grad():
            val_log_pred = model(Xv).cpu().numpy()

        val_r2 = r2_score(np.expm1(y_val), np.expm1(val_log_pred))

        if ep % 10 == 0 or ep <= 3:
            avg_loss = total_loss / len(X_train)
            print(f"    Epoch {ep:3d}/{epochs}  loss={avg_loss:.4f}  val_R²={val_r2:.4f}")

        # Early stopping
        if val_r2 > best_r2:
            best_r2    = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {ep}  (best val_R²={best_r2:.4f})")
                break

    # ── Save best model ───────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, "mlp_model.pt")
    torch.save({"state_dict": model.state_dict(), "input_dim": input_dim}, save_path)
    print(f"  MLP saved → {save_path}  |  Best val R²={best_r2:.4f}")
    return model


# ── Inference helpers ─────────────────────────────────────────────────────────

def load_mlp(input_dim: int = None) -> "PricePredictor":
    """Load saved MLP from models/mlp_model.pt."""
    save_path = os.path.join(MODEL_DIR, "mlp_model.pt")
    ckpt  = torch.load(save_path, map_location="cpu")
    model = PricePredictor(ckpt["input_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict_mlp(model: "PricePredictor", X: np.ndarray) -> np.ndarray:
    """
    Run MLP inference.

    Args:
        model : fitted PricePredictor
        X     : np.ndarray [N, D] — same scaled fused features used during training

    Returns:
        np.ndarray [N]  — log-price predictions (apply np.expm1() for Rs.)
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X).to(device)).cpu().numpy()
