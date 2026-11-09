# src/preprocess_text.py
# Text feature extraction using Sentence-BERT + numeric keyword features

import os, re, hashlib, joblib, numpy as np
from sentence_transformers import SentenceTransformer
from src.config import PROCESSED_DIR

# ── Sentence-BERT model ───────────────────────────────────────────────────────
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast, excellent for product text


# ── Helpers ───────────────────────────────────────────────────────────────────
def _num(pattern, text, default=0.0, group=1):
    """Safe regex numeric extractor."""
    m = re.search(pattern, text)
    return float(m.group(group)) if m else default


def extract_numeric_features(descriptions):
    """
    Extract 15 price-relevant numeric/keyword features.
    These encode quantity, size and quality signals that complement SBERT.

    Features (order matters — saved to text_feature_names.pkl for SHAP):
      pack_n, log_pack_n   — pack/set/bundle count
      vol_ml, log_vol      — total volume in ml
      wt_g,   log_wt       — total weight in grams
      pack_x_vol           — log1p(pack_n * vol)  interaction
      pack_x_wt            — log1p(pack_n * wt)   interaction
      is_bogo              — "buy N get N" deal
      is_combo             — combo / set product
      is_premium           — premium / luxury keyword
      is_organic           — organic / natural keyword
      is_sugar_free        — sugar free / no sugar
      is_discount          — discount / offer / sale
      is_multipack         — pack_n >= 6
    """
    rows = []
    for text in descriptions:
        t = str(text).lower()

        pack_n    = _num(r'\b(?:pack|set|box|bundle)\s*of\s*(\d+)', t, default=1.0)
        vol_ml    = _num(r'\b(\d+(?:\.\d+)?)\s*ml\b', t)
        vol_l     = _num(r'\b(\d+(?:\.\d+)?)\s*(?:l|ltr|litre|liter)s?\b', t)
        total_vol = vol_ml + vol_l * 1000
        wt_g      = _num(r'\b(\d+(?:\.\d+)?)\s*(?:gm?|gram)s?\b', t)
        wt_kg     = _num(r'\b(\d+(?:\.\d+)?)\s*kgs?\b', t)
        total_wt  = wt_g + wt_kg * 1000

        rows.append([
            pack_n,
            np.log1p(pack_n),
            total_vol,
            np.log1p(total_vol),
            total_wt,
            np.log1p(total_wt),
            np.log1p(pack_n * total_vol),
            np.log1p(pack_n * total_wt),
            float(bool(re.search(r'\bbuy\s*\d+\s*get\s*\d+\b', t))),
            float(bool(re.search(r'\bcombo\b|\bset\s*of\b', t))),
            float(bool(re.search(r'\bpremium\b|\bluxury\b', t))),
            float(bool(re.search(r'\borganic\b|\bnatural\b', t))),
            float(bool(re.search(r'\bsugar\s*free\b|\bno\s*sugar\b', t))),
            float(bool(re.search(r'\bdiscount\b|\boffer\b|\bsale\b', t))),
            float(pack_n >= 6),
        ])
    return np.array(rows, dtype=np.float32)


NUMERIC_FEATURE_NAMES = [
    "pack_n", "log_pack_n", "vol_ml", "log_vol", "wt_g", "log_wt",
    "pack_x_vol", "pack_x_wt",
    "is_bogo", "is_combo", "is_premium", "is_organic",
    "is_sugar_free", "is_discount", "is_multipack",
]


# ── Cache helpers ─────────────────────────────────────────────────────────────
def _cache_key(descriptions):
    content = "\n".join(str(d) for d in descriptions)
    return hashlib.md5(content.encode()).hexdigest()[:16]


def _load_or_encode(descriptions, cache_prefix="train"):
    """Load cached SBERT embeddings or compute and cache them."""
    cache_path = os.path.join(PROCESSED_DIR, f"sbert_{cache_prefix}_cache.npy")
    key_path   = os.path.join(PROCESSED_DIR, f"sbert_{cache_prefix}_key.txt")
    curr_key   = _cache_key(descriptions)

    if os.path.exists(cache_path) and os.path.exists(key_path):
        if open(key_path).read().strip() == curr_key:
            print(f"SBERT: loading cached embeddings from {cache_path}")
            return np.load(cache_path)

    print(f"SBERT: encoding {len(descriptions):,} descriptions with {SBERT_MODEL_NAME}...")
    print("       (first run takes ~10-20 min on CPU — cached for all subsequent runs)")
    model = SentenceTransformer(SBERT_MODEL_NAME)
    embeddings = model.encode(
        [str(d) for d in descriptions],
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.save(cache_path, embeddings)
    open(key_path, "w").write(curr_key)
    print(f"SBERT: embeddings cached to {cache_path}")
    return embeddings


# ── Public API ────────────────────────────────────────────────────────────────
def preprocess_text_train(descriptions):
    """
    Encode product descriptions with Sentence-BERT.
    Returns: np.ndarray [N, 399]  (384 SBERT dims + 15 numeric dims)
    """
    sbert_feats = _load_or_encode(descriptions, cache_prefix="train")
    kw          = extract_numeric_features(descriptions)
    result      = np.hstack([sbert_feats, kw])

    # Save feature names for SHAP analysis
    feat_names = (
        [f"sbert_{i}" for i in range(sbert_feats.shape[1])]
        + NUMERIC_FEATURE_NAMES
    )
    joblib.dump(feat_names, os.path.join(PROCESSED_DIR, "text_feature_names.pkl"))

    print(f"Text features: SBERT({sbert_feats.shape[1]}) "
          f"+ Numeric({kw.shape[1]}) = {result.shape[1]} dims")
    return result


def preprocess_text_infer(descriptions):
    """Encode at inference time — no caching needed for small batches."""
    model       = SentenceTransformer(SBERT_MODEL_NAME)
    sbert_feats = model.encode(
        [str(d) for d in descriptions],
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    kw = extract_numeric_features(descriptions)
    return np.hstack([sbert_feats, kw])
