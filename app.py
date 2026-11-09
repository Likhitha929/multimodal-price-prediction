"""
app.py — Multimodal Product Price Predictor
Streamlit web app for the MTech project.

Run: streamlit run app.py
"""

import os, sys, tempfile
import numpy as np
import streamlit as st
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Product Price Predictor",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
}
.main-header h1 { font-size: 2.4rem; font-weight: 700; margin: 0; }
.main-header p  { font-size: 1rem; opacity: 0.75; margin-top: 0.5rem; }

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 1.5rem;
    color: white;
    text-align: center;
    box-shadow: 0 8px 32px rgba(102,126,234,0.3);
}
.metric-value { font-size: 2.8rem; font-weight: 700; }
.metric-label { font-size: 0.9rem; opacity: 0.85; margin-top: 0.2rem; }

.range-card {
    background: #f8faff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
}

.info-badge {
    display: inline-block;
    background: #e8f4fd;
    color: #1565c0;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.82rem;
    margin: 0.2rem;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    cursor: pointer;
    transition: transform 0.1s;
}
.stButton > button:hover { transform: translateY(-1px); opacity: 0.92; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏷️ Product Price Predictor</h1>
    <p>Multimodal AI · Sentence-BERT + ResNet18 · Trained on 74,000 products</p>
</div>
""", unsafe_allow_html=True)


# ── Model info sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Model Info")
    st.markdown("""
    | | |
    |---|---|
    | **Text** | Sentence-BERT (384-dim) |
    | **Image** | ResNet18 + PCA (150-dim) |
    | **Models** | XGBoost + LightGBM + GB + RF + Ridge |
    | **Training data** | 74,249  products |
    | **Val R²** | 0.30 → 0.45+ (SBERT) |
    """)
    st.markdown("---")
    st.markdown("### 🔬 About")
    st.markdown("""
    This app uses a **multimodal ensemble** to predict product prices from:
    - Product description text
    - Product image
    - Extracted quantity/quality signals
    """)


# ── Main input area ───────────────────────────────────────────────────────────
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown("#### 📝 Product Description")
    desc = st.text_area(
        "Enter the full product description",
        placeholder=(
            "e.g. Heinz Tomato Ketchup, 500ml (Pack of 6), "
            "No artificial colours or preservatives, Gluten Free"
        ),
        height=180,
        label_visibility="collapsed",
    )
    st.markdown("#### 📸 Product Image")
    uploaded_img = st.file_uploader(
        "Upload a product image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if uploaded_img:
        st.image(uploaded_img, caption="Uploaded product image", use_column_width=True)

with col2:
    st.markdown("#### 🚀 Predict Price")
    st.markdown("<br>", unsafe_allow_html=True)

    predict_btn = st.button("🔍 Predict Price", use_container_width=True)

    if predict_btn:
        if not desc.strip():
            st.warning("⚠️ Please enter a product description.")
            st.stop()
        if not uploaded_img:
            st.warning("⚠️ Please upload a product image.")
            st.stop()

        # Save uploaded image to a temp file
        suffix = "." + uploaded_img.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_img.getbuffer())
            tmp_path = tmp.name

        with st.spinner("🤖 Running AI model..."):
            try:
                from src import predict as price_predict
                pred = price_predict.predict_single(desc.strip(), tmp_path)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                pred = None
            finally:
                try:
                    os.remove(tmp_path)
                except:
                    pass

        if pred is not None:
            # Main price display
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">Rs. {pred:.2f}</div>
                <div class="metric-label">Predicted Price</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Price range (±25%)
            low, high = pred * 0.75, pred * 1.25
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="range-card">
                    <div style="font-size:1.3rem; font-weight:600; color:#2d6a4f;">
                        Rs. {low:.2f}
                    </div>
                    <div style="font-size:0.8rem; color:#666;">Lower estimate</div>
                </div>""", unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div class="range-card">
                    <div style="font-size:1.3rem; font-weight:600; color:#9d4edd;">
                        Rs. {high:.2f}
                    </div>
                    <div style="font-size:0.8rem; color:#666;">Upper estimate</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Feature badges detected
            import re
            t = desc.lower()
            badges = []
            if re.search(r'\bpack\s*of\s*(\d+)', t):
                n = re.search(r'\bpack\s*of\s*(\d+)', t).group(1)
                badges.append(f"📦 Pack of {n}")
            if re.search(r'\b(\d+(?:\.\d+)?)\s*ml\b', t):
                v = re.search(r'\b(\d+(?:\.\d+)?)\s*ml\b', t).group(1)
                badges.append(f"💧 {v}ml")
            if re.search(r'\b(\d+(?:\.\d+)?)\s*kgs?\b', t):
                v = re.search(r'\b(\d+(?:\.\d+)?)\s*kgs?\b', t).group(1)
                badges.append(f"⚖️ {v}kg")
            if re.search(r'\borganic\b', t): badges.append("🌿 Organic")
            if re.search(r'\bpremium\b', t): badges.append("⭐ Premium")

            if badges:
                st.markdown("**Detected signals:**")
                st.markdown(" ".join([f'<span class="info-badge">{b}</span>'
                                       for b in badges]), unsafe_allow_html=True)

    else:
        # Show placeholder
        st.info("👆 Enter a description and upload an image, then click Predict Price.")
        st.markdown("""
        **Example products to try:**
        - Pantene Shampoo 400ml, Silky Smooth
        - Heinz Ketchup 500ml (Pack of 6)
        - Britannia NutriChoice Oats, 400g
        - Dove Beauty Bar Soap 100g Pack of 3
        """)



# SHAP plots if available
shap_bar = "outputs/shap_bar.png"
shap_summary = "outputs/shap_summary.png"

if os.path.exists(shap_bar) or os.path.exists(shap_summary):
    st.markdown("### 🔍 Model Explainability (SHAP)")
    c1, c2 = st.columns(2)
    if os.path.exists(shap_bar):
        c1.image(shap_bar, caption="Feature Importance (SHAP)", use_column_width=True)
    if os.path.exists(shap_summary):
        c2.image(shap_summary, caption="SHAP Summary (Beeswarm)", use_column_width=True)
    if os.path.exists("outputs/shap_waterfall.png"):
        st.image("outputs/shap_waterfall.png",
                 caption="Single Prediction Explanation", use_column_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; font-size:0.85rem; padding:1rem;">
    Multimodal Price Prediction · Sentence-BERT + ResNet18 + XGBoost Ensemble<br>
    Built with Python · PyTorch · scikit-learn · XGBoost · LightGBM · Streamlit
</div>
""", unsafe_allow_html=True)
