import streamlit as st
import numpy as np
from PIL import Image
import pickle
import os
import sys
import tensorflow as tf

# ===============================
# PAGE CONFIG (must be first st call)
# ===============================
st.set_page_config(
    page_title="FoodNetAI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# BASE PATH
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "foodnet_weights.weights.h5")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

sys.path.append(os.path.join(BASE_DIR, "src"))
from food_info import FOOD_INFO
from train import build_model

CONFIDENCE_THRESHOLD = 50.0

# Category -> accent color + icon, used throughout the UI
CATEGORY_STYLE = {
    "Fruit":     {"color": "#C2410C", "soft": "#FCE8DB", "icon": "🍊"},
    "Vegetable": {"color": "#15803D", "soft": "#DCEFE0", "icon": "🥕"},
    "Nut":       {"color": "#92400E", "soft": "#F1E4D3", "icon": "🌰"},
}
DEFAULT_STYLE = {"color": "#57534E", "soft": "#EDE9E3", "icon": "🍽️"}


# ===============================
# STYLING
# ===============================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #FBF7EF;
    }
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li {
        color: #2B2620;
    }

    /* enough top padding to clear Streamlit's fixed header/toolbar */
    .block-container {
        padding-top: 4rem;
        max-width: 1180px;
    }

    /* ---------- HERO ---------- */
    .hero-eyebrow {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #B45309;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-family: 'Fraunces', serif;
        font-size: 3.2rem;
        font-weight: 700;
        line-height: 1.05;
        color: #1F2A1F;
        margin: 0 0 0.6rem 0;
    }
    .hero-title em {
        font-style: italic;
        color: #3F6F4F;
    }
    .hero-subtitle {
        font-size: 1.12rem;
        color: #5C5346;
        max-width: 640px;
        line-height: 1.55;
        margin-bottom: 0.5rem;
    }
    .hero-divider {
        height: 2px;
        background: linear-gradient(90deg, #3F6F4F 0%, #D9C18C 45%, transparent 100%);
        border: none;
        margin: 1.6rem 0 1.8rem 0;
    }

    /* ---------- UPLOAD CARD (real container, not split div) ---------- */
    div[class*="st-key-upload_card"] {
        background: #FFFFFF;
        border: 1px solid #EAE2D2;
        border-radius: 18px;
        padding: 1.4rem;
        box-shadow: 0 1px 2px rgba(43,38,32,0.04);
    }
    .panel-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #8A7F6B;
        margin-bottom: 0.7rem;
    }

    [data-testid="stFileUploader"] section {
        background: #FBF7EF;
        border: 1.5px dashed #D9C18C;
        border-radius: 14px;
        padding: 1rem;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #3F6F4F;
    }

    /* ---------- RESULT CARDS (real containers) ---------- */
    div[class*="st-key-result_card"],
    div[class*="st-key-top3_card"] {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 1.9rem 2rem;
        box-shadow: 0 4px 24px rgba(43,38,32,0.07);
        border: 1px solid #EFE8D8;
        margin-bottom: 1.2rem;
    }
    .category-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.32rem 0.85rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
    }
    .prediction-name {
        font-family: 'Fraunces', serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1F2A1F;
        margin: 0 0 1.1rem 0;
        line-height: 1.1;
    }

    .confidence-row {
        display: flex;
        align-items: center;
        gap: 1.1rem;
        margin-bottom: 1.6rem;
    }
    .confidence-track {
        flex: 1;
        height: 12px;
        background: #F1EBDC;
        border-radius: 999px;
        overflow: hidden;
        position: relative;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #3F6F4F, #6FA17B);
    }
    .confidence-num {
        font-family: 'Fraunces', serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #1F2A1F;
        white-space: nowrap;
        min-width: 80px;
        text-align: right;
    }

    .nutrient-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.7rem;
        margin-bottom: 0.4rem;
    }
    .nutrient-box {
        background: #FBF7EF;
        border: 1px solid #EFE8D8;
        border-radius: 12px;
        padding: 0.85rem 0.5rem;
        text-align: center;
    }
    .nutrient-value {
        font-family: 'Fraunces', serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #1F2A1F;
        display: block;
    }
    .nutrient-label {
        font-size: 0.74rem;
        color: #8A7F6B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }

    .info-block-title {
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #3F6F4F;
        margin: 1.3rem 0 0.6rem 0;
    }
    .pill-list { display: flex; flex-wrap: wrap; gap: 0.45rem; }
    .pill {
        background: #DCEFE0;
        color: #1F2A1F;
        padding: 0.32rem 0.8rem;
        border-radius: 999px;
        font-size: 0.86rem;
        font-weight: 500;
    }
    .pill.use { background: #FCE8DB; }

    /* top-3 */
    .rank-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.5rem 0;
    }
    .rank-num {
        font-family: 'Fraunces', serif;
        font-weight: 700;
        font-size: 1rem;
        color: #B8AC92;
        min-width: 1.8rem;
        white-space: nowrap;
    }
    .rank-name { font-weight: 600; width: 130px; flex-shrink: 0; }
    .rank-track {
        flex: 1; height: 8px; background: #F1EBDC; border-radius: 999px; overflow: hidden;
    }
    .rank-fill { height: 100%; border-radius: 999px; background: #B8AC92; }
    .rank-fill.top { background: #3F6F4F; }
    .rank-pct { width: 54px; text-align: right; font-size: 0.86rem; color: #5C5346; font-weight: 600;}

    /* empty state */
    .empty-state {
        background: #FFFFFF;
        border: 1px dashed #D9C18C;
        border-radius: 18px;
        padding: 3rem 2rem;
        text-align: center;
        color: #8A7F6B;
    }
    .empty-state .icon { font-size: 2.4rem; margin-bottom: 0.6rem; }

    .low-conf-card {
        background: #FFF7EB;
        border: 1px solid #F0DCA8;
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
    }
    .low-conf-title {
        font-family: 'Fraunces', serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #92400E;
        margin-bottom: 0.4rem;
    }

    /* sidebar */
    section[data-testid="stSidebar"] {
        background: #1F2A1F;
        border-right: none;
    }
    section[data-testid="stSidebar"] * {
        color: #E9E4D6 !important;
    }
    section[data-testid="stSidebar"] h3 {
        font-family: 'Fraunces', serif;
        font-size: 1.5rem !important;
        color: #FBF7EF !important;
    }
    .sb-divider { height: 1px; background: #3A453A; margin: 1.1rem 0; border: none; }
    .sb-class-chip {
        display: inline-block;
        font-size: 0.82rem;
        background: #2C382C;
        color: #C9CDB8 !important;
        padding: 0.22rem 0.6rem;
        border-radius: 8px;
        margin: 0.15rem 0.2rem 0.15rem 0;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] div { color: #5C5346 !important; }
    button[kind="primary"], .stButton button {
        background: #3F6F4F !important;
        color: #FBF7EF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# ===============================
# LOAD MODEL + ENCODER
# ===============================
@st.cache_resource
def load_model_and_encoder():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder not found: {ENCODER_PATH}")

    model, _ = build_model(num_classes=15)
    model.load_weights(WEIGHTS_PATH)

    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)

    return model, encoder


with st.spinner("Loading model..."):
    model, label_encoder = load_model_and_encoder()


# =========================
# SIDEBAR
# ===============================
with st.sidebar:
    st.markdown("### 🌿 FoodNetAI")
    st.write("A deep learning classifier trained on the Fruits-360 dataset — recognizes 15 fruits, vegetables, and nuts from a photo.")

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown("**Model**")
    st.write("MobileNetV2 · transfer learning")
    st.markdown("**Input**")
    st.write("100 × 100 px RGB")
    st.markdown("**Classes**")
    st.write("15")

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown("**Recognized classes**")

    grouped = {"Fruit": [], "Vegetable": [], "Nut": []}
    for name, info in FOOD_INFO.items():
        grouped.setdefault(info.get("category", "Other"), []).append(name)

    sb_icons = {"Fruit": "🍊", "Vegetable": "🥕", "Nut": "🌰"}
    for cat in ["Fruit", "Vegetable", "Nut"]:
        names = sorted(grouped.get(cat, []))
        if not names:
            continue
        st.markdown(
            f'<div class="sb-cat-label">{sb_icons.get(cat, "")} {cat}s &nbsp;'
            f'<span class="sb-cat-count">{len(names)}</span></div>',
            unsafe_allow_html=True
        )
        chips = "".join(f'<span class="sb-class-chip">{n}</span>' for n in names)
        st.markdown(f'<div class="sb-chip-group">{chips}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.caption("Riphah International University — Programming for AI / Machine Learning project")
    
# ===============================
# HERO
# ===============================
st.markdown('<div class="hero-eyebrow">Image Classification · Nutrition Lookup</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">What\'s on your <em>plate?</em></h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Upload a photo of a fruit, vegetable, or nut. FoodNetAI identifies it and pulls up its '
    'full nutrition profile — calories, macros, health benefits, and common uses.</p>',
    unsafe_allow_html=True
)
st.markdown('<hr class="hero-divider">', unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1.35], gap="large")

with left_col:
    with st.container(key="upload_card"):
        st.markdown('<div class="panel-label">Upload Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)


# ===============================
# PREDICTION + RESULTS
# ===============================
with right_col:
    if uploaded_file is not None:
        img_resized = image.resize((100, 100))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing image..."):
            predictions = model.predict(img_array, verbose=0)[0]

        predicted_idx = int(np.argmax(predictions))
        predicted_class = label_encoder.inverse_transform([predicted_idx])[0]
        confidence = float(predictions[predicted_idx]) * 100

        if confidence < CONFIDENCE_THRESHOLD:
            st.markdown('<div class="low-conf-card">', unsafe_allow_html=True)
            st.markdown('<div class="low-conf-title">⚠️ Not confident enough</div>', unsafe_allow_html=True)
            st.write(
                f"Best guess was **{predicted_class}** at only **{confidence:.1f}%** confidence — "
                f"below the 50% threshold this app requires before showing a result. "
                f"FoodNetAI only recognizes 15 specific fruits, vegetables, and nuts; "
                f"try a clear, well-lit photo of one of those."
            )
            st.markdown('</div>', unsafe_allow_html=True)
            st.stop()

        info = FOOD_INFO.get(predicted_class, {})
        style = CATEGORY_STYLE.get(info.get("category"), DEFAULT_STYLE)

        with st.container(key="result_card"):
            st.markdown(
                f'<div class="category-chip" style="background:{style["soft"]}; color:{style["color"]};">'
                f'{style["icon"]} {info.get("category", "Unknown")}</div>',
                unsafe_allow_html=True
            )
            st.markdown(f'<div class="prediction-name">{predicted_class}</div>', unsafe_allow_html=True)

            st.markdown(
                f'<div class="confidence-row">'
                f'<div class="confidence-track"><div class="confidence-fill" style="width:{confidence}%;"></div></div>'
                f'<div class="confidence-num">{confidence:.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            nutrients = [
                ("Calories", f'{info.get("calories", "N/A")}'),
                ("Protein", info.get("protein", "N/A")),
                ("Carbs", info.get("carbohydrates", "N/A")),
                ("Fat", info.get("fat", "N/A")),
                ("Fiber", info.get("fiber", "N/A")),
            ]
            grid_html = '<div class="nutrient-grid">' + "".join(
                f'<div class="nutrient-box"><span class="nutrient-value">{val}</span>'
                f'<span class="nutrient-label">{label}</span></div>'
                for label, val in nutrients
            ) + '</div>'
            st.markdown(grid_html, unsafe_allow_html=True)

            if info.get("benefits"):
                st.markdown('<div class="info-block-title">Health Benefits</div>', unsafe_allow_html=True)
                pills = "".join(f'<span class="pill">{b}</span>' for b in info["benefits"])
                st.markdown(f'<div class="pill-list">{pills}</div>', unsafe_allow_html=True)

            if info.get("uses"):
                st.markdown('<div class="info-block-title">Common Uses</div>', unsafe_allow_html=True)
                pills = "".join(f'<span class="pill use">{u}</span>' for u in info["uses"])
                st.markdown(f'<div class="pill-list">{pills}</div>', unsafe_allow_html=True)

        # ---- Top-3 predictions ----
        with st.container(key="top3_card"):
            st.markdown('<div class="panel-label">Top 3 Predictions</div>', unsafe_allow_html=True)
            top3_idx = np.argsort(predictions)[-3:][::-1]
            for rank, idx in enumerate(top3_idx, start=1):
                cls_name = label_encoder.inverse_transform([idx])[0]
                conf = float(predictions[idx]) * 100
                fill_class = "rank-fill top" if rank == 1 else "rank-fill"
                st.markdown(
                    f'<div class="rank-row">'
                    f'<span class="rank-num">{rank:02d}</span>'
                    f'<span class="rank-name">{cls_name}</span>'
                    f'<div class="rank-track"><div class="{fill_class}" style="width:{conf}%;"></div></div>'
                    f'<span class="rank-pct">{conf:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    else:
        st.markdown(
            '<div class="empty-state">'
            '<div class="icon">🍓</div>'
            '<strong>No image yet</strong><br>'
            'Upload a photo on the left to see the prediction and nutrition info here.'
            '</div>',
            unsafe_allow_html=True
        )