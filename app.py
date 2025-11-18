import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import io

st.set_page_config(page_title="Garbage Classifier", layout="wide")
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #0f1720;
        color: #e6eef6;
    }
    .stApp {
        background-color: #0f1720;
        color: #e6eef6;
    }
    .card {
        background: #0b1220;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.6);
    }
    .small-muted {
        color: #9aa6b2;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("♻️ Garbage Classifier — Dashboard")

MODEL_FILE_OPTIONS = [
    "saved_models/garbage_efficientnet_finetuned.h5",
    "saved_models/garbage_efficientnet_final.h5",
    "saved_models/garbage_efficientnet_best.h5",
    "saved_models/garbage_efficientnet.h5",
]
LABELS_PATH = Path("artifacts/labels.npy")
TRAIN_CURVES_IMG = Path("artifacts/training_curves.png")
CONF_MATRIX_IMG = Path("artifacts/confusion_matrix.png")
CLASS_REPORT_TXT = Path("artifacts/class_report.txt")

def find_best_model():
    for p in MODEL_FILE_OPTIONS:
        if Path(p).exists():
            return Path(p)
    return None

model_path = find_best_model()

if model_path is None:
    st.warning("No trained model found in saved_models/. The app will still show artifacts if present.")
else:
    st.sidebar.header("Model Info")
    st.sidebar.write(f"Loaded model: **{model_path.name}**")
    st.sidebar.write(f"Model path: {str(model_path)}")

if not LABELS_PATH.exists():
    st.sidebar.warning("artifacts/labels.npy not found. Some features (like class table) may be missing.")
    labels = []
else:
    try:
        labels = list(np.load(LABELS_PATH))
    except Exception:
        labels = []

@st.cache_resource
def load_model(path: str):
    model = tf.keras.models.load_model(path, compile=False)
    try:
        dummy = tf.zeros([1, 224, 224, 3], dtype=tf.float32)
        _ = model(dummy, training=False)
    except Exception:
        pass
    return model

model = None
if model_path:
    with st.spinner("Loading model (for inference) ..."):
        try:
            model = load_model(str(model_path))
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")
            model = None

left_col, right_col = st.columns([2, 3], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Training curves")
    if TRAIN_CURVES_IMG.exists():
        st.image(str(TRAIN_CURVES_IMG), use_column_width=True)
        st.caption("Accuracy & Loss over epochs")
    else:
        st.info("training_curves.png not found in artifacts.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Classification report (test set)")
    if CLASS_REPORT_TXT.exists():
        txt = CLASS_REPORT_TXT.read_text()
        buf = io.StringIO(txt)
        lines = [l.rstrip() for l in buf.readlines() if l.strip()]
        parsed_rows = []
        for ln in lines:
            parts = ln.split()
            if len(parts) >= 5:
                first = parts[0]
                if first in labels:
                    try:
                        precision = float(parts[1])
                        recall = float(parts[2])
                        f1 = float(parts[3])
                        support = int(parts[4])
                        parsed_rows.append((first, precision, recall, f1, support))
                    except Exception:
                        continue
        if parsed_rows:
            df = pd.DataFrame(parsed_rows, columns=["class","precision","recall","f1-score","support"])
            df = df.set_index("class")
            st.dataframe(df.style.background_gradient(cmap="Blues_r"), height=320)
        else:
            st.text(txt)
    else:
        st.info("class_report.txt not found in artifacts.")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Confusion Matrix")
    if CONF_MATRIX_IMG.exists():
        st.image(str(CONF_MATRIX_IMG), use_column_width=True)
        st.caption("Confusion matrix (True vs Predicted)")
    else:
        st.info("confusion_matrix.png not found in artifacts.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.header("Predict an image (no preview shown)")
uploaded_file = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    if model is None:
        st.error("No model loaded for prediction. Place a valid model file in saved_models/ and restart the app.")
    else:
        img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
        x = np.array(img).astype("float32")
        x = tf.keras.applications.efficientnet.preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)[0]
        top_k = 5
        top_idx = np.argsort(preds)[::-1][:top_k]
        st.subheader("Top predictions")
        for i in top_idx:
            label_name = labels[i] if (i < len(labels)) else str(i)
            st.write(f"**{label_name}** — {preds[i]*100:.2f}%")

st.markdown("<div class='small-muted'>Note: this dashboard uses pre-generated artifact images found in the artifacts/ folder (training_curves.png and confusion_matrix.png). If you retrain the model, replace the artifacts folder and refresh the app to see updated visuals.</div>", unsafe_allow_html=True)
