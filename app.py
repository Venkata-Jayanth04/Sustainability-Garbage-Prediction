import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from pathlib import Path

st.set_page_config(page_title="Garbage Classifier", layout="wide")
st.title("♻️ Garbage Classifier — Sustainability Project")

# --- FIND MODEL FILE ---
MODEL_FILE_OPTIONS = [
    "saved_models/garbage_efficientnet_final.h5",
    "saved_models/garbage_efficientnet_finetuned.h5",
    "saved_models/garbage_efficientnet_best.h5",
    "saved_models/garbage_efficientnet.h5"
]

model_path = None
for p in MODEL_FILE_OPTIONS:
    if Path(p).exists():
        model_path = p
        break

if model_path is None:
    st.error("Model file NOT FOUND. Please place a model inside the saved_models folder.")
    st.stop()

LABELS_PATH = Path("artifacts/labels.npy")
if not LABELS_PATH.exists():
    st.error("labels.npy not found in artifacts/. Please copy artifacts folder next to this app.")
    st.stop()

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model(model_path)
labels = list(np.load(LABELS_PATH))

st.sidebar.header("Model Info")
st.sidebar.write("Model path: {{}}".format(model_path))
st.sidebar.write("Classes: {{}}".format(len(labels)))

st.header("Upload an image for prediction")
uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    img_resized = img.resize((224, 224))
    x = np.array(img_resized)[None, ...].astype("float32")
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    preds = model.predict(x)[0]
    top_idx = np.argsort(preds)[::-1][:5]

    st.subheader("Top Predictions")
    for i in top_idx:
        st.write("**{{}}** — {{:.2f}}%".format(labels[i], preds[i]*100))

st.markdown("---")
st.header("Training Artifacts")

col1, col2 = st.columns(2)

if Path("artifacts/training_curves.png").exists():
    col1.image("artifacts/training_curves.png")
else:
    col1.info("training_curves.png missing.")

if Path("artifacts/confusion_matrix.png").exists():
    col2.image("artifacts/confusion_matrix.png")
else:
    col2.info("confusion_matrix.png missing.")

if Path("artifacts/class_report.txt").exists():
    st.subheader("Classification Report")
    st.text(Path("artifacts/class_report.txt").read_text())
else:
    st.info("class_report.txt missing.")