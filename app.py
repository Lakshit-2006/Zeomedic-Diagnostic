import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="ZeaMedic - Live Corn Disease Diagnosis",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #2E7D32;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #555;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- Model Config & Knowledge Base -----------------
CLASS_NAMES = [
    "Common Rust",     # 0
    "Gray Leaf Spot",  # 1
    "Blight",          # 2
    "Healthy"          # 3
]

MODEL_PATH = "models/model2.h5"
IMG_SIZE = (224, 224)

DISEASE_GUIDE = {
    "Common Rust": {
        "pathogen": "Puccinia sorghi (Fungus)",
        "symptoms": "Small, powdery, cinnamon-brown pustules scattered across both leaf surfaces.",
        "management": "Apply strobilurin or triazole fungicides if severe before tassel stage. Use rust-resistant hybrids."
    },
    "Gray Leaf Spot": {
        "pathogen": "Cercospora zeae-maydis (Fungus)",
        "symptoms": "Rectangular, tan-to-gray lesions running parallel between leaf veins with sharp boundaries.",
        "management": "Apply strobilurin/triazole fungicides at onset. Implement crop rotation and tillage to break residue cycle."
    },
    "Blight": {
        "pathogen": "Exserohilum turcicum (Northern Corn Leaf Blight)",
        "symptoms": "Large, elongated, cigar-shaped grayish-green or tan lesions spreading along the blade.",
        "management": "Deploy resistant hybrids, apply foliar fungicides prior to silking, and rotate out of continuous corn."
    },
    "Healthy": {
        "pathogen": "None (Plant is healthy)",
        "symptoms": "Leaves are uniform green, fully expanded, and free of chlorosis or necrotic spotting.",
        "management": "Maintain balanced NPK fertilization, scheduled irrigation, and routine field scouting."
    }
}

# ----------------- Inference Functions -----------------
@st.cache_resource(show_spinner="Loading trained CNN weights...")
def get_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image: Image.Image, target_size=IMG_SIZE) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict(model, image: Image.Image):
    tensor = preprocess_image(image)
    raw_probs = model.predict(tensor)[0]
    
    top_idx = int(np.argmax(raw_probs))
    top_class = CLASS_NAMES[top_idx]
    confidence = float(raw_probs[top_idx])

    probabilities = {
        CLASS_NAMES[i]: float(raw_probs[i]) for i in range(len(CLASS_NAMES))
    }

    return {
        "class": top_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "info": DISEASE_GUIDE.get(top_class, {})
    }

# ----------------- Load Model -----------------
try:
    model = get_model()
except Exception as e:
    st.error(f"Error loading model weights. Ensure `{MODEL_PATH}` exists. Details: {e}")
    st.stop()

# ----------------- Sidebar -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/corn.png", width=65)
    st.title("ZeaMedic System")
    st.markdown("Automated pathology assessment & diagnostic engine for maize (*Zea mays*).")
    st.divider()
    confidence_threshold = st.slider("Confidence Warning Threshold", 0.50, 0.95, 0.65, 0.05)
    st.caption("Classified: Common Rust, Gray Leaf Spot, Blight, Healthy")

# ----------------- UI Layout -----------------
st.markdown('<p class="main-title">🌽 ZeaMedic: Real-Time Corn Leaf Disease Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Capture an image using your device camera or upload a field photograph for instant classification.</p>', unsafe_allow_html=True)

col_input, col_results = st.columns([1, 1.2], gap="large")

captured_image = None

with col_input:
    st.subheader("1. Provide Leaf Sample")
    tab_cam, tab_file = st.tabs(["📸 Live Camera", "📁 File Upload"])

    with tab_cam:
        cam_shot = st.camera_input("Aim camera at the corn leaf lesion")
        if cam_shot is not None:
            captured_image = Image.open(cam_shot)

    with tab_file:
        uploaded_file = st.file_uploader("Select image file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            captured_image = Image.open(uploaded_file)

    if captured_image is not None:
        st.image(captured_image, caption="Current Analysis Sample", use_container_width=True)

with col_results:
    st.subheader("2. Diagnostic Report")

    if captured_image is not None:
        with st.spinner("Classifying leaf pathology..."):
            res = predict(model, captured_image)

        pred_class = res["class"]
        confidence = res["confidence"]
        info = res["info"]

        # Alert banner
        status_color = "#2E7D32" if pred_class == "Healthy" else "#D32F2F"
        st.markdown(f"""
            <div style="background-color: #FAFAFA; border: 1px solid #E0E0E0; border-left: 6px solid {status_color}; padding: 14px; border-radius: 6px;">
                <h3 style="margin: 0; color: {status_color};">{pred_class}</h3>
                <p style="margin: 4px 0 0 0; font-size: 15px;"><strong>Inference Confidence:</strong> {confidence * 100:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

        if confidence < confidence_threshold:
            st.warning(f"⚠️ **Low Confidence Flag ({confidence * 100:.1f}%)**: Results are uncertain. Ensure proper lighting and leaf lesion focus.")

        st.write("")

        # Plotly horizontal probability bar
        probs_df = pd.DataFrame({
            "Pathology": list(res["probabilities"].keys()),
            "Probability (%)": [v * 100 for v in res["probabilities"].values()]
        })

        fig = px.bar(
            probs_df,
            x="Probability (%)",
            y="Pathology",
            orientation="h",
            text=[f"{v:.1f}%" for v in probs_df["Probability (%)"]],
            color="Probability (%)",
            color_continuous_scale="Greens"
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=220, margin=dict(l=0, r=0, t=15, b=15))
        st.plotly_chart(fig, use_container_width=True)

        # Agronomic guide
        with st.expander("📋 Pathological Profile & Treatment Plan", expanded=True):
            st.markdown(f"**Pathogen:** {info.get('pathogen', 'N/A')}")
            st.markdown(f"**Primary Symptoms:** {info.get('symptoms', 'N/A')}")
            st.markdown(f"**Agronomic Practice:** {info.get('management', 'N/A')}")
    else:
        st.info("Awaiting input image from camera or file upload.")