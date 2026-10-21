import sqlite3
from datetime import datetime
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# ==================================================
# PATHS AND CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_mobilenet_model.keras"
DB_PATH = BASE_DIR / "prediction_history.db"
MATRIX_PATH = BASE_DIR / "confusion_matrix.png"

IMAGE_SIZE = (224, 224)

# IMPORTANT: Keep this order identical to training.
CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary",
]

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
)

# ==================================================
# SQLITE DATABASE
# ==================================================

def get_connection():
    return sqlite3.connect(str(DB_PATH), timeout=10)


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_name TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL
            )
        """)


def save_prediction(image_name, prediction, confidence):
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO prediction_history
                (timestamp, image_name, prediction, confidence)
            VALUES (?, ?, ?, ?)
        """, (
            timestamp,
            image_name,
            prediction,
            float(confidence),
        ))


def load_history():
    with get_connection() as conn:
        return pd.read_sql_query("""
            SELECT
                timestamp AS 'Date & Time',
                image_name AS 'Image',
                prediction AS 'Prediction',
                ROUND(confidence, 2) AS 'Confidence (%)'
            FROM prediction_history
            ORDER BY id DESC
        """, conn)


def clear_history():
    with get_connection() as conn:
        conn.execute("DELETE FROM prediction_history")


initialize_database()

# ==================================================
# LOAD EXISTING TRAINED MODEL
# ==================================================

@st.cache_resource
def load_model():
    import tensorflow as tf

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Check the model file path in your project."
        )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )


# ==================================================
# MRI PREDICTION
# ==================================================

def predict_mri(image):
    """
    Uses the existing trained model.
    Expected input: RGB image.
    Expected preprocessing: resize to 224x224,
    convert to float32 and divide by 255.
    Expected model output: 4 class probabilities.
    """

    model = load_model()

    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    outputs = model.predict(image_array, verbose=0)
    probabilities = np.asarray(outputs[0]).reshape(-1)

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} model outputs, "
            f"but received {len(probabilities)}. "
            "Check the model and class order."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("The model returned invalid predictions.")

    # This assumes the trained model outputs softmax probabilities.
    if (
        np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.isclose(probabilities.sum(), 1.0, atol=0.02)
    ):
        raise ValueError(
            "Model output does not look like probabilities. "
            "Check whether your model uses a softmax output."
        )

    predicted_index = int(np.argmax(probabilities))
    prediction = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index] * 100)

    probability_percentages = probabilities * 100

    return prediction, confidence, probability_percentages


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>
/* Clean, responsive medical-AI dashboard styling */
.block-container { max-width: 1250px; padding-top: 2rem; padding-bottom: 3rem; }
.main-title { text-align: center; font-size: clamp(2rem, 4vw, 3.2rem); font-weight: 800; letter-spacing: -0.04em; margin: 0.4rem 0 0.3rem; }
.subtitle { text-align: center; color: #9ca3af; font-size: 1.08rem; margin-bottom: 1.6rem; }
.section-kicker { text-transform: uppercase; letter-spacing: .12em; font-size: .76rem; font-weight: 700; color: #7dd3fc; margin-bottom: .2rem; }
.result-card { background: linear-gradient(135deg, rgba(14,165,233,.14), rgba(99,102,241,.10)); border: 1px solid rgba(125,211,252,.42); border-radius: 18px; padding: 1.5rem; text-align: center; margin: .8rem 0 1rem; }
.result-label { font-size: .95rem; color: #cbd5e1; }
.result-value { font-size: clamp(1.8rem, 4vw, 2.5rem); font-weight: 800; margin: .45rem 0; }
.result-confidence { color: #cbd5e1; font-size: .95rem; }
div[data-testid="stMetric"] { background: rgba(148,163,184,.07); border: 1px solid rgba(148,163,184,.18); padding: 1rem; border-radius: 14px; }
div[data-testid="stFileUploader"] { border: 1px dashed rgba(125,211,252,.55); border-radius: 14px; padding: .5rem; }
.stButton > button, .stDownloadButton > button { border-radius: 10px; min-height: 2.8rem; font-weight: 650; }
hr { margin: 2rem 0; }
.small-note { color: #9ca3af; font-size: .9rem; }
@media (max-width: 700px) { .block-container { padding-left: 1rem; padding-right: 1rem; } }
</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="main-title">🧠 Brain Tumor Detection</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">AI-powered MRI image classification · MobileNetV2</div>',
    unsafe_allow_html=True,
)

intro_left, intro_right = st.columns([3, 1])
with intro_left:
    st.markdown(
        "Upload a brain MRI image to view the model's predicted category "
        "and class probability distribution."
    )
with intro_right:
    st.markdown("<div class='small-note'>🛡️ Educational use only</div>", unsafe_allow_html=True)

st.info(
    "This tool classifies images into Glioma, Meningioma, No Tumor, or Pituitary. "
    "It is a student project and does not provide a medical diagnosis."
)

# ==================================================
# MRI IMAGE UPLOAD AND PREDICTION
# ==================================================

st.divider()
st.markdown("<div class='section-kicker'>Image analysis</div>", unsafe_allow_html=True)
st.markdown("## 📤 Upload an MRI image")

uploaded_file = st.file_uploader(
    "Choose a brain MRI image",
    type=["jpg", "jpeg", "png"],
    key="mri_uploader",
)

if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        st.image(
            image,
            caption=f"Uploaded MRI Image: {uploaded_file.name}",
            use_container_width=True,
        )

        if st.button(
            "🔍 Predict Tumor Type",
            use_container_width=True,
            type="primary",
        ):
            try:
                with st.spinner("Analyzing MRI image..."):
                    prediction, confidence, probabilities = predict_mri(
                        image
                    )

                    # Save only after a successful prediction.
                    save_prediction(
                        uploaded_file.name,
                        prediction,
                        confidence,
                    )

                # Store the latest result for Streamlit reruns.
                st.session_state["latest_prediction"] = {
                    "image_name": uploaded_file.name,
                    "prediction": prediction,
                    "confidence": confidence,
                    "probabilities": probabilities.tolist(),
                }

                st.success(
                    "Prediction completed and saved to SQLite!"
                )

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    except Exception as exc:
        st.error(f"Could not open this image: {exc}")

# ==================================================
# LATEST PREDICTION RESULT
# ==================================================

latest = st.session_state.get("latest_prediction")

if latest:
    st.divider()
    st.markdown("<div class='section-kicker'>Analysis result</div>", unsafe_allow_html=True)
    st.markdown("## 🧠 Prediction summary")

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Prediction</div>
            <div class="result-value">{latest["prediction"]}</div>
            <div class="result-confidence">
                Model Confidence: {latest["confidence"]:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(f'Image: {latest["image_name"]}')

    st.markdown("### 📊 Class probability distribution")

    probability_df = pd.DataFrame({
        "Tumor Category": CLASS_NAMES,
        "Probability (%)": latest["probabilities"],
    })

    st.bar_chart(
        probability_df.set_index("Tumor Category"),
        y="Probability (%)",
    )

    st.caption(
        "Probabilities represent the model's outputs, "
        "not a medically confirmed diagnosis."
    )

# ==================================================
# MODEL PERFORMANCE
# ==================================================

st.divider()
st.markdown("<div class='section-kicker'>Evaluation</div>", unsafe_allow_html=True)
st.markdown("## 📊 Model performance")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", "91%")
col2.metric("Macro F1-score", "91%")
col3.metric("Test Images", "1,600")

st.caption("Reported test-set results; these are fixed evaluation results, not live dashboard calculations.")
st.markdown("### Class-wise F1-scores")

f1_data = {
    "Glioma": 86,
    "Meningioma": 86,
    "No Tumor": 96,
    "Pituitary": 96,
}

for class_name, score in f1_data.items():
    st.write(f"**{class_name}:** {score}%")

if MATRIX_PATH.exists():
    st.markdown("### 🟪 Confusion Matrix")

    st.image(
        str(MATRIX_PATH),
        caption="Confusion Matrix on Test Dataset",
        use_container_width=True,
    )
else:
    st.warning(
        "confusion_matrix.png was not found. "
        "Check that the file is in the project folder."
    )

# ==================================================
# PERMANENT PREDICTION HISTORY
# ==================================================

st.divider()
st.markdown("<div class='section-kicker'>Saved records</div>", unsafe_allow_html=True)
st.markdown("## 🕒 Prediction history")

st.caption(
    "Review predictions saved in SQLite. "
    "Records remain available after refreshing or restarting "
    "the app, provided the database file is preserved."
)

history = load_history()

if history.empty:
    st.info("No predictions have been saved yet.")

else:
    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = history.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="📥 Download Prediction History",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if st.button(
        "🗑️ Clear Prediction History",
        use_container_width=True,
    ):
        st.session_state["confirm_clear_history"] = True

    if st.session_state.get("confirm_clear_history", False):
        st.warning(
            "This will permanently delete all saved history records."
        )

        col_yes, col_no = st.columns(2)

        with col_yes:
            if st.button(
                "Confirm Delete",
                use_container_width=True,
            ):
                clear_history()
                st.session_state.pop(
                    "confirm_clear_history", None
                )
                st.session_state.pop(
                    "latest_prediction", None
                )
                st.rerun()

        with col_no:
            if st.button(
                "Cancel",
                use_container_width=True,
            ):
                st.session_state.pop(
                    "confirm_clear_history", None
                )
                st.rerun()

# ==================================================
# ANALYTICS DASHBOARD
# ==================================================

st.divider()
st.markdown("<div class='section-kicker'>Usage insights</div>", unsafe_allow_html=True)
st.markdown("## 📈 Prediction analytics dashboard")
st.caption(
    "Summary and charts are generated from predictions saved "
    "in your SQLite prediction history."
)

# Use the same history DataFrame loaded from SQLite above.
if history.empty:
    st.info("Analytics will appear after you make your first prediction.")
else:
    analytics = history.copy()
    analytics["Confidence (%)"] = pd.to_numeric(
        analytics["Confidence (%)"], errors="coerce"
    )
    analytics = analytics.dropna(subset=["Confidence (%)"])

    if analytics.empty:
        st.info("There are no valid prediction records to analyze.")
    else:
        total_predictions = len(analytics)
        average_confidence = analytics["Confidence (%)"].mean()
        class_counts = analytics["Prediction"].value_counts()
        most_predicted_class = (
            class_counts.idxmax() if not class_counts.empty else "N/A"
        )

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Total Predictions", f"{total_predictions}")
        metric2.metric(
            "Average Confidence", f"{average_confidence:.2f}%"
        )
        metric3.metric("Most Predicted Class", most_predicted_class)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("### Predictions by Class")
            counts_df = (
                class_counts.rename_axis("Tumor Category")
                .reset_index(name="Number of Predictions")
            )
            st.bar_chart(
                counts_df.set_index("Tumor Category"),
                y="Number of Predictions",
            )

        with chart_col2:
            st.markdown("### Prediction Distribution")
            fig, ax = plt.subplots()
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            ax.pie(
                class_counts.values,
                labels=class_counts.index,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.axis("equal")
            ax.tick_params(colors="white")
            st.pyplot(fig, transparent=True)
            plt.close(fig)

        st.markdown("### Average Confidence by Class")
        confidence_by_class = (
            analytics.groupby("Prediction")["Confidence (%)"]
            .mean()
            .sort_values(ascending=False)
            .rename_axis("Tumor Category")
            .reset_index(name="Average Confidence (%)")
        )
        st.bar_chart(
            confidence_by_class.set_index("Tumor Category"),
            y="Average Confidence (%)",
        )
        st.dataframe(
            confidence_by_class.round(2),
            use_container_width=True,
            hide_index=True,
        )

# ==================================================
# DISCLAIMER
# ==================================================

st.divider()

st.caption(
    "⚠️ Educational AI project only. Predictions are not "
    "medical diagnoses and must not replace evaluation by "
    "a qualified healthcare professional."
)