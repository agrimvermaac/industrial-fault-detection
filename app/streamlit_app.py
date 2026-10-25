"""
Streamlit Interactive Dashboard for Bearing Fault Detection and Maintenance Prioritization.

Provides an operator-facing interface to:
1. Inspect uploaded or sample vibration signals (waveform + FFT spectrum).
2. Run fault classification via trained baseline (Random Forest/SVM) or 1D-CNN.
3. Calculate and display dynamic Maintenance Priority Scores and recommended actions.
"""

from pathlib import Path
import streamlit as st
import numpy as np


# Page configuration
st.set_page_config(
    page_title="Industrial Fault Intelligence & Maintenance Prioritization",
    page_icon="⚙️",
    layout="wide"
)

def main():
    st.title("⚙️ Industrial Fault Detection & Maintenance Prioritization System")
    st.markdown(
        """
        **System Overview**: This platform monitors industrial motor bearing health using high-frequency vibration signals.
        It uses signal processing and machine learning to diagnose fault conditions and compute operational maintenance urgency.
        """
    )

    # Sidebar controls
    st.sidebar.header("🕹️ Control & Configuration")
    selected_model_type = st.sidebar.selectbox(
        "Select Diagnostic Model",
        ["1D-CNN Deep Learning", "Random Forest Baseline", "Support Vector Machine"]
    )
    motor_hp = st.sidebar.slider("Motor Operating Load (HP)", min_value=0.0, max_value=3.0, value=1.0, step=1.0)
    uploaded_file = st.sidebar.file_uploader("Upload Vibration Signal (.mat / .npy / .csv)", type=["mat", "npy", "csv"])

    # Main dashboard tabs
    tab1, tab2, tab3 = st.tabs(["📊 Vibration Signal Analysis", "🎯 Fault Diagnosis", "🚨 Maintenance Prioritization"])

    with tab1:
        st.subheader("Vibration Signal Monitoring")
        if uploaded_file is None:
            st.info(
                "📁 No vibration file uploaded yet. Place CWRU `.mat` files in `data/raw/` "
                "or upload a vibration recording to inspect time-domain and frequency-domain waveforms."
            )
        else:
            st.success(f"Loaded file: {uploaded_file.name}")
            # Real signal visualization will render here once raw files are loaded

    with tab2:
        st.subheader("Model Diagnostic Inference")
        models_dir = Path("models")
        available_models = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.joblib"))

        if not available_models:
            st.warning(
                "⚠️ No trained model checkpoints found in `models/`. "
                "Per project design constraints, models will be trained on inspected CWRU data in the next phase."
            )
        else:
            st.write(f"Available model checkpoints: {[m.name for m in available_models]}")

    with tab3:
        st.subheader("Maintenance Prioritization & Action Matrix")
        st.markdown(
            """
            The Maintenance Priority Index (MPI) assesses fault severity, vibration energy (RMS),
            and operating load to assign urgency tiers:
            - **Low (< 25)**: Routine condition monitoring
            - **Medium (25 - 55)**: Increased surveillance & scheduled inspection
            - **High (55 - 80)**: Urgent replacement planning within 48h
            - **Critical (> 80)**: Immediate shutdown & isolation
            """
        )

if __name__ == "__main__":
    main()
