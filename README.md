# Industrial Fault Detection and Maintenance Prioritization System

A college engineering Data Science project focused on industrial condition monitoring, fault diagnosis, and operational maintenance prioritization for rolling element bearings.

---

## 📌 Project Overview

In industrial manufacturing, unexpected motor bearing failures cause severe production downtimes, equipment damage, and maintenance overhead. This project implements an end-to-end Data Science pipeline to:
1. **Analyze vibration measurements** acquired from industrial motor bearings.
2. **Classify 4 bearing health conditions**:
   - `Normal Baseline`
   - `Inner Race Fault`
   - `Ball Fault`
   - `Outer Race Fault`
3. **Compare Machine Learning and Deep Learning approaches**:
   - Classical ML baseline (Random Forest / SVM) trained on statistical time-domain and FFT spectral features.
   - Deep Learning (1D-CNN) operating directly on vibration time-series windows.
4. **Compute Maintenance Priorities (MPI)**:
   - Combine diagnosed fault kinematic risk, vibration energy (RMS), and motor load into an actionable priority score (0–100) and operational recommendations.
5. **Interactive Dashboard**:
   - Streamlit interface for vibration signal visualization, model inference, and maintenance urgency dispatching.

---

## 🔬 Dataset: Case Western Reserve University (CWRU) Bearing Center

This project utilizes the official CWRU Bearing Data Center benchmark:
- **Official Portal**: [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file)
- **Vibration Signals**: Accelerometer measurements recorded at 12,000 samples/second (Drive End) and 48,000 samples/second.
- **Motor Loads**: 0 to 3 Horsepower (approx. 1797 to 1730 RPM).
- **Fault Diameters**: 0.007", 0.014", and 0.021" electro-discharge machined (EDM) faults.

> **Data Integrity Notice**: Raw MATLAB (`.mat`) vibration recordings are directly ingested from the official repository. No fabricated or artificial datasets are substituted.

---

## 🛡️ Critical Data Science Design: Leakage Prevention

In vibration signal analysis, slicing a continuous time-series into sliding windows with high overlap (e.g., 50% or 75%) and then randomly shuffling windows into training and test splits causes **severe data leakage**. Because adjacent overlapping windows share up to 75% of the same timestamp sequence, standard random splits yield artificially inflated accuracies (e.g., 99.9%) that completely fail when deployed on unseen continuous operational data.

**Our Evaluation Protocol**:
- Continuous time-series recordings are partitioned **temporally** or by **distinct recording sessions / motor loads** *before* windowing.
- Normalization parameters (mean, standard deviation) are computed strictly on the training partition and applied to validation/test partitions.
- No window in the test set shares timestamps or overlaps with any window in the training set.

---

## 📁 Project Directory Structure

```text
d:/DS project/
├── data/
│   ├── raw/                # Downloaded official CWRU .mat vibration files
│   └── processed/          # Cached feature matrices and preprocessed splits
├── notebooks/              # Exploratory Data Analysis & signal visualization
│   └── README.md
├── src/                    # Modular source code
│   ├── __init__.py
│   ├── data_loader.py      # CWRU .mat file downloader and reader
│   ├── preprocessing.py    # Leakage-safe temporal splitting & windowing
│   ├── feature_extraction.py # Time-domain (RMS, Kurtosis, Crest) & FFT spectral features
│   ├── train_baseline.py   # Random Forest & SVM baseline model training
│   ├── train_cnn.py        # PyTorch 1D-CNN architecture & training loop
│   ├── evaluate.py         # Precision, Recall, F1, Confusion Matrix
│   └── prioritization.py   # Maintenance Priority Index (MPI) scoring engine
├── models/                 # Saved model weights (.pt, .joblib)
├── app/
│   └── streamlit_app.py    # Interactive Streamlit dashboard
├── tests/                  # Automated verification tests
│   ├── __init__.py
│   └── test_basic.py
├── requirements.txt        # Project dependencies
├── .gitignore              # Git ignore rules for data, models, and cache
└── README.md               # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- `pip` package manager

### 2. Environment Setup & Dependency Installation
```bash
# Clone or navigate to the project directory
cd "d:/DS project"

# Install required dependencies
pip install -r requirements.txt
```

### 3. Verify Setup with Automated Tests
```bash
pytest
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 🚀 Next Steps: Data Acquisition & EDA

1. **Download CWRU .mat files**:
   - Download representative normal and fault files from the CWRU repository into `data/raw/`.
2. **Exploratory Data Analysis**:
   - Inspect raw signals, plot time-domain vibrations, and calculate FFT spectra to observe fault-characteristic frequencies.
3. **Model Development**:
   - Extract features and train baseline models, followed by 1D-CNN training and measured evaluation.
