# Notebooks Directory

This directory is dedicated to interactive Exploratory Data Analysis (EDA) and experimental prototyping for the **Industrial Fault Detection and Maintenance Prioritization System**.

## Structure & Purpose

- `01_exploratory_data_analysis.ipynb`:
  - Inspect raw vibration signals from CWRU MATLAB (`.mat`) files.
  - Visualize time-domain waveforms for each of the 4 bearing health states:
    1. Normal Baseline
    2. Inner Race Fault
    3. Ball Fault
    4. Outer Race Fault
  - Perform Fast Fourier Transform (FFT) to identify characteristic defect frequencies (BPFI, BPFO, BSF, FTF).
  - Analyze statistical metrics across different operating loads (0, 1, 2, 3 HP) and defect sizes (0.007", 0.014", 0.021").

## Guidelines

- All core functions (loading, filtering, feature calculation) should reside in `src/` so notebooks remain clean and reproducible.
- Keep notebooks focused on insights, visualizations, and experimental validation.
