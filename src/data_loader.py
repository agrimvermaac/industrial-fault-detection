"""
Data Loader Module for Case Western Reserve University (CWRU) Bearing Data.

Responsible for downloading and reading MATLAB (.mat) vibration signal files
from the official CWRU Bearing Data Center.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import urllib.request
import scipy.io


# Class labels mapping for the 4-class classification problem
FAULT_CLASSES = {
    0: "Normal",
    1: "Inner Race Fault",
    2: "Ball Fault",
    3: "Outer Race Fault"
}

# Standard sampling frequencies in CWRU dataset
SAMPLING_RATE_12K = 12000  # 12,000 samples/second for Drive End (DE) data
SAMPLING_RATE_48K = 48000  # 48,000 samples/second for Fan End (FE) data


def download_cwru_mat_file(download_url: str, save_path: Path) -> Path:
    """
    Downloads a single .mat file from the CWRU repository if it does not already exist.

    Args:
        download_url: URL to download the .mat file from.
        save_path: Destination path on the local disk.

    Returns:
        Path: Verified path to the downloaded file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        print(f"[DataLoader] File already exists at {save_path}, skipping download.")
        return save_path

    print(f"[DataLoader] Downloading from {download_url} to {save_path}...")
    urllib.request.urlretrieve(download_url, str(save_path))
    print(f"[DataLoader] Download complete: {save_path.name}")
    return save_path


def load_cwru_mat(file_path: Path) -> Dict[str, Any]:
    """
    Loads and parses a raw MATLAB (.mat) file from the CWRU bearing dataset.

    Extracts:
        - Drive End vibration signal (key typically contains 'DE_time')
        - Fan End vibration signal (key typically contains 'FE_time')
        - Base plate vibration signal (key typically contains 'BA_time')
        - Motor rotational speed (RPM)

    Args:
        file_path: Path to the .mat file.

    Returns:
        Dict[str, Any]: Parsed dictionary containing extracted vibration arrays and metadata.

    Raises:
        FileNotFoundError: If the specified file does not exist on disk.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"CWRU data file not found at {file_path}. "
            "Please ensure real CWRU .mat files are downloaded into 'data/raw/'."
        )

    mat_contents = scipy.io.loadmat(str(file_path))
    parsed_record: Dict[str, Any] = {
        "file_name": file_path.name,
        "de_signal": None,
        "fe_signal": None,
        "ba_signal": None,
        "rpm": None
    }

    for key, value in mat_contents.items():
        if "DE_time" in key:
            parsed_record["de_signal"] = value.squeeze()
        elif "FE_time" in key:
            parsed_record["fe_signal"] = value.squeeze()
        elif "BA_time" in key:
            parsed_record["ba_signal"] = value.squeeze()
        elif "RPM" in key:
            parsed_record["rpm"] = float(value.squeeze())

    return parsed_record
