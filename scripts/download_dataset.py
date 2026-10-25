"""
Dataset downloader utility for UCI Machine Learning Repository: Online Retail II.
Dataset URL: https://archive.ics.uci.edu/dataset/502/online+retail+ii
Direct download URL: https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm


UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download a file with a visual progress indicator."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract zip archive."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def ensure_dataset(raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    """
    Ensure the dataset is present in raw_dir.
    Looks for online_retail_II.xlsx or online_retail_II.csv.
    If neither exists, downloads and extracts the official zip from UCI.
    Returns path to the primary dataset file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = raw_dir / "online_retail_II.xlsx"
    csv_path = raw_dir / "online_retail_II.csv"

    if xlsx_path.exists():
        print(f"Found existing raw Excel dataset: {xlsx_path}")
        return xlsx_path
    if csv_path.exists():
        print(f"Found existing raw CSV dataset: {csv_path}")
        return csv_path

    zip_path = raw_dir / "online_retail_ii.zip"
    print(f"Dataset not found locally in {raw_dir}.")
    print(f"Downloading official dataset from UCI ML Repository: {UCI_ZIP_URL}")

    download_file(UCI_ZIP_URL, zip_path)
    print(f"Extracting {zip_path.name}...")
    extract_zip(zip_path, raw_dir)

    # Clean up zip after extraction to save disk space
    if zip_path.exists():
        try:
            zip_path.unlink()
        except OSError:
            pass

    if xlsx_path.exists():
        print(f"Successfully downloaded and extracted: {xlsx_path}")
        return xlsx_path
    elif csv_path.exists():
        print(f"Successfully downloaded and extracted: {csv_path}")
        return csv_path
    else:
        # Search for any .xlsx or .csv in raw_dir
        found_files = list(raw_dir.glob("*.xlsx")) + list(raw_dir.glob("*.csv"))
        if found_files:
            print(f"Identified dataset file: {found_files[0]}")
            return found_files[0]
        raise FileNotFoundError(
            f"Extraction completed but could not find dataset file in {raw_dir}. "
            "Please check the extracted files."
        )


if __name__ == "__main__":
    try:
        data_file = ensure_dataset()
        print(f"Ready: {data_file} ({data_file.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"Error downloading dataset: {e}", file=sys.stderr)
        sys.exit(1)
