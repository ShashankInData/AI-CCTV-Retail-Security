"""
Download and extract the UCF-Crime dataset from Kaggle.
Dataset: https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset
~15GB download — extraction takes 10-20 minutes.

Setup (one-time):
    1. Go to https://www.kaggle.com/settings -> API -> Create New Token
    2. Save kaggle.json to C:\\Users\\reddy\\.kaggle\\kaggle.json
    3. pip install kaggle

Run:
    python scripts/download_dataset.py
"""

import zipfile
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

DATASET = "odins0n/ucf-crime-dataset"
ZIP_NAME = "ucf-crime-dataset.zip"
EXTRACT_DIR = DATA_RAW / "ucf-crime"


def check_credentials():
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(
            f"Kaggle credentials not found at {kaggle_json}\n"
            "  1. Go to https://www.kaggle.com/settings -> API -> Create New Token\n"
            "  2. Save the downloaded kaggle.json to C:\\Users\\reddy\\.kaggle\\kaggle.json"
        )
    print(f"[+] Kaggle credentials found: {kaggle_json}")


def setup_dirs():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    print(f"[+] Directories ready")
    print(f"    raw       -> {DATA_RAW}")
    print(f"    processed -> {DATA_PROCESSED}")


def download():
    zip_path = DATA_RAW / ZIP_NAME
    if zip_path.exists():
        size_gb = zip_path.stat().st_size / 1e9
        print(f"[!] ZIP already exists ({size_gb:.1f} GB) — skipping download.")
        return

    print(f"[+] Downloading dataset: {DATASET}")
    print(f"    Destination: {DATA_RAW}")
    print(f"    Expected size: ~15 GB — this will take a while...\n")

    from kaggle import KaggleApi
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(DATASET, path=str(DATA_RAW), unzip=False, quiet=False)

    print("\n[+] Download complete.")


def extract():
    zip_path = DATA_RAW / ZIP_NAME
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found at {zip_path} — run download first.")

    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        print(f"[!] Already extracted at {EXTRACT_DIR} — skipping.")
        return

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    size_gb = zip_path.stat().st_size / 1e9
    print(f"[+] Extracting {size_gb:.1f} GB ZIP -> {EXTRACT_DIR}")
    print(f"    This may take 10-20 minutes...\n")

    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        for i, member in enumerate(members, 1):
            z.extract(member, EXTRACT_DIR)
            if i % 500 == 0 or i == len(members):
                print(f"    Extracted {i}/{len(members)} files...", end="\r")

    print(f"\n[+] Extraction complete.")


def print_summary():
    print("\n" + "=" * 50)
    print("[OK] Dataset ready!")
    print(f"    Location : {EXTRACT_DIR}")
    if EXTRACT_DIR.exists():
        subdirs = [d for d in EXTRACT_DIR.iterdir() if d.is_dir()]
        print(f"    Folders  : {len(subdirs)}")
    print("=" * 50)


def main():
    check_credentials()
    setup_dirs()
    download()
    extract()
    print_summary()


if __name__ == "__main__":
    main()
