# AI CCTV Retail Security

An AI-powered CCTV surveillance system for retail security that uses deep learning to detect and track people in real time.

## Features

- **Person Detection** — YOLOv8-based real-time person detection
- **Person Tracking** — ByteTrack multi-object tracking with persistent IDs
- **Real-time Processing** — Live webcam/CCTV feed analysis with annotated output

## Tech Stack

- **YOLOv8** (Ultralytics) — Object detection
- **Supervision** — Tracking (ByteTrack) and annotation
- **OpenCV** — Video capture and display
- **PyTorch** — Deep learning backend
- **FastAPI** — API server
- **Streamlit** — Dashboard UI
- **SQLAlchemy** — Database ORM

## Project Structure

```
AI-CCTV-Retail-Security/
├── src/
│   └── pipeline/
│       ├── detector.py      # YOLOv8 person detection
│       └── tracker.py       # ByteTrack person tracking
├── data/                    # Dataset (not tracked)
├── models/                  # Saved models (not tracked)
├── notebooks/               # Jupyter notebooks
├── outputs/                 # Output files (not tracked)
├── scripts/                 # Utility scripts
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/ShashankInData/AI-CCTV-Retail-Security.git
cd AI-CCTV-Retail-Security

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python test_imports.py
```

## Usage

### Run Person Detection

```bash
python src/pipeline/detector.py
```

### Run Person Tracking

```bash
cd src
python -m pipeline.tracker
```

Press `q` to quit the live preview.

## License

See [LICENSE](LICENSE) for details.
