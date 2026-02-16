# AI CCTV Retail Security

Real-time shoplifting detection system using deep learning video understanding. Analyzes CCTV footage to detect suspicious behavior by learning temporal motion patterns — how items are concealed, how body posture shifts — rather than classifying single frames.

## Model Performance

Evaluated on 1,011 test clips from UCF-Crime dataset at threshold 0.80:

| Metric | Value |
|--------|-------|
| Detection Rate (Recall) | 97.4% |
| Precision | 98.9% |
| F1 Score | 98.1% |
| False Positive Rate | 0.9% |
| Est. False Alarms/Hour | ~122 (at 30fps continuous) |

Confusion matrix (1,011 clips):
```
                Predicted Normal    Predicted Shoplifting
Actual Normal        546 (TN)            5 (FP)
Actual Shoplift       12 (FN)          448 (TP)
```

## How It Works

**VideoMAE-Base** (Video Masked Autoencoder) processes 16 frames at once and learns temporal motion patterns across the clip. This is fundamentally different from image classifiers that look at single frames:

1. **Frame sampling** — 16 frames spaced 10 apart (covering ~5s of footage at 30fps)
2. **Sliding window** — Inference runs every 8 raw frames for continuous monitoring
3. **Temporal smoothing** — Rolling average of last 3 predictions filters single-frame noise
4. **Alert threshold** — Smoothed probability must exceed 0.80 to trigger an alert

Person detection uses **YOLOv8-nano** for real-time bounding boxes around each individual.

## Dataset

**UCF-Crime** — a large-scale anomaly detection dataset containing 1,900+ real-world surveillance videos across 13 anomaly categories. This project uses the **Shoplifting** subset (shoplifting vs. normal behavior).

- Training: pre-extracted frames with stride-10 sampling into 16-frame clips
- Classes: `normal` (0), `shoplifting` (1)
- Class weighting: 2x weight on shoplifting class to handle imbalance

The dataset is not included in this repository (~12GB). To reproduce:
1. Download UCF-Crime from [Kaggle](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset)
2. Extract to `data/raw/ucf-crime/`
3. Run `python scripts/prepare_frame_dataset.py`

## Project Structure

```
AI-CCTV-Retail-Security/
├── scripts/
│   ├── inference.py              # Inference: video, webcam, or batch evaluate
│   ├── train_local.py            # Full fine-tuning with layer-wise LR
│   ├── prepare_frame_dataset.py  # Extract frames + build clips.txt
│   └── analyze_errors.py         # Visualize FP/FN as contact sheets
├── src/
│   ├── dashboard/
│   │   └── video_detection_app.py  # Streamlit web UI
│   ├── data/
│   │   └── clip_dataset.py         # PyTorch dataset for frame clips
│   └── pipeline/
│       ├── detector.py             # YOLOv8 person detection
│       └── tracker.py              # ByteTrack person tracking
├── run_streamlit.bat              # One-click Windows launcher
├── test_streamlit_setup.py        # Pre-launch environment check
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended; CPU works but is slow)

### Installation

```bash
git clone https://github.com/ShashankInData/AI-CCTV-Retail-Security.git
cd AI-CCTV-Retail-Security

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

### Download Model Weights

The trained VideoMAE model (~350MB) is not tracked in git. To use the pre-trained model:
1. Download from [Releases](https://github.com/ShashankInData/AI-CCTV-Retail-Security/releases) (or train your own)
2. Place in `models/videomae-shoplifting-best/`

YOLOv8-nano weights (`yolov8n.pt`) auto-download on first run.

### Verify Setup

```bash
python test_streamlit_setup.py
```

## Usage

### Streamlit Dashboard (recommended)

```bash
streamlit run src/dashboard/video_detection_app.py
```
Or double-click `run_streamlit.bat` on Windows. Upload a video, adjust threshold in the sidebar, click Start Detection.

### Command-Line Inference

```bash
# Evaluate on test set
python scripts/inference.py --evaluate data/processed/frame_classification/test/clips.txt --threshold 0.8

# Process a video file
python scripts/inference.py --input path/to/video.mp4 --output annotated_output.mp4

# Live webcam
python scripts/inference.py --threshold 0.8
```

### Training

```bash
python scripts/train_local.py
```
Full fine-tuning of VideoMAE-Base with layer-wise learning rates (backbone: 2e-5, classifier head: 2e-4). Trains for 20 epochs, saves best model by F1 score.

### Error Analysis

```bash
python scripts/analyze_errors.py --threshold 0.8
```
Generates contact sheet images in `outputs/error_analysis/` showing the 16 frames of each misclassified clip.

## Retail Industry Application

This system is designed for real-world retail deployment:

**Current capability:**
- Upload recorded CCTV footage and get alerts with timestamps
- Works on standard surveillance camera output (any resolution, any FPS)
- Threshold tuning per store — high-traffic stores can raise threshold to reduce false alarms

**Production deployment path:**
- **Edge deployment** — Run on an NVIDIA Jetson or local GPU box connected to store cameras via RTSP streams
- **Multi-camera support** — One GPU can process 4-8 camera feeds at stride-8 inference
- **Alert integration** — Connect alerts to existing store security systems (email, SMS, POS integration)
- **Retraining on store data** — Fine-tune on footage from your specific store layout to improve accuracy for your environment

**What it does NOT do:**
- It does not identify individuals (no face recognition)
- It does not track across cameras (single-camera per instance)
- It requires a GPU for real-time processing (~15-25 fps on RTX 5070)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Video classification | VideoMAE-Base (HuggingFace Transformers) |
| Person detection | YOLOv8-nano (Ultralytics) |
| Person tracking | ByteTrack (Supervision) |
| Frontend | Streamlit |
| Training | PyTorch + CUDA |
| Video processing | OpenCV |

## License

See [LICENSE](LICENSE) for details.
