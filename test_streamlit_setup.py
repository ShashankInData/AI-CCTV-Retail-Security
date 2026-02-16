"""Quick sanity check before launching the Streamlit dashboard."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
errors = []


def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if not ok:
        errors.append(label)


print("=" * 50)
print("  Streamlit Dashboard Setup Check")
print("=" * 50)

# 1. Python packages
pkgs = {
    "streamlit": "streamlit",
    "torch": "torch",
    "transformers": "transformers",
    "cv2": "opencv-python",
    "torchvision": "torchvision",
    "PIL": "Pillow",
    "ultralytics": "ultralytics",
    "supervision": "supervision",
    "numpy": "numpy",
}
for mod, pip_name in pkgs.items():
    try:
        __import__(mod)
        check(f"import {mod}", True)
    except ImportError:
        check(f"import {mod}", False, f"pip install {pip_name}")

# 2. Model files
model_dir = ROOT / "models" / "videomae-shoplifting-best"
check("Model directory exists", model_dir.is_dir(), str(model_dir))
check("config.json", (model_dir / "config.json").is_file())
has_weights = (
    (model_dir / "model.safetensors").is_file()
    or (model_dir / "pytorch_model.bin").is_file()
)
check("Model weights", has_weights)

# 3. YOLO weights
yolo_path = ROOT / "yolov8n.pt"
check("YOLOv8 weights", yolo_path.is_file(), str(yolo_path))

# 4. GPU
import torch
gpu = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if gpu else "none"
check("CUDA GPU", gpu, gpu_name)

# 5. Streamlit version
import streamlit
check("Streamlit version", True, streamlit.__version__)

print()
if errors:
    print(f"  ** {len(errors)} check(s) failed. Fix them before launching.")
    sys.exit(1)
else:
    print("  All checks passed! Run:")
    print("    streamlit run src\\dashboard\\video_detection_app.py")
    print("  Or double-click run_streamlit.bat")
