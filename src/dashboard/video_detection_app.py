"""
Streamlit dashboard for AI shoplifting detection.
Uploads a video, runs VideoMAE + YOLOv8 person detection,
and displays annotated results with alerts.

Launch:
    streamlit run src/dashboard/video_detection_app.py
"""
import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import time
from pathlib import Path
from collections import deque
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from transformers import VideoMAEForVideoClassification
from torchvision import transforms
from PIL import Image
from src.pipeline.detector import PersonDetector


# ---------------------------------------------------------------------------
# Model loading (cached so it only runs once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_videomae(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoMAEForVideoClassification.from_pretrained(model_path)
    model.to(device).eval()
    return model, device


@st.cache_resource
def load_yolo():
    return PersonDetector(conf_threshold=0.5)


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def predict_clip(model, frames, device):
    """Run VideoMAE on a list of 16 preprocessed tensors."""
    tensor = torch.stack(frames).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(pixel_values=tensor).logits
        probs = torch.softmax(logits, dim=1)[0]
    return float(probs[1])  # shoplifting probability


def annotate_frame(frame, detections, alert, smoothed_prob):
    """Draw bounding boxes and status bar on an OpenCV frame."""
    h, w = frame.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        if alert:
            color, label = (0, 0, 255), "SHOPLIFTING"
        else:
            color, label = (0, 200, 0), "Normal"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    status = "ALERT" if alert else "Normal"
    bar_color = (0, 0, 255) if alert else (0, 200, 0)
    cv2.putText(frame, f"{status}  prob={smoothed_prob*100:.1f}%",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)

    if alert:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

    return frame


def deduplicate_alerts(alerts, min_gap_frames=60):
    """Merge alerts that are within min_gap_frames of each other."""
    if not alerts:
        return []
    deduped = [alerts[0]]
    for a in alerts[1:]:
        if a["frame"] - deduped[-1]["frame"] >= min_gap_frames:
            deduped.append(a)
        elif a["smoothed_prob"] > deduped[-1]["smoothed_prob"]:
            deduped[-1] = a
    return deduped


# ---------------------------------------------------------------------------
# Processing -- stores everything in session_state
# ---------------------------------------------------------------------------
def run_detection(video_path, fps, total_frames,
                  threshold, frame_stride, window_stride, smooth_window):
    """Process video and save all results to st.session_state."""

    model, device = load_videomae(str(ROOT / "models" / "videomae-shoplifting-best"))
    yolo = load_yolo()

    cap = cv2.VideoCapture(video_path)
    clip_length = 16
    buffer_size = clip_length * frame_stride
    frame_buffer = deque(maxlen=buffer_size)
    recent_probs = deque(maxlen=smooth_window)

    alerts = []
    annotated_frames = []  # list of (frame_number, jpeg_bytes)
    raw_count = 0
    last_smoothed = 0.0
    last_alert = False

    progress = st.progress(0, text="Processing...")
    status_text = st.empty()
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        raw_count += 1

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_buffer.append(TRANSFORM(pil))

        if len(frame_buffer) >= buffer_size and raw_count % window_stride == 0:
            buf = list(frame_buffer)
            sampled = [buf[i * frame_stride] for i in range(clip_length)]
            prob = predict_clip(model, sampled, device)
            recent_probs.append(prob)
            last_smoothed = float(np.mean(recent_probs))
            last_alert = last_smoothed >= threshold

        detections = []
        if raw_count % 3 == 0:
            detections = yolo.detect(frame)

        if last_alert and detections:
            alerts.append({
                "frame": raw_count,
                "time": raw_count / fps,
                "smoothed_prob": last_smoothed,
                "n_persons": len(detections),
            })

        # Store ~2 annotated frames per second as compressed JPEG bytes
        if raw_count % max(fps // 2, 1) == 0:
            annotated = annotate_frame(frame.copy(), detections,
                                       last_alert, last_smoothed)
            _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_frames.append((raw_count, jpeg.tobytes()))

        pct = raw_count / total_frames if total_frames else 0
        elapsed = time.time() - t0
        proc_fps = raw_count / elapsed if elapsed > 0 else 0
        progress.progress(min(pct, 1.0),
                          text=f"Frame {raw_count}/{total_frames}  "
                               f"({proc_fps:.1f} fps)")

    cap.release()
    elapsed = time.time() - t0
    progress.progress(1.0, text="Done!")
    status_text.success(
        f"Processed {raw_count} frames in {elapsed:.1f}s "
        f"({raw_count/elapsed:.1f} fps)"
    )

    # Persist results in session_state so they survive reruns
    unique_alerts = deduplicate_alerts(alerts, min_gap_frames=fps * 2)

    # Pre-compute alert -> closest stored frame index
    alert_frame_indices = []
    for a in unique_alerts:
        target = a["frame"]
        closest_idx = min(
            range(len(annotated_frames)),
            key=lambda j: abs(annotated_frames[j][0] - target),
        ) if annotated_frames else 0
        alert_frame_indices.append(closest_idx)

    st.session_state["results"] = {
        "annotated_frames": annotated_frames,
        "alerts": unique_alerts,
        "alert_frame_indices": alert_frame_indices,
        "total_frames": raw_count,
        "elapsed": elapsed,
        "fps": fps,
    }


# ---------------------------------------------------------------------------
# Display -- reads from session_state, no reprocessing on slider change
# ---------------------------------------------------------------------------
def decode_frame(jpeg_bytes):
    """Decode stored JPEG bytes to RGB numpy array."""
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_results():
    """Render results from session_state."""
    res = st.session_state["results"]
    annotated_frames = res["annotated_frames"]
    unique_alerts = res["alerts"]
    alert_frame_indices = res["alert_frame_indices"]
    raw_count = res["total_frames"]
    elapsed = res["elapsed"]
    fps = res["fps"]

    st.markdown("---")

    # ---- Alert gallery: show each alert as a thumbnail you can click ----
    if unique_alerts:
        st.error(f"SHOPLIFTING DETECTED: {len(unique_alerts)} alert(s)")

        # Show all alerts as a grid with image previews
        cols_per_row = min(len(unique_alerts), 4)
        for row_start in range(0, len(unique_alerts), cols_per_row):
            row_alerts = unique_alerts[row_start:row_start + cols_per_row]
            row_indices = alert_frame_indices[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)

            for col_i, (a, frame_idx) in enumerate(zip(row_alerts, row_indices)):
                alert_num = row_start + col_i + 1
                _, jpeg_bytes = annotated_frames[frame_idx]
                img_rgb = decode_frame(jpeg_bytes)

                with cols[col_i]:
                    st.image(img_rgb, width=320,
                             caption=f"Alert #{alert_num} | t={a['time']:.1f}s | "
                                     f"{a['smoothed_prob']*100:.0f}% conf")
                    detail = (
                        f"Frame {a['frame']} | "
                        f"{a['n_persons']} person(s) | "
                        f"Confidence: {a['smoothed_prob']*100:.1f}%"
                    )
                    st.caption(detail)
                    if st.button(f"Inspect Alert #{alert_num}", key=f"jump_{alert_num}"):
                        st.session_state["viewer_idx"] = frame_idx
    else:
        st.success("No shoplifting detected in this video.")

    # ---- Frame browser (slider) ----
    if annotated_frames:
        st.markdown("---")
        st.subheader("Frame Browser")
        st.caption("Scroll through all stored frames, or click an alert above to jump there.")

        default_idx = st.session_state.get("viewer_idx", 0)
        # Clamp in case stored index is out of range
        default_idx = min(default_idx, len(annotated_frames) - 1)

        idx = st.slider(
            "Frame position",
            0,
            len(annotated_frames) - 1,
            default_idx,
            key="frame_slider",
        )
        st.session_state["viewer_idx"] = idx

        fnum, jpeg_bytes = annotated_frames[idx]
        img_rgb = decode_frame(jpeg_bytes)
        # Capped width so the image doesn't stretch across the full page
        st.image(img_rgb, caption=f"Frame {fnum}  (t={fnum/fps:.1f}s)",
                 width=720)

    # ---- Stats ----
    st.markdown("---")
    st.subheader("Detection Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total frames", raw_count)
    col2.metric("Alerts", len(unique_alerts))
    col3.metric("Processing speed", f"{raw_count/elapsed:.1f} fps")


# ---------------------------------------------------------------------------
# Sidebar with descriptive help text
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.header("Settings")

        threshold = st.slider(
            "Detection threshold", 0.50, 1.00, 0.80, 0.05,
            help="Minimum smoothed shoplifting probability to trigger an alert."
        )
        st.caption(
            "**0.80 (recommended):** Best balance found during evaluation -- "
            "97.4% detection rate with only 0.9% false positives. "
            "Lower it (0.6-0.7) to catch more subtle theft at the cost of "
            "more false alarms. Raise it (0.9+) for very high-confidence-only alerts."
        )

        frame_stride = st.slider(
            "Frame stride", 4, 16, 10, 1,
            help="Spacing between the 16 frames sampled for each clip."
        )
        st.caption(
            "**10 (default):** Matches training data where frames were extracted "
            "10 apart. A clip of 16 frames x stride 10 spans 160 raw frames "
            "(~5.3s at 30fps). Lower values (4-6) look at shorter time windows; "
            "higher values (12-16) look at longer actions."
        )

        window_stride = st.slider(
            "Window stride", 4, 24, 8, 1,
            help="Run inference every N raw frames."
        )
        st.caption(
            "**8 (default):** Run the model every 8 frames (~0.27s at 30fps). "
            "This balances latency vs GPU cost. Set to 4 for near-real-time "
            "responsiveness, or 16-24 to reduce GPU load on longer videos."
        )

        smooth_window = st.slider(
            "Temporal smoothing", 1, 7, 3, 1,
            help="Average the last N predictions before deciding."
        )
        st.caption(
            "**3 (default):** Averages the last 3 model outputs to filter "
            "single-frame spikes. A person must look suspicious across ~3 "
            "consecutive windows before an alert fires. "
            "Set to 1 to disable smoothing (raw model output). "
            "Set to 5-7 for very stable alerts with slower reaction."
        )

        st.markdown("---")
        st.subheader("About this model")
        st.markdown(
            "**VideoMAE-Base** -- a Video Masked Autoencoder pretrained on "
            "Kinetics-400 (400K video clips of human actions), then "
            "fine-tuned on UCF-Crime shoplifting vs. normal footage.\n\n"
            "Unlike image-based models (ResNet, EfficientNet) that classify "
            "single frames, VideoMAE processes **16 frames at once** and "
            "learns **temporal motion patterns** -- how hands move toward "
            "pockets, how items disappear, how body posture shifts. "
            "This is why it reaches 97%+ accuracy where single-frame "
            "models plateau around 70-80%.\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            "| Detection rate (recall) | 97.4% |\n"
            "| Precision | 98.9% |\n"
            "| F1 Score | 98.1% |\n"
            "| False positive rate | 0.9% |\n"
            "| Current threshold | {:.2f} |\n\n"
            "**Person detection** uses YOLOv8-nano for real-time bounding "
            "boxes around each person in the frame.".format(threshold)
        )

        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        st.info(f"Running on: **{device_name}**")

    return threshold, frame_stride, window_stride, smooth_window


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Shoplifting Detection", layout="wide")
    st.title("AI Shoplifting Detection System")

    threshold, frame_stride, window_stride, smooth_window = render_sidebar()

    # ---- upload ----
    uploaded = st.file_uploader("Upload surveillance video",
                                type=["mp4", "avi", "mov", "mkv"])
    if uploaded is None:
        st.session_state.pop("results", None)
        st.info("Upload a video file to start detection.")
        return

    # Save upload to temp file (only once per file)
    file_id = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("uploaded_file_id") != file_id:
        suffix = Path(uploaded.name).suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.read())
        tmp.flush()
        st.session_state["video_path"] = tmp.name
        st.session_state["uploaded_file_id"] = file_id
        st.session_state.pop("results", None)

    video_path = st.session_state["video_path"]

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = total_frames / fps if fps else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duration", f"{duration:.1f}s")
    col2.metric("FPS", fps)
    col3.metric("Frames", total_frames)
    col4.metric("Resolution", f"{width}x{height}")

    # ---- process or show results ----
    if "results" not in st.session_state:
        if st.button("Start Detection", type="primary"):
            run_detection(video_path, fps, total_frames,
                          threshold, frame_stride, window_stride, smooth_window)
            st.rerun()
    else:
        show_results()
        if st.button("Re-process video"):
            st.session_state.pop("results", None)
            st.rerun()


if __name__ == "__main__":
    main()
