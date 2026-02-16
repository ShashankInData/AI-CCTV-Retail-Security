import torch
import cv2
import numpy as np
from pathlib import Path
import sys
from collections import deque
from datetime import datetime
import argparse
sys.path.append(str(Path(__file__).parent.parent))

from transformers import VideoMAEForVideoClassification
from torchvision import transforms
from PIL import Image


class ShopliftingDetector:
    def __init__(self, model_path="models/videomae-shoplifting-best",
                 clip_length=16,
                 frame_stride=10,    # Fix 1: spacing between sampled frames (matches training: stride=10)
                 window_stride=8,    # Fix 1: run inference every N raw frames
                 confidence_threshold=0.7,
                 smooth_window=3,    # Fix 3: temporal smoothing over N predictions
                 device=None):
        """
        Args:
            clip_length:          frames per clip (16 for VideoMAE)
            frame_stride:         gap between sampled frames within a clip (match training data)
            window_stride:        advance sliding window by this many raw frames between inferences
            confidence_threshold: minimum smoothed prob to trigger alert
            smooth_window:        number of recent predictions to average (temporal smoothing)
        """
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.window_stride = window_stride
        self.confidence_threshold = confidence_threshold
        self.smooth_window = smooth_window
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[Device] Using: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

        print(f"\n[Config] frame_stride={frame_stride}, window_stride={window_stride}, "
              f"smooth_window={smooth_window}, threshold={confidence_threshold}")

        # Load model
        print(f"\n[Model] Loading from {model_path}...")
        self.model = VideoMAEForVideoClassification.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("   Model loaded!")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Fix 2: Sliding window buffer holds clip_length * frame_stride raw frames
        buffer_size = clip_length * frame_stride
        self.frame_buffer = deque(maxlen=buffer_size)
        self.raw_frame_count = 0  # counts frames since last inference

        # Fix 3: Temporal smoothing — queue of recent shoplifting probabilities
        self.recent_probs = deque(maxlen=smooth_window)
        self.last_result = None       # most recent raw model result
        self.last_smoothed_prob = 0.0

        # Stats
        self.total_inferences = 0
        self.shoplifting_detections = 0
        self.alerts = []

    def preprocess_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(frame_rgb))

    def predict_clip(self, frames):
        """Run model on a list of clip_length preprocessed frame tensors."""
        if len(frames) < self.clip_length:
            return None
        frames_tensor = torch.stack(frames).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values=frames_tensor)
            probs = torch.softmax(outputs.logits, dim=1)
            shoplifting_prob = probs[0][1].item()
            normal_prob = probs[0][0].item()
            confidence, predicted = torch.max(probs, 1)
        return {
            'prediction': "shoplifting" if predicted.item() == 1 else "normal",
            'confidence': confidence.item(),
            'shoplifting_prob': shoplifting_prob,
            'normal_prob': normal_prob,
        }

    def _sample_clip_from_buffer(self):
        """Fix 2: sample clip_length frames spaced frame_stride apart from the buffer."""
        buf = list(self.frame_buffer)
        # Take the most recent clip_length * frame_stride frames
        # sample at indices: 0, frame_stride, 2*frame_stride, ...
        indices = [i * self.frame_stride for i in range(self.clip_length)]
        # buf[-buffer_size:] is already in the deque; pick evenly spaced
        return [buf[i] for i in indices]

    def process_frame(self, frame):
        """
        Add frame to buffer. Run inference every window_stride frames once buffer is full.
        Returns dict with raw + smoothed results, or None if not yet due.
        """
        processed = self.preprocess_frame(frame)
        self.frame_buffer.append(processed)
        self.raw_frame_count += 1

        # Buffer must be full before we can sample a properly-strided clip
        buffer_size = self.clip_length * self.frame_stride
        if len(self.frame_buffer) < buffer_size:
            return None

        # Fix 1: only run inference every window_stride frames
        if self.raw_frame_count % self.window_stride != 0:
            return self.last_result  # return last result for display continuity

        # Sample strided clip and predict
        sampled = self._sample_clip_from_buffer()
        result = self.predict_clip(sampled)
        if result is None:
            return None

        self.total_inferences += 1

        # Fix 3: temporal smoothing
        self.recent_probs.append(result['shoplifting_prob'])
        smoothed_prob = float(np.mean(self.recent_probs))
        self.last_smoothed_prob = smoothed_prob

        result['smoothed_prob'] = smoothed_prob
        result['alert'] = smoothed_prob >= self.confidence_threshold
        self.last_result = result

        if result['alert']:
            self.shoplifting_detections += 1
            self.alerts.append({
                'timestamp': datetime.now(),
                'smoothed_prob': smoothed_prob,
                'raw_confidence': result['confidence'],
            })

        return result

    def draw_prediction(self, frame, result):
        if result is None:
            return frame

        pred = result['prediction']
        raw_conf = result['confidence']
        smoothed = result.get('smoothed_prob', result['shoplifting_prob'])
        alert = result.get('alert', False)

        color = (0, 0, 255) if alert else (0, 200, 0)

        # Top bar
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
        cv2.putText(frame, f"Prediction: {pred.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Raw conf: {raw_conf*100:.1f}%  Smoothed: {smoothed*100:.1f}%",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Inferences: {self.total_inferences}  Alerts: {self.shoplifting_detections}",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        if alert:
            cv2.putText(frame, "ALERT: SHOPLIFTING DETECTED!",
                        (frame.shape[1]//2 - 200, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 5)

        return frame


# ─── Batch evaluation on clips.txt (measures detection rate + FP rate) ────────

def evaluate_on_clips(clips_txt, detector, max_clips=None):
    """
    Run inference on pre-extracted frame clips and report metrics.
    Clips.txt format:  label<TAB>frame1.png,frame2.png,...
    """
    from tqdm import tqdm

    clips = []
    with open(clips_txt) as f:
        for line in f:
            parts = line.strip().split("\t")
            label = int(parts[0])
            paths = parts[1].split(",")
            clips.append((label, paths))

    if max_clips:
        clips = clips[:max_clips]

    print(f"\n[Evaluate] {len(clips)} clips from {clips_txt}")
    tp = fp = tn = fn = 0
    all_probs = []

    for label, paths in tqdm(clips, desc="Evaluating clips"):
        # Load clip frames
        frames = []
        for p in paths[:detector.clip_length]:
            img = Image.open(p).convert("RGB")
            frames.append(detector.transform(img))

        if len(frames) < detector.clip_length:
            continue

        result = detector.predict_clip(frames)
        if result is None:
            continue

        prob = result['shoplifting_prob']
        all_probs.append(prob)
        predicted_positive = prob >= detector.confidence_threshold

        if label == 1 and predicted_positive:
            tp += 1
        elif label == 1 and not predicted_positive:
            fn += 1
        elif label == 0 and predicted_positive:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    shoplifting_clips = tp + fn
    normal_clips = fp + tn

    detection_rate = tp / shoplifting_clips if shoplifting_clips > 0 else 0.0
    fp_rate = fp / normal_clips if normal_clips > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * detection_rate / (precision + detection_rate) if (precision + detection_rate) > 0 else 0.0

    # Estimate false alarms per hour (assumes 30fps, clip stride = window_stride)
    # Each inference covers window_stride raw frames at ~30fps
    fps = 30
    inferences_per_hour = (fps * 3600) / detector.window_stride
    fp_per_hour = fp_rate * inferences_per_hour

    print(f"\n{'='*50}")
    print(f"  EVALUATION RESULTS  (threshold={detector.confidence_threshold})")
    print(f"{'='*50}")
    print(f"  Total clips:         {total}")
    print(f"  Shoplifting clips:   {shoplifting_clips}")
    print(f"  Normal clips:        {normal_clips}")
    print(f"")
    print(f"  Detection Rate:      {detection_rate*100:.1f}%   (recall / sensitivity)")
    print(f"  False Positive Rate: {fp_rate*100:.1f}%   (FP / total normal)")
    print(f"  Precision:           {precision*100:.1f}%")
    print(f"  F1 Score:            {f1*100:.1f}%")
    print(f"")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"")
    print(f"  Est. false alarms/hour @ 30fps: {fp_per_hour:.1f}")
    print(f"{'='*50}")

    return {
        'detection_rate': detection_rate,
        'fp_rate': fp_rate,
        'precision': precision,
        'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'fp_per_hour': fp_per_hour,
    }


# ─── Video / webcam processing ────────────────────────────────────────────────

def process_video(video_path, detector, output_path=None, display=True):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] Could not open video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n[Video] {width}x{height} @ {fps}fps  ({total_frames} frames)")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"   Saving to: {output_path}")

    frame_count = 0
    last_result = None
    print("[Processing] Press 'q' to quit, 's' to save frame\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        result = detector.process_frame(frame)
        if result is not None:
            last_result = result

        annotated = detector.draw_prediction(frame.copy(), last_result)

        if display:
            cv2.imshow('Shoplifting Detection', annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f"alert_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(save_path, annotated)
                print(f"   Saved: {save_path}")

        if writer:
            writer.write(annotated)

        if frame_count % 100 == 0:
            pct = (frame_count / total_frames) * 100 if total_frames else 0
            print(f"   {pct:.1f}%  frame={frame_count}  inferences={detector.total_inferences}"
                  f"  alerts={detector.shoplifting_detections}")

    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    duration_sec = frame_count / fps
    fp_per_hour = (detector.shoplifting_detections / duration_sec * 3600) if duration_sec > 0 else 0

    print(f"\n[Summary]")
    print(f"   Frames:          {frame_count}")
    print(f"   Inferences:      {detector.total_inferences}")
    print(f"   Alerts:          {detector.shoplifting_detections}")
    print(f"   Duration:        {duration_sec:.1f}s")
    print(f"   Est alerts/hour: {fp_per_hour:.1f}  (on this video)")


def process_webcam(detector, camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Error] Could not open camera {camera_id}")
        return

    print("[Webcam] Starting... Press 'q' to quit")
    last_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.process_frame(frame)
        if result is not None:
            last_result = result

        annotated = detector.draw_prediction(frame.copy(), last_result)
        cv2.imshow('Shoplifting Detection - Webcam', annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[Summary] Inferences: {detector.total_inferences}  Alerts: {detector.shoplifting_detections}")


def main():
    parser = argparse.ArgumentParser(description='Shoplifting Detection Inference')
    parser.add_argument('--input', type=str, default=None,
                        help='Video file path (default: webcam)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save annotated video to path')
    parser.add_argument('--model', type=str, default='models/videomae-shoplifting-best',
                        help='Trained model path')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Smoothed probability threshold for alerts (default: 0.7)')
    parser.add_argument('--frame-stride', type=int, default=10,
                        help='Frames between sampled frames in clip (default: 10, matches training)')
    parser.add_argument('--window-stride', type=int, default=8,
                        help='Run inference every N raw frames (default: 8)')
    parser.add_argument('--smooth-window', type=int, default=3,
                        help='Temporal smoothing: average over N predictions (default: 3)')
    parser.add_argument('--evaluate', type=str, default=None,
                        help='Path to clips.txt to evaluate (e.g. data/processed/frame_classification/test/clips.txt)')
    parser.add_argument('--max-clips', type=int, default=None,
                        help='Max clips for evaluation (default: all)')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable real-time display')
    parser.add_argument('--camera', type=int, default=0,
                        help='Webcam ID (default: 0)')

    args = parser.parse_args()

    detector = ShopliftingDetector(
        model_path=args.model,
        frame_stride=args.frame_stride,
        window_stride=args.window_stride,
        confidence_threshold=args.threshold,
        smooth_window=args.smooth_window,
    )

    if args.evaluate:
        evaluate_on_clips(args.evaluate, detector, max_clips=args.max_clips)
    elif args.input:
        process_video(args.input, detector,
                      output_path=args.output,
                      display=not args.no_display)
    else:
        process_webcam(detector, camera_id=args.camera)


if __name__ == "__main__":
    main()
