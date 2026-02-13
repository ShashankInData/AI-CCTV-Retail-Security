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
                 clip_length=16, confidence_threshold=0.7, device=None):
        """
        Initialize the shoplifting detector.
        
        Args:
            model_path: Path to trained model
            clip_length: Number of frames per clip (16 for VideoMAE)
            confidence_threshold: Minimum confidence to trigger alert
            device: torch device (auto-detected if None)
        """
        self.clip_length = clip_length
        self.confidence_threshold = confidence_threshold
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[Device] Using: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # Load model
        print(f"\n[Model] Loading model from {model_path}...")
        self.model = VideoMAEForVideoClassification.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("   Model loaded successfully!")
        
        # Transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Frame buffer for sliding window
        self.frame_buffer = deque(maxlen=clip_length)
        
        # Statistics
        self.total_clips = 0
        self.shoplifting_detections = 0
        self.alerts = []
    
    def preprocess_frame(self, frame):
        """Convert OpenCV frame to PIL Image and apply transforms."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        # Apply transforms
        return self.transform(pil_image)
    
    def predict_clip(self, frames):
        """
        Predict shoplifting on a clip of frames.
        
        Args:
            frames: List of preprocessed frame tensors (length = clip_length)
        
        Returns:
            dict with 'prediction', 'confidence', 'probabilities'
        """
        if len(frames) < self.clip_length:
            return None
        
        # Stack frames: (clip_length, C, H, W)
        frames_tensor = torch.stack(frames)
        # Add batch dimension: (1, clip_length, C, H, W)
        frames_tensor = frames_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(pixel_values=frames_tensor)
            probs = torch.softmax(outputs.logits, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            prediction = "shoplifting" if predicted.item() == 1 else "normal"
            confidence = confidence.item()
            shoplifting_prob = probs[0][1].item()
            normal_prob = probs[0][0].item()
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'shoplifting_prob': shoplifting_prob,
            'normal_prob': normal_prob
        }
    
    def process_frame(self, frame):
        """
        Process a single frame and return prediction if clip is ready.
        
        Args:
            frame: OpenCV frame (BGR format)
        
        Returns:
            dict with prediction info or None if clip not ready
        """
        # Preprocess and add to buffer
        processed_frame = self.preprocess_frame(frame)
        self.frame_buffer.append(processed_frame)
        
        # Only predict when buffer is full
        if len(self.frame_buffer) == self.clip_length:
            self.total_clips += 1
            result = self.predict_clip(list(self.frame_buffer))
            
            if result and result['prediction'] == 'shoplifting' and result['confidence'] >= self.confidence_threshold:
                self.shoplifting_detections += 1
                self.alerts.append({
                    'timestamp': datetime.now(),
                    'confidence': result['confidence'],
                    'shoplifting_prob': result['shoplifting_prob']
                })
            
            return result
        
        return None
    
    def draw_prediction(self, frame, prediction_result):
        """
        Draw prediction results on frame.
        
        Args:
            frame: OpenCV frame
            prediction_result: dict from predict_clip()
        
        Returns:
            Annotated frame
        """
        if prediction_result is None:
            return frame
        
        # Status text
        pred = prediction_result['prediction']
        conf = prediction_result['confidence']
        shop_prob = prediction_result['shoplifting_prob']
        
        # Color: red for shoplifting, green for normal
        color = (0, 0, 255) if pred == 'shoplifting' else (0, 255, 0)
        text_color = (255, 255, 255)
        
        # Draw status bar at top
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        
        # Prediction text
        pred_text = f"Prediction: {pred.upper()}"
        cv2.putText(frame, pred_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Confidence text
        conf_text = f"Confidence: {conf*100:.1f}%"
        cv2.putText(frame, conf_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Probability breakdown
        prob_text = f"Shoplifting: {shop_prob*100:.1f}% | Normal: {prediction_result['normal_prob']*100:.1f}%"
        cv2.putText(frame, prob_text, (frame.shape[1] - 400, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # Alert indicator
        if pred == 'shoplifting' and conf >= self.confidence_threshold:
            alert_text = "ALERT: SHOPLIFTING DETECTED!"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            cv2.putText(frame, alert_text, (text_x, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            # Draw red border
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 5)
        
        # Statistics
        stats_text = f"Clips processed: {self.total_clips} | Alerts: {self.shoplifting_detections}"
        cv2.putText(frame, stats_text, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        return frame

def process_video(video_path, detector, output_path=None, display=True):
    """
    Process a video file.
    
    Args:
        video_path: Path to input video file
        detector: ShopliftingDetector instance
        output_path: Optional path to save output video
        display: Whether to display video in real-time
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"[Error] Could not open video: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n[Video Info]")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    
    # Video writer if saving
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"   Saving to: {output_path}")
    
    frame_count = 0
    print(f"\n[Processing] Press 'q' to quit, 's' to save current frame")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame
        prediction = detector.process_frame(frame)
        
        # Draw prediction
        annotated_frame = detector.draw_prediction(frame, prediction)
        
        # Display
        if display:
            cv2.imshow('Shoplifting Detection', annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f"alert_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(save_path, annotated_frame)
                print(f"   Frame saved to: {save_path}")
        
        # Write to output video
        if writer:
            writer.write(annotated_frame)
        
        # Progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()
    
    print(f"\n[Summary]")
    print(f"   Frames processed: {frame_count}")
    print(f"   Clips analyzed: {detector.total_clips}")
    print(f"   Alerts triggered: {detector.shoplifting_detections}")
    if detector.total_clips > 0:
        alert_rate = (detector.shoplifting_detections / detector.total_clips) * 100
        print(f"   Alert rate: {alert_rate:.2f}%")

def process_webcam(detector, camera_id=0):
    """Process webcam feed."""
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"[Error] Could not open camera {camera_id}")
        return
    
    print(f"\n[Webcam] Starting detection... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        prediction = detector.process_frame(frame)
        
        # Draw prediction
        annotated_frame = detector.draw_prediction(frame, prediction)
        
        # Display
        cv2.imshow('Shoplifting Detection - Webcam', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n[Summary]")
    print(f"   Clips analyzed: {detector.total_clips}")
    print(f"   Alerts triggered: {detector.shoplifting_detections}")

def main():
    parser = argparse.ArgumentParser(description='Shoplifting Detection Inference')
    parser.add_argument('--input', type=str, default=None,
                       help='Path to input video file (default: webcam)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output video')
    parser.add_argument('--model', type=str, default='models/videomae-shoplifting-best',
                       help='Path to trained model')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Confidence threshold for alerts (default: 0.7)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display (useful for batch processing)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID for webcam (default: 0)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ShopliftingDetector(
        model_path=args.model,
        confidence_threshold=args.threshold
    )
    
    # Process input
    if args.input:
        # Process video file
        process_video(
            video_path=args.input,
            detector=detector,
            output_path=args.output,
            display=not args.no_display
        )
    else:
        # Process webcam
        process_webcam(detector, camera_id=args.camera)

if __name__ == "__main__":
    main()
