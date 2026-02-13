"""
Simple example script for using the shoplifting detector.

Usage examples:
    # Webcam detection
    python scripts/inference.py
    
    # Video file detection
    python scripts/inference.py --input path/to/video.mp4
    
    # Save output video
    python scripts/inference.py --input input.mp4 --output output.mp4
    
    # Custom threshold
    python scripts/inference.py --input video.mp4 --threshold 0.8
    
    # Batch process without display
    python scripts/inference.py --input video.mp4 --no-display --output output.mp4
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.inference import ShopliftingDetector, process_video, process_webcam

def example_video():
    """Example: Process a video file."""
    print("="*70)
    print("Example: Video File Processing")
    print("="*70)
    
    # Initialize detector
    detector = ShopliftingDetector(
        model_path="models/videomae-shoplifting-best",
        confidence_threshold=0.7
    )
    
    # Process video (replace with your video path)
    video_path = "path/to/your/video.mp4"
    output_path = "output_detection.mp4"
    
    if Path(video_path).exists():
        process_video(
            video_path=video_path,
            detector=detector,
            output_path=output_path,
            display=True
        )
    else:
        print(f"Video file not found: {video_path}")
        print("Please update the video_path variable with a valid video file.")

def example_webcam():
    """Example: Process webcam feed."""
    print("="*70)
    print("Example: Webcam Processing")
    print("="*70)
    
    # Initialize detector
    detector = ShopliftingDetector(
        model_path="models/videomae-shoplifting-best",
        confidence_threshold=0.7
    )
    
    # Process webcam
    process_webcam(detector, camera_id=0)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Shoplifting Detection Examples')
    parser.add_argument('--mode', type=str, choices=['video', 'webcam'], 
                       default='webcam', help='Example mode to run')
    
    args = parser.parse_args()
    
    if args.mode == 'video':
        example_video()
    else:
        example_webcam()
