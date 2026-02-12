import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List
import numpy as np
import supervision as sv
from pipeline.detector import PersonDetector


class PersonTracker:
    def __init__(self) -> None:
        # ByteTrack tracker from supervision
        self.tracker = sv.ByteTrack()

    def update(self, detections: List[dict]) -> sv.Detections:
        """Update tracker with new detections and return tracked objects."""
        if not detections:
            return sv.Detections.empty()

        xyxy = np.array([d["bbox"] for d in detections], dtype=float)
        confidence = np.array([d["confidence"] for d in detections], dtype=float)

        detections_sv = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=np.zeros(len(detections), dtype=int),
        )

        tracked = self.tracker.update_with_detections(detections_sv)
        return tracked


if __name__ == "__main__":
    import cv2

    detector = PersonDetector(conf_threshold=0.6)
    tracker = PersonTracker()

    # Annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    cap = cv2.VideoCapture(0)

    print("Testing tracker... press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracked = tracker.update(detections)

        # Annotate frame
        if len(tracked) > 0:
            labels = [f"ID: {tid}" for tid in tracked.tracker_id]
            annotated = box_annotator.annotate(scene=frame.copy(), detections=tracked)
            annotated = label_annotator.annotate(
                scene=annotated, detections=tracked, labels=labels
            )
        else:
            annotated = frame

        cv2.imshow("Tracking Test", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Tracking test completed!")
