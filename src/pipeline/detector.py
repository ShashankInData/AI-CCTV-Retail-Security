from typing import List

import numpy as np
from ultralytics import YOLO


class PersonDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> List[dict]:
        """Detect people in a frame using YOLO."""
        results = self.model(frame, verbose=False)[0]

        detections: List[dict] = []
        for box in results.boxes:
            # COCO class 0 is 'person'
            if int(box.cls[0]) == 0 and float(box.conf[0]) >= self.conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(box.conf[0]),
                    }
                )
        return detections


if __name__ == "__main__":
    import cv2

    detector = PersonDetector(conf_threshold=0.6)

    # Test webcam
    cap = cv2.VideoCapture(0)

    print("Testing Person Detector... press q to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        # Draw boxes
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            conf = det["confidence"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Person Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Person Detection Test Complete")
