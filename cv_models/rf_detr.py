import cv2

from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import supervision as sv


class RFDetrDetector:
    """
    RF-DETR detector adapter.

    RESPONSIBILITY
    --------------
    - Wraps RF-DETR inference behind a stable `detect()` interface
    - Converts raw model outputs into a supervision.Detections object
    - Acts as a pluggable detection component for tracking pipelines

    DESIGN NOTES
    ------------
    - This class owns *only* detection logic
    - No tracking, no visualization, no persistence
    - Output format is intentionally generic to allow tracker interchangeability
    """

    def __init__(self, confidence_threshold: float):
        """
        Initializes the RF-DETR model and detection threshold.

        Args:
            confidence_threshold:
                Minimum confidence score required for a detection
                to be considered valid.
        """
        self.confidence_threshold = confidence_threshold

        # RF-DETR model initialization
        # Heavy operation, intentionally done once per process
        self.model = RFDETRBase()

    def detect(self, frame):
        """
        Runs object detection on a single video frame.

        CONTRACT
        --------
        - Input  : BGR frame (OpenCV format)
        - Output : supervision.Detections instance
        - This method must be fast and side-effect free

        Args:
            frame:
                Single video frame in BGR color space.

        Returns:
            sv.Detections:
                Bounding boxes, confidence scores, and class IDs
                for all detections above the confidence threshold.
        """
        # Convert frame from OpenCV BGR to RGB
        # RF-DETR expects RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run model inference
        detections = self.model.predict(
            rgb_frame,
            threshold=self.confidence_threshold
        )

        # Normalize model output into supervision's Detections format
        sv_detections = sv.Detections(
            xyxy=detections.xyxy,
            confidence=detections.confidence,
            class_id=detections.class_id,
        )

        # Attach human-readable class names for downstream consumers
        # (visualization, JSON serialization, analytics)
        sv_detections.data["class_name"] = [
            COCO_CLASSES[c] for c in detections.class_id
        ]

        return sv_detections


