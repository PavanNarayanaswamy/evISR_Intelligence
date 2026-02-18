import cv2
import time
import torch
import supervision as sv

from rfdetr import RFDETRMedium
from rfdetr.util.coco_classes import COCO_CLASSES


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

        self.total_inference_time = 0
        self.total_frames = 0

        self.model_name = "rfdetr_medium"

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self.model = RFDETRMedium()

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------
    def get_metrics(self):
        if self.total_frames == 0:
            return {}

        avg_latency = self.total_inference_time / self.total_frames

        metrics = {
            "model_name": self.model_name,
            "avg_inference_latency_ms": avg_latency * 1000,
            "detector_fps": 1 / avg_latency if avg_latency > 0 else 0,
            "detector_total_frames": self.total_frames,
        }

        if torch.cuda.is_available():
            metrics["gpu_peak_memory_mb"] = (
                torch.cuda.max_memory_allocated() / 1024 ** 2
            )

        return metrics

    # --------------------------------------------------
    # DETECT
    # --------------------------------------------------
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
        start = time.perf_counter()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run model inference
        detections = self.model.predict(
            rgb_frame,
            threshold=self.confidence_threshold
        )

        latency = time.perf_counter() - start
        self.total_inference_time += latency
        self.total_frames += 1

        sv_detections = sv.Detections(
            xyxy=detections.xyxy,
            confidence=detections.confidence,
            class_id=detections.class_id,
        )

        sv_detections.data["class_name"] = [
            COCO_CLASSES[c] for c in detections.class_id
        ]

        return sv_detections
 