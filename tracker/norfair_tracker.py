import numpy as np
import cv2
from norfair import Detection, Tracker
from rfdetr.util.coco_classes import COCO_CLASSES


class NorfairTrackerAnnotator:
    """
    Norfair-based tracking and annotation adapter.

    RESPONSIBILITY
    --------------
    - Converts detector outputs into Norfair-compatible detections
    - Maintains object identities across frames
    - Annotates frames with tracking IDs and class labels

    DESIGN NOTES
    ------------
    - Tracking is centroid-based (single point per object)
    - Distance metric: Euclidean distance between centroids
    - Annotation is intentionally lightweight and optional
    - Compatible with ObjectTracker orchestrator
    """

    def __init__(
        self,
        distance_function: str,
        distance_threshold: int ,
        hit_counter_max: int ,
        initialization_delay: int ,
        
    ):
        self.distance_function=distance_function
        self.distance_threshold=distance_threshold
        self.hit_counter_max=hit_counter_max
        self.initialization_delay=initialization_delay
        """
        Initializes the Norfair tracker with stable defaults.

        Args:
            distance_threshold:
                Maximum allowed distance (in pixels) between detections
                across frames to be considered the same object.

            hit_counter_max:
                Number of consecutive missed frames before a track
                is considered dead.

            initialization_delay:
                Minimum number of consecutive detections required
                before a track is assigned a permanent ID.
        """
        # Norfair Tracker initialization
        # Tracker state persists across frames
        self.tracker = Tracker(
            distance_function=self.distance_function,
            distance_threshold=self.distance_threshold,
            hit_counter_max=self.hit_counter_max,
            initialization_delay=self.initialization_delay,
        )

    def track_and_annotate(self, frame, detections):
        """
        Performs tracking update and draws annotations on the frame.

        CONTRACT
        --------
        - Input  :
            - frame      : OpenCV BGR frame (mutated in-place)
            - detections : supervision.Detections from detector
        - Output :
            - annotated frame
            - list of Norfair tracked objects

        This method is intended to be called once per frame.
        """
        norfair_detections = []

        # --------------------------------------------------
        # Convert detector outputs into Norfair detections
        # --------------------------------------------------
        for box, conf, cls in zip(
            detections.xyxy,
            detections.confidence,
            detections.class_id,
        ):
            x1, y1, x2, y2 = box

            # Compute centroid (used for tracking association)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            norfair_detections.append(
                Detection(
                    # Norfair tracks points, not boxes
                    points=np.array([[cx, cy]]),
                    scores=np.array([conf]),
                    data={
                        # Store metadata for downstream use
                        "bbox": box,
                        "class_id": int(cls),
                    },
                )
            )

        # --------------------------------------------------
        # Update tracker state with current frame detections
        # --------------------------------------------------
        tracked_objects = self.tracker.update(norfair_detections)

        # --------------------------------------------------
        # Draw tracking results (DEV / visualization only)
        # --------------------------------------------------
        for obj in tracked_objects:
            # Skip tracks that are not currently visible
            if not obj.live_points.any():
                continue

            track_id = obj.id
            bbox = obj.last_detection.data["bbox"]
            class_id = obj.last_detection.data["class_id"]

            x1, y1, x2, y2 = map(int, bbox)
            label = f"ID {track_id} | {COCO_CLASSES[class_id]}"

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            # Draw label above bounding box
            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return frame, tracked_objects
