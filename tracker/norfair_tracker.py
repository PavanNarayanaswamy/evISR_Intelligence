import numpy as np
import cv2

from norfair import Detection, Tracker
from norfair.camera_motion import MotionEstimator
from rfdetr.util.coco_classes import COCO_CLASSES

# ==================================================
# HISTOGRAM EMBEDDING (ReID)
# ==================================================
def get_hist_embedding(cutout):
    hsv = cv2.cvtColor(cutout, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [16, 16],
        [0, 180, 0, 256],
    )

    cv2.normalize(hist, hist)
    return hist


# ==================================================
# ReID DISTANCE FUNCTION
# ==================================================
def embedding_distance(matched_not_init_trackers, unmatched_trackers):
    snd_embedding = unmatched_trackers.last_detection.embedding

    if snd_embedding is None:
        for detection in reversed(unmatched_trackers.past_detections):
            if detection.embedding is not None:
                snd_embedding = detection.embedding
                break
        else:
            return 1

    for detection_fst in matched_not_init_trackers.past_detections:
        if detection_fst.embedding is None:
            continue

        distance = 1 - cv2.compareHist(
            snd_embedding,
            detection_fst.embedding,
            cv2.HISTCMP_CORREL,
        )

        if distance < 0.5:
            return distance

    return 1


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

    Norfair-based tracking adapter with:
    - Relative (image-space) motion
    - Absolute (camera-compensated) motion
    """

    STATIONARY_SPEED_THRESHOLD = 0.3   # px/frame (ABSOLUTE)
    DIRECTION_EPS = 0.3                # noise tolerance

    def __init__(
        self,
        distance_function: str,
        distance_threshold: int,
        hit_counter_max: int,
        initialization_delay: int,
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
        # -------------------------------
        # Norfair tracker WITH ReID enabled
        # -------------------------------
        self.tracker = Tracker(
            distance_function=distance_function,
            distance_threshold=distance_threshold,
            hit_counter_max=hit_counter_max,
            initialization_delay=initialization_delay,
            
            # ReID configuration
            past_detections_length=5,
            reid_distance_function=embedding_distance,
            reid_distance_threshold=0.5,
            reid_hit_counter_max=500,
        )

        self.motion_estimator = MotionEstimator(
            max_points=200,
            min_distance=15,
            block_size=3,
            draw_flow=False,
        )

        # -------------------------------
        # Tracking state
        # -------------------------------
        self.track_age = {}

        # RAW (image-space)
        self.prev_raw_centroids = {}

        # ABSOLUTE (camera-compensated)
        self.prev_abs_centroids = {}

    # --------------------------------------------------
    # Direction helper (uses ABSOLUTE velocity)
    # --------------------------------------------------
    def _get_direction(self, vx: float, vy: float) -> str:
        if abs(vx) < self.DIRECTION_EPS and abs(vy) < self.DIRECTION_EPS:
            return "STATIC"
        if vx > 0 and vy < 0:
            return "RIGHT_UP"
        if vx > 0 and vy > 0:
            return "RIGHT_DOWN"
        if vx < 0 and vy < 0:
            return "LEFT_UP"
        if vx < 0 and vy > 0:
            return "LEFT_DOWN"
        return "UNKNOWN"

    # --------------------------------------------------
    # Main API
    # --------------------------------------------------
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

        # -----------------------------------
        # Estimate camera motion
        # -----------------------------------         
        coord_transform = self.motion_estimator.update(frame)

        # --------------------------------------------------
        # Convert detections → Norfair format (WITH EMBEDDING)
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

            # ReID embedding
            x1i, y1i, x2i, y2i = map(int, box)
            cutout = frame[y1i:y2i, x1i:x2i]

            embedding = None
            if cutout.size > 0:
                embedding = get_hist_embedding(cutout)

            norfair_detections.append(
                Detection(
                    # Norfair tracks points, not boxes
                    points=np.array([[cx, cy]]),
                    scores=np.array([conf]),
                    embedding=embedding,
                    data={
                        # Store metadata for downstream use
                        "bbox": box,
                        "class_id": int(cls),
                        "centroid": (cx, cy),  # RAW centroid
                    },
                )
            )

        # -----------------------------------
        # Update tracker (camera-aware)
        # -----------------------------------
        tracked_objects = self.tracker.update(
            norfair_detections,
            coord_transformations=coord_transform,
        )

        # -----------------------------------
        # Process tracked objects
        # -----------------------------------
        for obj in tracked_objects:
            if not obj.live_points.any():
                continue

            track_id = obj.id
            det = obj.last_detection

            bbox = det.data["bbox"]
            class_id = det.data["class_id"]

            # RAW centroid (image space)
            raw_cx, raw_cy = det.data["centroid"]

            # ABSOLUTE centroid (after camera motion)
            abs_cx, abs_cy = obj.estimate[0]

            # -------------------------------
            # Track age
            # -------------------------------
            self.track_age[track_id] = self.track_age.get(track_id, 0) + 1
            age = self.track_age[track_id]

            # ==================================================
            # RELATIVE (IMAGE-SPACE) MOTION
            # ==================================================
            if track_id in self.prev_raw_centroids:
                prx, pry = self.prev_raw_centroids[track_id]
                rvx = raw_cx - prx
                rvy = raw_cy - pry
                relative_speed = float(np.sqrt(rvx ** 2 + rvy ** 2))
            else:
                rvx, rvy = 0.0, 0.0
                relative_speed = 0.0

            self.prev_raw_centroids[track_id] = (raw_cx, raw_cy)

            # ==================================================
            # ABSOLUTE (CAMERA-COMPENSATED) MOTION
            # ==================================================
            if track_id in self.prev_abs_centroids:
                pax, pay = self.prev_abs_centroids[track_id]
                avx = abs_cx - pax
                avy = abs_cy - pay
                absolute_speed = float(np.sqrt(avx ** 2 + avy ** 2))
            else:
                avx, avy = 0.0, 0.0
                absolute_speed = 0.0

            self.prev_abs_centroids[track_id] = (abs_cx, abs_cy)

            # -------------------------------
            # Semantic attributes (ABSOLUTE)
            # -------------------------------
            is_stationary = absolute_speed < self.STATIONARY_SPEED_THRESHOLD
            direction = self._get_direction(avx, avy)
            is_confirmed = age >= self.initialization_delay

            # -------------------------------
            # Attach everything to object
            # -------------------------------
            obj.relative_velocity = (rvx, rvy)
            obj.relative_speed = relative_speed

            obj.absolute_velocity = (avx, avy)
            obj.absolute_speed = absolute_speed

            obj.age = age
            obj.is_stationary = is_stationary
            obj.direction = direction
            obj.is_confirmed = is_confirmed

            # -------------------------------
            # Visualization (optional)
            # -------------------------------
            x1, y1, x2, y2 = map(int, bbox)
            label = (
                f"ID {track_id} | {COCO_CLASSES[class_id]} | "
                f"ABS {absolute_speed:.2f}px"
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
 