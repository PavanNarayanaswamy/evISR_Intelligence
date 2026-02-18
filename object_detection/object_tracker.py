import cv2
import os
import json
import numpy as np
from datetime import datetime
from typing import Any

from minio import Minio
from rfdetr.util.coco_classes import COCO_CLASSES
from zenml_pipeline.minio_utils import upload_output


class ObjectTracker:
    """
    Object Detection + Tracking Orchestrator.
    """

    def __init__(
        self,
        clip_id: str,
        minio_client: Minio,
        video_path: str,
        detector,
        tracker,
        forced_fps: int = 25,
        output_bucket_detection: str | None = None,
        output_path: str | None = None,
        enable_json_output: bool = True,
    ):
        self.clip_id = clip_id
        self.minio = minio_client
        self.video_path = video_path
        self.detector = detector
        self.tracker = tracker
        self.forced_fps = forced_fps
        self.output_bucket_detection = output_bucket_detection
        self.output_path = output_path

        self.enable_json_output = enable_json_output and output_path is not None

        self.cap = None
        self.fps = None
        self.width = None
        self.height = None

        self.run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.run_dir = None

        self._json_data = {
            "video_metadata": {},
            "frames": {},
        }

        self._mp4_writer = None
        self._mp4_path = None

        # -----------------------
        # METRIC STORAGE (ADDED)
        # -----------------------
        self.total_detections = 0
        self.confidences = []

        self.total_tracks = set()
        self.confirmed_tracks = set()
        self.track_lifetimes = {}

        self.total_frames = 0

        self.SHORT_TRACK_THRESHOLD = 5

    # --------------------------------------------------
    def start(self, enable_mp4: bool = False) -> None:
        self.cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open video")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps >= 5 else self.forced_fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.enable_json_output:
            self.run_dir = os.path.join(
                self.output_path, f"run_{self.run_timestamp}"
            )
            os.makedirs(self.run_dir, exist_ok=True)

            self._json_data["video_metadata"] = {
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
                "run_timestamp": self.run_timestamp,
            }

        if enable_mp4:
            self._init_mp4_writer()

    # --------------------------------------------------
    def process(self) -> None:
        frame_index = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            detections = self.detector.detect(frame)
            annotated, tracked = self.tracker.track_and_annotate(
                frame, detections
            )

            # -----------------------
            # METRIC COLLECTION (ADDED)
            # -----------------------
            for conf in detections.confidence:
                self.total_detections += 1
                self.confidences.append(float(conf))

            for obj in tracked:
                track_id = obj.id

                self.total_tracks.add(track_id)

                if getattr(obj, "is_confirmed", False):
                    self.confirmed_tracks.add(track_id)

                if hasattr(obj, "age"):
                    self.track_lifetimes[track_id] = obj.age

            self.total_frames += 1

            # JSON sink
            if self.enable_json_output:
                self._json_data["frames"][frame_index] = \
                    self._serialize_frame(tracked)

            if self._mp4_writer:
                self._mp4_writer.write(annotated)

            frame_index += 1

    # --------------------------------------------------
    def write_outputs(self) -> str | None:
        if not self.enable_json_output:
            return None

        if self.run_dir is None:
            raise RuntimeError(
                "write_outputs() called before start()/process()"
            )

        now = datetime.now()

        json_uri = None

        if self.enable_json_output:
            json_path = os.path.join(
                self.run_dir, f"{self.clip_id}.json"
            )

            with open(json_path, "w") as f:
                json.dump(self._json_data, f, indent=2)

            object_name = (
                f"detection/{now.strftime('%Y/%m/%d/%H')}/detection_{self.clip_id}.json"
            )

            upload_output(
                bucket=self.output_bucket_detection,
                object_name=object_name,
                file_path=json_path,
            )

            json_uri = f"minio://{self.output_bucket_detection}/{object_name}"

        # ==================================================
        # METRICS COMPUTATION (ADDED)
        # ==================================================

        avg_confidence = float(np.mean(self.confidences)) if self.confidences else 0.0
        confidence_std = float(np.std(self.confidences)) if self.confidences else 0.0

        total_tracks = len(self.total_tracks)
        confirmed_tracks = len(self.confirmed_tracks)

        confirmation_ratio = (
            confirmed_tracks / total_tracks if total_tracks else 0.0
        )

        avg_track_age = (
            float(np.mean(list(self.track_lifetimes.values())))
            if self.track_lifetimes else 0.0
        )

        short_tracks = sum(
            1 for age in self.track_lifetimes.values()
            if age < self.SHORT_TRACK_THRESHOLD
        )

        short_track_ratio = (
            short_tracks / total_tracks if total_tracks else 0.0
        )

        normalized_avg_track_age = (
            avg_track_age / self.total_frames if self.total_frames else 0.0
        )

        accuracy_score = (
            0.30 * avg_confidence +
            0.25 * confirmation_ratio +
            0.25 * normalized_avg_track_age +
            0.10 * (1 - confidence_std) +
            0.10 * (1 - short_track_ratio)
        )

        detector_metrics = (
            self.detector.get_metrics()
            if hasattr(self.detector, "get_metrics")
            else {}
        )

        metrics = {
            "run_timestamp": self.run_timestamp,
            **detector_metrics,

            "avg_confidence": avg_confidence,
            "confidence_std": confidence_std,
            "total_detections": self.total_detections,

            "total_tracks": total_tracks,
            "confirmed_tracks": confirmed_tracks,
            "confirmation_ratio": confirmation_ratio,

            "avg_track_age": avg_track_age,
            "short_track_ratio": short_track_ratio,

            "accuracy_score": accuracy_score,
        }

        metrics_filename = f"metrics_{self.run_timestamp}.json"
        metrics_path = os.path.join(self.run_dir, metrics_filename)

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return json_uri, self.fps, metrics


# --------------------------------------------------
#   MP4 WRITER INIT
# --------------------------------------------------
    def _init_mp4_writer(self) -> None:
        if self.output_path is None:
            raise RuntimeError("output_path required for MP4")

        now = datetime.now()
        output_dir = os.path.join(
            self.output_path,
            "detection",
            now.strftime("%Y/%m/%d/%H"),
        )
        os.makedirs(output_dir, exist_ok=True)

        self._mp4_path = os.path.join(output_dir, f"{self.clip_id}.mp4")

        self._mp4_writer = cv2.VideoWriter(
            self._mp4_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height),
        )

    # --------------------------------------------------
    # CLEANUP
    # --------------------------------------------------
    def cleanup(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None

        if self._mp4_writer:
            self._mp4_writer.release()
            self._mp4_writer = None

    # --------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------
    def _serialize_frame(self, tracked_objects: Any) -> dict:
        objects = []

        for obj in tracked_objects:
            if not obj.live_points.any():
                continue

            det = obj.last_detection
            bbox = det.data["bbox"]
            class_id = det.data["class_id"]

            # -------------------------------
            # Time-based metrics
            # -------------------------------
            track_age = int(obj.age) if hasattr(obj, "age") else 0
            dwell_time_sec = (
                float(track_age / self.fps)
                if self.fps and track_age > 0
                else 0.0
            )

            objects.append(
                {
                    "track_id": int(obj.id),
                    "class_id": int(class_id),
                    "class_name": COCO_CLASSES[class_id],
                    "confidence": float(det.scores[0]),

                    # -------------------------------
                    # Spatial
                    # -------------------------------
                    "bbox": [float(v) for v in bbox],
                    "centroid": [float(v) for v in det.points[0]],

                    # -------------------------------
                    # ABSOLUTE (camera-compensated)
                    # -------------------------------
                    "absolute_velocity": [
                        float(obj.absolute_velocity[0]),
                        float(obj.absolute_velocity[1]),
                    ] if hasattr(obj, "absolute_velocity") else [0.0, 0.0],

                    "absolute_speed": float(obj.absolute_speed)
                    if hasattr(obj, "absolute_speed")
                    else 0.0,

                    # -------------------------------
                    # RELATIVE (image-space)
                    # -------------------------------
                    "relative_velocity": [
                        float(obj.relative_velocity[0]),
                        float(obj.relative_velocity[1]),
                    ] if hasattr(obj, "relative_velocity") else [0.0, 0.0],

                    "relative_speed": float(obj.relative_speed)
                    if hasattr(obj, "relative_speed")
                    else 0.0,

                    # -------------------------------
                    # Semantics
                    # -------------------------------
                    "track_age": track_age,
                    "dwell_time_sec": dwell_time_sec,
                    "is_stationary": bool(obj.is_stationary)
                    if hasattr(obj, "is_stationary")
                    else False,
                    "direction": str(obj.direction)
                    if hasattr(obj, "direction")
                    else "UNKNOWN",
                    "is_confirmed": bool(obj.is_confirmed)
                    if hasattr(obj, "is_confirmed")
                    else False,
                }
            )

        return {"objects": objects}
