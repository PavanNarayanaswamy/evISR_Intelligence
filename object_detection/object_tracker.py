import cv2
import os
import json
import shutil
from datetime import datetime
from typing import Generator, Tuple, Any

from minio import Minio
from rfdetr.util.coco_classes import COCO_CLASSES
from zenml_pipeline.minio_utils import upload_output


class ObjectTracker:
    """
    Object Detection + Tracking Orchestrator.

    DESIGN PRINCIPLES
    -----------------
    - run()      : Core engine (PROD-safe, streaming)
    - JSON       : Single consolidated JSON per run
    - save_mp4() : DEV-only visualization utility

    RULES
    -----
    - output_path is decided at construction time
    - No per-frame JSON files written to disk
    - All frames aggregated into one JSON object
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
        # ----------------------------
        # External identifiers & I/O
        # ----------------------------
        self.clip_id = clip_id
        self.minio = minio_client
        self.output_bucket_detection = output_bucket_detection

        self.video_path = video_path
        self.detector = detector
        self.tracker = tracker
        self.forced_fps = forced_fps

        # JSON output is only enabled if output_path is provided
        self.output_path = output_path
        self.enable_json_output = enable_json_output and output_path is not None

        # ----------------------------
        # Video runtime state
        # ----------------------------
        self.cap = None
        self.fps = None
        self.width = None
        self.height = None

        # ----------------------------
        # Run metadata
        # ----------------------------
        self.run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.run_dir = None

        # ----------------------------
        # Consolidated JSON accumulator
        # ----------------------------
        self._json_data = {
            "video_metadata": {},
            "frames": {},
        }
        self.out_path=None

    # ======================================================
    # Video lifecycle management
    # ======================================================
    def open_video(self) -> None:
        """
        Opens the video stream and initializes metadata.
        Must be called before run().
        """
        self.cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open video stream")

        # Use actual FPS if valid, otherwise fallback
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps >= 5 else self.forced_fps

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Prepare output directory & metadata
        if self.enable_json_output:
            self.run_dir = os.path.join(
                self.output_path, f"run_{self.run_timestamp}"
            )
            os.makedirs(self.run_dir, exist_ok=True)

            self._json_data["video_metadata"] = {
                "video_path": self.video_path,
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
                "run_timestamp": self.run_timestamp,
            }

    def close_video(self) -> None:
        """
        Releases video resources, writes consolidated JSON,
        uploads it to MinIO, and cleans up local files.
        """
        if self.cap:
            self.cap.release()
            self.cap = None

        # Skip JSON logic if disabled
        if not self.enable_json_output or self.run_dir is None:
            return

        now = datetime.now()  # Local time for object path

        # ----------------------------
        # Write consolidated JSON
        # ----------------------------
        json_name = f"{self.clip_id}.json"
        json_path = os.path.join(self.run_dir, json_name)

        with open(json_path, "w") as f:
            json.dump(self._json_data, f, indent=2)

        # ----------------------------
        # Upload to MinIO
        # ----------------------------
        object_name = (
            f"detection/"
            f"{now.strftime('%Y/%m/%d/%H')}/"
            f"{self.clip_id}.json"
        )
        upload_output(
            bucket=self.output_bucket_detection,
            object_name=object_name,
            file_path=json_path,
        )

        # ----------------------------
        # Cleanup local artifacts
        # ----------------------------
        shutil.rmtree(self.output_path, ignore_errors=True)
        return f"minio://{self.output_bucket_detection}/{object_name}"

    # ======================================================
    # Core processing engine 
    # ======================================================
    def run(
        self,
    ) -> Generator[Tuple[Any, Any, int], None, None]:
        """
        Streams video frames through detector + tracker.

        Yields:
            annotated_frame : Frame with drawn annotations
            tracked_objects : Norfair tracked objects
            frame_index     : Zero-based frame index
        """
        if self.cap is None:
            raise RuntimeError("Call open_video() before run()")

        frame_index = 0

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Object detection
                detections = self.detector.detect(frame)

                # Tracking + annotation
                annotated_frame, tracked_objects = (
                    self.tracker.track_and_annotate(frame, detections)
                )

                # Aggregate JSON (no disk writes per frame)
                if self.enable_json_output:
                    self._json_data["frames"][frame_index] = (
                        self._serialize_frame(tracked_objects)
                    )

                yield annotated_frame, tracked_objects, frame_index
                frame_index += 1

        finally:
            # Guaranteed cleanup even if consumer breaks early
            path= self.close_video()
            self.out_path=path

    # ======================================================
    # Serialization helpers
    # ======================================================
    def _serialize_frame(self, tracked_objects):
        """
        Converts tracked objects into a JSON-safe structure
        for a single frame.
        """
        objects = []

        for obj in tracked_objects:
            # Skip dead or invalid tracks
            if not obj.live_points.any():
                continue

            det = obj.last_detection
            bbox = det.data["bbox"]
            class_id = det.data["class_id"]
            class_name = COCO_CLASSES[class_id]
            cx, cy = det.points[0]

            objects.append(
                {
                    "track_id": int(obj.id),
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": float(det.scores[0]),
                    "bbox": [float(v) for v in bbox],
                    "centroid": [float(cx), float(cy)],
                }
            )

        return {"objects": objects}

    # ======================================================
    # DEV-only visualization utility
    # ======================================================
    def save_mp4(self, save_frames: bool = True) -> None:
        """
        Saves an annotated MP4 for debugging / development.
        Not intended for production use.
        """
        if self.run_dir is None:
            raise RuntimeError("output_path not set; DEV mode unavailable")

        if self.cap is None:
            self.open_video()

        mp4_path = os.path.join(self.run_dir, "annotated.mp4")

        writer = cv2.VideoWriter(
            mp4_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height),
        )

        for frame, _, _ in self.run():
            writer.write(frame)

        writer.release()
        return self.out_path
