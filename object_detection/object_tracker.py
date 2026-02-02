import cv2
import os
import json
from datetime import datetime
from typing import Any

from minio import Minio
from rfdetr.util.coco_classes import COCO_CLASSES
from zenml_pipeline.minio_utils import upload_output


class ObjectTracker:
    """
    Object Detection + Tracking Orchestrator.

    Lifecycle:
        start()   -> open video, init metadata
        process() -> detect + track
        write_outputs() -> JSON / MP4
        cleanup() -> release resources
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

    # --------------------------------------------------
    # START
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

        # DEV-only: init MP4 writer BEFORE processing
        if enable_mp4:
            self._init_mp4_writer()


    # --------------------------------------------------
    # PROCESS
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

            # JSON sink
            if self.enable_json_output:
                self._json_data["frames"][frame_index] = \
                    self._serialize_frame(tracked)

            # MP4 sink (DEV only)
            if self._mp4_writer:
                self._mp4_writer.write(annotated)

            frame_index += 1


    # --------------------------------------------------
    # OUTPUTS
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
                f"detection/{now.strftime('%Y/%m/%d/%H')}/{self.clip_id}.json"
            )

            upload_output(
                bucket=self.output_bucket_detection,
                object_name=object_name,
                file_path=json_path,
            )

            json_uri = f"minio://{self.output_bucket_detection}/{object_name}"

        return json_uri


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

            objects.append(
                {
                    "track_id": int(obj.id),
                    "class_id": int(class_id),
                    "class_name": COCO_CLASSES[class_id],
                    "confidence": float(det.scores[0]),
                    "bbox": [float(v) for v in bbox],
                    "centroid": [float(v) for v in det.points[0]],
                }
            )

        return {"objects": objects}
