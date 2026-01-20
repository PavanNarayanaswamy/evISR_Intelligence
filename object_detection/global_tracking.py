import cv2
import supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import torch

class ObjectTracker:
    def __init__(
        self,
        video_path: str,
        output_path: str,
        confidence_threshold: float = 0.4
    ):
        self.video_path = video_path
        self.output_path = output_path
        self.confidence_threshold = confidence_threshold

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device.upper()}")

        self.model = None
        self.cap = None
        self.writer = None
        self.byte_tracker = None
        self.box_annotator = None
        self.label_annotator = None

    # Model Initialization
    # ------------------------------------------------------
    def load_model(self):
        self.model = RFDETRBase()

    # Stream Setup
    # ------------------------------------------------------
    def setup_stream(self):
        self.cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Could not open TS file")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps > 0 else 25

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )

    # Tracking + Annotation Setup
    # ------------------------------------------------------
    def setup_tracking(self):
        self.byte_tracker = sv.ByteTrack()

        self.box_annotator = sv.BoxAnnotator(
            color=sv.ColorPalette.ROBOFLOW,
            thickness=2
        )
        self.label_annotator = sv.LabelAnnotator(text_scale=0.6)

    # Frame Processing
    # ------------------------------------------------------
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = self.model.predict(
            rgb_frame,
            threshold=self.confidence_threshold
        )

        sv_detections = sv.Detections(
            xyxy=detections.xyxy,
            confidence=detections.confidence,
            class_id=detections.class_id
        )

        sv_detections.data["class_name"] = [
            COCO_CLASSES[c] for c in detections.class_id
        ]

        tracked = self.byte_tracker.update_with_detections(sv_detections)

        labels = [
            f"ID {track_id} | {name} {conf:.2f}"
            for track_id, name, conf in zip(
                tracked.tracker_id,
                tracked.data.get("class_name", []),
                tracked.confidence
            )
        ]

        annotated = self.box_annotator.annotate(frame.copy(), tracked)
        annotated = self.label_annotator.annotate(
            annotated, tracked, labels
        )

        return annotated

    # ------------------------------------------------------
    # Main Processing Loop
    # ------------------------------------------------------
    def run(self):
        print("🚀 Starting RTSP stream processing...")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ RTSP stream ended or frame drop")
                break

            annotated_frame = self.process_frame(frame)

            self.writer.write(annotated_frame)
            cv2.imshow("Stream", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cleanup()

    # ------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------
    def cleanup(self):
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()

        cv2.destroyAllWindows()
        print("🎉 RTSP stream processing completed")