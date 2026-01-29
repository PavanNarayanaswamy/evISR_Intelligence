import json
import math
import subprocess
from typing import Dict, Any, List


class TemporalFusion:
    """
    Per-second, time-anchored fusion of KLV telemetry and object detections.
    """

    # -----------------------------
    # Helpers
    # -----------------------------

    @staticmethod
    def klv_timestamp_to_seconds(ts_micro: str) -> float:
        return int(ts_micro) / 1_000_000

    @staticmethod
    def get_video_duration_sec(video_path: str) -> float:
        """
        Get duration of a video segment using ffprobe.
        """
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
        return float(result.stdout.strip())

    @staticmethod
    def find_detection_frames_with_buffer(
        frames: List[Dict[str, Any]],
        anchor_time: float,
        max_buffer: float = 0.5,
        step: float = 0.1,
        max_frames: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Find up to max_frames detection frames near anchor_time.
        Only frames WITH objects are considered.
        """
        buffer = step

        frames_with_objects = [f for f in frames if f["objects"]]

        while buffer <= max_buffer:
            candidates = [
                f for f in frames_with_objects
                if abs(f["relative_time_sec"] - anchor_time) <= buffer
            ]

            if candidates:
                return sorted(
                    candidates,
                    key=lambda f: abs(f["relative_time_sec"] - anchor_time)
                )[:max_frames]

            buffer += step

        return []

    # -----------------------------
    # Fusion Implementation
    # -----------------------------

    @classmethod
    def fuse_klv_and_detections(
        cls,
        clip_id: str,
        klv_json: Dict[str, Any],
        det_json: Dict[str, Any],
        segment_duration_sec: int
    ) -> Dict[str, Any]:

        # -----------------------------
        # Preprocess KLV packets
        # -----------------------------
        klv_packets = []
        raw_packets = klv_json.get("packets", [])
        if not raw_packets:
            raise ValueError("No KLV packets found")

        t0 = cls.klv_timestamp_to_seconds(
            raw_packets[0]["fields"]["PrecisionTimeStamp"]
        )

        for pkt in raw_packets:
            ts = pkt["fields"].get("PrecisionTimeStamp")
            if not ts:
                continue

            rel_time = cls.klv_timestamp_to_seconds(ts) - t0
            klv_packets.append({
                "packet_index": pkt.get("packet_index"),
                "type": pkt.get("type"),
                "relative_time_sec": round(rel_time, 6),
                "fields": pkt["fields"]  # ALL KLV FIELDS PRESERVED
            })

        # -----------------------------
        # Preprocess detection frames
        # -----------------------------
        fps = det_json["video_metadata"]["fps"]
        frames = []

        for frame_idx_str, frame_data in det_json.get("frames", {}).items():
            frame_idx = int(frame_idx_str)
            time_sec = frame_idx / fps

            frames.append({
                "frame_index": frame_idx,
                "relative_time_sec": round(time_sec, 6),
                "objects": frame_data.get("objects", [])
            })

        # -----------------------------
        # Build per-second anchored fusion
        # -----------------------------
        fusion = []

        for sec in range(segment_duration_sec):
            anchor_time = float(sec)

            # ---- KLV closest to this second
            klv = min(
                klv_packets,
                key=lambda k: abs(k["relative_time_sec"] - anchor_time),
                default=None
            )

            # ---- Detection anchor (KLV time preferred)
            det_anchor = klv["relative_time_sec"] if klv else anchor_time

            nearest_frames = cls.find_detection_frames_with_buffer(
                frames=frames,
                anchor_time=det_anchor,
                max_buffer=0.9,
                step=0.1,
                max_frames=2
            )

            detections = []
            for frame in nearest_frames:
                for obj in frame["objects"]:
                    detections.append({
                        "frame_index": frame["frame_index"],
                        "relative_time_sec": frame["relative_time_sec"],
                        "track_id": obj["track_id"],
                        "class_id": obj["class_id"],
                        "class_name": obj["class_name"],
                        "confidence": obj["confidence"],
                        "bbox": obj["bbox"],
                        "centroid": obj["centroid"]
                    })

            fusion.append({
                "second": sec,
                "anchor_time_sec": anchor_time,
                "klv": klv,
                "detections": detections
            })

        return {
            "clip_id": clip_id,
            "segment_duration_sec": segment_duration_sec,
            "fusion": fusion
        }
