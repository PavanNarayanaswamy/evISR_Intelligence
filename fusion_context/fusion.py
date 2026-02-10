from typing import Dict, Any


class TemporalFusion:
    """
    Track-centric fusion of KLV telemetry and object detections.
    Each track contains time-ordered observations with nearest KLV geo data.
    """

    # -----------------------------
    # Helpers
    # -----------------------------

    @staticmethod
    def klv_timestamp_to_seconds(ts_micro: str) -> float:
        return int(ts_micro) / 1_000_000

    @staticmethod
    def parse_float(value: str):
        """Extract float from values like '41.0933°' or '2932.7m'"""
        if value is None:
            return None
        return float(
            value.replace("°", "").replace("m", "")
        )

    # -----------------------------
    # Fusion Implementation
    # -----------------------------

    @classmethod
    def fuse_klv_and_detections(
        cls,
        klv_json: Dict[str, Any],
        det_json: Dict[str, Any],
        klv_time_window: float = 0.5,
    ) -> Dict[str, Any]:

        # -----------------------------
        # Preprocess KLV packets
        # -----------------------------
        raw_packets = klv_json.get("packets", [])
        if not raw_packets:
            raise ValueError("No KLV packets found")

        t0 = cls.klv_timestamp_to_seconds(
            raw_packets[0]["fields"]["PrecisionTimeStamp"]
        )

        klv_packets = []
        for pkt in raw_packets:
            ts = pkt["fields"].get("PrecisionTimeStamp")
            if not ts:
                continue

            rel_time = cls.klv_timestamp_to_seconds(ts) - t0
            fields = pkt["fields"]

            klv_packets.append({
                "time_sec": rel_time,
                "geo": {
                    "latitude": cls.parse_float(fields.get("SensorLatitude")),
                    "longitude": cls.parse_float(fields.get("SensorLongitude")),
                    "altitude": cls.parse_float(fields.get("SensorTrueAltitude")),
                },
            })

        # -----------------------------
        # Preprocess detections into tracks
        # -----------------------------
        fps = det_json["video_metadata"]["fps"]
        frames = det_json.get("frames", {})

        tracks: Dict[int, Dict[str, Any]] = {}

        for frame_idx_str, frame_data in frames.items():
            frame_idx = int(frame_idx_str)
            time_sec = frame_idx / fps

            for obj in frame_data.get("objects", []):
                track_id = obj["track_id"]

                if track_id not in tracks:
                    tracks[track_id] = {
                        "track_id": track_id,
                        "label": obj["class_name"],
                        "observations": []
                    }

                # ---- Find nearest KLV packet
                nearest_klv = min(
                    klv_packets,
                    key=lambda k: abs(k["time_sec"] - time_sec),
                    default=None
                )

                geo = None
                if nearest_klv and abs(nearest_klv["time_sec"] - time_sec) <= klv_time_window:
                    geo = nearest_klv["geo"]

                tracks[track_id]["observations"].append({
                    "frame_index": frame_idx,
                    "time_sec": round(time_sec, 6),
                    "bbox": obj["bbox"],
                    "confidence": obj["confidence"],
                    "centroid": obj["centroid"],
                    "absolute_velocity": obj.get("absolute_velocity", None),
                    "absolute_speed": obj.get("absolute_speed", None),
                    "relative_velocity": obj.get("relative_velocity", None),
                    "relative_speed": obj.get("relative_speed", None),
                    "track_age": obj.get("track_age", None),
                    "dwell_time_sec": obj.get("dwell_time_sec", None),
                    "is_stationary": obj.get("is_stationary", None),
                    "direction": obj.get("direction", None),
                    "is_confirmed": obj.get("is_confirmed", None),
                    "geo": geo
                })

        for track in tracks.values():
            track["observations"].sort(key=lambda o: o["frame_index"])

        # -----------------------------
        # NEW: Sort tracks by track_id
        # -----------------------------
        sorted_tracks = sorted(
            tracks.values(),
            key=lambda t: t["track_id"]
        )

        return {
            "tracks": sorted_tracks
        }