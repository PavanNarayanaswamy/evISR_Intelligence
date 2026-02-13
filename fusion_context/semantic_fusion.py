from typing import Dict, Any, Optional, List
import statistics
import math
from collections import Counter
import reverse_geocoder as rg
import logging
import json
logger = logging.getLogger(__name__)

class SemanticFusion:

    # --------------------------------------------------
    # Helper: Mean velocity vector
    # --------------------------------------------------
    @staticmethod
    def compute_mean_velocity(observations: List[Dict[str, Any]]) -> Dict[str, float]:
        vxs = [o["absolute_velocity"][0] for o in observations]
        vys = [o["absolute_velocity"][1] for o in observations]

        return {
            "vx": statistics.mean(vxs) if vxs else 0.0,
            "vy": statistics.mean(vys) if vys else 0.0,
        }

    # --------------------------------------------------
    # Helper: Mean speed
    # --------------------------------------------------
    @staticmethod
    def compute_mean_speed(observations: List[Dict[str, Any]]) -> float:
        speeds = [o["absolute_speed"] for o in observations]
        return statistics.mean(speeds) if speeds else 0.0

    # --------------------------------------------------
    # Helper: Acceleration std
    # --------------------------------------------------
    @staticmethod
    def compute_acceleration_std(observations: List[Dict[str, Any]]) -> float:
        speeds = [o["absolute_speed"] for o in observations]

        if len(speeds) < 2:
            return 0.0

        accelerations = [
            speeds[i] - speeds[i - 1]
            for i in range(1, len(speeds))
        ]

        return statistics.pstdev(accelerations) if len(accelerations) > 1 else 0.0

    # --------------------------------------------------
    # Helper: Path linearity
    # --------------------------------------------------
    @staticmethod
    def compute_path_linearity(observations: List[Dict[str, Any]]) -> float:
        if len(observations) < 2:
            return 1.0

        centroids = [o["centroid"] for o in observations]

        # straight-line distance
        x0, y0 = centroids[0]
        xN, yN = centroids[-1]
        straight_dist = math.sqrt((xN - x0) ** 2 + (yN - y0) ** 2)

        # total path distance
        total_dist = 0.0
        for i in range(1, len(centroids)):
            x1, y1 = centroids[i - 1]
            x2, y2 = centroids[i]
            total_dist += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if total_dist == 0:
            return 1.0

        return round(straight_dist / total_dist, 4)

    # --------------------------------------------------
    # Helper: Confidence stats
    # --------------------------------------------------
    @staticmethod
    def compute_confidence_stats(observations: List[Dict[str, Any]]) -> Dict[str, float]:
        confs = [o["confidence"] for o in observations]

        return {
            "mean": round(statistics.mean(confs), 4) if confs else 0.0,
            "min": round(min(confs), 4) if confs else 0.0,
            "max": round(max(confs), 4) if confs else 0.0,
        }

    @staticmethod
    def geo_distance_m(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )

        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    @staticmethod
    def reverse_geocode(latitude: Optional[float], longitude: Optional[float]) -> Optional[str]:
        if latitude is None or longitude is None:
            return None

        try:
            coords = (float(latitude), float(longitude))
            result = rg.search([coords])[0]

            return (
                f"{result['name']}, "
                f"{result['admin1']}, "
                f"{result['admin2']}, "
                f"{result['cc']}"
            )
        except Exception as e:
            logger.warning(f"Offline geolocation lookup failed: {e}")
            return None
        
    @classmethod
    def build_geo_events(
        cls,
        observations: List[Dict[str, Any]],
        movement_threshold_m: float = 300.0,
    ) -> List[Dict[str, Any]]:
        geo_events = []

        last_lat = None
        last_lon = None
        last_location = None

        for obs in observations:
            geo = obs.get("geo")
            if not geo:
                continue

            lat = geo.get("latitude")
            lon = geo.get("longitude")

            if lat is None or lon is None:
                continue

            if last_lat is None:
                location_name = cls.reverse_geocode(lat, lon)

                geo_events.append({
                    "type": "start_location",
                    "frame": obs["frame_index"],
                    "timestamp_sec": obs["time_sec"],
                    "latitude": lat,
                    "longitude": lon,
                    "location_name": location_name
                })

                last_lat = lat
                last_lon = lon
                last_location = location_name
                continue

            dist = cls.geo_distance_m(last_lat, last_lon, lat, lon)

            if dist < movement_threshold_m:
                continue

            location_name = cls.reverse_geocode(lat, lon)

            geo_events.append({
                "type": "location_change",
                "frame": obs["frame_index"],
                "timestamp_sec": obs["time_sec"],
                "latitude": lat,
                "longitude": lon,
                "location_name": location_name
            })

            last_lat = lat
            last_lon = lon
            last_location = location_name

        return geo_events

    # --------------------------------------------------
    # Main Scene Builder
    # --------------------------------------------------
    @classmethod
    def build_semantic_fusion(
        cls,
        raw_fusion_output: Dict[str, Any],
        fps: float,
        clip_id: str,
        time_window_sec: int = 30,
    ) -> Dict[str, Any]:
        
        frame_rate = round(fps)
        scene_id = clip_id
        structured_tracks = []
        object_counter = Counter()

        for track in raw_fusion_output.get("tracks", []):
            track_id = track["track_id"]
            class_label = track.get("label")
            observations = track["observations"]

            if not observations:
                continue

            # ----------------------------------------
            # Confidence (compute FIRST)
            # ----------------------------------------
            confidence_stats = cls.compute_confidence_stats(observations)

            # Drop low-confidence tracks
            if confidence_stats["mean"] < 0.70 or confidence_stats["min"] < 0.50:
                continue

            # Only count valid tracks
            object_counter[class_label] += 1

            # ----------------------------------------
            # Frame bounds
            # ----------------------------------------
            frame_indices = [o["frame_index"] for o in observations]
            start_frame = min(frame_indices)
            end_frame = max(frame_indices)
            duration_sec = round((end_frame - start_frame) / frame_rate, 2)

            # ----------------------------------------
            # Trajectory Summary (start, mid, end)
            # ----------------------------------------
            if len(observations) == 1:
                start_obs = mid_obs = end_obs = observations[0]
            elif len(observations) == 2:
                start_obs = observations[0]
                mid_obs = observations[0]
                end_obs = observations[-1]
            else:
                mid_index = len(observations) // 2
                start_obs = observations[0]
                mid_obs = observations[mid_index]
                end_obs = observations[-1]

            trajectory_summary = {
                "start": {
                    "frame": start_obs["frame_index"],
                    "x": float(start_obs["centroid"][0]),
                    "y": float(start_obs["centroid"][1]),
                },
                "mid": {
                    "frame": mid_obs["frame_index"],
                    "x": float(mid_obs["centroid"][0]),
                    "y": float(mid_obs["centroid"][1]),
                },
                "end": {
                    "frame": end_obs["frame_index"],
                    "x": float(end_obs["centroid"][0]),
                    "y": float(end_obs["centroid"][1]),
                },
            }

            # ----------------------------------------
            # Motion
            # ----------------------------------------
            mean_velocity = cls.compute_mean_velocity(observations)
            mean_speed = cls.compute_mean_speed(observations)
            acceleration_std = cls.compute_acceleration_std(observations)

            # ----------------------------------------
            # Spatial
            # ----------------------------------------
            path_linearity = cls.compute_path_linearity(observations)
            geo_events = cls.build_geo_events(observations)

            # ----------------------------------------
            # Append Valid Track
            # ----------------------------------------
            structured_tracks.append({
                "track_id": track_id,
                "class_label": class_label,

                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_sec": duration_sec,

                "trajectory_summary": trajectory_summary,

                "motion": {
                    "mean_velocity_vector": mean_velocity,
                    "mean_speed_px": round(mean_speed, 4),
                    "acceleration_std": round(acceleration_std, 4),
                },

                "spatial": {
                    "path_linearity": path_linearity
                },

                "confidence": confidence_stats,
                "geo_events": geo_events

            })

        # --------------------------------------------
        # Scene Metrics
        # --------------------------------------------
        scene_metrics = {
            "object_counts": dict(object_counter)
        }
        geo_timeline = []

        for t in structured_tracks:
            for e in t.get("geo_events", []):
                if e.get("location_name"):
                    geo_timeline.append((e["frame"], e["location_name"]))

        geo_timeline.sort(key=lambda x: x[0])

        geo_context = {
            "start_location": geo_timeline[0][1] if geo_timeline else None,
            "end_location": geo_timeline[-1][1] if geo_timeline else None,
            "unique_locations": list({loc for _, loc in geo_timeline})
        }


        # --------------------------------------------
        # Final Structure
        # --------------------------------------------
        return {
            "scene_id": scene_id,
            "time_window_sec": time_window_sec,
            "frame_rate": frame_rate,
            "tracks": structured_tracks,
            "scene_metrics": scene_metrics,
            "geo_context": geo_context,
        } 

class CompactListEncoder(json.JSONEncoder):
    def encode(self, obj):
        if isinstance(obj, list):
            return "[" + ", ".join(self.encode(x) for x in obj) + "]"
        return json.JSONEncoder.encode(self, obj)