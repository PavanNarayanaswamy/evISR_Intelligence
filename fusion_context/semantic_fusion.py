# Semantic_fusion.py
from typing import Dict, Any, Optional, List
import reverse_geocoder as rg
import logging
import math
import statistics
import json

logger = logging.getLogger(__name__)


class SemanticFusion:

    # --------------------------------------------------
    # Geo helpers
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Semantic helpers Thresholds and mappings
    # --------------------------------------------------
    @staticmethod
    def speed_level(speed: float) -> str:
        if speed <= 0.2:
            return "stationary"
        elif speed <= 1.5:
            return "low"
        elif speed <= 4.0:
            return "moderate"
        else:
            return "high"

    @staticmethod
    def movement_state(speed_level: str, is_stationary: bool) -> str:
        if is_stationary or speed_level == "stationary":
            return "stationary"
        elif speed_level == "low":
            return "slow moving"
        elif speed_level == "moderate":
            return "moving"
        return "fast moving"

    @staticmethod
    def confidence_level(conf: float) -> str:
        if conf >= 0.75:
            return "high"
        elif conf >= 0.5:
            return "medium"
        return "low"

    @staticmethod
    def visibility_pattern(duration: float) -> str:
        if duration <= 0.3:
            return "brief"
        elif duration <= 1.5:
            return "intermittent"
        return "persistent"

    @staticmethod
    def track_maturity(track_age: int) -> str:
        if track_age < 3:
            return "new"
        elif track_age < 8:
            return "developing"
        return "mature"

    @staticmethod
    def position_stability(centroids: List[List[float]]) -> str:
        xs = [c[0] for c in centroids]
        ys = [c[1] for c in centroids]

        var = statistics.pvariance(xs) + statistics.pvariance(ys)

        if var < 2:
            return "highly stable"
        elif var < 10:
            return "moderately stable"
        return "unstable"

    @staticmethod
    def velocity_consistency(speeds: List[float]) -> str:
        if len(speeds) < 2:
            return "consistent"

        std = statistics.pstdev(speeds)

        if std < 0.5:
            return "consistent"
        elif std < 1.5:
            return "variable"
        return "erratic"
    
    # --------------------------------------------------
    # Main semantic fusion
    # --------------------------------------------------
    @classmethod
    def build_semantic_fusion(
        cls,
        raw_fusion_output: Dict[str, Any],
        movement_threshold_m: float = 300.0,
    ) -> Dict[str, Any]:

        semantic_tracks = []

        for track in raw_fusion_output.get("tracks", []):
            track_id = track["track_id"]
            label = track.get("label")
            observations = track["observations"]

            if not observations:
                continue

            times = [o["time_sec"] for o in observations]
            duration = round(times[-1] - times[0], 2)

            speeds = [o["absolute_speed"] for o in observations if o["absolute_speed"] is not None]
            avg_speed = statistics.mean(speeds) if speeds else 0

            centroids = [o["centroid"] for o in observations]

            confidences = [o["confidence"] for o in observations]
            avg_conf = statistics.mean(confidences)

            is_stationary = all(o.get("is_stationary") for o in observations)

            directions = [o["direction"] for o in observations if o.get("direction")]
            dominant_direction = max(set(directions), key=directions.count) if directions else "UNKNOWN"

            last_obs = observations[-1]

            speed_lvl = cls.speed_level(avg_speed)
            move_state = cls.movement_state(speed_lvl, is_stationary)
            conf_lvl = cls.confidence_level(avg_conf)
            vis_pattern = cls.visibility_pattern(duration)
            maturity = cls.track_maturity(last_obs.get("track_age", 1))
            pos_stability = cls.position_stability(centroids)
            vel_consistency = cls.velocity_consistency(speeds)

            loitering = "observed" if duration > 2 and speed_lvl in ["stationary", "low"] else "not observed"

            # -----------------------------
            # GEO SUMMARY
            # -----------------------------
            geo_points = [o.get("geo") for o in observations if o.get("geo")]

            geo_summary = None
            if geo_points:
                first = geo_points[0]
                last = geo_points[-1]

                first_loc = cls.reverse_geocode(first["latitude"], first["longitude"])
                last_loc = cls.reverse_geocode(last["latitude"], last["longitude"])

                if first_loc and first_loc == last_loc:
                    geo_summary = f"Observed within {first_loc}."
                elif last_loc:
                    geo_summary = f"Movement observed toward {last_loc}."

            # -----------------------------
            # Build track object (Semantic Fusion Output)
            # -----------------------------
            semantic_tracks.append({
                "track_id": track_id,
                "label": label,
                "observations": [
                    {
                        "frame_index": [o["frame_index"] for o in observations],
                        "time_sec": times,

                        "visibility_summary":
                            f"The {label} is persistently visible with {vis_pattern} visibility for {duration} seconds.",

                        "motion_summary":
                            f"The {label} shows {speed_lvl} speed and is classified as {move_state}.",

                        "behavior_summary":
                            f"The {label} is {move_state} with lingering movement in the {dominant_direction} direction.",

                        "loitering_summary":
                            f"Loitering behavior is {loitering} for the {label}.",

                        "confidence_summary":
                            f"The {label} is detected with a {conf_lvl} confidence level (confidence score: {round(avg_conf,2)}).",

                        "track_lifecycle_summary":
                            f"The {label} track is {maturity} and observed for {duration} seconds.",

                        "position_stability_summary":
                            f"The {label} maintains a {pos_stability} position on the screen.",

                        # "screen_region_summary":
                        #     f"The {label} is primarily located in the central region of the frame.",

                        "motion_intensity_summary":
                            f"The {label} exhibits {speed_lvl} motion during the observation period.",

                        "velocity_consistency_summary":
                            f"The {label} shows {vel_consistency} velocity behavior.",

                        "relative_motion_summary":
                            f"The {label} shows minor relative motion with respect to the camera.",

                        "track_confirmation_summary":
                            f"The {label} track is currently {'confirmed' if last_obs.get('is_confirmed') else 'tentative'}.",

                        "geo_summary": geo_summary
                    }
                ]
            })

        return {
            "tracks": semantic_tracks
        }