from typing import Dict, Any, Optional
import reverse_geocoder as rg
import logging
import math

logger = logging.getLogger(__name__)


class SemanticFusion:

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

    @classmethod
    def build_semantic_geo(
        cls,
        raw_fusion_output: Dict[str, Any],
        movement_threshold_m: float = 300.0,
    ) -> Dict[str, Any]:

        semantic_tracks = {}

        for track in raw_fusion_output.get("tracks", []):
            track_id = track["track_id"]
            label = track.get("label")
            observations = track.get("observations", [])

            last_lat = None
            last_lon = None
            last_location = None

            first_location = None
            last_known_location = None

            location_summary_parts = []

            for obs in observations:
                geo = obs.get("geo")
                if not geo:
                    continue

                latitude = geo.get("latitude")
                longitude = geo.get("longitude")

                if latitude is None or longitude is None:
                    continue

                if last_lat is None:
                    location_name = cls.reverse_geocode(latitude, longitude)
                    if not location_name:
                        continue

                    first_location = location_name
                    last_location = location_name
                    last_known_location = location_name
                    last_lat = latitude
                    last_lon = longitude

                    location_summary_parts.append(
                        f"Started at {location_name}"
                    )
                    continue

                dist = cls.geo_distance_m(last_lat, last_lon, latitude, longitude)

                if dist < movement_threshold_m:
                    continue

                location_name = cls.reverse_geocode(latitude, longitude)
                if not location_name:
                    continue

                last_lat = latitude
                last_lon = longitude
                last_known_location = location_name

                if location_name != last_location:
                    location_summary_parts.append(
                        f"Location changed to {location_name} "
                        f"at frame {obs['frame_index']}"
                    )
                    last_location = location_name

            # -----------------------------
            # UPDATED SEMANTIC SUMMARY LOGIC
            # -----------------------------
            if first_location and first_location == last_known_location:
                geo_summary = f"Observed within {first_location}."
            else:
                if last_known_location:
                    location_summary_parts.append(
                        f"Ended near {last_known_location}"
                    )

                geo_summary = (
                    ". ".join(location_summary_parts) + "."
                    if location_summary_parts else None
                )

            semantic_tracks[track_id] = {
                "label": label,
                "geo_location": geo_summary
            }

        return {
            "tracks": semantic_tracks
        }
