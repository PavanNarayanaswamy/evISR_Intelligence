import logging
from pathlib import Path
from typing import List, Dict

from openscenesense_ollama.models import AnalysisPrompts
from openscenesense_ollama.analyzer import OllamaVideoAnalyzer
from openscenesense_ollama.frame_selectors import DynamicFrameSelector
import reverse_geocoder as rg


from utils.logger import get_logger
from pytoony import json2toon
import json

logger = get_logger(__name__)


class VideoLLMSummarizer:
    """
    Generates a video-level ISR intelligence summary using:
    - direct video input
    - OpenSceneSense Ollama pipeline
    - fusion-context (TOON compressed)
    """

    
    @staticmethod
    def fusion_json_to_toon(fusion_context: dict) -> str:
        json_str = json.dumps(fusion_context, separators=(",", ":"))
        return json2toon(json_str)

    @staticmethod
    def tune_fusion_context_for_llm(fusion_context: Dict) -> Dict:

        tuned = {
            "clip_id": fusion_context.get("clip_id"),
            "segment_duration_sec": fusion_context.get("segment_duration_sec"),
            "fusion": []
        }

        for entry in fusion_context.get("fusion", []):
            klv = entry.get("klv", {})
            fields = klv.get("fields", {}) if klv else {}

            latitude = fields.get("SensorLatitude")
            longitude = fields.get("SensorLongitude")

            if latitude:
                latitude = str(latitude).replace("°", "").strip()

            if longitude:
                longitude = str(longitude).replace("°", "").strip()

            resolved_location = "Unknown location"

            # ---- Offline reverse geocoding ----
            if latitude and longitude:
                try:
                    coords = (float(latitude), float(longitude))
                    result = rg.search([coords])[0]

                    resolved_location = (
                        f"{result['name']}, "
                        f"{result['admin1']}, "
                        f"{result['admin2']}, "
                        f"{result['cc']}"
                    )
                except Exception as e:
                    logger.warning(f"Offline geolocation lookup failed: {e}")

            tuned_entry = {
                "second": entry.get("second"),
                "geo_location": {
                    "relative_time_sec": klv.get("relative_time_sec"),
                    "location": resolved_location,
                    "latitude": latitude,
                    "longitude": longitude,
                },
                "detections": []
            }

            for det in entry.get("detections", []):
                tuned_entry["detections"].append({
                    "class_name": det.get("class_name"),
                    "track_id": det.get("track_id"),
                    "bbox": det.get("bbox"),
                    "centroid": det.get("centroid"),
                    "absolute_velocity": det.get("absolute_velocity", [0.0, 0.0]),
                    "absolute_speed": det.get("absolute_speed", 0.0),
                    "relative_velocity": det.get("relative_velocity", [0.0, 0.0]),
                    "relative_speed": det.get("relative_speed", 0.0),
                    "track_age": det.get("track_age", 0),
                    "is_stationary": det.get("is_stationary", False),
                    "direction": det.get("direction", "UNKNOWN"),
                    "is_confirmed": det.get("is_confirmed", False), 
                    "dwell_time_sec": det.get("dwell_time_sec", 0.0),
                })

            tuned["fusion"].append(tuned_entry)

        return tuned

    @staticmethod
    def escape_for_format(text: str) -> str:
        """Escape braces so Python .format() won't break"""
        return text.replace("{", "{{").replace("}", "}}")


    # -------------------------------------------------
    # LLM Summary (Video-native)
    # -------------------------------------------------
    @staticmethod
    def summarize(
        fusion_context: Dict,
        model: str = "qwen3-vl:30b",
        video_path: str | None = None
    ) -> str:
        """
        Generates ISR intelligence summary directly from video
        using OpenSceneSense Ollama + Qwen3-VL.
        """

        if not video_path:
            raise ValueError("video_path must be provided for video-based analysis")

        # ---- TOON Context ----
        tuned_fusion = VideoLLMSummarizer.tune_fusion_context_for_llm(fusion_context)
        raw_toon = VideoLLMSummarizer.fusion_json_to_toon(tuned_fusion)
        toon_context = VideoLLMSummarizer.escape_for_format(raw_toon)
        # ---- Custom Prompts (ISR aware) ----
        prompts = AnalysisPrompts(
        frame_analysis=(
                "Analyze this frame as airborne ISR imagery. "
                "Describe observable activity in natural language, focusing on behavior, "
                "movement, terrain, infrastructure, and spatial relationships. "
                "Note any visible changes, transitions, or anomalies. "
                "Base all observations strictly on visual evidence."
            ),
        detailed_summary = f"""
            System Instruction:
            You are a senior ISR Intelligence Analyst producing an operational intelligence
            summary from airborne video surveillance.

            Write in clear, human ISR-report style.
            Do NOT mention data structures, IDs, bounding boxes, or model artifacts.
            Visual evidence is authoritative; sensor/TOON context may support reasoning
            but must not be quoted or exposed.

            ====================================================
            ACTUAL SENSOR CONTEXT (TOON) – FOR ANALYST USE ONLY
            ====================================================
            {toon_context}

            ====================================================
            ANALYSIS TASK (USE ACTUAL CONTEXT ABOVE)
            ====================================================

            Primary Activity:
            - Provide a clear, time-aware narrative of what happens across the segment.
            - Describe actions, movement, and changes; avoid object enumeration.

            Geospatial Intelligence:
            - Use the resolved location field from the sensor context as the primary geographic reference.
            - If the resolved location is generic (for example only country, state, or city),
              use the latitude and longitude to infer a more precise place such as:
              road name, neighborhood, industrial area, rural zone, highway segment,
              landmark proximity, or terrain-based description.
            - If precision is still uncertain, describe the environment type
              (urban residential area, highway corridor, farmland, industrial zone, etc.).
            - Avoid raw coordinate dumps unless necessary.

            Analyst Observations:
            - Add ISR-relevant context (persistence, environment, infrastructure).
            - Use TOON internally if helpful, but do NOT mention tracks, IDs, or detections.
            - Explicitly state uncertainty where confirmation is limited.

            Constraints:
            - No speculative intent.
            - No technical language.

            ====================================================
            RESPONSE FORMAT (STRICT)
            ====================================================

            [Concise Summary Title]

            Primary Activity:
            (3–5 sentences)

            Geospatial Intelligence:
            (Location and movement)

            Analyst Observations:
            (Additional ISR insights or caveats)
            """,
            brief_summary=(
                "Generate a concise ISR intelligence summary focusing on the "
                "primary observed activity and key geospatial context."
            )
        )

        # ---- Analyzer ----
        analyzer = OllamaVideoAnalyzer(
            frame_analysis_model=model,
            summary_model=model,
            host="http://localhost:11434",
            request_timeout=1000.0,   # ⬅️ increase to 5 minutes
            min_frames=10,
            max_frames=30,
            frames_per_minute=12,
            frame_selector=DynamicFrameSelector(),
            prompts=prompts,
            log_level=logging.INFO
        )

        logger.info(
            f"[OpenSceneSense] Running ISR video summary on: {video_path}"
        )

        results = analyzer.analyze_video(video_path)

        return results["summary"]
