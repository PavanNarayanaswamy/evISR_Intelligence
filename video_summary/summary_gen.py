import logging
from pathlib import Path
from typing import List, Dict

from openscenesense_ollama.models import AnalysisPrompts
from openscenesense_ollama.analyzer import OllamaVideoAnalyzer
from openscenesense_ollama.frame_selectors import DynamicFrameSelector

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

            tuned_entry = {
                "second": entry.get("second"),
                "geo_location": {
                    "relative_time_sec": klv.get("relative_time_sec"),
                    "location": {
                        "latitude": fields.get("FrameCenterLatitude"),
                        "longitude": fields.get("FrameCenterLongitude"),
                        "elevation": fields.get("FrameCenterElevation"),
                    }
                },
                "detections": []
            }

            for det in entry.get("detections", []):
                tuned_entry["detections"].append({
                    "class_name": det.get("class_name"),
                    "track_id": det.get("track_id"),
                    "bbox": det.get("bbox"),
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
        raw_example="""
            Sensor Context (TOON):
            clip_id: example-camera-01_20260101_120000
            segment_duration_sec: 30
            fusion[6]{second, geo_location, detections}:
            0,{"relative_time_sec":0.0,"location":{"latitude":"41.1000°","longitude":"-104.8000°","elevation":"1900m"}},[
                {"class_name":"truck","track_id":1,"bbox":[1100,300,1300,450]},
                {"class_name":"truck","track_id":2,"bbox":[500,280,950,460]}
            ]
            5,{"relative_time_sec":5.0,"location":{"latitude":"41.0997°","longitude":"-104.8003°","elevation":"1901m"}},[
                {"class_name":"truck","track_id":1,"bbox":[1000,340,1200,470]},
                {"class_name":"truck","track_id":2,"bbox":[600,300,1050,480]}
            ]
            10,{"relative_time_sec":10.0,"location":{"latitude":"41.0993°","longitude":"-104.8006°","elevation":"1902m"}},[
                {"class_name":"bus","track_id":2,"bbox":[900,290,1200,500]},
                {"class_name":"car","track_id":5,"bbox":[750,390,960,470]}
            ]
            15,{"relative_time_sec":15.0,"location":{"latitude":"41.0989°","longitude":"-104.8009°","elevation":"1903m"}},[
                {"class_name":"bus","track_id":2,"bbox":[880,270,1210,505]},
                {"class_name":"car","track_id":7,"bbox":[720,310,840,380]}
            ]
            20,{"relative_time_sec":20.0,"location":{"latitude":"41.0985°","longitude":"-104.8012°","elevation":"1904m"}},[
                {"class_name":"car","track_id":7,"bbox":[780,140,870,210]}
            ]
            28,{"relative_time_sec":28.0,"location":{"latitude":"41.0981°","longitude":"-104.8016°","elevation":"1905m"}},[
                {"class_name":"traffic light","track_id":19,"bbox":[1120,430,1250,690]}
            ]"""
        formatted_example = VideoLLMSummarizer.escape_for_format(raw_example)
        # ---- Custom Prompts (ISR aware) ----
        prompts = AnalysisPrompts(
        frame_analysis=(
            "Analyze this frame as ISR imagery. "
            "Identify vehicles, people, terrain, infrastructure, "
            "movement patterns, and any visible anomalies. "
            "Base observations strictly on visual evidence."
        ),

        detailed_summary = f"""
            System Instruction:
            You are a specialist ISR Intelligence Analyst. Your task is to generate a
            high-fidelity intelligence summary for a 30-second video segment.

            You must cross-reference visual evidence with the provided
            TOON (Token-Oriented Object Notation) sensor fusion context and tracking logs.
            Visual confirmation takes precedence over sensor data.

            ====================================================
            ACTUAL SENSOR CONTEXT (TOON) – USE FOR ANALYSIS
            ====================================================
            {toon_context}

            ====================================================
            EXAMPLE (FOR STRUCTURE AND REASONING ONLY — NOT REAL DATA)
            ====================================================
            {formatted_example}

            Example Response:
            [Multi-Vehicle Transit Along Urban Roadway]

            Primary Activity:
            Multiple vehicles, including trucks and a bus, are observed traveling along a paved urban roadway,
            with consistent forward motion over the duration of the segment.

            Geospatial Intelligence:
            Observed activity progresses southward relative to the sensor platform, consistent with gradual
            changes in latitude and a stable sensor heading.

            Analyst Observations:
            Truck track IDs 1 and 2 persist across early frames, later transitioning to a bus-dominated scene.
            A stationary traffic light appears in the latter portion of the segment.

            ====================================================
            ANALYSIS TASK (USE ACTUAL TOON CONTEXT ABOVE)
            ====================================================
            Tasks:
            1. Correlate:
            - Identify visually observed objects and confirm whether they align
                with TOON track IDs, classes, and temporal windows.
            - Do NOT assume TOON tracks are valid unless visually confirmed.

            2. Geospatial Summary:
            - Describe observed activity relative to sensor latitude, longitude,
                altitude, and heading when available.

            3. Narrative Intelligence Summary:
            - Identify the primary target activity.
            - Describe relevant environmental context (terrain, road types,
                infrastructure, urban/rural setting).
            - Call out any visual anomalies, ambiguities, or activities NOT
                represented in the TOON tracking data.

            Constraints:
            - Visual evidence is authoritative.
            - Explicitly state uncertainty where confirmation is not possible.
            - Avoid speculative language.

            Response Format (STRICT):
            [Concise Summary Title]

            Primary Activity:
            (1–2 concise sentences)

            Geospatial Intelligence:
            (Spatial reasoning grounded in coordinates and heading)

            Analyst Observations:
            (Additional ISR-relevant details, anomalies, or caveats)
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
            min_frames=3,
            max_frames=12,
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
