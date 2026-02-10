import logging
from typing import Dict
import json

from openscenesense_ollama.models import AnalysisPrompts
from openscenesense_ollama.analyzer import OllamaVideoAnalyzer
from openscenesense_ollama.frame_selectors import DynamicFrameSelector

from utils.logger import get_logger
from pytoony import json2toon

logger = get_logger(__name__)


class VideoLLMSummarizer:
    """
    Generates a video-level ISR intelligence summary using:
    - video frames (primary signal)
    - semantic fusion context (TOON compressed)
    """

    # -------------------------------------------------
    # TOON helpers
    # -------------------------------------------------

    @staticmethod
    def fusion_json_to_toon(fusion_context: dict) -> str:
        json_str = json.dumps(fusion_context, separators=(",", ":"))
        return json2toon(json_str)

    @staticmethod
    def escape_for_format(text: str) -> str:
        return text.replace("{", "{{").replace("}", "}}")

    # -------------------------------------------------
    # LLM Summary (Semantic Fusion Driven)
    # -------------------------------------------------

    @staticmethod
    def summarize(
        fusion_context: Dict,
        model: str = "qwen3-vl:30b",
        video_path: str | None = None,
    ) -> str:
        """
        Generates ISR intelligence summary using:
        - semantic fusion context
        - video frames
        """

        if not video_path:
            raise ValueError("video_path must be provided")

        # -------------------------------------------------
        # Convert semantic fusion directly to TOON
        # -------------------------------------------------
        raw_toon = VideoLLMSummarizer.fusion_json_to_toon(fusion_context)
        toon_context = VideoLLMSummarizer.escape_for_format(raw_toon)

        logger.info(f"Semantic TOON Context:\n{toon_context}\n")

        # -------------------------------------------------
        # ISR Prompt
        # -------------------------------------------------
        prompts = AnalysisPrompts(
            frame_analysis=(
                "Analyze this frame as airborne ISR imagery. "
                "Describe visible activity, movement, terrain, infrastructure, "
                "and object appearance."
            ),

            detailed_summary=f"""
            SYSTEM ROLE:
            You are a senior ISR Intelligence Analyst producing an operational
            intelligence summary from airborne surveillance video.

            Write clearly using ISR-report language.
            Use only observable activity.
            Do NOT mention detections, track IDs, JSON, models, or TOON.

            --------------------------------------------------
            SEMANTIC TRACK CONTEXT
            --------------------------------------------------
            {toon_context}

            --------------------------------------------------
            TASK
            --------------------------------------------------

            Primary Activity:
            Describe the timeline of activity using frame analysis.
            Focus on movement, interactions, persistence, and scene dynamics.

            Geospatial Intelligence:
            Use geo_summary information to describe where the activity occurs.

            Analyst Observations:
            Use visibility_summary, motion_summary, behavior_summary, loitering_summary, track_lifecycle_summary, confidence_summary
            position_stability_summary, motion_intensity_summary, velocity_consistency_summary, 
            relative_motion_summary, track_confirmation_summary to identify patterns,
            persistence, or anomalies.

            --------------------------------------------------
            OUTPUT FORMAT (STRICT)
            --------------------------------------------------

            [Concise Summary Title]

            Primary Activity:
            3–5 sentences

            Geospatial Intelligence:
            Location and movement context

            Analyst Observations:
            Behavioral and motion insights
            """,

            brief_summary=(
                "Generate a concise ISR intelligence summary focusing on "
                "primary activity and geospatial context."
            ),
        )

        # -------------------------------------------------
        # Analyzer
        # -------------------------------------------------
        analyzer = OllamaVideoAnalyzer(
            frame_analysis_model=model,
            summary_model=model,
            host="http://localhost:11434",
            request_timeout=1000.0,
            min_frames=10,
            max_frames=30,
            frames_per_minute=12,
            frame_selector=DynamicFrameSelector(),
            prompts=prompts,
            log_level=logging.INFO,
        )

        logger.info(
            f"[OpenSceneSense] Running semantic ISR summary on: {video_path}"
        )

        results = analyzer.analyze_video(video_path)

        return results["summary"]
 