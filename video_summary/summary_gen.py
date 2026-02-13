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
    # LLM Summary
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
        # Convert semantic fusion → TOON
        # -------------------------------------------------
        raw_toon = VideoLLMSummarizer.fusion_json_to_toon(fusion_context)
        toon_context = VideoLLMSummarizer.escape_for_format(raw_toon)

        logger.info(f"Semantic TOON Context:\n{toon_context}\n")

        # -------------------------------------------------
        # Prompt definition
        # NOTE:
        # {timeline} and {transcript} are injected later
        # by OpenSceneSense analyzer
        # -------------------------------------------------
        detailed_prompt_template = """
        SYSTEM ROLE:
        You are a senior ISR Intelligence Analyst tasked with producing a clear, Precise and concise operational
        intelligence summary from surveillance video.

        Use professional ISR-report language. Base conclusions only on Frame timeline and semantic scene context.
        Avoid mentioning technical details such as detections, track IDs, JSON, TOON, frame numbers,
        timestamps, coordinates, velocities, pixel values, or raw numeric data.

        --------------------------------------------------
        SEMANTIC SCENE CONTEXT
        --------------------------------------------------
        {toon_context}

        --------------------------------------------------
        FRAME TIMELINE
        --------------------------------------------------
        {{timeline}}

        --------------------------------------------------
        TASK
        --------------------------------------------------

        1. **Primary Activity:**
           - Use FRAME TIMELINE to identify and summarize the main activities occurring in the video.
           - Summarize the main activities observed in the scene.
           - Focus on movement patterns, object interactions, direction of travel,
             entry and exit behavior, and flow consistency.

        2. **Geospatial Intelligence:**
           - Describe the locations where activities occur using geo_events and geo_context information provided in semantic scene context.
           - Indicate whether the activity is localized or spans multiple locations.
           - Decode location using latitude/longitude only if exact location names are not provided in semantic scene context.

        3. **Analyst Observations:**
           - Use both FRAME TIMELINE and SEMANTIC SCENE CONTEXT to highlight key observations, anomalies, and patterns.
           - Highlight motion consistency, persistence of objects, relative movement, and anomalies.
           - Identify any unusual or unexpected behaviors.
           - state anomalies if they are present, do not hallucinate anomalies if they are not present.

        --------------------------------------------------
        OUTPUT FORMAT (STRICT)
        --------------------------------------------------

        [Operational Activity Title]

        Primary Activity:
        - 3-5 sentences summarizing the main activities.

        Geospatial Intelligence:
        - 1-2 sentences describing the spatial context of the activity.

        Analyst Observations:
        - 2-4 sentences highlighting key observations and anomalies.
        """

        detailed_prompt = detailed_prompt_template.format(
            toon_context=toon_context
        )

        prompts = AnalysisPrompts(
            frame_analysis=(
                "Analyze this frame as surveillance imagery from a fixed monitoring camera. "
                "Describe visible activity, movement, terrain, infrastructure, "
                "and object appearance."
            ),
            detailed_summary=detailed_prompt,
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
            min_frames=15,
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
 