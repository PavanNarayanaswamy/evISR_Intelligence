import logging
from pathlib import Path
from typing import List, Dict

from openscenesense_ollama.models import AnalysisPrompts
from openscenesense_ollama.analyzer import OllamaVideoAnalyzer
from openscenesense_ollama.frame_selectors import DynamicFrameSelector
import reverse_geocoder as rg

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
        logger.info(f"TOON Context:\n{toon_context}\n")
        # ---- Custom Prompts (ISR aware) ----
        prompts = AnalysisPrompts(
            frame_analysis=(
                "Analyze this frame as airborne ISR imagery. "
                "Describe visible activity, movement, terrain, infrastructure, and object appearance. "
                "Include relative size and color when clearly observable."
            ),

            detailed_summary = f"""
            SYSTEM ROLE:
            You are a senior ISR Intelligence Analyst producing an operational intelligence
            summary from airborne video surveillance.

            Write in clear ISR-report style. Use only observable activity.
            Do NOT mention data structures, detections, IDs, bounding boxes, models, or TOON.
            Avoid speculation and technical language.

            --------------------------------------------------
            SENSOR CONTEXT (FOR REASONING ONLY)
            --------------------------------------------------
            {toon_context}

            --------------------------------------------------
            TASK
            --------------------------------------------------

            Primary Activity:
            Describe the timeline of activity across the segment (movement, changes, behavior).

            Geospatial Intelligence:
            Use the resolved location as the main geographic reference.
            If the location is generic, infer environment type (highway, rural area,
            industrial zone, residential area, etc.) using coordinates if needed.
            Do not output raw coordinates unless necessary.

            Analyst Observations:
            Add persistence, infrastructure, or environmental context.
            State uncertainty when visibility is limited.

            --------------------------------------------------
            OUTPUT FORMAT (STRICT)
            --------------------------------------------------

            [Concise Summary Title]

            Primary Activity:
            3–5 sentences

            Geospatial Intelligence:
            Location and movement

            Analyst Observations:
            Additional ISR insight
            """
            ,
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

# import os
# import json
# import base64
# import logging
# from typing import Dict, List
# from openai import OpenAI
# import reverse_geocoder as rg
# import numpy as np
# from PIL import Image
# import io

# from openscenesense_ollama.frame_selectors import DynamicFrameSelector
# from openscenesense_ollama.models import AnalysisPrompts
# from openscenesense_ollama.analyzer import OllamaVideoAnalyzer
# from pytoony import json2toon
# from utils.logger import get_logger

# logger = get_logger(__name__)


# class LiteLLMVideoAnalyzer:
#     def __init__(self, model_vision="gpt-4o", model_summary="gpt-5"):
#         self.client = OpenAI(
#             base_url="https://evgpt.evertz.com:24000/v1",
#             api_key="sk-ABTuxPbTDxbSXWIvVAF5ZA",
#         )
#         self.model_vision = model_vision
#         self.model_summary = model_summary
#         self.selector = DynamicFrameSelector()

#     def _frame_to_base64(self, frame: np.ndarray) -> str:
#         """Convert a frame to base64 string"""
#         image = Image.fromarray(frame)
#         buffered = io.BytesIO()
#         image.save(buffered, format="PNG")
#         return base64.b64encode(buffered.getvalue()).decode()


#     # -------------------------------------------------
#     # Frame analysis
#     # -------------------------------------------------
#     def analyze_frame(self, frame, prompt: str) -> str:
#         base64_image = self._frame_to_base64(frame.image)
#         response = self.client.chat.completions.create(
#             model=self.model_vision,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": prompt},
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/png;base64,{base64_image}"
#                             },
#                         },
#                     ],
#                 }
#             ],
#             temperature=0.2,
#         )

#         return response.choices[0].message.content

#     # -------------------------------------------------
#     # Video analysis
#     # -------------------------------------------------
#     def analyze_video(self, video_path: str, prompts: AnalysisPrompts) -> List[str]:
#         analyzer_instance = OllamaVideoAnalyzer()
#         frames = self.selector.select_frames(video_path, analyzer_instance)

#         logger.info(f"Selected {len(frames)} frames")

#         frame_reports = []

#         for frame in frames:
#             logger.info(f"Analyzing frame: {frame}")
#             report = self.analyze_frame(frame, prompts.frame_analysis)
#             frame_reports.append(report)

#         return frame_reports

#     # -------------------------------------------------
#     # Summary generation
#     # -------------------------------------------------
#     def summarize(self, frame_reports: List[str], detailed_prompt: str) -> str:
#         combined = "\n\n".join(frame_reports)

#         response = self.client.chat.completions.create(
#             model=self.model_summary,
#             messages=[
#                 {"role": "system", "content": "You are an ISR intelligence analyst."},
#                 {"role": "user", "content": detailed_prompt + "\n\n" + combined},
#             ],
#             temperature=0.2,
#         )

#         return response.choices[0].message.content


# class VideoLLMSummarizer:
#     @staticmethod
#     def fusion_json_to_toon(fusion_context: dict) -> str:
#         json_str = json.dumps(fusion_context, separators=(",", ":"))
#         return json2toon(json_str)

#     @staticmethod
#     def escape_for_format(text: str) -> str:
#         return text.replace("{", "{{").replace("}", "}}")

#     @staticmethod
#     def tune_fusion_context_for_llm(fusion_context: Dict) -> Dict:
#         tuned = {
#             "clip_id": fusion_context.get("clip_id"),
#             "segment_duration_sec": fusion_context.get("segment_duration_sec"),
#             "fusion": [],
#         }

#         for entry in fusion_context.get("fusion", []):
#             klv = entry.get("klv", {})
#             fields = klv.get("fields", {}) if klv else {}

#             latitude = fields.get("SensorLatitude")
#             longitude = fields.get("SensorLongitude")

#             resolved_location = "Unknown location"

#             if latitude and longitude:
#                 try:
#                     coords = (float(latitude), float(longitude))
#                     result = rg.search([coords])[0]
#                     resolved_location = f"{result['name']}, {result['admin1']}, {result['cc']}"
#                 except Exception:
#                     pass

#             tuned["fusion"].append(
#                 {
#                     "second": entry.get("second"),
#                     "geo_location": {
#                         "resolved_location": resolved_location,
#                         "latitude": latitude,
#                         "longitude": longitude,
#                     },
#                     "detections": entry.get("detections", []),
#                 }
#             )

#         return tuned

#     # -------------------------------------------------
#     # Main ISR summarization pipeline
#     # -------------------------------------------------
#     @staticmethod
#     def summarize(fusion_context: Dict, video_path: str) -> str:
#         tuned_fusion = VideoLLMSummarizer.tune_fusion_context_for_llm(fusion_context)
#         toon_context = VideoLLMSummarizer.escape_for_format(
#             VideoLLMSummarizer.fusion_json_to_toon(tuned_fusion)
#         )
#         logger.info(f"TOON Context:\n{toon_context}\n")

#         prompts = AnalysisPrompts(
#             frame_analysis=(
#                 "Analyze this frame as airborne ISR imagery. "
#                 "Describe visible activity, movement, terrain, infrastructure, and object appearance. "
#                 "Include relative size and color when clearly observable."
#             ),
 
#             detailed_summary = f"""
#             SYSTEM ROLE:
#             You are a senior ISR Intelligence Analyst producing an operational intelligence
#             summary from airborne video surveillance.
 
#             Write in clear ISR-report style. Use only observable activity.
#             Do NOT mention data structures, detections, IDs, bounding boxes, models, or TOON.
#             Avoid speculation and technical language.
            
#             IMPORTANT:
#             Resolved geographic location strings provided in the sensor context are AUTHORITATIVE and must be used as the primary location reference.
#             Do NOT replace a resolved location with "Unknown location" unless the location field itself is explicitly missing.
            
#             --------------------------------------------------
#             SENSOR CONTEXT (FOR REASONING ONLY)
#             --------------------------------------------------
#             {toon_context}
#             --------------------------------------------------
#             TASK
#             --------------------------------------------------
#             Primary Activity:
#             Describe the timeline of activity across the segment (movement, changes, behavior).
 
#             Geospatial Intelligence:
#             Use the resolved location as the main geographic reference.
#             If a resolved location is provided, state it directly.
#             If the location is generic, infer environment type (highway, rural area,
#             industrial zone, residential area, etc.) using coordinates if needed.
#             Do not output raw coordinates unless necessary.
#             Do NOT invent road names, junction names, or regions not explicitly supported.
            
#             Analyst Observations:
#             Add persistence, infrastructure, traffic patterns, or environmental context.
#             State uncertainty when visibility is limited or positioning cannot be confirmed.
 
#             --------------------------------------------------
#             OUTPUT FORMAT (STRICT)
#             --------------------------------------------------
 
#             [Concise Summary Title]
 
#             Primary Activity:
#             5-7 sentences
 
#             Geospatial Intelligence:
#             Location and movement
 
#             Analyst Observations:
#             Additional ISR insight
#             """
#             ,
#             brief_summary=(
#                 "Generate a concise ISR intelligence summary focusing on the "
#                 "primary observed activity and key geospatial context."
#             )
#         )

#         analyzer = LiteLLMVideoAnalyzer()

#         frame_reports = analyzer.analyze_video(video_path, prompts)

#         summary = analyzer.summarize(frame_reports, prompts.detailed_summary)

#         return summary


