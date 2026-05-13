# zenml_pipeline/steps.py
from zenml import step, log_metadata
from pathlib import Path
import requests
from typing import Dict, Any
from typing_extensions import Annotated
from utils.downloads import download_video_from_uri
from utils.logger import get_logger
from utils import config

from typing import Tuple, List
from typing_extensions import Annotated

#Agent Wrapper Imports
from agents.klv.state import KLVState
from agents.detection.state import DetectionState
from agents.fusion.state import FusionState
from agents.summary.state import SummaryState

logger = get_logger(__name__)
    
# -------------------------------------------------
# DOWNLOAD STEP
# -------------------------------------------------
def minio_segmented_clip(clip_id: str, clip_uri: str) -> Tuple[Annotated[str, "video_path"], Annotated[float, "video_duration"]]:
    """
    Downloads TS from MinIO by calling the utility download function.
    """
    return download_video_from_uri(clip_id, clip_uri)


def klv_extraction_agent(
    ts_path: str,
    clip_id: str,
    jars: List[str],
    output_bucket: str,
) -> Tuple[
    Annotated[str, "klv_extraction_uri"],
    Annotated[str, "klv_decoding_uri"],
]:
    """ZenML step wrapper with Pydantic validation."""
    
    state = KLVState(
        clip_id=clip_id,
        ts_path=ts_path,
        output_bucket=output_bucket,
        jars=jars,
    )
    
    response = requests.post("http://localhost:8000/tools/klv", json=state.model_dump(mode='json'))
    response.raise_for_status()
    final_state = KLVState.model_validate(response.json())
    
    assert final_state.is_complete, "KLV agent failed to complete"
    
    return (
        final_state.extraction_uri,
        final_state.decoding_uri,
    )


def object_detection_agent(
    clip_id: str, ts_path: str, output_bucket_detection: str, output_path: str,
    confidence_threshold: float, distance_threshold: int, hit_counter_max: int,
    initialization_delay: int, distance_function: str,
) -> Tuple[
    Annotated[str, "detection_uri"],
    Annotated[float, "fps"],
]:
    state = DetectionState(
        clip_id=clip_id, ts_path=ts_path, output_bucket_detection=output_bucket_detection,
        output_path=output_path, confidence_threshold=confidence_threshold,
        distance_threshold=distance_threshold, hit_counter_max=hit_counter_max,
        initialization_delay=initialization_delay, distance_function=distance_function,
        save_mp4=bool(getattr(config, "SAVE_MP4", False)),
    )
    
    response = requests.post("http://localhost:8000/tools/detection", json=state.model_dump(mode='json'))
    response.raise_for_status()
    final_state = DetectionState.model_validate(response.json())

    assert final_state.is_complete, "Detection agent failed"
    
    log_metadata({"Model_Metrics": final_state.metrics or {}})
    return final_state.det_json_uri, final_state.fps


def fusion_context_agent(
    clip_id: str,
    video_duration: float,
    klv_json_uri: str,
    det_json_uri: str,
    output_bucket: str,
    fps: float,
) -> tuple[
    Annotated[str, "fusion_uri"],
    Annotated[Dict[str, float], "geo_coordinates"],
]:

    logger.info(f"[ZENML] Fusion context step clip_id={clip_id}")

    state = FusionState(
        clip_id=clip_id,
        video_duration=video_duration,
        klv_json_uri=klv_json_uri,
        det_json_uri=det_json_uri,
        output_bucket=output_bucket,
        fps=fps,
    )

    response = requests.post("http://localhost:8000/tools/fusion", json=state.model_dump(mode='json'))
    response.raise_for_status()
    final_state = FusionState.model_validate(response.json())

    assert final_state.is_complete, "Fusion agent failed"

    # Extract geo context
    semantic_data = final_state.semantic_fusion
    geo_context = semantic_data.get("geo_context", {})

    geo_coordinates = {
        "start_latitude": geo_context.get("start_latitude"),
        "start_longitude": geo_context.get("start_longitude"),
        "end_latitude": geo_context.get("end_latitude"),
        "end_longitude": geo_context.get("end_longitude"),
    }

    return final_state.fusion_uri, geo_coordinates

def llm_summary_agent(clip_id: str, ts_path: str, fusion_json_uri: str,
                     output_bucket: str, model: str = "qwen3-vl:32b") -> tuple[
    Annotated[str, "summary_uri"],
    Annotated[float, "severity_score"],
    Annotated[str, "severity_label"],
]:
    """
    ZenML step boundary for LLM video summary.
    Actual summarization runs inside the LLM summary agent.
    """
    logger.info(
        f"[ZENML] LLM summary step clip_id={clip_id} "
        f"fusion_json_uri={fusion_json_uri} model={model}"
    )

    state = SummaryState(clip_id=clip_id, ts_path=ts_path,
                        fusion_json_uri=fusion_json_uri, output_bucket=output_bucket, model=model)
    
    response = requests.post("http://localhost:8000/tools/summary", json=state.model_dump(mode='json'))
    response.raise_for_status()
    final_state = SummaryState.model_validate(response.json())

    assert final_state.is_complete, "Summary agent failed"
    try:
        ts_file = Path(ts_path)
        if ts_file.exists():
            ts_file.unlink()
            logger.info(f"[CLEANUP] Removed TS file: {ts_file}")
    except Exception as e:
        logger.warning(f"[CLEANUP] Failed to remove TS file: {e}")
    return (
        final_state.summary_uri,
        final_state.severity_score,
        final_state.severity_label,
    )
