from zenml import pipeline
from .steps import (
    download_clip,
    extract_metadata,
    decode_metadata,
    object_detection,
    fusion_context,
    llm_summary

)
from utils.logger import get_logger

logger = get_logger(__name__)

@pipeline(enable_cache=False)
def isr_pipeline(
    clip_id: str,
    clip_uri: str,
    jars: list[str],
    output_bucket: str,
    output_bucket_detection: str,
    output_bucket_fusion: str,
    output_bucket_summary: str,
    output_path: str,
    confidence_threshold: float,
    distance_threshold: int,
    hit_counter_max: int,
    initialization_delay: int,
    distance_function: str,
):
    logger.info(f"Pipeline started for clip_id: {clip_id}")

    ts_path, video_duration = download_clip(clip_id, clip_uri)
    logger.info(f"Downloaded clip for clip_id: {clip_id}")

    # Extract Metadata (KLV )
    # -----------------------------
    klv_path = extract_metadata(
        ts_path=ts_path,
        clip_id=clip_id,
        output_bucket=output_bucket,
    )
    logger.info(f"Extracted metadata for clip_id: {clip_id}")

    # Decode Metadata (KLV )
    # -----------------------------
    dec_json= decode_metadata(
        ts_path=ts_path,
        klv_path=klv_path,
        jars=jars,
        clip_id=clip_id,
        output_bucket=output_bucket,
    )
    logger.info(f"Decoded metadata for clip_id: {clip_id}")

    # Object Detection (TS )
    # -----------------------------
    obj_json,fps = object_detection(
        clip_id=clip_id,
        ts_path=ts_path,
        output_bucket_detection=output_bucket_detection,
        output_path=output_path,
        confidence_threshold=confidence_threshold,
        distance_threshold=distance_threshold,
        hit_counter_max=hit_counter_max,
        initialization_delay=initialization_delay,
        distance_function=distance_function,
    )
    logger.info(f"Pipeline completed for clip_id: {clip_id}")

    # Fusion Context (TS + KLV + Detections)
    # -----------------------------
    fusion_json = fusion_context(
        clip_id=clip_id,
        video_duration=video_duration,
        klv_json_uri=dec_json,
        det_json_uri=obj_json,
        output_bucket=output_bucket_fusion,
        fps=fps,
    )
    logger.info(f"Fusion context completed for clip_id: {clip_id}")
    # LLM Video Summary (TS + Fusion Context)
    # -----------------------------
    summary_uri = llm_summary(
        clip_id=clip_id,
        ts_path=ts_path,
        fusion_json_uri=fusion_json,
        output_bucket=output_bucket_summary,
        model="qwen3-vl:30b",
    )
    logger.info(f"LLM summary completed for clip_id: {clip_id}")