from zenml import pipeline
from .steps import (
    download_clip_step,
    extract_metadata_step,
    decode_metadata_step,
    object_detection,
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
    output_path: str,
    confidence_threshold: float,
    distance_threshold: int,
    hit_counter_max: int,
    initialization_delay: int,
    distance_function: str,
):
    logger.info(f"Pipeline started for clip_id: {clip_id}")

    ts_path = download_clip_step(clip_id, clip_uri)
    logger.info(f"Downloaded clip for clip_id: {clip_id}")

    # Extract Metadata (KLV )
    # -----------------------------
    klv_path = extract_metadata_step(
        ts_path=ts_path,
        clip_id=clip_id,
        output_bucket=output_bucket,
    )
    logger.info(f"Extracted metadata for clip_id: {clip_id}")

    # Decode Metadata (KLV )
    # -----------------------------
    decode_metadata_step(
        ts_path=ts_path,
        klv_path=klv_path,
        jars=jars,
        clip_id=clip_id,
        output_bucket=output_bucket,
    )
    logger.info(f"Decoded metadata for clip_id: {clip_id}")

    # Object Detection (TS )
    # -----------------------------
    object_detection(
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