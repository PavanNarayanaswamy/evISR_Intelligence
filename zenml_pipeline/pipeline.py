from zenml import pipeline
from .steps import (
    download_clip_step,
    extract_metadata_step,
    decode_metadata_step,
    object_detection,
)

@pipeline(enable_cache=False)
def isr_pipeline(
    clip_id: str,
    clip_uri: str,
    jars: list[str],
    output_bucket: str,
    output_path: str,
    confidence_threshold: float = 0.4,
):
    ts_path = download_clip_step(clip_id, clip_uri)

    # Extract Metadata (KLV )
    # -----------------------------
    klv_path = extract_metadata_step(
        ts_path=ts_path,
        clip_id=clip_id,
        output_bucket=output_bucket,
    )

    # Decode Metadata (KLV )
    # -----------------------------
    decode_metadata_step(
        ts_path=ts_path,
        klv_path=klv_path,
        jars=jars,
        clip_id=clip_id,
        output_bucket=output_bucket,
    )

    # # Object Detection (TS )
    # # -----------------------------
    # object_detection(
    #     ts_path=ts_path,
    #     output_path=output_path,
    #     confidence_threshold=confidence_threshold,
    # )