from zenml import step
from pathlib import Path
import json
import os
import datetime

from .minio_utils import download_clip, upload_output, get_minio_client, download_klv
from klv_metadata_extraction.decoding import JmisbDecoder
from klv_metadata_extraction.extraction import Extraction
from object_detection.global_tracking import ObjectTracker
from utils.logger import get_logger

logger = get_logger(__name__)

# -------------------------------------------------
# DOWNLOAD STEP
# -------------------------------------------------
@step
def download_clip_step(clip_id: str, clip_uri: str) -> str:
    """
    Downloads TS from MinIO as ./<clip_id>.ts
    """
    ts_path = Path.cwd() / f"{clip_id}.ts"
    logger.info(f"Downloading clip for clip_id: {clip_id} from {clip_uri}")
    download_clip(clip_uri, ts_path)
    logger.info(f"Downloaded clip to {ts_path}")
    return str(ts_path)


# -------------------------------------------------
# EXTRACT KLV STEP
# -------------------------------------------------
@step
def extract_metadata_step(
    ts_path: str,
    clip_id: str,
    output_bucket: str,
) -> str | None:
    """
    Extracts KLV if present.
    Returns local KLV path or None.
    """
    logger.info(f"Extracting KLV metadata for clip_id: {clip_id}")
    extractor = Extraction(
        minio_client=get_minio_client(),
        output_bucket=output_bucket,
    )

    klv_path = extractor.extract_klv(
        ts_path=ts_path,
        clip_id=clip_id,
    )
    logger.info(f"Extracted KLV metadata for clip_id: {clip_id} to {klv_path}")
    return klv_path


# -------------------------------------------------
# DECODE + UPLOAD + CLEANUP STEP
# -------------------------------------------------
@step
def decode_metadata_step(
    klv_path: str,          # s3://bucket/file.klv
    jars: list[str],
    clip_id: str,
    output_bucket: str,
    ts_path: str,
) -> str:

    local_klv = Path("/tmp") / f"{clip_id}.klv"
    output_json = Path("/tmp") / f"{clip_id}.json"

    try:
        logger.info(f"Decoding KLV metadata for clip_id: {clip_id}")
        # DOWNLOAD FROM MINIO
        download_klv(klv_path, local_klv)

        decoder = JmisbDecoder(jars)
        decoder.start_jvm()
        decoded = decoder.decode_file(str(local_klv))
        # decoder.shutdown_jvm()

        with open(output_json, "w") as f:
            json.dump(decoded, f, indent=2)
        now = datetime.datetime.now()  # local time
        object_name = (
            f"decoding/"
            f"{now.strftime('%Y/%m/%d/%H')}/"
            f"{clip_id}.json"
        )
        upload_output(
            output_bucket,
            object_name,
            output_json,
        )
        logger.info(f"Decoded metadata uploaded for clip_id: {clip_id} to {object_name}")

        return f"minio://{output_bucket}/{object_name}"

    except Exception as e:
        logger.error(f"Error during decoding/upload for clip_id: {clip_id}: {e}", exc_info=True)
        raise

    finally:
        # CLEANUP
        for path in [ts_path, local_klv, output_json]:
            if Path(path).exists():
                try:
                    os.remove(path)
                    logger.debug(f"Removed temporary file: {path}")
                except Exception as ce:
                    logger.error(f"Error cleaning up {path} for clip_id: {clip_id}: {ce}", exc_info=True)

@step(enable_cache=False)
def object_detection(ts_path: str, output_path: str, confidence_threshold: float,) -> None:
    """
    Run the full RTSP tracking job.
    Live resources must stay inside ONE step.
    """
    logger.info(f"Starting object detection for {ts_path}")
    tracker = ObjectTracker(
        video_path=ts_path,
        output_path=output_path,
        confidence_threshold=confidence_threshold,
    )
    tracker.load_model()
    tracker.setup_stream()
    tracker.setup_tracking()
    tracker.run()
    tracker.cleanup()
    logger.info(f"Object detection completed for {ts_path}")