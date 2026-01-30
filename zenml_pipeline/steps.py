from zenml import step
from pathlib import Path
import json
import os
import datetime
import subprocess
import math

from .minio_utils import download_segment, upload_output, get_minio_client, download_file
from klv_metadata_extraction.decoding import JmisbDecoder
from klv_metadata_extraction.extraction import Extraction
from object_detection.object_tracker import ObjectTracker
from cv_models.rf_detr import RFDetrDetector
from tracker.norfair_tracker import NorfairTrackerAnnotator
from utils.logger import get_logger
from fusion_context.fusion import TemporalFusion
from fusion_context.semantic_fusion import SemanticFusion
from video_summary.summary_gen import VideoLLMSummarizer
from utils import config

from agents.klv.agent import klv_graph
from typing import Tuple
from typing_extensions import Annotated

logger = get_logger(__name__)

class CompactListEncoder(json.JSONEncoder):
    def encode(self, obj):
        if isinstance(obj, list):
            return "[" + ", ".join(self.encode(x) for x in obj) + "]"
        return json.JSONEncoder.encode(self, obj)
    
# -------------------------------------------------
# DOWNLOAD STEP
# -------------------------------------------------
@step
def download_clip(clip_id: str, clip_uri: str) -> tuple[str, float]:
    """
    Downloads TS from MinIO as ./<clip_id>.ts
    """
    ts_path = Path.cwd() / f"{clip_id}.ts"
    logger.info(f"Downloading clip for clip_id: {clip_id} from {clip_uri}")
    download_segment(clip_uri, ts_path)
    logger.info(f"Downloaded clip to {ts_path}")
    result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                ts_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
    return str(ts_path), float(result.stdout.strip())


# -------------------------------------------------
# EXTRACT KLV STEP
# -------------------------------------------------
'''
@step
def extract_metadata(
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
def decode_metadata(
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
        download_file(klv_path, local_klv)

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
        for path in [local_klv, output_json]:
            if Path(path).exists():
                try:
                    os.remove(path)
                    logger.debug(f"Removed temporary file: {path}")
                except Exception as ce:
                    logger.error(f"Error cleaning up {path} for clip_id: {clip_id}: {ce}", exc_info=True)
'''

@step
def klv_agent_step(
    ts_path: str,
    clip_id: str,
    jars: list[str],
    output_bucket: str,
) -> Tuple[
    Annotated[str, "klv_extraction_uri"],
    Annotated[str, "klv_decoding_uri"],
]:
    out = klv_graph.invoke({  # invoke is the standard compiled-graph call.
        "ts_path": ts_path,
        "clip_id": clip_id,
        "jars": jars,
        "output_bucket": output_bucket,
    })
    # Returning a tuple literal => ZenML treats as multiple output artifacts.
    return out["extraction_uri"], out["decoding_uri"]

# --------------------------------------------------
# Object Detection STEP
# --------------------------------------------------
@step(enable_cache=False)
def object_detection(
    clip_id: str,
    ts_path: str,
    output_bucket_detection :str,
    output_path: str,
    confidence_threshold: float,
    distance_threshold: int,
    hit_counter_max: int,
    initialization_delay: int,
    distance_function: str,
) -> tuple[str, float]:
    """
    ZenML step for object detection + tracking .
    All components are explicitly constructed here.
    """
    logger.info(f"Starting object detection for {ts_path}")
    # ------------------------------
    # RF-DETR MODEL
    # ------------------------------
    detector = RFDetrDetector(confidence_threshold)

    # ------------------------------
    # TRACKING METHOD 
    # ------------------------------
    #tracker_algo = ByteTrackerAnnotator()
    tracker_algo = NorfairTrackerAnnotator(
        distance_function=distance_function,
        distance_threshold=distance_threshold,
        hit_counter_max=hit_counter_max,
        initialization_delay=initialization_delay,
    )

    # ------------------------------
    # Orchestrator
    # ------------------------------
    tracker = ObjectTracker(
        clip_id=clip_id,
        minio_client=get_minio_client(),
        video_path=ts_path,
        detector=detector,
        tracker=tracker_algo,
        output_bucket_detection=output_bucket_detection,
        output_path=output_path,
    )
    tracker.start(enable_mp4=config.SAVE_MP4)
    tracker.process()
    det_json_uri,fps = tracker.write_outputs()
    tracker.cleanup()

    return det_json_uri,fps

@step(enable_cache=False)
def fusion_context(
    clip_id: str,
    video_duration: float,
    klv_json_uri: str,
    det_json_uri: str,
    output_bucket: str,
    fps: float,
) -> str:
    """
    Performs:
    1. Temporal fusion → raw fusion output
    2. Semantic fusion → semantic geo output
    Uploads both to MinIO.
    """

    logger.info(f"Starting fusion for clip_id: {clip_id}")

    local_klv = Path("/tmp") / f"{clip_id}_klv.json"
    local_det = Path("/tmp") / f"{clip_id}_det.json"
    raw_fusion_path = Path("/tmp") / f"{clip_id}_fusion.json"
    semantic_fusion_path = Path("/tmp") / f"{clip_id}_semantic.json"

    try:
        logger.info("Downloading KLV JSON from MinIO")
        download_file(klv_json_uri, local_klv)

        logger.info("Downloading detection JSON from MinIO")
        download_file(det_json_uri, local_det)

        with open(local_klv, "r") as f:
            klv_json = json.load(f)

        with open(local_det, "r") as f:
            det_json = json.load(f)

        # -----------------------------
        # RAW FUSION
        # -----------------------------
        klv_time_window = 0.5

        fusion_output = TemporalFusion.fuse_klv_and_detections(
            klv_json=klv_json,
            det_json=det_json,
            klv_time_window=klv_time_window,
        )

        with open(raw_fusion_path, "w") as f:
            json.dump(fusion_output, f, indent=2)

        now = datetime.datetime.now()

        raw_object_name = (
            f"raw_fusion/"
            f"{now.strftime('%Y/%m/%d/%H')}/"
            f"{clip_id}.json"
        )

        upload_output(
            bucket=output_bucket,
            object_name=raw_object_name,
            file_path=raw_fusion_path,
        )

        logger.info(f"Raw fusion uploaded → {raw_object_name}")

        # -----------------------------
        # SEMANTIC FUSION
        # -----------------------------
        semantic_output = SemanticFusion.build_semantic_fusion(
            raw_fusion_output=fusion_output,
            fps=fps,
            clip_id=clip_id,
        )

        with open(semantic_fusion_path, "w") as f:
            json.dump(
                semantic_output,
                f,
                cls=CompactListEncoder,
                indent=2,
                ensure_ascii=False
            )

        semantic_object_name = (
            f"semantic_fusion/"
            f"{now.strftime('%Y/%m/%d/%H')}/"
            f"{clip_id}.json"
        )

        upload_output(
            bucket=output_bucket,
            object_name=semantic_object_name,
            file_path=semantic_fusion_path,
        )

        logger.info(f"Semantic fusion uploaded → {semantic_object_name}")

        return f"minio://{output_bucket}/{semantic_object_name}"

    except Exception as e:
        logger.error(
            f"Fusion failed for clip_id {clip_id}: {e}",
            exc_info=True
        )
        raise

    finally:
        for path in [
            local_klv,
            local_det,
            raw_fusion_path,
            semantic_fusion_path,
        ]:
            if path.exists():
                try:
                    os.remove(path)
                    logger.debug(f"Removed temporary file: {path}")
                except Exception as ce:
                    logger.warning(
                        f"Failed to remove {path}: {ce}",
                        exc_info=True
                    )


@step(enable_cache=False)
def llm_summary(
    clip_id: str,
    ts_path: str,
    fusion_json_uri: str,
    output_bucket: str,
    model: str = "qwen3-vl:30b",
) -> str:
    """
    Generates LLM-based video summary using:
    - fusion-selected frames (1 per second)
    - fusion-context JSON
    """

    local_fusion = Path("/tmp") / f"{clip_id}_fusion.json"
    frames_dir = Path("/tmp") / f"{clip_id}_llm_frames"
    summary_path = Path("/tmp") / f"{clip_id}_summary.txt"

    try:
        # -------------------------------------------------
        # Download fusion JSON
        # -------------------------------------------------
        logger.info(f"Downloading fusion JSON for clip_id: {clip_id}")
        download_file(fusion_json_uri, local_fusion)

        with open(local_fusion, "r") as f:
            fusion_context = json.load(f)

        logger.info("Extracting fusion-aligned frames for LLM")


        # -------------------------------------------------
        # Run LLM summarization
        # -------------------------------------------------
        logger.info(f"Running LLM summarization using model: {model}")
        summary = VideoLLMSummarizer.summarize(
            fusion_context=fusion_context,
            # model=model,
            video_path=ts_path,
        )

        summary_path.write_text(summary)

        # -------------------------------------------------
        # Upload summary
        # -------------------------------------------------
        now = datetime.datetime.now()
        object_name = (
            f"summary/"
            f"{now.strftime('%Y/%m/%d/%H')}/"
            f"{clip_id}.txt"
        )

        upload_output(
            bucket=output_bucket,
            object_name=object_name,
            file_path=summary_path,
        )

        logger.info(
            f"LLM summary uploaded for clip_id {clip_id} to {object_name}"
        )

        return f"minio://{output_bucket}/{object_name}"

    finally:
        # -------------------------------------------------
        # Cleanup
        # -------------------------------------------------
        for path in [local_fusion, summary_path, ts_path]:
            path = Path(path) 
            if path.exists():
                try:
                    os.remove(path)
                except Exception:
                    pass
