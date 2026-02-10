# agents/fusion/agent.py
import datetime
import json
import math
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from typing import Dict, Any

from utils.logger import get_logger
from fusion_context.fusion import TemporalFusion
from zenml_pipeline.minio_utils import download_file, upload_output

from .state import FusionState

logger = get_logger(__name__)


def download_inputs_node(state: FusionState) -> Dict[str, Any]:
    """
    Download KLV + detection JSON from MinIO into /tmp.
    """
    clip_id = state["clip_id"]
    logger.info(f"[FUSION_AGENT] Downloading inputs for clip_id={clip_id}")

    local_klv = Path("/tmp") / f"{clip_id}_klv.json"
    local_det = Path("/tmp") / f"{clip_id}_det.json"

    download_file(state["klv_json_uri"], local_klv)
    logger.info(f"[FUSION_AGENT] Downloaded KLV JSON to {local_klv}")

    download_file(state["det_json_uri"], local_det)
    logger.info(f"[FUSION_AGENT] Downloaded detection JSON to {local_det}")

    return {
        "local_klv_path": str(local_klv),
        "local_det_path": str(local_det),
    }


def load_and_fuse_node(state: FusionState) -> Dict[str, Any]:
    """
    Load JSON, run TemporalFusion, and write fusion output locally.
    """
    clip_id = state["clip_id"]
    local_klv = Path(state["local_klv_path"])
    local_det = Path(state["local_det_path"])

    logger.info(f"[FUSION_AGENT] Loading JSON for clip_id={clip_id}")
    with open(local_klv, "r") as f:
        klv_json = json.load(f)

    with open(local_det, "r") as f:
        det_json = json.load(f)

    segment_duration_sec = int(math.ceil(state["video_duration"]))
    logger.info(
        f"[FUSION_AGENT] Running TemporalFusion clip_id={clip_id} "
        f"segment_duration_sec={segment_duration_sec}"
        f"klv_time_window={klv_time_window}"
    )

    fusion_output = TemporalFusion.fuse_klv_and_detections(
        klv_json=klv_json,
        det_json=det_json,
        klv_time_window=klv_time_window,
    )

    fusion_output_path = Path("/tmp") / f"{clip_id}_fusion.json"
    with open(fusion_output_path, "w") as f:
        json.dump(fusion_output, f, indent=2)

    logger.info(f"[FUSION_AGENT] Wrote fusion output to {fusion_output_path}")

    return {
        "klv_json": klv_json,
        "det_json": det_json,
        "local_fusion_path": str(fusion_output_path),
    }

def upload_node(state: FusionState) -> Dict[str, Any]:
    """
    Upload fusion JSON to MinIO and return minio:// URI.
    """
    clip_id = state["clip_id"]
    output_bucket = state["output_bucket"]
    fusion_output_path = Path(state["local_fusion_path"])

    now = datetime.datetime.now()
    object_name = (
        f"fusion/"
        f"{now.strftime('%Y/%m/%d/%H')}/"
        f"{clip_id}.json"
    )

    logger.info(
        f"[FUSION_AGENT] Uploading fusion output clip_id={clip_id} "
        f"to bucket={output_bucket}, object={object_name}"
    )
    upload_output(
        bucket=output_bucket,
        object_name=object_name,
        file_path=fusion_output_path,
    )

    fusion_uri = f"minio://{output_bucket}/{object_name}"
    logger.info(f"[FUSION_AGENT] Fusion URI: {fusion_uri}")
    return {"fusion_uri": fusion_uri}


def cleanup_node(state: FusionState) -> Dict[str, Any]:
    """
    Delete local temp files.
    """
    for key in ("local_klv_path", "local_det_path", "local_fusion_path"):
        path_str = state.get(key)
        if not path_str:
            continue
        p = Path(path_str)
        if p.exists():
            try:
                p.unlink()
                logger.debug(f"[FUSION_AGENT] Removed temp file: {p}")
            except Exception as e:
                logger.warning(
                    f"[FUSION_AGENT] Failed to remove {p}: {e}",
                    exc_info=True,
                )
    return {}


def build_fusion_graph():
    """
    Linear graph: download -> load+fuse -> upload -> cleanup -> END.
    Kept simple and synchronous.
    """
    g = StateGraph(FusionState)
    g.add_node("download_inputs", download_inputs_node)
    g.add_node("load_and_fuse", load_and_fuse_node)
    g.add_node("upload", upload_node)
    g.add_node("cleanup", cleanup_node)

    g.add_edge(START, "download_inputs")
    g.add_edge("download_inputs", "load_and_fuse")
    g.add_edge("load_and_fuse", "upload")
    g.add_edge("upload", "cleanup")
    g.add_edge("cleanup", END)

    return g.compile()


fusion_graph = build_fusion_graph()
