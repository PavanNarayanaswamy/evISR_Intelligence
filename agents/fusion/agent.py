# agents/fusion/agent.py
import datetime
import json
import math
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from typing import Dict, Any

from utils.logger import get_logger
from fusion_context.fusion import TemporalFusion
from fusion_context.semantic_fusion import SemanticFusion, CompactListEncoder
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
    
    klv_time_window = 0.5
    segment_duration_sec = int(math.ceil(state["video_duration"]))
    logger.info(
        f"[FUSION_AGENT] Running TemporalFusion clip_id={clip_id} "
        f"segment_duration_sec={segment_duration_sec}"
        f"klv_time_window={klv_time_window}"
    )

    raw_fusion = TemporalFusion.fuse_klv_and_detections(
        klv_json=klv_json,
        det_json=det_json,
        klv_time_window=klv_time_window,
    )
    logger.info(f"[FUSION_AGENT] Raw fusion complete clip_id={clip_id}")
    # Semantic fusion (uses fps + clip_id)
    semantic_fusion = SemanticFusion.build_semantic_fusion(
        raw_fusion_output=raw_fusion,
        fps=state["fps"],
        clip_id=clip_id,
    )
    logger.info(f"[FUSION_AGENT] Semantic fusion complete clip_id={clip_id}")

    # Write both
    raw_fusion_path = Path("/tmp") / f"{clip_id}_fusion.json"
    semantic_fusion_path = Path("/tmp") / f"{clip_id}_semantic.json"
    with open(raw_fusion_path, "w") as f:
        json.dump(raw_fusion, f, indent=2)

    with open(semantic_fusion_path, "w") as f:
        json.dump(
            semantic_fusion,
            f,
            cls=CompactListEncoder,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(f"[FUSION_AGENT] Wrote raw+semantic fusion for clip_id={clip_id}")

    return {
        "klv_json": klv_json,
        "det_json": det_json,
        "raw_fusion": raw_fusion,
        "semantic_fusion": semantic_fusion,
        "raw_fusion_path": str(raw_fusion_path),
        "semantic_fusion_path": str(semantic_fusion_path),
    }

def upload_node(state: FusionState) -> Dict[str, Any]:
    """
    Upload raw_fusion.json → raw_fusion/... and semantic_fusion.json → semantic_fusion/...
    Return semantic URI (matching step behavior).
    """
    clip_id = state["clip_id"]
    output_bucket = state["output_bucket"]
    now = datetime.datetime.now()

    # Upload raw fusion
    raw_path = Path(state["raw_fusion_path"])
    raw_object_name = (
        f"raw_fusion/"
        f"{now.strftime('%Y/%m/%d/%H')}/"
        f"{clip_id}.json"
    )
    upload_output(
        bucket=output_bucket,
        object_name=raw_object_name,
        file_path=raw_path,
    )
    logger.info(f"[FUSION_AGENT] Raw fusion uploaded → {raw_object_name}")

    # Upload semantic fusion
    semantic_path = Path(state["semantic_fusion_path"])
    semantic_object_name = (
        f"semantic_fusion/"
        f"{now.strftime('%Y/%m/%d/%H')}/"
        f"semantic_fusion_{clip_id}.json"
    )
    upload_output(
        bucket=output_bucket,
        object_name=semantic_object_name,
        file_path=semantic_path,
    )
    logger.info(f"[FUSION_AGENT] Semantic fusion uploaded → {semantic_object_name}")

    semantic_uri = f"minio://{output_bucket}/{semantic_object_name}"
    return {"fusion_uri": semantic_uri}


def cleanup_node(state: FusionState) -> Dict[str, Any]:
    """
    Delete local temp files.
    """
    for key in ("local_klv_path", "local_det_path", "raw_fusion_path", "semantic_fusion_path"):
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
