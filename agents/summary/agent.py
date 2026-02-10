# agents/summary/agent.py
import datetime
import json
import os
from pathlib import Path
from typing import Dict, Any

from langgraph.graph import StateGraph, START, END  # [web:13]
from utils.logger import get_logger

from zenml_pipeline.minio_utils import download_file, upload_output
from video_summary.summary_gen import VideoLLMSummarizer

from .state import SummaryState

logger = get_logger(__name__)


def download_and_load_node(state: SummaryState) -> Dict[str, Any]:
    """
    Download fusion JSON and load into memory.
    """
    clip_id = state["clip_id"]
    logger.info(f"[SUMMARY_AGENT] Downloading fusion JSON for clip_id={clip_id}")

    local_fusion = Path("/tmp") / f"{clip_id}_fusion.json"
    download_file(state["fusion_json_uri"], local_fusion)
    logger.info(f"[SUMMARY_AGENT] Downloaded fusion JSON to {local_fusion}")

    with open(local_fusion, "r") as f:
        fusion_context = json.load(f)

    return {
        "local_fusion_path": str(local_fusion),
        "fusion_context": fusion_context,
    }


def run_llm_node(state: SummaryState) -> Dict[str, Any]:
    """
    Call VideoLLMSummarizer to generate the summary text.
    """
    clip_id = state["clip_id"]
    model = state["model"]
    ts_path = state["ts_path"]

    logger.info(
        f"[SUMMARY_AGENT] Running LLM summarization clip_id={clip_id} model={model}"
    )

    summary = VideoLLMSummarizer.summarize(
        fusion_context=state["fusion_context"],
        model=model,
        video_path=ts_path,
    )

    summary_path = Path("/tmp") / f"{clip_id}_summary.txt"
    summary_path.write_text(summary)
    logger.info(f"[SUMMARY_AGENT] Wrote summary to {summary_path}")

    return {"summary_path": str(summary_path)}


def upload_node(state: SummaryState) -> Dict[str, Any]:
    """
    Upload summary file to MinIO and return URI.
    """
    clip_id = state["clip_id"]
    output_bucket = state["output_bucket"]
    summary_path = Path(state["summary_path"])

    now = datetime.datetime.now()
    object_name = (
        f"summary/"
        f"{now.strftime('%Y/%m/%d/%H')}/"
        f"{clip_id}.txt"
    )

    logger.info(
        f"[SUMMARY_AGENT] Uploading summary clip_id={clip_id} "
        f"to bucket={output_bucket}, object={object_name}"
    )
    upload_output(
        bucket=output_bucket,
        object_name=object_name,
        file_path=summary_path,
    )

    summary_uri = f"minio://{output_bucket}/{object_name}"
    logger.info(f"[SUMMARY_AGENT] Summary URI: {summary_uri}")
    return {"summary_uri": summary_uri}


def cleanup_node(state: SummaryState) -> Dict[str, Any]:
    """
    Remove local temp files, including the TS copy if it still exists.
    """
    paths = [
        state.get("local_fusion_path"),
        state.get("summary_path"),
        state.get("ts_path"),
    ]

    for p_str in paths:
        if not p_str:
            continue
        p = Path(p_str)
        if p.exists():
            try:
                os.remove(p)
                logger.debug(f"[SUMMARY_AGENT] Removed temp file: {p}")
            except Exception as e:
                logger.warning(
                    f"[SUMMARY_AGENT] Failed to remove {p}: {e}",
                    exc_info=True,
                )
    return {}


def build_summary_graph():
    """
    Linear graph: download+load -> run_llm -> upload -> cleanup -> END. [web:13]
    """
    g = StateGraph(SummaryState)
    g.add_node("download_and_load", download_and_load_node)
    g.add_node("run_llm", run_llm_node)
    g.add_node("upload", upload_node)
    g.add_node("cleanup", cleanup_node)

    g.add_edge(START, "download_and_load")
    g.add_edge("download_and_load", "run_llm")
    g.add_edge("run_llm", "upload")
    g.add_edge("upload", "cleanup")
    g.add_edge("cleanup", END)

    return g.compile()  # [web:13]


llm_summary_graph = build_summary_graph()
