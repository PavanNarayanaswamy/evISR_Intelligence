# agents/klv/agent.py
import datetime
import json
from pathlib import Path
from typing import Literal

from langgraph.graph import StateGraph, START, END
from utils.logger import get_logger

from klv_metadata_extraction.extraction import Extraction
from klv_metadata_extraction.decoding import JmisbDecoder
from zenml_pipeline.minio_utils import download_file, upload_output, get_minio_client

from .state import KLVState

logger = get_logger(__name__)


def extract_node(state: KLVState) -> KLVState:
    logger.info(f"[KLV_AGENT] Extracting KLV for clip_id={state['clip_id']}")
    extractor = Extraction(
        minio_client=get_minio_client(),
        output_bucket=state["output_bucket"],
    )
    extraction_uri = extractor.extract_klv(
        ts_path=state["ts_path"],
        clip_id=state["clip_id"],
    )
    state["extraction_uri"] = extraction_uri
    return state


def decode_node(state: KLVState) -> KLVState:
    clip_id = state["clip_id"]
    output_bucket = state["output_bucket"]

    local_klv = Path("/tmp") / f"{clip_id}.klv"
    output_json = Path("/tmp") / f"{clip_id}.json"

    logger.info(f"[KLV_AGENT] Downloading KLV for clip_id={clip_id}")
    download_file(state["extraction_uri"], local_klv)

    logger.info(f"[KLV_AGENT] Decoding KLV for clip_id={clip_id}")
    decoder = JmisbDecoder(state["jars"])
    decoder.start_jvm()
    decoded = decoder.decode_file(str(local_klv))

    with open(output_json, "w") as f:
        json.dump(decoded, f, indent=2)

    now = datetime.datetime.now()
    object_name = f"decoding/{now.strftime('%Y/%m/%d/%H')}/klv_decoded_{clip_id}.json"

    logger.info(f"[KLV_AGENT] Uploading decoded JSON for clip_id={clip_id}")
    upload_output(output_bucket, object_name, output_json)

    state["decoding_uri"] = f"minio://{output_bucket}/{object_name}"
    return state


def cleanup_node(state: KLVState) -> KLVState:
    # IMPORTANT: do NOT delete ts_path here; object detection still needs it.
    clip_id = state["clip_id"]
    for p in [Path("/tmp") / f"{clip_id}.klv", Path("/tmp") / f"{clip_id}.json"]:
        try:
            if p.exists():
                p.unlink()
                logger.debug(f"[KLV_AGENT] Removed temp file: {p}")
        except Exception as e:
            logger.warning(f"[KLV_AGENT] Cleanup failed for {p}: {e}")
    return state


def emit_gate(state: KLVState) -> Literal["emit", "skip"]:
    """
    Routing decision.
    For now: always emit, but keep this hook for delta filtering later.
    """
    # Example future logic:
    # if state.get("emit") is False: return "skip"
    return "emit"


def skip_node(state: KLVState) -> KLVState:
    # If you ever skip, you can still return extraction_uri but omit decoding_uri.
    logger.info(f"[KLV_AGENT] Skipping decoded output for clip_id={state['clip_id']}")
    state["decoding_uri"] = None  # type: ignore
    return state


def build_klv_graph():
    g = StateGraph(KLVState)
    g.add_node("extract", extract_node)
    g.add_node("decode", decode_node)
    g.add_node("cleanup", cleanup_node)
    g.add_node("skip", skip_node)

    g.add_edge(START, "extract")
    g.add_edge("extract", "decode")
    g.add_edge("decode", "cleanup")

    # Conditional edges are the LangGraph-native way to route based on state.
    g.add_conditional_edges("cleanup", emit_gate, {"emit": END, "skip": "skip"})
    g.add_edge("skip", END)

    return g.compile()


klv_graph = build_klv_graph()
