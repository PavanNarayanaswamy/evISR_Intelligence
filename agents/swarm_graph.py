# agents/swarm_graph.py
from langgraph.graph import StateGraph, END
from langfuse import observe
from typing import Dict, Any
import os
from utils.logger import get_logger
from kafka_consumer import consumer_config as config

# Import the state model for the graph
from .swarm_state import SwarmState

logger = get_logger(__name__)

@observe()  # Langfuse tracing for the graph
def download_clip_node(state: SwarmState) -> Dict[str, Any]:
    """Downloads the video clip by calling the ZenML step function."""
    from zenml_pipeline.steps import minio_segmented_clip
    
    logger.info(f"[SWARM] Downloading clip: {state.clip_id}")
    # This step function returns: ts_path, video_duration
    ts_path, video_duration = minio_segmented_clip(
        clip_id=state.clip_id,
        clip_uri=state.clip_uri
    )
    return {"ts_path": ts_path, "video_duration": video_duration}

@observe()  # Langfuse tracing for the node
def klv_tool_node(state: SwarmState) -> Dict[str, Any]:
    """Calls the KLV extraction step function."""
    from zenml_pipeline.steps import klv_extraction_agent
    
    logger.info(f"[SWARM] Calling KLV tool for clip: {state.clip_id}")
    # This step function returns: klv_extraction_uri, klv_decoding_uri
    klv_extraction_uri, klv_decoding_uri = klv_extraction_agent(
        ts_path=state.ts_path,
        clip_id=state.clip_id,
        jars=[
            "jars/jmisb-api-1.12.0.jar",
            "jars/jmisb-core-common-1.12.0.jar",
            "jars/slf4j-api-1.7.36.jar",
            "jars/slf4j-simple-1.7.36.jar",
        ],
        output_bucket=config.OUTPUT_BUCKET,
    )
    return {
        "klv_extraction_uri": klv_extraction_uri,
        "klv_decoding_uri": klv_decoding_uri,
    }
@observe()  # Langfuse tracing for the node
def detection_tool_node(state: SwarmState) -> Dict[str, Any]:
    """Calls the object detection step function."""
    from zenml_pipeline.steps import object_detection_agent
    
    logger.info(f"[SWARM] Calling detection tool for clip: {state.clip_id}")
    # This step function returns: detection_uri, fps
    det_json_uri, fps = object_detection_agent(
        clip_id=state.clip_id,
        ts_path=state.ts_path,
        output_bucket_detection=config.OUTPUT_BUCKET_DETECTION,
        output_path=config.OUTPUT_PATH,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        distance_threshold=config.DISTANCE_THRESHOLD,
        hit_counter_max=config.HIT_COUNTER_MAX,
        initialization_delay=config.INITIALIZATION_DELAY,
        distance_function=config.DISTANCE_FUNCTION,
    )
    return {
        "det_json_uri": det_json_uri,
        "fps": fps
    }
@observe()  # Langfuse tracing for the node
def fusion_tool_node(state: SwarmState) -> Dict[str, Any]:
    """Calls the fusion context step function."""
    from zenml_pipeline.steps import fusion_context_agent
    
    logger.info(f"[SWARM] Calling fusion tool for clip: {state.clip_id}")
    # This step function returns: fusion_uri, geo_coordinates
    fusion_uri, geo_coordinates = fusion_context_agent(
        clip_id=state.clip_id,
        video_duration=state.video_duration,
        klv_json_uri=state.klv_decoding_uri,
        det_json_uri=state.det_json_uri,
        output_bucket=config.OUTPUT_BUCKET_FUSION,
        fps=state.fps,
    )
    return {"fusion_uri": fusion_uri, "geo_coordinates": geo_coordinates}
@observe()  # Langfuse tracing for the node
def summary_tool_node(state: SwarmState) -> Dict[str, Any]:
    """Calls the LLM summary step function."""
    from zenml_pipeline.steps import llm_summary_agent
    
    logger.info(f"[SWARM] Calling summary tool for clip: {state.clip_id}")
    # This step function returns: summary_uri, severity_score, severity_label
    summary_uri, severity_score, severity_label = llm_summary_agent(
        clip_id=state.clip_id,
        ts_path=state.ts_path,
        fusion_json_uri=state.fusion_uri,
        output_bucket=config.OUTPUT_BUCKET_SUMMARY,
        model="qwen3-vl:30b",  # This should ideally be in config
    )
    return {
        "summary_uri": summary_uri,
        "severity_score": severity_score,
        "severity_label": severity_label,
    }

# --- Graph Definition ---

def create_swarm_graph():
    """Builds the main agentic swarm graph."""
    workflow = StateGraph(SwarmState)

    workflow.add_node("download_clip", download_clip_node)
    workflow.add_node("klv", klv_tool_node)
    workflow.add_node("detection", detection_tool_node)
    workflow.add_node("fusion", fusion_tool_node)
    workflow.add_node("summary", summary_tool_node)

    workflow.set_entry_point("download_clip")
    
    # This defines the sequence of operations.
    workflow.add_edge("download_clip", "klv")
    workflow.add_edge("klv", "detection")
    workflow.add_edge("detection", "fusion")
    workflow.add_edge("fusion", "summary")
    workflow.add_edge("summary", END)

    return workflow.compile()

# A single compiled instance to be used by the client.
swarm_graph = create_swarm_graph()
