from dotenv import load_dotenv
load_dotenv()

import datetime
from zenml import pipeline, step

from utils.logger import get_logger
from .swarm_graph import swarm_graph
from .swarm_state import SwarmState

logger = get_logger(__name__)


# -----------------------------
# ZenML Step
# -----------------------------
@step(enable_cache=False)
def invoke_langgraph(event: dict) -> dict:
    """
    Runs the LangGraph swarm inside a ZenML step.
    This allows nested ZenML steps inside graph nodes
    to appear in the ZenML UI.
    """

    clip_id = event["clip_id"]

    logger.info(f"[ZENML] Running swarm for clip_id={clip_id}")

    # Prepare initial state
    initial_state = SwarmState(
        clip_id=clip_id,
        clip_uri=event["clip_uri"],
    )

    # Run LangGraph
    final_state_obj = swarm_graph.invoke(
        initial_state.model_dump(),
        config={"run_name": f"swarm-{clip_id}"}
    )

    final_state = SwarmState.model_validate(final_state_obj)

    geo_coords = final_state.geo_coordinates or {}

    # Final event output
    output_event = {
        "clip_id": final_state.clip_id,
        "clip_uri": final_state.clip_uri,
        "klv_extraction_uri": final_state.klv_extraction_uri,
        "klv_decoding_uri": final_state.klv_decoding_uri,
        "object_detection_uri": final_state.det_json_uri,
        "fusion_uri": final_state.fusion_uri,
        "summary_uri": final_state.summary_uri,
        "start_latitude": geo_coords.get("start_latitude"),
        "start_longitude": geo_coords.get("start_longitude"),
        "end_latitude": geo_coords.get("end_latitude"),
        "end_longitude": geo_coords.get("end_longitude"),
        "severity_score": final_state.severity_score,
        "severity_label": final_state.severity_label,
        "status": "success",
        "processed_at": datetime.datetime.now().isoformat(),
    }

    return output_event


# -----------------------------
# Dynamic Pipeline Creator
# -----------------------------
def create_swarm_pipeline(pipeline_name: str):
    """
    Dynamically creates a ZenML pipeline with the given name.
    """

    @pipeline(enable_cache=False, name=pipeline_name)
    def swarm_pipeline(event: dict):
        return invoke_langgraph(event)

    return swarm_pipeline


# -----------------------------
# Swarm Client
# -----------------------------
class SwarmClient:
    """
    Client used by Kafka consumer to run the swarm.
    """

    def run_pipeline(self, event: dict, pipeline_name: str) -> dict:

        clip_id = event["clip_id"]

        logger.info(
            f"[SWARM_CLIENT] Triggering swarm pipeline '{pipeline_name}' for {clip_id}"
        )

        try:

            # Create pipeline dynamically
            swarm_pipeline = create_swarm_pipeline(pipeline_name)

            # Run pipeline EXACTLY like old project
            run = swarm_pipeline(event=event)

            # Extract step output
            output_event = run.steps["invoke_langgraph"].output.load()

            logger.info(f"[SWARM_CLIENT] Successfully processed {clip_id}")

            return output_event

        except Exception as e:

            logger.error(
                f"[SWARM_CLIENT] Error processing clip {clip_id}: {e}",
                exc_info=True,
            )

            return {
                "clip_id": clip_id,
                "clip_uri": event["clip_uri"],
                "status": "error",
                "error": str(e),
                "processed_at": datetime.datetime.now().isoformat(),
            }