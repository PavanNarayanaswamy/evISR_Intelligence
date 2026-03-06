# agents/swarm_client.py
import datetime
from utils.logger import get_logger
from .swarm_graph import swarm_graph
from .swarm_state import SwarmState

logger = get_logger(__name__)

class SwarmClient:
    """
    A client that uses the agentic swarm graph to process video clips.
    This is the main entry point called by the Kafka consumer.
    """

    def run_pipeline(self, event: dict, pipeline_name: str) -> dict:
        """
        Triggers the swarm graph and returns the consolidated result.
        The `pipeline_name` argument is kept for compatibility with the consumer, but is unused.
        """
        clip_id = event["clip_id"]
        logger.info(f"[SWARM_CLIENT] Starting swarm for clip_id: {clip_id}")

        try:
            # 1. Prepare the initial state for the graph
            initial_state = SwarmState(
                clip_id=clip_id,
                clip_uri=event["clip_uri"],
            )

            # 2. Invoke the swarm graph
            final_state_obj = swarm_graph.invoke(initial_state.model_dump())
            final_state = SwarmState.model_validate(final_state_obj)

            # 3. Format the successful output event
            geo_coords = final_state.geo_coordinates or {}
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
            logger.info(f"[SWARM_CLIENT] Successfully processed clip {clip_id}")
            return output_event

        except Exception as e:
            logger.error(f"[SWARM_CLIENT] Error processing clip {clip_id}: {e}", exc_info=True)
            
            # 4. Format the error output event
            error_event = {
                "clip_id": clip_id,
                "clip_uri": event["clip_uri"],
                "status": "error",
                "error": str(e),
                "processed_at": datetime.datetime.now().isoformat(),
            }
            return error_event
