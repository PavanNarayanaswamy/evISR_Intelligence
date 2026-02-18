# agents/detection/agent.py
from langgraph.graph import StateGraph, START, END  # [web:13]
from utils.logger import get_logger

from zenml_pipeline.minio_utils import get_minio_client
from cv_models.rf_detr import RFDetrDetector
from tracker.norfair_tracker import NorfairTrackerAnnotator
from object_detection.object_tracker import ObjectTracker

from .state import DetectionState

logger = get_logger(__name__)


def _require_tracker(state: DetectionState) -> ObjectTracker:
    """Small helper so failures are obvious and logged."""
    tracker = state.get("_tracker")  # type: ignore
    if tracker is None:
        raise KeyError(
            "Detection agent internal error: '_tracker' missing. "
            "Ensure graph executes build -> start and build returns {'_tracker': ...}."
        )
    return tracker


def build_components_node(state: DetectionState) -> dict:
    """
    Build detector + tracker + ObjectTracker and return them as a state update.
    Important: return an update dict instead of mutating state in-place. [web:50]
    """
    clip_id = state["clip_id"]
    logger.info(f"[DETECTION_AGENT] Building components clip_id={clip_id}")

    detector = RFDetrDetector(state["confidence_threshold"])
    tracker_algo = NorfairTrackerAnnotator(
        distance_function=state["distance_function"],
        distance_threshold=state["distance_threshold"],
        hit_counter_max=state["hit_counter_max"],
        initialization_delay=state["initialization_delay"],
    )

    ot = ObjectTracker(
        clip_id=clip_id,
        minio_client=get_minio_client(),
        video_path=state["ts_path"],
        detector=detector,
        tracker=tracker_algo,
        output_bucket_detection=state["output_bucket_detection"],
        output_path=state["output_path"],
    )

    logger.debug(f"[DETECTION_AGENT] Components ready clip_id={clip_id}")
    return {"_tracker": ot}  # type: ignore


def start_node(state: DetectionState) -> dict:
    """Open video + optionally init mp4 writer."""
    ot = _require_tracker(state)
    clip_id = state["clip_id"]

    logger.info(f"[DETECTION_AGENT] Starting clip_id={clip_id} path={state['ts_path']}")
    ot.start(enable_mp4=bool(state.get("save_mp4", False)))

    # Return small debug updates (optional)
    meta = {}
    if getattr(ot, "fps", None) is not None:
        meta["fps"] = ot.fps
    if getattr(ot, "width", None) is not None:
        meta["width"] = ot.width
    if getattr(ot, "height", None) is not None:
        meta["height"] = ot.height

    if meta:
        logger.debug(f"[DETECTION_AGENT] Video meta clip_id={clip_id} meta={meta}")
    return meta


def process_node(state: DetectionState) -> dict:
    """Run detect+track over frames."""
    ot = _require_tracker(state)
    logger.info(f"[DETECTION_AGENT] Processing frames clip_id={state['clip_id']}")
    ot.process()
    return {}


def write_outputs_node(state: DetectionState) -> dict:
    """Write JSON and upload to MinIO."""
    ot = _require_tracker(state)
    clip_id = state["clip_id"]

    logger.info(f"[DETECTION_AGENT] Writing outputs clip_id={clip_id}")
    det_json_uri, fps, metrics = ot.write_outputs()
    logger.info(
        f"[DETECTION_AGENT] Uploaded detection json clip_id={clip_id} "
        f"uri={det_json_uri} fps={fps}"
    )

    return {
        "det_json_uri": det_json_uri,
        "fps": fps,
        "metrics": metrics,
    }


def cleanup_node(state: DetectionState) -> dict:
    """Always release resources."""
    ot = _require_tracker(state)
    logger.debug(f"[DETECTION_AGENT] Cleanup clip_id={state['clip_id']}")
    ot.cleanup()
    return {"_tracker": None}  # type: ignore


def build_detection_graph():
    """
    Linear graph: build -> start -> process -> write -> cleanup.
    """
    g = StateGraph(DetectionState)  # [web:13]
    g.add_node("build", build_components_node)
    g.add_node("start", start_node)
    g.add_node("process", process_node)
    g.add_node("write", write_outputs_node)
    g.add_node("cleanup", cleanup_node)

    g.add_edge(START, "build")  # [web:13]
    g.add_edge("build", "start")
    g.add_edge("start", "process")
    g.add_edge("process", "write")
    g.add_edge("write", "cleanup")
    g.add_edge("cleanup", END)

    return g.compile()  # [web:13]


detection_graph = build_detection_graph()
