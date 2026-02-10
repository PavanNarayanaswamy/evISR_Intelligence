# agents/fusion/state.py
from typing_extensions import TypedDict
from typing import Any, Dict


class FusionState(TypedDict, total=False):
    # Inputs
    clip_id: str
    video_duration: float
    klv_json_uri: str
    det_json_uri: str
    output_bucket: str

    # Local paths (for debug / cleanup)
    local_klv_path: str
    local_det_path: str
    local_fusion_path: str

    # Parsed JSON
    klv_json: Dict[str, Any]
    det_json: Dict[str, Any]

    # Outputs
    fusion_uri: str
