# agents/summary/state.py
from typing_extensions import TypedDict
from typing import Any, Dict


class SummaryState(TypedDict, total=False):
    # Inputs
    clip_id: str
    ts_path: str
    fusion_json_uri: str
    output_bucket: str
    model: str

    # Local paths
    local_fusion_path: str
    summary_path: str

    # Parsed fusion context
    fusion_context: Dict[str, Any]

    # Output
    summary_uri: str
