# agents/detection/state.py
from typing_extensions import TypedDict
from typing import Any

class DetectionState(TypedDict, total=False):
    # Inputs
    clip_id: str
    ts_path: str
    output_bucket_detection: str
    output_path: str

    # Config
    confidence_threshold: float
    distance_threshold: int
    hit_counter_max: int
    initialization_delay: int
    distance_function: str
    save_mp4: bool

    # Outputs
    det_json_uri: str

    # Internal debug info
    fps: float
    width: int
    height: int
    
    _tracker: Any
