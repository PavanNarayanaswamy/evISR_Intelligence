from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class SwarmState(BaseModel):
    """
    State for the main swarm agent.
    It orchestrates the calls to the other agents.
    """
    clip_id: str
    clip_uri: str
    
    # Downloaded video path
    ts_path: Optional[str] = None
    video_duration: Optional[float] = None

    # KLV agent output
    klv_extraction_uri: Optional[str] = None
    klv_decoding_uri: Optional[str] = None
    
    # Detection agent output
    det_json_uri: Optional[str] = None
    fps: Optional[float] = None
    
    # Fusion agent output
    fusion_uri: Optional[str] = None
    geo_coordinates: Optional[Dict[str, float]] = None

    # Summary agent output
    summary_uri: Optional[str] = None
    severity_score: Optional[float] = None
    severity_label: Optional[str] = None

    # Final output data for kafka
    final_output: Optional[Dict[str, Any]] = None
