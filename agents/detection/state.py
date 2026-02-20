from datetime import datetime
from typing import Optional, Any, Literal
from pydantic import BaseModel, Field, validator


class DetectionState(BaseModel):
    """Pydantic-validated state for object detection agent."""
    
    # Required inputs
    clip_id: str = Field(..., description="Unique clip identifier")
    ts_path: str = Field(..., description="Local path to TS file")
    output_bucket_detection: str = Field(..., description="MinIO detection bucket")
    output_path: str = Field(..., description="Output path prefix")
    
    # Config
    confidence_threshold: float = Field(..., ge=0.0, le=1.0)
    distance_threshold: int = Field(..., ge=0)
    hit_counter_max: int = Field(..., ge=1)
    initialization_delay: int = Field(..., ge=0)
    distance_function: Literal["euclidean", "iou"] = Field(..., description="Tracker distance")
    save_mp4: bool = Field(default=False)
    
    # Outputs
    det_json_uri: Optional[str] = Field(default=None, description="MinIO detection JSON URI")
    fps: float = Field(default=0.0, ge=0.0)
    
    # Debug info
    width: Optional[int] = Field(default=None, ge=0)
    height: Optional[int] = Field(default=None, ge=0)
    metrics: Optional[dict] = Field(default=None)
    
    # Internal (private)
    tracker: Optional[Any] = Field(default=None, exclude=True)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @validator('clip_id')
    def validate_clip_id(cls, v):
        if not v or len(v) > 100:
            raise ValueError('clip_id must be non-empty and <= 100 chars')
        return v
    
    @property
    def is_complete(self) -> bool:
        return bool(self.det_json_uri and self.fps > 0)
