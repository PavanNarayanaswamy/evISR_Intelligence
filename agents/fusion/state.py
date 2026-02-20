from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator


class FusionState(BaseModel):
    # Inputs
    clip_id: str = Field(..., description="Unique clip identifier")
    video_duration: float = Field(..., ge=0.0)
    klv_json_uri: str = Field(..., description="MinIO KLV JSON URI")
    det_json_uri: str = Field(..., description="MinIO detection JSON URI")
    output_bucket: str = Field(..., description="MinIO output bucket")
    fps: float = Field(..., ge=0.0)
    
    # Local paths
    local_klv_path: Optional[str] = Field(default=None)
    local_det_path: Optional[str] = Field(default=None)
    raw_fusion_path: Optional[str] = Field(default=None)
    semantic_fusion_path: Optional[str] = Field(default=None)
    
    # Parsed JSON
    klv_json: Optional[Dict[str, Any]] = Field(default=None)
    det_json: Optional[Dict[str, Any]] = Field(default=None)
    
    # Intermediate fusion outputs
    raw_fusion: Optional[Dict[str, Any]] = Field(default=None)
    semantic_fusion: Optional[Dict[str, Any]] = Field(default=None)
    
    # Output
    fusion_uri: Optional[str] = Field(default=None, description="Semantic fusion MinIO URI")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @validator('clip_id')
    def validate_clip_id(cls, v): 
        if not v or len(v) > 100: raise ValueError('Invalid clip_id')
        return v
    
    @property
    def is_complete(self) -> bool:
        return bool(self.fusion_uri)
