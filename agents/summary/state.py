from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class SummaryState(BaseModel):
    # Inputs
    clip_id: str = Field(..., description="Unique clip identifier")
    ts_path: str = Field(..., description="Local TS path")
    fusion_json_uri: str = Field(..., description="MinIO fusion JSON URI")
    output_bucket: str = Field(..., description="MinIO output bucket")
    model: str = Field(default="qwen3-vl:30b", description="LLM model")
    
    # Local paths
    local_fusion_path: Optional[str] = Field(default=None)
    summary_path: Optional[str] = Field(default=None)
    
    # Parsed data
    fusion_context: Optional[Dict[str, Any]] = Field(default=None)
    
    # Output
    summary_uri: Optional[str] = Field(default=None, description="MinIO summary URI")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @validator('clip_id')
    def validate_clip_id(cls, v):
        if not v or len(v) > 100:
            raise ValueError('Invalid clip_id')
        return v
    
    @property
    def is_complete(self) -> bool:
        return bool(self.summary_uri)
