# agents/klv/state.py
from datetime import datetime
from typing import List, Optional  # ✅ Use Optional[str] instead
from pydantic import BaseModel, Field, validator


class KLVState(BaseModel):
    """Pydantic-validated state for KLV extraction agent."""
    
    # Required inputs (from ZenML step)
    clip_id: str = Field(..., description="Unique clip identifier")
    ts_path: str = Field(..., description="Local path to TS file")
    output_bucket: str = Field(..., description="MinIO output bucket")
    jars: List[str] = Field(..., description="JVM JAR paths for JmisbDecoder")
    
    # Outputs (set by nodes)
    extraction_uri: str = Field(default=None, description="MinIO URI for extracted KLV")
    decoding_uri: str = Field(default=None, description="MinIO URI for decoded KLV JSON")
    
    # Control/debugging
    emit: bool = Field(default=True, description="Whether to emit decoding_uri")
    error: Optional[str] = Field(default=None, description="Error message if any")  # ✅ Fixed
    
    # Timestamps (auto-set)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @validator('clip_id')
    def validate_clip_id(cls, v):
        if not v or len(v) > 100:
            raise ValueError('clip_id must be non-empty and <= 100 chars')
        return v
    
    @validator('ts_path')
    def validate_ts_path(cls, v):
        if not v.startswith('/'):
            raise ValueError('ts_path must be absolute local path')
        return v
    
    @property
    def is_complete(self) -> bool:
        """Validation for agent completion."""
        return bool(self.extraction_uri and self.decoding_uri)
