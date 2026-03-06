# agents/mcp_server.py
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Tuple
import datetime
import logging

# Import agent graphs and states
from agents.klv.agent import klv_graph
from agents.klv.state import KLVState
from agents.detection.agent import detection_graph
from agents.detection.state import DetectionState
from agents.fusion.agent import fusion_graph
from agents.fusion.state import FusionState
from agents.summary.agent import llm_summary_graph
from agents.summary.state import SummaryState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="evISR Multi-Agent Control Plane (MCP)",
    description="A central server for orchestrating ISR agent swarm.",
    version="0.1.0",
)

@app.get("/")
def read_root():
    return {"message": "MCP server is running"}

# --- KLV Tool ---
@app.post("/tools/klv", response_model=KLVState)
async def run_klv_agent(state: KLVState):
    """
    Run the KLV extraction and decoding agent.
    """
    logger.info(f"Received KLV request for clip_id: {state.clip_id}")
    try:
        # The input state is already a valid KLVState object thanks to FastAPI
        raw_state = klv_graph.invoke(state.dict())
        final_state = KLVState.model_validate(raw_state)
        if not final_state.is_complete:
            raise HTTPException(status_code=500, detail="KLV agent failed to complete")
        return final_state
    except Exception as e:
        logger.error(f"KLV agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Detection Tool ---
@app.post("/tools/detection", response_model=DetectionState)
async def run_detection_agent(state: DetectionState):
    """
    Run the object detection and tracking agent.
    """
    logger.info(f"Received detection request for clip_id: {state.clip_id}")
    try:
        raw_state = detection_graph.invoke(state.dict())
        final_state = DetectionState.model_validate(raw_state)
        if not final_state.is_complete:
            raise HTTPException(status_code=500, detail="Detection agent failed to complete")
        return final_state
    except Exception as e:
        logger.error(f"Detection agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Fusion Tool ---
@app.post("/tools/fusion", response_model=FusionState)
async def run_fusion_agent(state: FusionState):
    """
    Run the temporal and semantic fusion agent.
    """
    logger.info(f"Received fusion request for clip_id: {state.clip_id}")
    try:
        raw_state = fusion_graph.invoke(state.dict())
        final_state = FusionState.model_validate(raw_state)
        if not final_state.is_complete:
            raise HTTPException(status_code=500, detail="Fusion agent failed to complete")
        return final_state
    except Exception as e:
        logger.error(f"Fusion agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Summary Tool ---
@app.post("/tools/summary", response_model=SummaryState)
async def run_summary_agent(state: SummaryState):
    """
    Run the LLM video summarization agent.
    """
    logger.info(f"Received summary request for clip_id: {state.clip_id}")
    try:
        raw_state = llm_summary_graph.invoke(state.dict())
        final_state = SummaryState.model_validate(raw_state)
        if not final_state.is_complete:
            raise HTTPException(status_code=500, detail="Summary agent failed to complete")
        return final_state
    except Exception as e:
        logger.error(f"Summary agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
