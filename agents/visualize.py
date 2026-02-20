# agents/visualize.py
"""Common LangGraph visualizer for all agents."""
import tempfile
import datetime
from pathlib import Path
from typing import Dict, Any
from langgraph.graph import StateGraph
from zenml_pipeline.minio_utils import get_minio_client, upload_output
from utils.logger import get_logger


logger = get_logger(__name__)


def visualize_agent_graph(
    graph: StateGraph,
    output_bucket: str,
    agent_name: str,
    clip_id: str = "demo",
    title: str = "Agent Graph",
    node_labels: Dict[str, str] = None 
) -> Dict[str, str]:
    """Generate clean, human-readable Mermaid + PNG."""
    
    client = get_minio_client()
    
    # 1. Get raw Mermaid
    mermaid_raw = graph.get_graph().draw_mermaid()
    
    # 2. Clean: Remove config + improve labels
    lines = mermaid_raw.strip().split('\n')
    
    # Remove config block
    mermaid_lines = []
    for line in lines:
        if line.strip().startswith('---'):
            continue
        # Replace short names → human readable
        if node_labels:
            for short, readable in node_labels.items():
                line = line.replace(f"({short})", f"|{readable}|")
        mermaid_lines.append(line)
    
    mermaid_clean = '\n'.join(mermaid_lines)
    
    # 3. Temp files + PNG
    mermaid_path = Path(tempfile.mktemp(suffix=f"_{agent_name}.mermaid"))
    png_path = Path(tempfile.mktemp(suffix=f"_{agent_name}.png"))
    
    mermaid_path.write_text(mermaid_clean)
    graph.get_graph().draw_png(png_path)
    
    # 4. Upload
    now = datetime.datetime.now()
    prefix = f"graphs/{agent_name}/{now.strftime('%Y/%m/%d/%H')}/{clip_id}_"
    mermaid_object = f"{prefix}{agent_name}.mermaid"
    png_object = f"{prefix}{agent_name}.png"
    
    upload_output(output_bucket, mermaid_object, mermaid_path)
    upload_output(output_bucket, png_object, png_path)
    
    mermaid_uri = f"minio://{output_bucket}/{mermaid_object}"
    png_uri = f"minio://{output_bucket}/{png_object}"
    
    return {
        "mermaid_uri": mermaid_uri,
        "png_uri": png_uri,
        "mermaid_text": mermaid_clean,
        "agent_name": agent_name,
    }
