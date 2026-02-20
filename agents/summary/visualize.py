# agents/summary/visualize.py
from .agent import build_summary_graph
from agents.visualize import visualize_agent_graph


def visualize_summary_agent(output_bucket: str, clip_id: str = "demo"):
    graph = build_summary_graph()
    return visualize_agent_graph(
        graph=graph,
        output_bucket=output_bucket,
        agent_name="summary_agent",
        clip_id=clip_id,
        title="Summary Agent Graph",
        node_labels={
            "download_and_load": "Download Fusion JSON",
            "run_llm": "Generate Summary", 
            "upload": "Upload Summary",
            "cleanup": "Cleanup Files"
        }
    )
