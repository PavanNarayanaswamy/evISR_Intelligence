# agents/fusion/visualize.py
from .agent import build_fusion_graph
from agents.visualize import visualize_agent_graph


def visualize_fusion_agent(output_bucket: str, clip_id: str = "demo"):
    graph = build_fusion_graph()
    return visualize_agent_graph(
        graph=graph,
        output_bucket=output_bucket,
        agent_name="fusion_agent",
        clip_id=clip_id,
        title="Fusion Agent Graph",
        node_labels={
            "download_inputs": "Download JSONs",
            "load_and_fuse": "Raw+Semantic Fusion", 
            "upload": "Upload Both",
            "cleanup": "Cleanup"
        }
    )
