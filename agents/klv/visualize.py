# agents/klv/visualize.py
from .agent import build_klv_graph
from agents.visualize import visualize_agent_graph


def visualize_klv_agent(output_bucket: str, clip_id: str = "demo"):
    graph = build_klv_graph()
    return visualize_agent_graph(
        graph=graph,
        output_bucket=output_bucket,
        agent_name="klv_agent",
        clip_id=clip_id,
        title="KLV Agent",
        node_labels={
            "extract": "Extract KLV",
            "decode": "Decode JSON", 
            "cleanup": "Cleanup Temp",
            "skip": "Skip Branch"
        }
    )
