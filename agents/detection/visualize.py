# agents/detection/visualize.py
from .agent import build_detection_graph
from agents.visualize import visualize_agent_graph


def visualize_detection_agent(output_bucket: str, clip_id: str = "demo"):
    graph = build_detection_graph()
    return visualize_agent_graph(
        graph=graph,
        output_bucket=output_bucket,
        agent_name="detection_agent",
        clip_id=clip_id,
        title="Object Detection Agent Graph",
        node_labels={
            "build": "Build Tracker",
            "start": "Start Video", 
            "process": "Detect+Track",
            "write": "Write JSON",
            "cleanup": "Cleanup"
        }
    )
