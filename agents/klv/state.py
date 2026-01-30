# agents/klv/state.py
from typing_extensions import TypedDict

class KLVState(TypedDict, total=False):
    clip_id: str
    ts_path: str
    output_bucket: str
    jars: list[str]

    # Outputs
    extraction_uri: str
    decoding_uri: str

    # Control / debugging
    emit: bool
    error: str
