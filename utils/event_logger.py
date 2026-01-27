import json
import os
from utils.logger import get_logger

logger = get_logger(__name__)

def append_event_to_file(event: dict, file_path: str) -> None:
    """
    Append an event to a JSON file (list-based).
    Safe, reusable, and service-agnostic.
    """
    if not os.path.exists(file_path):
        data = []
    else:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        except Exception:
            data = []

    data.append(event)

    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Event appended to {file_path}")
    except Exception as e:
        logger.error(f"Failed to write event log {file_path}: {e}", exc_info=True)
