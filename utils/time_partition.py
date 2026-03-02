from datetime import datetime
import re

TS_PATTERN = re.compile(r"(\d{8}_\d{6})")

def extract_event_datetime_from_clip_id(clip_id: str) -> datetime:
    """
    Extract datetime from clip_id.
    Example:
        port-5000_20260218_121630
        → 2026-02-18 12:16:30
    """
    match = TS_PATTERN.search(clip_id)
    if not match:
        raise ValueError(f"Cannot extract timestamp from clip_id: {clip_id}")

    return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")


def build_partition_path_from_clip_id(clip_id: str) -> str:
    dt = extract_event_datetime_from_clip_id(clip_id)
    return dt.strftime("%Y/%m/%d/%H")