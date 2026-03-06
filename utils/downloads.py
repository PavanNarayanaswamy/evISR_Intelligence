
from pathlib import Path
import subprocess
from .logger import get_logger
from zenml_pipeline.minio_utils import download_segment

logger = get_logger(__name__)

def download_video_from_uri(clip_id: str, clip_uri: str) -> tuple[str, float]:
    """
    Downloads TS from MinIO as ./<clip_id>.ts
    """
    ts_path = Path.cwd() / f"{clip_id}.ts"
    logger.info(f"Downloading clip for clip_id: {clip_id} from {clip_uri}")
    download_segment(clip_uri, ts_path)
    logger.info(f"Downloaded clip to {ts_path}")
    result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                ts_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
    return str(ts_path), float(result.stdout.strip())
