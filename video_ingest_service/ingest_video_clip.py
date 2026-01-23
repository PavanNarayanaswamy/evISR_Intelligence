import os
import subprocess
import json
from minio import Minio
from utils import config
import datetime
from utils.logger import get_logger

logger = get_logger(__name__)


def get_video_duration(video_file):
    """
    Returns total duration of video in seconds (int)
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    duration = float(json.loads(result.stdout)["format"]["duration"])
    return int(duration)



# Setup
os.makedirs(config.VIDEO_CLIP_TMP_DIR, exist_ok=True)

minio_client = Minio(
    config.MINIO_ENDPOINT,
    access_key=config.MINIO_ACCESS_KEY,
    secret_key=config.MINIO_SECRET_KEY,
    secure=config.MINIO_SECURE
)

# Ensure bucket exists
if not minio_client.bucket_exists(config.VIDEO_CLIP_BUCKET):
    minio_client.make_bucket(config.VIDEO_CLIP_BUCKET)
    logger.info(f"Created bucket: {config.VIDEO_CLIP_BUCKET}")
else:
    logger.info(f"Bucket already exists: {config.VIDEO_CLIP_BUCKET}")

logger.info("Chunking video clip into 30-second segments...")

ffmpeg_cmd = [
    "ffmpeg",

    # Input file containing video + KLV
    "-i", config.VIDEO_CLIP_FILE,

    # Explicitly keep video and KLV data streams
    # "-map", "0:v",
    # "-map", "0:d?",
    "-map", "0",

    # Copy streams exactly (no re-encode)
    "-c", "copy",

    # Segment into MPEG-TS files
    "-f", "segment",
    "-segment_time", str(config.VIDEO_CLIP_SEGMENT_SECONDS),
    "-segment_format", "mpegts",

    # 🔑 CRITICAL: keep continuous timestamps
    "-reset_timestamps", "1",

    # Output pattern
    f"{config.VIDEO_CLIP_TMP_DIR}/segment_%03d.ts"
]




try:
    subprocess.run(ffmpeg_cmd, check=True)
except Exception as e:
    logger.error(f"FFmpeg chunking failed: {e}")
    raise

# Upload with Accurate Time-Based Naming


total_duration = get_video_duration(config.VIDEO_CLIP_FILE)
logger.info(f"Detected video duration: {total_duration} seconds")

start_time = 0

for file in sorted(os.listdir(config.VIDEO_CLIP_TMP_DIR)):
    if not file.endswith(".ts"):
        logger.debug(f"Skipping non-ts file: {file}")
        continue

    # Clamp end_time to actual video duration
    end_time = min(
        start_time + config.VIDEO_CLIP_SEGMENT_SECONDS,
        total_duration
    )

    object_name = (
        f"{config.VIDEO_CLIP_STREAM_ID}/"
        f"{config.VIDEO_CLIP_STREAM_ID}_"
        f"{start_time:06d}_{end_time:06d}.ts"
    )
    
    # ADDED: directory time logic (UTC, 1-hour lap)
    ingest_time = datetime.datetime.utcnow()
    year = ingest_time.strftime("%Y")
    month = ingest_time.strftime("%m")
    day = ingest_time.strftime("%d")
    hour = ingest_time.strftime("%H")

    object_name = (
        f"{config.VIDEO_CLIP_STREAM_ID}/"
        f"{year}/{month}/{day}/{hour}/"
        f"{config.VIDEO_CLIP_STREAM_ID}_"
        f"{start_time:06d}_{end_time:06d}.ts"
    )

    file_path = os.path.join(config.VIDEO_CLIP_TMP_DIR, file)


    try:
        minio_client.fput_object(
            config.VIDEO_CLIP_BUCKET,
            object_name,
            file_path
        )
        logger.info(f"Uploaded: {object_name}")
    except Exception as e:
        logger.error(f"Failed to upload {file_path} to MinIO as {object_name}: {e}")
        continue

    # Move window forward
    start_time = end_time

    # Stop if we've reached the end of the video
    if start_time >= total_duration:
        break

# Cleanup

for f in os.listdir(config.VIDEO_CLIP_TMP_DIR):
    try:
        os.remove(os.path.join(config.VIDEO_CLIP_TMP_DIR, f))
    except Exception as e:
        logger.error(f"Failed to remove file {f}: {e}")

logger.info("✅ Video clip ingestion completed successfully")
