# ingest_video_streaming.py
import os
import re
import subprocess
import time
import datetime

from utils import config
from minio_client import MinIOClient
from utils.logger import get_logger

logger = get_logger(__name__)

os.makedirs(config.LOCAL_BUFFER_DIR, exist_ok=True)
TS_NAME_RE = re.compile(r"^\d{8}_\d{6}\.ts$")

def is_file_complete(path, wait=1):
    size1 = os.path.getsize(path)
    time.sleep(wait)
    size2 = os.path.getsize(path)
    return size1 == size2

def main():
    minio_client = MinIOClient(
        config.MINIO_ENDPOINT,
        config.MINIO_ACCESS_KEY,
        config.MINIO_SECRET_KEY,
        config.MINIO_SECURE
    )
    minio_client.ensure_bucket(config.STREAMING_MINIO_BUCKET)

    # Output filename pattern with timestamps
    output_pattern = os.path.join(config.LOCAL_BUFFER_DIR, "%Y%m%d_%H%M%S.ts")

    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", "udp://127.0.0.1:5000?fifo_size=5000000&overrun_nonfatal=1",
        "-map", "0",                   # Keep all streams (video + KLV)
        "-c", "copy",                  # No re-encoding
        "-f", "segment",
        "-segment_time", str(config.ROLLING_SECONDS),
        "-segment_format", "mpegts",
        "-segment_format_options", "mpegts_flags=+resend_headers",
        "-segment_atclocktime", "1",   # Align segments to real-time
        "-strftime", "1",              # Timestamped filenames
        output_pattern
    ]

    logger.info("Starting UDP KLV Video Ingest Service...")
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    try:
        while True:
            for file in os.listdir(config.LOCAL_BUFFER_DIR):
                if not file.endswith(".ts"):
                    logger.debug(f"Skipping non-ts file: {file}")
                    continue

                if not TS_NAME_RE.match(file):
                    logger.debug(f"Skipping file not matching pattern: {file}")
                    continue

                file_path = os.path.join(config.LOCAL_BUFFER_DIR, file)

                if not is_file_complete(file_path):
                    logger.debug(f"File not complete yet: {file}")
                    continue

                # Parse timestamp from filename
                segment_time = datetime.datetime.strptime(file.replace(".ts", ""), "%Y%m%d_%H%M%S")
                # Adjust IST if needed
                ist_time = segment_time + datetime.timedelta(hours=5, minutes=30)

                object_name = (
                    f"{config.STREAMING_STREAM_ID}/"
                    f"{ist_time.strftime('%Y/%m/%d/%H')}/"
                    f"{file}"
                )

                try:
                    minio_client.upload(
                        config.STREAMING_MINIO_BUCKET,
                        object_name,
                        file_path
                    )
                    logger.info(f"Uploaded to MinIO: {object_name}")
                except Exception as e:
                    logger.error(f"Failed to upload {file_path} to MinIO as {object_name}: {e}", exc_info=True)
                    continue

                try:
                    os.remove(file_path)
                    logger.debug(f"Removed local file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove file {file_path}: {e}", exc_info=True)
            time.sleep(2)

    except KeyboardInterrupt:
        logger.info("Stopping ingest service...")
        process.terminate()
    except Exception as e:
        logger.error(f"Unexpected error in ingest service: {e}", exc_info=True)
        process.terminate()

if __name__ == "__main__":
    main()