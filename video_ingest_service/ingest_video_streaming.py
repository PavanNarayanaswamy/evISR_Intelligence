# video_ingest_service/ingest_video_streaming.py
import os
import subprocess
import time
import datetime

from config import *
from minio_client import MinIOClient

os.makedirs(LOCAL_BUFFER_DIR, exist_ok=True)

def main():
    minio_client = MinIOClient(
        MINIO_ENDPOINT,
        MINIO_ACCESS_KEY,
        MINIO_SECRET_KEY,
        MINIO_SECURE
    )
    minio_client.ensure_bucket(STREAMING_MINIO_BUCKET)

    output_pattern = os.path.join(
        LOCAL_BUFFER_DIR,
        "rolling_%03d.ts"
    )

    # ✅ MPEG-TS over UDP input (KLV SAFE)
    ffmpeg_cmd = [
    "ffmpeg",

    # Input: MPEG-TS over UDP (video + KLV)
    "-i", "udp://127.0.0.1:5000?fifo_size=5000000&overrun_nonfatal=1",

    # Explicitly keep video and KLV data streams
    # "-map", "0:v",
    # "-map", "0:d?",
    "-map", "0",

    # Copy streams exactly (NO re-encoding)
    "-c", "copy",

    # Segment into MPEG-TS files
    "-f", "segment",
    "-segment_time", str(ROLLING_SECONDS),
    "-segment_format", "mpegts",

    # 🔑 CRITICAL: keep continuous timestamps (KLV SAFE)
    "-reset_timestamps", "1",

    # Output filename pattern
    output_pattern
    ]

    print("Starting Video Ingest Service (UDP MPEG-TS)...")
    process = subprocess.Popen(ffmpeg_cmd)

    uploaded = set()

    try:
        while True:
            for file in os.listdir(LOCAL_BUFFER_DIR):
                if file.endswith(".ts") and file not in uploaded:
                    local_path = os.path.join(LOCAL_BUFFER_DIR, file)

                    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    object_name = f"{STREAMING_STREAM_ID}/{timestamp}.ts"

                    minio_client.upload(
                        STREAMING_MINIO_BUCKET,
                        object_name,
                        local_path
                    )

                    print(f"Uploaded to MinIO: {object_name}")

                    uploaded.add(file)
                    os.remove(local_path)

            time.sleep(2)

    except KeyboardInterrupt:
        print("Stopping ingest service...")
        process.terminate()

if __name__ == "__main__":
    main()
