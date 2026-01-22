# ingest_video_streaming.py
import os
import re
import subprocess
import time
import datetime

from config import *
from minio_client import MinIOClient
 
os.makedirs(LOCAL_BUFFER_DIR, exist_ok=True)
TS_NAME_RE = re.compile(r"^\d{8}_\d{6}\.ts$")

def is_file_complete(path, wait=1):
    size1 = os.path.getsize(path)
    time.sleep(wait)
    size2 = os.path.getsize(path)
    return size1 == size2
 
def main():
    minio_client = MinIOClient(
        MINIO_ENDPOINT,
        MINIO_ACCESS_KEY,
        MINIO_SECRET_KEY,
        MINIO_SECURE
    )
    minio_client.ensure_bucket(STREAMING_MINIO_BUCKET)
 
    # Output filename pattern with timestamps
    output_pattern = os.path.join(LOCAL_BUFFER_DIR, "%Y%m%d_%H%M%S.ts")
 
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", "udp://127.0.0.1:5000?fifo_size=5000000&overrun_nonfatal=1",
        "-map", "0",                   # Keep all streams (video + KLV)
        "-c", "copy",                  # No re-encoding
        "-f", "segment",
        "-segment_time", str(ROLLING_SECONDS),
        "-segment_format", "mpegts",
        "-segment_format_options", "mpegts_flags=+resend_headers",
        "-segment_atclocktime", "1",   # Align segments to real-time
        "-strftime", "1",              # Timestamped filenames
        output_pattern
    ]
 
    print("Starting UDP KLV Video Ingest Service...")
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
 
    try:
        while True:
            for file in os.listdir(LOCAL_BUFFER_DIR):
                if not file.endswith(".ts"):
                    continue

                if not TS_NAME_RE.match(file):
                    continue
                
                file_path = os.path.join(LOCAL_BUFFER_DIR, file)
 
                if not is_file_complete(file_path):
                    continue
 
                # Parse timestamp from filename
                segment_time = datetime.datetime.strptime(file.replace(".ts", ""), "%Y%m%d_%H%M%S")
                # Adjust IST if needed
                ist_time = segment_time + datetime.timedelta(hours=5, minutes=30)
 
                object_name = (
                    f"{STREAMING_STREAM_ID}/"
                    f"{ist_time.strftime('%Y/%m/%d/%H')}/"
                    f"{file}"
                )
 
                minio_client.upload(
                    STREAMING_MINIO_BUCKET,
                    object_name,
                    file_path
                )
 
                print(f"Uploaded to MinIO: {object_name}")
                os.remove(file_path)
            time.sleep(2)
 
    except KeyboardInterrupt:
        print("Stopping ingest service...")
        process.terminate()

if __name__ == "__main__":
    main()