# stream_video_embedded.py
import subprocess
import time
import signal
import sys
from video_ingest_service.stream_registry import register_stream
from video_ingest_service.stream_registry import load_registry
import threading
from video_ingest_service.stream_registry import register_stream, heartbeat

PORT = 5001
register_stream(PORT, "embedded")

def heartbeat_loop():
    while True:
        heartbeat(PORT)
        time.sleep(5)

threading.Thread(target=heartbeat_loop, daemon=True).start()


VIDEO_FILE = "embedded.ts"
UDP_URL = "udp://127.0.0.1:5001?pkt_size=1316"

print("Starting FFmpeg MPEG-TS UDP stream (embedded.ts)...")

ffmpeg_command = [
    "ffmpeg",

    # Read input in realtime
    "-re",

    # Loop the input forever
    "-stream_loop", "-1",

    # Input file with video + KLV
    "-i", VIDEO_FILE,

    # Map all streams (video + metadata)
    "-map", "0",

    # Copy without re-encoding
    "-c", "copy",

    # Output as MPEG-TS over UDP
    "-f", "mpegts",

    UDP_URL
]

ffmpeg_process = subprocess.Popen(ffmpeg_command)


def shutdown(signum, frame):
    print("\nStopping embedded UDP stream...")
    ffmpeg_process.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

ffmpeg_process.wait()
