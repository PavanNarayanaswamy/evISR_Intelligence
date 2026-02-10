# stream_video.py
import subprocess
import time
import signal
import sys
from video_ingest_service.stream_registry import register_stream
import threading
from video_ingest_service.stream_registry import register_stream, heartbeat

PORT = 5000
register_stream(PORT, "standalone")

def heartbeat_loop():
    while True:
        heartbeat(PORT)
        time.sleep(5)

threading.Thread(target=heartbeat_loop, daemon=True).start()


VIDEO_FILE = "embedded.ts"
UDP_URL = "udp://127.0.0.1:5000?pkt_size=1316"

print("Starting FFmpeg MPEG-TS UDP stream...")

ffmpeg_command = [
    "ffmpeg",

    # Read input in realtime (important for UDP pacing)
    "-re",

    # Loop the input forever
    "-stream_loop", "-1",

    # Input file with video + KLV
    "-i", VIDEO_FILE,

    # Explicitly map video and KLV data streams
    # "-map", "0:v",
    # "-map", "0:d?",
    "-map", "0",

    # Copy streams exactly (no re-encode)
    "-c", "copy",

    # Output as MPEG-TS over UDP
    "-f", "mpegts",

    UDP_URL
]


ffmpeg_process = subprocess.Popen(ffmpeg_command)


def shutdown(signum, frame):
    print("\nStopping UDP stream...")
    ffmpeg_process.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

ffmpeg_process.wait()
