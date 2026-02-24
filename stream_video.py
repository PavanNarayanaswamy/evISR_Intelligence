import subprocess
import time
import signal
import sys
import argparse
import threading
import os

from video_ingest_service.stream_registry import (
    register_stream,
    heartbeat,
    cleanup_stale_streams,
)

VIDEO_FILE = "Truck.ts"
BASE_PORT = 5000


# ---------------------------------------------------
# FORCE FREE PORT
# ---------------------------------------------------

def free_port(port):
    """
    Kill any process using this UDP port.
    """
    try:
        os.system(
            f"lsof -ti udp:{port} | xargs -r kill -9 >/dev/null 2>&1"
        )
    except Exception:
        pass


# ---------------------------------------------------
# HEARTBEAT
# ---------------------------------------------------

def start_heartbeat(port):

    def loop():
        while True:
            heartbeat(port)
            time.sleep(5)

    threading.Thread(target=loop, daemon=True).start()


# ---------------------------------------------------
# STREAM START
# ---------------------------------------------------

def start_stream(port):

    free_port(port)

    stream_id = f"stream-{port}"

    print(f"""
====================================
STARTING STREAM
Port      : {port}
Stream ID : {stream_id}
====================================
""")

    register_stream(port, stream_id)
    start_heartbeat(port)

    udp_url = f"udp://127.0.0.1:{port}?pkt_size=1316"

    ffmpeg_command = [
        "ffmpeg",
        "-re",
        "-stream_loop", "-1",
        "-i", VIDEO_FILE,
        "-map", "0",
        "-c", "copy",
        "-f", "mpegts",
        udp_url,
    ]

    process = subprocess.Popen(
        ffmpeg_command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return process


# ---------------------------------------------------
# PORT RANGE
# ---------------------------------------------------

def get_ports(count):
    return [BASE_PORT + i for i in range(count)]


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="UDP ISR Stream Simulator"
    )

    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of streams in multi mode",
    )
    parser.add_argument(
    "--port",
    type=int,
    default=BASE_PORT,
    help="Port for single stream mode"
    )

    args = parser.parse_args()

    processes = []

    print("\nResetting simulator environment...\n")

    # ✅ cleanup registry
    cleanup_stale_streams()

    # ✅ kill leftover ffmpeg globally
    os.system("pkill -9 -f 'ffmpeg.*mpegts' >/dev/null 2>&1")

    time.sleep(2)

    # ---------------- SINGLE ----------------
    if args.mode == "single":

        port = args.port
        processes.append(start_stream(port))

    # ---------------- MULTI ----------------
    elif args.mode == "multi":

        ports = get_ports(args.count)

        for port in ports:
            processes.append(start_stream(port))
            time.sleep(1)

    # ---------------------------------------------------
    # CLEAN SHUTDOWN
    # ---------------------------------------------------

    def shutdown(signum, frame):

        print("\nStopping all streams...")

        for p in processes:
            try:
                p.terminate()
            except Exception:
                pass

        os.system("pkill -9 -f 'ffmpeg.*mpegts' >/dev/null 2>&1")

        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for p in processes:
        p.wait()


if __name__ == "__main__":
    main()