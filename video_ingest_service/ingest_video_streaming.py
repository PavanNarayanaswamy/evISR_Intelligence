# ingest_video_streaming.py
import os
import re
import subprocess
import time
import datetime
from typing import Dict

from utils import config
from minio_client import MinIOClient
from utils.logger import get_logger
from stream_registry import load_registry
from stream_registry import load_registry, cleanup_stale_streams


logger = get_logger(__name__)

os.makedirs(config.LOCAL_BUFFER_DIR, exist_ok=True)

TS_NAME_RE = re.compile(r"^\d{8}_\d{6}\.ts$")


def is_file_complete(path, wait=1):
    size1 = os.path.getsize(path)
    time.sleep(wait)
    size2 = os.path.getsize(path)
    return size1 == size2


class StreamIngestWorker:
    """
    Handles ingestion for ONE UDP stream (ONE port)
    """

    def __init__(self, port: int, minio_client: MinIOClient):
        self.port = port

        # Bucket naming logic
        self.bucket = (
            config.STREAMING_MINIO_BUCKET
            if port == 5000
            else f"{config.STREAMING_MINIO_BUCKET}-{port}"
        )

        # Per-stream buffer directory
        self.buffer_dir = os.path.join(config.LOCAL_BUFFER_DIR, str(port))
        os.makedirs(self.buffer_dir, exist_ok=True)

        self.minio = minio_client
        self.minio.ensure_bucket(self.bucket)

        self.process = None
        
        self.last_activity_ts = time.time()
        self.active = True
        
        self.restart_attempts = 0
        self.last_start_ts = None

    def start_ffmpeg(self):
        output_pattern = os.path.join(self.buffer_dir, "%Y%m%d_%H%M%S.ts")

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-i", f"udp://127.0.0.1:{self.port}?fifo_size=5000000&overrun_nonfatal=1",
            "-map", "0",
            "-c", "copy",
            "-f", "segment",
            "-segment_time", str(config.ROLLING_SECONDS),
            "-segment_format", "mpegts",
            "-segment_format_options", "mpegts_flags=+resend_headers",
            "-segment_atclocktime", "1",
            "-strftime", "1",
            output_pattern,
        ]

        logger.info(f"Starting FFmpeg ingest on UDP port {self.port}")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )
        self.restart_attempts = 0
        self.last_start_ts = time.time()

    def ingest_loop(self):
        """
        Upload completed TS segments to MinIO
        """
        for file in os.listdir(self.buffer_dir):
            if not file.endswith(".ts"):
                continue

            if not TS_NAME_RE.match(file):
                continue

            file_path = os.path.join(self.buffer_dir, file)

            if not is_file_complete(file_path):
                continue

            segment_time = datetime.datetime.strptime(
                file.replace(".ts", ""), "%Y%m%d_%H%M%S"
            )

            ist_time = segment_time + datetime.timedelta(hours=5, minutes=30)

            object_name = (
                f"port-{self.port}/"
                f"{ist_time.strftime('%Y/%m/%d/%H')}/"
                f"{file}"
            )

            try:
                self.minio.upload(self.bucket, object_name, file_path)
                logger.info(f"[PORT {self.port}] Uploaded {object_name}")
                self.last_activity_ts = time.time()
                os.remove(file_path)
            except Exception as e:
                logger.error(
                    f"[PORT {self.port}] Upload failed for {file}",
                    exc_info=True
                )
                
    def check_liveness(self):
        if not self.active:
            return

        idle_time = time.time() - self.last_activity_ts
        if idle_time > config.STREAM_IDLE_TIMEOUT_SECONDS:
            logger.warning(
                f"[PORT {self.port}] Stream inactive for {idle_time:.1f}s. Stopping FFmpeg."
            )
            if self.process:
                self.process.terminate()
            self.active = False
            
    def check_ffmpeg_health(self):
        if not self.process:
            return

        retcode = self.process.poll()
        if retcode is not None:
            logger.error(
                f"[PORT {self.port}] FFmpeg crashed with code {retcode}"
            )
            self.active = False

    def restart_ffmpeg(self):
        self.restart_attempts += 1

        backoff = min(10 * self.restart_attempts, 60)
        logger.warning(
            f"[PORT {self.port}] Restarting FFmpeg in {backoff}s "
            f"(attempt {self.restart_attempts})"
        )

        time.sleep(backoff)
        self.start_ffmpeg()
        self.active = True
        self.last_activity_ts = time.time()




def main():
    logger.info("Starting Dynamic UDP Video Ingestion Supervisor")

    minio_client = MinIOClient(
        config.MINIO_ENDPOINT,
        config.MINIO_ACCESS_KEY,
        config.MINIO_SECRET_KEY,
        config.MINIO_SECURE,
    )

    workers: Dict[int, StreamIngestWorker] = {}

    while True:
        cleanup_stale_streams()
        registry = load_registry()
        
        # Remove expired streams (TTL cleanup)
        for port in list(workers.keys()):
            if str(port) not in registry:
                logger.warning(f"[PORT {port}] Registry TTL expired. Stopping ingestion.")
                workers[port].stop_ffmpeg()
                del workers[port]

        # Detect new streams
        for port_str in registry.keys():
            port = int(port_str)

            if port not in workers:
                logger.info(f"Detected new stream on UDP port {port}")
                worker = StreamIngestWorker(port, minio_client)
                worker.start_ffmpeg()
                workers[port] = worker
            else:
                worker = workers[port]
                if not worker.active:
                    logger.info(f"[PORT {port}] Stream reappeared, restarting ingest")
                    worker.start_ffmpeg()
                    worker.active = True
                    worker.last_activity_ts = time.time()

        # Run ingest loop for all active streams
        for worker in workers.values():
            worker.ingest_loop()
            worker.check_liveness()
            worker.check_ffmpeg_health()
            if not worker.active:
                worker.restart_ffmpeg()

        time.sleep(2)


if __name__ == "__main__":
    main()
