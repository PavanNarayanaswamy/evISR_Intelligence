# extraction.py
import subprocess
from pathlib import Path
from minio import Minio
import datetime
from utils.logger import get_logger

logger = get_logger(__name__)

class Extraction:
    """
    Extracts KLV metadata from TS files and stores it directly in MinIO.
    """

    def __init__(
        self,
        minio_client: Minio,
        output_bucket: str,
        work_dir: str | Path = "/tmp/klv",
    ):
        self.minio = minio_client
        self.output_bucket = output_bucket
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        if not self.minio.bucket_exists(output_bucket):
            self.minio.make_bucket(output_bucket)
            logger.info(f"Created bucket: {output_bucket}")
        else:
            logger.info(f"Bucket already exists: {output_bucket}")

    def extract_klv(self, ts_path: str | Path, clip_id: str) -> str:
        """
        Extract KLV (all data streams) from TS and upload to MinIO.
        """
        ts_path = Path(ts_path)
        klv_file = self.work_dir / f"{clip_id}.klv"

        logger.info(f"Extracting KLV from {ts_path.name}")

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", str(ts_path),

            # ✅ Extract ALL data streams (KLV)
            "-map", "0:d",

            "-c", "copy",
            "-f", "data",
            str(klv_file),
        ]

        logger.debug(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Failed to extract KLV from {ts_path}: {result.stderr}", exc_info=True)
            raise RuntimeError(
                f"Failed to extract KLV from {ts_path}\n{result.stderr}"
            )

        # Basic validation
        if not klv_file.exists() or klv_file.stat().st_size == 0:
            logger.error(f"No KLV data extracted from {ts_path}")
            raise RuntimeError(f"No KLV data extracted from {ts_path}")

        logger.info(f"Extracted {klv_file.stat().st_size:,} bytes of KLV data")

        # Upload to MinIO
        now = datetime.datetime.now()  # local time

        object_name = (
            f"extraction/"
            f"{now.strftime('%Y/%m/%d/%H')}/"
            f"{clip_id}.klv"
        )
        try:
            self.minio.fput_object(
                self.output_bucket,
                object_name,
                str(klv_file),
            )
            logger.info(f"Uploaded to: minio://{self.output_bucket}/{object_name}")
        except Exception as e:
            logger.error(f"Failed to upload {klv_file} to MinIO: {e}", exc_info=True)
            raise

        # Cleanup
        try:
            klv_file.unlink(missing_ok=True)
            logger.debug(f"Cleaned up temporary file: {klv_file}")
        except Exception as e:
            logger.error(f"Failed to remove temporary file {klv_file}: {e}", exc_info=True)

        return f"minio://{self.output_bucket}/{object_name}"