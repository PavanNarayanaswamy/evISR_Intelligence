# extraction.py
import subprocess
from pathlib import Path
from minio import Minio
 
 
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
 
    def extract_klv(self, ts_path: str | Path, clip_id: str) -> str:
        """
        Extract KLV (all data streams) from TS and upload to MinIO.
        """
        ts_path = Path(ts_path)
        klv_file = self.work_dir / f"{clip_id}.klv"
 
        print(f"\n=== Extracting KLV from {ts_path.name} ===")
 
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
 
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
 
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to extract KLV from {ts_path}\n{result.stderr}"
            )
 
        # Basic validation
        if not klv_file.exists() or klv_file.stat().st_size == 0:
            raise RuntimeError(f"No KLV data extracted from {ts_path}")
 
        print(f"✅ Extracted {klv_file.stat().st_size:,} bytes of KLV data")
 
        # Upload to MinIO
        object_name = f"{clip_id}.klv"
        self.minio.fput_object(
            self.output_bucket,
            object_name,
            str(klv_file),
        )
 
        print(f"✅ Uploaded to: minio://{self.output_bucket}/{object_name}")
 
        # Cleanup
        klv_file.unlink(missing_ok=True)
 
        return f"minio://{self.output_bucket}/{object_name}"