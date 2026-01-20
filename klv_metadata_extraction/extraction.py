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
        Extract KLV and upload to MinIO.

        Returns:
            MinIO URI of uploaded KLV file
        """
        ts_path = Path(ts_path)
        klv_file = self.work_dir / f"{clip_id}.klv"

        print(f"\n=== Extracting KLV from {ts_path.name} ===")
        
        # Use stream #1 specifically (the KLV stream from ffprobe output)
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(ts_path),
            "-map", "0:1",  # ← CHANGED: Extract ONLY stream #1 (KLV)
            "-c", "copy",
            "-f", "data",
            str(klv_file),
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"⚠️  First attempt failed: {result.stderr[:200]}")
            
            # Fallback: try stream #0:0x101 (by PID)
            print("Trying fallback: extracting by PID 0x101")
            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(ts_path),
                "-map", "0:0x101",  # Extract by MPEG-TS PID
                "-c", "copy",
                "-f", "data",
                str(klv_file),
            ]
            subprocess.run(cmd, capture_output=True)

        # Verify we got data
        if not klv_file.exists() or klv_file.stat().st_size == 0:
            raise RuntimeError(f"No KLV data extracted from {ts_path}")

        print(f"✅ Extracted {klv_file.stat().st_size:,} bytes")
        
        # Check for MISB UL
        with open(klv_file, 'rb') as f:
            first_1k = f.read(1024)
        
        misb_ul = bytes.fromhex("060E2B34020B0101")
        if misb_ul in first_1k:
            ul_pos = first_1k.find(misb_ul)
            print(f"✅ Found MISB UL at offset {ul_pos}")
            
            # If UL not at position 0, trim the file
            if ul_pos > 0:
                print(f"⚠️  Trimming {ul_pos} bytes from start")
                with open(klv_file, 'rb') as f:
                    all_data = f.read()
                with open(klv_file, 'wb') as f:
                    f.write(all_data[ul_pos:])
        else:
            print("⚠️  No MISB UL found in extracted data")
            # Continue anyway - might have different UL
            
        # Upload to MinIO
        object_name = f"{clip_id}.klv"
        self.minio.fput_object(
            self.output_bucket,
            object_name,
            str(klv_file),
        )

        print(f"✅ Uploaded to: minio://{self.output_bucket}/{object_name}")

        # Cleanup local temp file
        klv_file.unlink(missing_ok=True)

        return f"minio://{self.output_bucket}/{object_name}"