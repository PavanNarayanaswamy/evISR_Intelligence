from minio import Minio
from urllib.parse import urlparse
from pathlib import Path
import os
import sys

from utils import config
from utils.logger import get_logger

logger = get_logger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from kafka_consumer import consumer_config


# Global MinIO client
client = Minio(
    consumer_config.MINIO_ENDPOINT,
    access_key=consumer_config.MINIO_ACCESS_KEY,
    secret_key=consumer_config.MINIO_SECRET_KEY,
    secure=consumer_config.MINIO_SECURE,
)


def get_minio_client():
    """Create and return a new MinIO client instance."""
    logger.info("Creating new MinIO client instance.")
    return Minio(
        consumer_config.MINIO_ENDPOINT,
        access_key=consumer_config.MINIO_ACCESS_KEY,
        secret_key=consumer_config.MINIO_SECRET_KEY,
        secure=consumer_config.MINIO_SECURE,
    )


def parse_minio_uri(uri: str):
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def download_clip(clip_uri: str, local_path: Path):
    logger.info(f"Downloading clip from {clip_uri} to {local_path}")
    client = get_minio_client()
    bucket, object_name = parse_minio_uri(clip_uri)

    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        client.fget_object(
            bucket_name=bucket,
            object_name=object_name,
            file_path=str(local_path),
        )
        logger.info(f"Successfully downloaded clip: {object_name}")
    except Exception as e:
        logger.error(f"Failed to download clip {object_name}: {e}", exc_info=True)


def download_klv(s3_uri: str, local_path: Path):
    logger.info(f"Downloading KLV from {s3_uri} to {local_path}")
    parsed = urlparse(s3_uri)

    bucket = parsed.netloc
    object_name = parsed.path.lstrip("/")

    client = get_minio_client()
    try:
        client.fget_object(bucket, object_name, str(local_path))
        logger.info(f"Successfully downloaded KLV: {object_name}")
    except Exception as e:
        logger.error(f"Failed to download KLV {object_name}: {e}", exc_info=True)


def upload_output(bucket: str, object_name: str, file_path: Path):
    logger.info(f"Uploading output to bucket: {bucket}, object: {object_name}, file: {file_path}")
    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            logger.info(f"Created new bucket: {bucket}")

        client.fput_object(bucket, object_name, str(file_path))
        logger.info(f"Successfully uploaded: {object_name}")
    except Exception as e:
        logger.error(f"Failed to upload {file_path} to {bucket}/{object_name}: {e}", exc_info=True)