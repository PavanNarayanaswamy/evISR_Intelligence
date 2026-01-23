
from minio import Minio
from utils.logger import get_logger

logger = get_logger(__name__)

class MinIOClient:
    def __init__(self, endpoint, access_key, secret_key, secure=False):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

    def ensure_bucket(self, bucket):
        try:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
                logger.info(f"Created bucket: {bucket}")
            else:
                logger.debug(f"Bucket already exists: {bucket}")
        except Exception as e:
            logger.error(f"Error ensuring bucket {bucket}: {e}")
            raise

    def upload(self, bucket, object_name, file_path):
        try:
            self.client.fput_object(bucket, object_name, file_path)
            logger.info(f"Uploaded {file_path} to {bucket}/{object_name}")
        except Exception as e:
            logger.error(f"Error uploading {file_path} to {bucket}/{object_name}: {e}")
            raise
