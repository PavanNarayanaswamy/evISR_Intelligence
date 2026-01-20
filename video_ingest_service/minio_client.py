from minio import Minio

class MinIOClient:
    def __init__(self, endpoint, access_key, secret_key, secure=False):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

    def ensure_bucket(self, bucket):
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)

    def upload(self, bucket, object_name, file_path):
        self.client.fput_object(bucket, object_name, file_path)
