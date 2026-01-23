# RTSP / Ingest
RTSP_URL = "rtsp://localhost:8554/live"

# Rolling window (seconds)
ROLLING_SECONDS = 30

# Temporary local storage for ingest
LOCAL_BUFFER_DIR = "/tmp/evisr_ingest"

# MinIO
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin123"
MINIO_SECURE = False

# Bucket where streaming clips are uploaded
STREAMING_MINIO_BUCKET = "raw-streaming-video"
STREAMING_STREAM_ID = "standalone-camera-01"

# Offline Video Clip Ingest Configuration
VIDEO_CLIP_FILE = "embedded.ts"

# Each clip chunk duration (seconds)
VIDEO_CLIP_SEGMENT_SECONDS = 30

# MinIO bucket for offline clips
VIDEO_CLIP_BUCKET = "raw-video-clips"

# Stream / source identity for clip
VIDEO_CLIP_STREAM_ID = "embedding-camera-01"

# Temp directory for clip chunking
VIDEO_CLIP_TMP_DIR = "/tmp/evisr_video_clips"

# Kafka
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "videoclips"

# Polling
POLL_INTERVAL_SECONDS = 3
STATE_FILE = "state.json"