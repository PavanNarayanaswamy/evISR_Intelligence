#event_service.py:
import os
import time
import json
import datetime
from typing import Dict, Any

from minio import Minio

from utils import config

from eventing.kafka_admin import KafkaAdmin
from eventing.kafka_producer import KafkaProducerClient
from utils.logger import get_logger
from utils.event_logger import append_event_to_file

logger = get_logger(__name__)

DEFAULT_STATE_FILE = "state.json"
DEFAULT_EVENTS_LOG_FILE = "events_log.json"

class EventingService:
    """
    Polls MinIO buckets for new .ts clips and publishes Kafka events.
    """
    def __init__(
        self,
        minio_endpoint: str,
        minio_access_key: str,
        minio_secret_key: str,
        minio_secure: bool,
        kafka_bootstrap_servers: str,
        kafka_topic: str,
        state_file: str = DEFAULT_STATE_FILE,
        events_log_file: str = DEFAULT_EVENTS_LOG_FILE,
        poll_interval_seconds: int = 3,
    ):
        self.kafka_topic = kafka_topic
        self.poll_interval_seconds = poll_interval_seconds
        self.state_file = state_file
        self.events_log_file = events_log_file

        # MinIO client
        self.minio_client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=minio_secure,
        )

        # Kafka admin + producer
        self.kafka_admin = KafkaAdmin(kafka_bootstrap_servers)
        self.kafka_producer = KafkaProducerClient(kafka_bootstrap_servers)

    # State handling
    def load_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.state_file):
            logger.info(f"State file {self.state_file} does not exist. Starting with empty state.")
            return {}
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            logger.info(f"Loaded state from {self.state_file}")
            return state
        except Exception as e:
            logger.error(f"Failed to load state file {self.state_file}: {e}", exc_info=True)
            return {}

    def save_state(self, state: Dict[str, Any]) -> None:
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
            logger.debug(f"Saved state to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state file {self.state_file}: {e}", exc_info=True)

    def ensure_stream_state(self, state: Dict[str, Any], stream_id: str) -> None:
        if stream_id not in state:
            logger.info(f"Initializing state for stream_id: {stream_id}")
            state[stream_id] = {
                "last_sequence": 0,
                "processed_objects": []
            }

    # Event log handling
    def append_event_log(self, event: dict) -> None:
        if not os.path.exists(self.events_log_file):
            log_data = []
        else:
            try:
                with open(self.events_log_file, "r") as f:
                    log_data = json.load(f)
                    if not isinstance(log_data, list):
                        log_data = []
            except Exception as e:
                logger.error(f"Failed to read event log file {self.events_log_file}: {e}", exc_info=True)
                log_data = []

        log_data.append(event)

        try:
            with open(self.events_log_file, "w") as f:
                json.dump(log_data, f, indent=2)
            logger.debug(f"Appended event to log file {self.events_log_file}")
        except Exception as e:
            logger.error(f"Failed to write event log file {self.events_log_file}: {e}", exc_info=True)

    # Event building
    def build_event(
        self,
        stream_id: str,
        bucket: str,
        object_name: str,
        clip_name: str,
        sequence_number: int,
        duration_seconds: int,
    ) -> dict:
        base_name = os.path.splitext(clip_name)[0]
        clip_id = f"{stream_id}_{base_name}"

        return {
            "stream_id": stream_id,
            "clip_id": clip_id,
            "sequence_number": sequence_number,
            "clip_uri": f"minio://{bucket}/{object_name}",
            "duration_seconds": duration_seconds,
            "has_klv": True,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        }

    # Processing
    def process_bucket(
        self,
        state: Dict[str, Any],
        bucket: str,
        stream_id: str,
        duration_seconds: int,
        ) -> None:
        self.ensure_stream_state(state, stream_id)

        try:
            objects = self.minio_client.list_objects(
                bucket,
                prefix=f"{stream_id}/",
                recursive=True
            )
        except Exception as e:
            logger.error(f"Failed to list objects in bucket {bucket}: {e}", exc_info=True)
            return

        for obj in objects:
            object_name = obj.object_name
            clip_name = os.path.basename(object_name)

            if not clip_name.endswith(".ts"):
                logger.debug(f"Skipping non-ts file: {clip_name}")
                continue

            # ✅ Deduplicate using full object path
            if object_name in state[stream_id]["processed_objects"]:
                logger.debug(f"Already processed: {object_name}")
                continue

            state[stream_id]["last_sequence"] += 1

            event = self.build_event(
                stream_id=stream_id,
                bucket=bucket,
                object_name=object_name,
                clip_name=clip_name,
                sequence_number=state[stream_id]["last_sequence"],
                duration_seconds=duration_seconds,
            )

            # Publish to Kafka
            try:
                self.kafka_producer.publish(
                    topic=self.kafka_topic,
                    key=event["stream_id"],
                    value=event,
                )
                logger.info(f"Published event for clip_id: {event['clip_id']}")
            except Exception as e:
                logger.error(f"Failed to publish event for {event['clip_id']}: {e}", exc_info=True)
                continue

            append_event_to_file(event, self.events_log_file)

            logger.debug(f"Event appended and state updated for: {event['clip_id']}")

            # ✅ Store full object name
            state[stream_id]["processed_objects"].append(object_name)
            self.save_state(state)

    def start(self) -> None:
        """
        Ensures topic exists and starts the polling loop.
        """
        logger.info("Starting Eventing Service...")
        logger.info(f"Kafka topic: {self.kafka_topic}")
        logger.info(f"Writing state to: {self.state_file}")
        logger.info(f"Writing event payloads to: {self.events_log_file}")

        # Ensure Kafka topic exists
        try:
            self.kafka_admin.ensure_topic(
                topic_name=self.kafka_topic,
                num_partitions=1,
                replication_factor=1,
            )
            logger.info(f"Topic ready: {self.kafka_topic}")
        except Exception as e:
            logger.error(f"Failed to ensure Kafka topic {self.kafka_topic}: {e}", exc_info=True)
            return

        state = self.load_state()

        while True:
            try:
                # Streaming RTSP clips ingest bucket
                self.process_bucket(
                    state=state,
                    bucket=config.STREAMING_MINIO_BUCKET,
                    stream_id=config.STREAMING_STREAM_ID,
                    duration_seconds=config.ROLLING_SECONDS,
                )

                # Offline clips bucket
                self.process_bucket(
                    state=state,
                    bucket=config.VIDEO_CLIP_BUCKET,
                    stream_id=config.VIDEO_CLIP_STREAM_ID,
                    duration_seconds=config.VIDEO_CLIP_SEGMENT_SECONDS,
                )
            except Exception as e:
                logger.error(f"Error in polling loop: {e}", exc_info=True)
            time.sleep(self.poll_interval_seconds)