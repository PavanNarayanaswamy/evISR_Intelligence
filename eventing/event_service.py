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
from io import BytesIO

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
        self.stream_partition_map = self.load_partition_map()
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
        # Ensure partition-map bucket exists
        if not self.minio_client.bucket_exists(config.MAP_BUCKET):
            logger.info(
                f"Creating partition map bucket: {config.MAP_BUCKET}"
            )
            self.minio_client.make_bucket(config.MAP_BUCKET)

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
        duration_seconds: int,
    ) -> None:

        try:
            objects = self.minio_client.list_objects(
                bucket,
                prefix="port-",
                recursive=True,
            )
        except Exception as e:
            logger.error(
                f"Failed to list objects in bucket {bucket}: {e}",
                exc_info=True,
            )
            return

        for obj in objects:

            object_name = obj.object_name
            clip_name = os.path.basename(object_name)

            if not clip_name.endswith(".ts"):
                continue

            # ----------------------------------------
            # ⭐ DERIVE STREAM ID FROM OBJECT PATH
            # ----------------------------------------
            # object example:
            # port-5001/2026/02/23/...ts

            port_prefix = object_name.split("/")[0]
            port = port_prefix.replace("port-", "")

            stream_id = f"stream-{port}"

            # ensure state AFTER stream detected
            self.ensure_stream_state(state, stream_id)

            # Deduplication
            if object_name in state[stream_id]["processed_objects"]:
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

            # ----------------------------------------
            # Kafka Publish
            # ----------------------------------------
            try:
                partition = self.get_or_create_partition(
                    stream_id
                )

                self.kafka_producer.publish(
                    topic=self.kafka_topic,
                    partition=partition,
                    value=event,
                )

                logger.info(
                    f"[{stream_id}] Published clip_id: {event['clip_id']}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to publish event {event['clip_id']}: {e}",
                    exc_info=True,
                )
                continue

            append_event_to_file(event, self.events_log_file)

            state[stream_id]["processed_objects"].append(object_name)
            self.save_state(state)

    def start(self) -> None:
        """
        Ensures topic exists and starts dynamic polling loop.
        """

        logger.info("Starting Eventing Service...")
        logger.info(f"Kafka topic: {self.kafka_topic}")
        logger.info(f"Writing state to: {self.state_file}")
        logger.info(f"Writing event payloads to: {self.events_log_file}")

        # --------------------------------------------------
        # Ensure Kafka topic exists
        # --------------------------------------------------
        try:
            self.kafka_admin.ensure_topic(
                topic_name=self.kafka_topic,
                num_partitions=config.num_partitions,
                replication_factor=1,
            )
            logger.info(f"Topic ready: {self.kafka_topic}")

        except Exception as e:
            logger.error(
                f"Failed to ensure Kafka topic {self.kafka_topic}: {e}",
                exc_info=True,
            )
            return

        state = self.load_state()

        # --------------------------------------------------
        # Dynamic polling loop
        # --------------------------------------------------
        while True:

            try:
                buckets = self.minio_client.list_buckets()

                for bucket in buckets:

                    # ------------------------------------------
                    # STREAMING CAMERAS
                    # raw-streaming-video*
                    # ------------------------------------------
                    if bucket.name.startswith(
                        config.STREAMING_MINIO_BUCKET
                    ):

                        self.process_bucket(
                            state=state,
                            bucket=bucket.name,
                            duration_seconds=config.ROLLING_SECONDS,
                        )

                    # ------------------------------------------
                    # OFFLINE VIDEO INGEST
                    # ------------------------------------------
                    elif bucket.name == config.VIDEO_CLIP_BUCKET:

                        self.process_bucket(
                            state=state,
                            bucket=bucket.name,
                            duration_seconds=config.VIDEO_CLIP_SEGMENT_SECONDS,
                        )

            except Exception as e:
                logger.error(
                    f"Error in polling loop: {e}",
                    exc_info=True,
                )

            time.sleep(self.poll_interval_seconds)

    def get_or_create_partition(self, stream_id):

        self.stream_partition_map = self.load_partition_map()

        if stream_id in self.stream_partition_map:
            return self.stream_partition_map[stream_id]

        logger.info(f"New camera detected: {stream_id}")

        current = self.kafka_admin.get_partition_count(
            self.kafka_topic
        )

        # ✅ FIRST CAMERA USES EXISTING PARTITION 0
        if current == 1 and len(self.stream_partition_map) == 0:
            new_partition = 0
        else:
            new_partition = current
            self.kafka_admin.increase_partitions(
                self.kafka_topic,
                current + 1,
            )

            # force metadata refresh
            self.kafka_producer.producer.list_topics(
                topic=self.kafka_topic,
                timeout=10,
            )

            import time
            time.sleep(1)

        self.stream_partition_map[stream_id] = new_partition
        self.save_partition_map(self.stream_partition_map)

        return new_partition

    def load_partition_map(self):
        try:
            obj = self.minio_client.get_object(
                config.MAP_BUCKET,
                "stream_partition_map.json",
            )

            data = json.loads(obj.read().decode())
            logger.info("Loaded partition map from MinIO")
            return data

        except Exception:
            logger.info("Partition map not found. Creating new.")
            return {}
        
    def save_partition_map(self, mapping):

        data = json.dumps(mapping, indent=2).encode()

        self.minio_client.put_object(
            config.MAP_BUCKET,
            "stream_partition_map.json",
            BytesIO(data),
            length=len(data),
            content_type="application/json",
        )

        logger.info("Partition map updated in MinIO")