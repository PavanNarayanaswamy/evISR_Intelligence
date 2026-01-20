import os
import time
import json
import datetime
from typing import Dict, Any

from minio import Minio

import config

from eventing.kafka_admin import KafkaAdmin
from eventing.kafka_producer import KafkaProducerClient

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
            return {}
        with open(self.state_file, "r") as f:
            return json.load(f)

    def save_state(self, state: Dict[str, Any]) -> None:
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def ensure_stream_state(self, state: Dict[str, Any], stream_id: str) -> None:
        if stream_id not in state:
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
            except Exception:
                log_data = []

        log_data.append(event)

        with open(self.events_log_file, "w") as f:
            json.dump(log_data, f, indent=2)

    # Event building
    def build_event(
        self,
        stream_id: str,
        bucket: str,
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
            "clip_uri": f"minio://{bucket}/{stream_id}/{clip_name}",
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

        objects = self.minio_client.list_objects(
            bucket,
            prefix=f"{stream_id}/",
            recursive=True
        )

        for obj in objects:
            clip_name = os.path.basename(obj.object_name)

            if not clip_name.endswith(".ts"):
                continue

            if clip_name in state[stream_id]["processed_objects"]:
                continue

            state[stream_id]["last_sequence"] += 1

            event = self.build_event(
                stream_id=stream_id,
                bucket=bucket,
                clip_name=clip_name,
                sequence_number=state[stream_id]["last_sequence"],
                duration_seconds=duration_seconds,
            )

            # Publish to Kafka
            self.kafka_producer.publish(
                topic=self.kafka_topic,
                key=event["stream_id"],
                value=event,
            )

            # Store event payload locally (temporary reference)
            self.append_event_log(event)

            print(f"[EVENTING] Published event: {event['clip_id']}")

            state[stream_id]["processed_objects"].append(clip_name)
            self.save_state(state)

    def start(self) -> None:
        """
        Ensures topic exists and starts the polling loop.
        """
        print("[EVENTING] Starting Eventing Service...")
        print(f"[EVENTING] Kafka topic: {self.kafka_topic}")
        print(f"[EVENTING] Writing state to: {self.state_file}")
        print(f"[EVENTING] Writing event payloads to: {self.events_log_file}")

        # Ensure Kafka topic exists
        self.kafka_admin.ensure_topic(
            topic_name=self.kafka_topic,
            num_partitions=1,
            replication_factor=1,
        )
        print(f"[EVENTING] Topic ready: {self.kafka_topic}")

        state = self.load_state()

        while True:
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
            
            time.sleep(self.poll_interval_seconds)
