import json
import argparse
import os
import sys
import time
import threading
from queue import Queue
from kafka import KafkaConsumer, KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import TopicAlreadyExistsError
from utils.logger import get_logger
import kafka_consumer.consumer_config as config

logger = get_logger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from zenml_pipeline.pipeline_trigger import trigger_pipeline

# ==================================================
# EVENT QUEUE
# ==================================================
event_queue = Queue()


# ==================================================
# PIPELINE WORKER (SEQUENTIAL EXECUTION)
# ==================================================
def pipeline_worker():
    logger.info("[WORKER] Pipeline worker started")

    while True:
        event = event_queue.get()

        if event is None:
            break

        try:
            logger.info(f"[WORKER] Running pipeline for {event['clip_id']}")
            trigger_pipeline(event)
            logger.info(f"[WORKER] Completed pipeline for {event['clip_id']}")
        except Exception as e:
            logger.error(f"Pipeline failed for {event['clip_id']}: {e}")

        event_queue.task_done()


# ==================================================
# ENSURE KAFKA TOPICS
# ==================================================
def ensure_topics_exist():
    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            client_id="klv-metadata-admin",
        )
        existing_topics = admin_client.list_topics()
        topics_to_create = []
        if config.KAFKA_TOPIC_INPUT not in existing_topics:
            logger.info(f"Creating input topic: {config.KAFKA_TOPIC_INPUT}")
            topics_to_create.append(
                NewTopic(
                    name=config.KAFKA_TOPIC_INPUT,
                    num_partitions=1,
                    replication_factor=1,
                )
            )
        if config.KAFKA_TOPIC_OUTPUT not in existing_topics:
            logger.info(f"Creating output topic: {config.KAFKA_TOPIC_OUTPUT}")
            topics_to_create.append(
                NewTopic(
                    name=config.KAFKA_TOPIC_OUTPUT,
                    num_partitions=1,
                    replication_factor=1,
                )
            )
        if topics_to_create:
            admin_client.create_topics(topics_to_create)
            logger.info(f"Created {len(topics_to_create)} topic(s)")
        else:
            logger.info("All required topics already exist")
    except TopicAlreadyExistsError:
        logger.info("Topics already exist")
    except Exception as e:
        logger.warning(f"Could not create topics: {e}")
        logger.warning("Make sure Kafka is running and accessible")
    finally:
        if "admin_client" in locals():
            admin_client.close()

# ==================================================
# HISTORICAL MODE
# ==================================================
def process_historical_events():
    consumer = KafkaConsumer(
        config.KAFKA_TOPIC_INPUT,
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        group_id=None,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=3000,
    )

    logger.info("[KAFKA] Processing ALL historical events...")

    processed = 0

    for message in consumer:
        event = message.value

        if not event.get("has_klv", False):
            continue

        logger.info(f"[KAFKA] Queueing pipeline for {event['clip_id']}")
        event_queue.put(event)
        processed += 1

    consumer.close()
    logger.info(f"[KAFKA] Done. Total queued: {processed}")


# ==================================================
# MAIN
# ==================================================
def main():
    ensure_topics_exist()

    # Start pipeline worker thread
    worker_thread = threading.Thread(target=pipeline_worker, daemon=True)
    worker_thread.start()

    parser = argparse.ArgumentParser(
        description="Kafka Consumer for KLV Metadata Extraction"
    )
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Process all historical events from beginning",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Process only new events (default)",
    )

    args = parser.parse_args()

    if args.historical:
        process_historical_events()
        event_queue.join()
        return

    # ==================================================
    # LIVE MODE
    # ==================================================
    consumer = KafkaConsumer(
        config.KAFKA_TOPIC_INPUT,
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )

    logger.info("[KAFKA] Listening for new clip events (live mode)...")

    for message in consumer:
        event = message.value

        if not event.get("has_klv", False):
            continue

        logger.info(f"[KAFKA] Queueing pipeline for {event['clip_id']}")
        event_queue.put(event)


if __name__ == "__main__":
    main()