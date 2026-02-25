import time
import subprocess
import sys
import os
from kafka import KafkaAdminClient
from utils.logger import get_logger
import kafka_consumer.consumer_config as config

logger = get_logger(__name__)

CHECK_INTERVAL = 10

MAX_CONSUMERS = 50

consumer_processes = []

def spawn_consumer():
    if len(consumer_processes) >= MAX_CONSUMERS:
        return

    logger.info("Spawning consumer process")

    p = subprocess.Popen(
        [sys.executable, "kafka_consumer/consumer.py", "--live"],
        env=os.environ.copy(),
    )
    consumer_processes.append(p)


def kill_consumer():
    if not consumer_processes:
        return

    p = consumer_processes.pop()
    logger.info(f"Stopping consumer pid={p.pid}")
    p.terminate()


def get_partition_count(admin):
    topic_md = admin.describe_topics(
        [config.KAFKA_TOPIC_INPUT]
    )[0]

    return len(topic_md["partitions"])


def main():
    admin = KafkaAdminClient(
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS
    )

    logger.info("Partition-based autoscaler started")

    while True:
        try:
            partitions = get_partition_count(admin)
            consumers = len(consumer_processes)

            logger.info(
                f"[AUTOSCALER] partitions={partitions} consumers={consumers}"
            )

            # SCALE UP until 1:1 achieved
            while consumers < partitions:
                spawn_consumer()
                consumers += 1

            # SCALE DOWN if partitions reduced
            while consumers > partitions:
                kill_consumer()
                consumers -= 1

        except Exception as e:
            logger.error(f"Autoscaler error: {e}", exc_info=True)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()