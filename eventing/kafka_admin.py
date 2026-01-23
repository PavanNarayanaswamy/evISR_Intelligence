
import time
from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka import KafkaException
from utils.logger import get_logger

logger = get_logger(__name__)

class KafkaAdmin:
    def __init__(self, bootstrap_servers: str):
        self.admin = AdminClient({"bootstrap.servers": bootstrap_servers})

    def topic_exists(self, topic_name: str) -> bool:
        try:
            md = self.admin.list_topics(timeout=10)
            exists = topic_name in md.topics
            logger.debug(f"Checked topic existence: {topic_name} exists={exists}")
            return exists
        except Exception as e:
            logger.error(f"Error checking if topic exists {topic_name}: {e}")
            return False
        
    def ensure_topic(
        self,
        topic_name: str,
        num_partitions: int = 1,
        replication_factor: int = 1,
    ) -> None:
        """
        Ensure a Kafka topic exists. If missing, create it.
        """

        try:
            md = self.admin.list_topics(timeout=10)
            if topic_name in md.topics:
                logger.info(f"Kafka topic already exists: {topic_name}")
                return
        except Exception as e:
            logger.error(f"Error listing topics: {e}")
            raise

        new_topic = NewTopic(
            topic=topic_name,
            num_partitions=num_partitions,
            replication_factor=replication_factor,
        )

        futures = self.admin.create_topics([new_topic])

        # Wait for result
        for topic, future in futures.items():
            try:
                future.result(timeout=15)
                logger.info(f"Created Kafka topic: {topic}")
            except KafkaException as e:
                msg = str(e)
                if "TOPIC_ALREADY_EXISTS" in msg:
                    logger.info(f"Kafka topic already exists (race): {topic}")
                    return
                logger.error(f"Failed to create topic {topic}: {e}")
                raise
