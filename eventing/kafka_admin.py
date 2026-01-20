import time
from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka import KafkaException

class KafkaAdmin:
    def __init__(self, bootstrap_servers: str):
        self.admin = AdminClient({"bootstrap.servers": bootstrap_servers})

    def topic_exists(self, topic_name: str) -> bool:
        md = self.admin.list_topics(timeout=10)
        return topic_name in md.topics
        
    def ensure_topic(
        self,
        topic_name: str,
        num_partitions: int = 1,
        replication_factor: int = 1,
    ) -> None:
        """
        Ensure a Kafka topic exists. If missing, create it.
        """
        md = self.admin.list_topics(timeout=10)
        if topic_name in md.topics:
            return

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
            except KafkaException as e:
                # If topic already exists due to race condition, ignore
                # Otherwise, raise
                msg = str(e)
                if "TOPIC_ALREADY_EXISTS" in msg:
                    return
                raise
