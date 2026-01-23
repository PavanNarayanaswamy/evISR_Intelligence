
import json
from confluent_kafka import Producer
from utils.logger import get_logger

logger = get_logger(__name__)

class KafkaProducerClient:
    def __init__(self, bootstrap_servers: str):
        self.producer = Producer({"bootstrap.servers": bootstrap_servers})

    def _delivery_report(self, err, msg):
        if err is not None:
            logger.error(f"[KAFKA-PRODUCER] Delivery failed: {err}")
        else:
            logger.info(
                f"[KAFKA-PRODUCER] Delivered to {msg.topic()} "
                f"[{msg.partition()}] offset={msg.offset()}"
            )

    def publish(self, topic: str, key: str, value: dict) -> None:
        self.producer.produce(
            topic=topic,
            key=key,
            value=json.dumps(value),
            callback=self._delivery_report,
        )
        self.producer.flush()
