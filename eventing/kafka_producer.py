
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

    def publish(
        self,
        topic: str,
        partition: int,
        value: dict,
        max_retries: int = 3,
    ) -> None:
        """Publish message with retry logic for unknown partition errors."""
        import time
        
        for attempt in range(max_retries):
            try:
                self.producer.produce(
                    topic=topic,
                    partition=partition,
                    value=json.dumps(value),
                    callback=self._delivery_report,
                )
                self.producer.poll(0)
                return
            except Exception as e:
                error_str = str(e)
                # Check for unknown partition error
                if "_UNKNOWN_PARTITION" in error_str or "unknown partition" in error_str.lower():
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Unknown partition {partition} on attempt {attempt + 1}/{max_retries}. "
                            f"Refreshing metadata and retrying..."
                        )
                        self.refresh_metadata()
                        time.sleep(1)
                        continue
                # Re-raise if not a partition error or all retries exhausted
                raise

    def refresh_metadata(self):
        """
        Force Kafka producer metadata refresh
        """
        logger.info("Refreshing Kafka producer metadata")

        # poll forces metadata request
        self.producer.poll(0)

        # wait broker propagation
        import time
        time.sleep(2)