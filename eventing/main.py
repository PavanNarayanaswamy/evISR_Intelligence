
from utils import config
from eventing.event_service import EventingService
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    logger.info("Starting EventingService main entry point.")
    try:
        service = EventingService(
            minio_endpoint=config.MINIO_ENDPOINT,
            minio_access_key=config.MINIO_ACCESS_KEY,
            minio_secret_key=config.MINIO_SECRET_KEY,
            minio_secure=config.MINIO_SECURE,
            kafka_bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            kafka_topic=config.KAFKA_TOPIC,
            state_file=config.STATE_FILE,
            events_log_file="events_log.json",
            poll_interval_seconds=config.POLL_INTERVAL_SECONDS,
        )
        service.start()
    except Exception as e:
        logger.error(f"EventingService failed: {e}")
    finally:
        logger.info("EventingService main entry point exiting.")

if __name__ == "__main__":
    main()
