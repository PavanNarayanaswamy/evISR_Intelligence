# zenml_pipeline/pipeline_trigger.py
from zenml_pipeline.pipeline import run_isr_pipeline
from zenml.client import Client
import os
import sys
import json
import datetime
from kafka import KafkaProducer

from utils import config
from utils.logger import get_logger
from utils.event_logger import append_event_to_file

logger = get_logger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import kafka_consumer.consumer_config as config

JARS = [
    "jars/jmisb-api-1.12.0.jar",
    "jars/jmisb-core-common-1.12.0.jar",
    "jars/slf4j-api-1.7.36.jar",
    "jars/slf4j-simple-1.7.36.jar",
]
PIPELINE_EVENTS_FILE = "pipeline_events.json"

def get_artifact_value(artifact_response):
    """Extract artifact value from ZenML StepRunResponse output."""
    try:
        # ZenML v2 returns list[ArtifactVersionResponse]
        if isinstance(artifact_response, list):
            artifact_response = artifact_response[0]

        return artifact_response.load()

    except Exception as e:
        logger.error(f"Failed to load artifact: {e}")
        return None


def trigger_pipeline(event: dict, pipeline_name: str) -> None:
    """Trigger ZenML pipeline and send result to output topic"""
    
    producer = KafkaProducer(
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    
    try:
        logger.info(f"[PIPELINE] Starting pipeline for clip: {event['clip_id']}")

        pipeline_run = run_isr_pipeline(
            clip_id=event["clip_id"],
            clip_uri=event["clip_uri"],
            jars=JARS,
            output_bucket=config.OUTPUT_BUCKET,
            output_bucket_detection=config.OUTPUT_BUCKET_DETECTION,
            output_bucket_fusion=config.OUTPUT_BUCKET_FUSION,
            output_bucket_summary=config.OUTPUT_BUCKET_SUMMARY,
            output_path=config.OUTPUT_PATH,
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
            distance_threshold=config.DISTANCE_THRESHOLD,
            hit_counter_max=config.HIT_COUNTER_MAX,
            initialization_delay=config.INITIALIZATION_DELAY,
            distance_function=config.DISTANCE_FUNCTION,
            pipeline_name=pipeline_name
        )

        extraction_uri = None
        decoding_uri = None
        detection_uri = None
        fusion_uri = None
        summary_uri = None

        steps = pipeline_run.steps

        # ---- KLV STEP ----
        if "klv_extraction_agent" in steps:
            klv_step = steps["klv_extraction_agent"]
            extraction_uri = get_artifact_value(
                klv_step.outputs["klv_extraction_uri"]
            )
            decoding_uri = get_artifact_value(
                klv_step.outputs["klv_decoding_uri"]
            )

        # ---- DETECTION STEP ----
        if "object_detection_agent" in steps:
            det_step = steps["object_detection_agent"]
            detection_uri = get_artifact_value(
                det_step.outputs["detection_uri"]
            )

        # ---- FUSION STEP ----
        if "fusion_context_agent" in steps:
            fusion_step = steps["fusion_context_agent"]
            fusion_uri = get_artifact_value(
                fusion_step.outputs["fusion_uri"]
            )

        # ---- SUMMARY STEP ----
        if "llm_summary_agent" in steps:
            summary_step = steps["llm_summary_agent"]
            summary_uri = get_artifact_value(
                summary_step.outputs["summary_uri"]
            )


        # -------------------------------------------------
        # WARNINGS
        # -------------------------------------------------
        if extraction_uri is None:
            logger.warning(f"KLV extraction missing for {event['clip_id']}")

        if decoding_uri is None:
            logger.warning(f"KLV decoding missing for {event['clip_id']}")

        if detection_uri is None:
            logger.warning(f"Detection output missing for {event['clip_id']}")

        if fusion_uri is None:
            logger.warning(f"Fusion output missing for {event['clip_id']}")

        if summary_uri is None:
            logger.warning(f"Summary output missing for {event['clip_id']}")

        output_event = {
            "clip_id": event["clip_id"],
            "clip_uri": event["clip_uri"],
            "klv_extraction_uri": extraction_uri,
            "klv_decoding_uri": decoding_uri,
            "object_detection_uri": detection_uri,
            "fusion_uri": fusion_uri,
            "summary_uri": summary_uri,
            "status": "success",
            "processed_at": datetime.datetime.now().isoformat(),
        }

        producer.send(config.KAFKA_TOPIC_OUTPUT, output_event)
        producer.flush()
        append_event_to_file(output_event, PIPELINE_EVENTS_FILE)

        logger.info(f"[PIPELINE] Successfully processed clip {event['clip_id']}")

    except Exception as e:
        error_event = {
            "clip_id": event["clip_id"],
            "clip_uri": event["clip_uri"],
            "status": "error",
            "error": str(e),
            "processed_at": datetime.datetime.now().isoformat(),
        }

        producer.send(config.KAFKA_TOPIC_OUTPUT, error_event)
        producer.flush()
        append_event_to_file(error_event, PIPELINE_EVENTS_FILE)

        logger.error(f"[PIPELINE] Error processing clip {event['clip_id']}: {e}")

    finally:
        producer.close()
