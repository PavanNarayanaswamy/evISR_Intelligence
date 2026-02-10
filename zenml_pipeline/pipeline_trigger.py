from zenml_pipeline.pipeline import isr_pipeline
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
    """Extract the actual artifact value from ArtifactVersionResponse"""
    try:
        # Method 1: Try to get artifact content
        return artifact_response.load()
    except Exception:
        try:
            # Method 2: Try to get from artifact metadata
            client = Client()
            artifact = client.get_artifact_version(artifact_response.id)
            return artifact.load()
        except Exception as e:
            logger.error(f"Failed to load artifact: {e}")
            return None

def trigger_pipeline(event: dict):
    """Trigger ZenML pipeline and send result to output topic"""
    
    producer = KafkaProducer(
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    try:
        logger.info(f"[PIPELINE] Starting pipeline for clip: {event['clip_id']}")

        pipeline_run = isr_pipeline(
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
        )

        extraction_uri = None
        decoding_uri = None
        detection_uri = None
        fusion_uri = None
        summary_uri = None


        for step_name, step_output in pipeline_run.steps.items():
            #Skip multi-output steps we don't care about
            if step_name == "download_clip":
                continue

            try:
                if step_output.output is not None:
                    artifact_value = get_artifact_value(step_output.output)
                    
                    # if step_name == "decode_metadata":
                    #     decoding_uri = artifact_value
                    # elif step_name == "extract_metadata":
                    #     extraction_uri = artifact_value
                    if step_name == "klv_extraction_agent":
                        # In many ZenML versions, multiple outputs are accessible via .outputs or dict-like.
                        # Try dict-style first:
                        try:
                            extraction_uri = get_artifact_value(step_output.output["klv_extraction_uri"])
                            decoding_uri = get_artifact_value(step_output.output["klv_decoding_uri"])
                        except Exception:
                            # Fallback: if ZenML returns a tuple-like single artifact, load it then unpack
                            val = get_artifact_value(step_output.output)
                            if isinstance(val, (tuple, list)) and len(val) == 2:
                                extraction_uri, decoding_uri = val
                    elif step_name == "object_detection_agent":
                        detection_uri = artifact_value
                    elif step_name == "fusion_context_agent":
                        fusion_uri = artifact_value
                    elif step_name == "llm_summary_agent":
                        summary_uri = artifact_value

                    logger.info(f"[PIPELINE] Step {step_name} output: {artifact_value}")

            except Exception as e:
                logger.error(f"[PIPELINE] Error getting output from step {step_name}: {e}")



        if extraction_uri is None:
            logger.warning(
                f"[PIPELINE] decode_metadata_step did not produce output "
                f"for clip {event['clip_id']}"
            )

        if decoding_uri is None:
            logger.warning(
                f"[PIPELINE] extract_metadata_step did not produce output "
                f"for clip {event['clip_id']}"
            )

        if detection_uri is None:
            logger.warning(
                f"[PIPELINE] object_detection step did not produce output "
                f"for clip {event['clip_id']}"
            )
        if fusion_uri is None:
            logger.warning(
                f"[PIPELINE] fusion_context step did not produce output "
                f"for clip {event['clip_id']}"
            )
        if summary_uri is None:
            logger.warning(
                f"[PIPELINE] llm_summary step did not produce output "
                f"for clip {event['clip_id']}"
            )


        output_event = {
            "clip_id": event["clip_id"],
            "clip_uri": event["clip_uri"],
            "klv_extraction_uri": extraction_uri,
            "klv_decoding_uri": decoding_uri,
            "object_detection_uri": detection_uri,
            "fusion_uri": fusion_uri,
            "summary_uri": summary_uri,
            "status": "success",
            "processed_at": datetime.datetime.now().isoformat()
        }

        producer.send(config.KAFKA_TOPIC_OUTPUT, output_event)
        producer.flush()
        append_event_to_file(output_event, PIPELINE_EVENTS_FILE)

        logger.info(f"[PIPELINE] Successfully processed clip {event['clip_id']}")
        logger.info(f"[PIPELINE] Result sent to {config.KAFKA_TOPIC_OUTPUT}")

    except Exception as e:
        # Send error event to output topic
        error_event = {
            "clip_id": event["clip_id"],
            "clip_uri": event["clip_uri"],
            "status": "error",
            "error": str(e),
            "processed_at": datetime.datetime.now().isoformat()
        }

        producer.send(config.KAFKA_TOPIC_OUTPUT, error_event)
        producer.flush()
        append_event_to_file(error_event, PIPELINE_EVENTS_FILE)

        logger.error(f"[PIPELINE] Error processing clip {event['clip_id']}: {e}")

    finally:
        producer.close()