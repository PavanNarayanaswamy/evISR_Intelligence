from zenml_pipeline.pipeline import isr_pipeline
import os
import sys
import json
import datetime
from kafka import KafkaProducer

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

def trigger_pipeline(event: dict):
    """Trigger ZenML pipeline and send result to output topic"""
    
    # Initialize Kafka producer for output topic
    producer = KafkaProducer(
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    try:
        print(f"[PIPELINE] Starting pipeline for clip: {event['clip_id']}")
        
        # Run the ZenML pipeline
        pipeline_run = isr_pipeline(
            clip_id=event["clip_id"],
            clip_uri=event["clip_uri"],
            jars=JARS,
            output_bucket=config.OUTPUT_BUCKET,
            output_path=f"output/{event['clip_id']}.mp4",
            confidence_threshold=0.4,
        ).run()
        
        # Get the result from the pipeline
        # Assuming decode_metadata_step returns the output URI
        # You might need to adjust this based on your actual pipeline output
        output_uri = None
        for step_name, step_output in pipeline_run.steps.items():
            if step_name == "decode_metadata_step":
                output_uri = step_output.output
                break
        
        # Create output message
        output_event = {
            "clip_id": event["clip_id"],
            "clip_uri": event["clip_uri"],
            "klv_metadata_uri": output_uri or f"minio://{config.OUTPUT_BUCKET}/{event['clip_id']}.json",
            "status": "success",
            "timestamp": event.get("timestamp", ""),
            "processed_at": datetime.datetime.now().isoformat()
        }
        
        # Send to output topic
        producer.send(config.KAFKA_TOPIC_OUTPUT, output_event)
        producer.flush()
        
        print(f"[PIPELINE] Successfully processed clip {event['clip_id']}")
        print(f"[PIPELINE] Result sent to {config.KAFKA_TOPIC_OUTPUT}")
        
    except Exception as e:
        # Send error event to output topic
        error_event = {
            "clip_id": event["clip_id"],
            "clip_uri": event["clip_uri"],
            "status": "error",
            "error": str(e),
            "timestamp": event.get("timestamp", ""),
            "processed_at": datetime.datetime.now().isoformat()
        }
        
        producer.send(config.KAFKA_TOPIC_OUTPUT, error_event)
        producer.flush()
        
        print(f"[PIPELINE] Error processing clip {event['clip_id']}: {e}")
        
    finally:
        producer.close()