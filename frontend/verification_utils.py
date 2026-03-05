import json
import tempfile
import os
import time
from datetime import datetime
from typing import List, Dict, Any

from confluent_kafka import Consumer, Producer
from zenml_pipeline.minio_utils import get_minio_client, parse_minio_uri
from kafka_consumer.consumer_config import KAFKA_BOOTSTRAP_SERVERS
from io import BytesIO
from datetime import datetime

PIPELINE_TOPIC = "pipeline-results"
VERIFICATION_TOPIC = "verification-results"

APPROVED_BUCKET = "approved-clips"
REJECTED_BUCKET = "rejected-clips"


# -------------------------------------------------
# KAFKA CONSUMER
# -------------------------------------------------
def create_consumer():

    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": "summary-verification-ui",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True
    })

    consumer.subscribe([PIPELINE_TOPIC])

    return consumer


# -------------------------------------------------
# FETCH EVENTS FROM KAFKA
# -------------------------------------------------
def consume_pipeline_results(timeout=10) -> List[Dict[str, Any]]:

    print("Connecting to Kafka...")

    consumer = create_consumer()

    events = []
    start_time = time.time()

    while time.time() - start_time < timeout:

        msg = consumer.poll(1.0)

        if msg is None:
            continue

        if msg.error():
            print(msg.error())
            continue

        event = json.loads(msg.value().decode())

        print("Received:", event)

        # ✅ ONLY SUCCESS EVENTS
        if (
            event.get("status") == "success"
            and event.get("clip_uri")
            and event.get("summary_uri")
        ):
            event["verification_status"] = "pending"
            events.append(event)

    consumer.close()

    print(f"Loaded {len(events)} clips")

    events.sort(
        key=lambda x: x.get("processed_at", ""),
        reverse=True
    )

    return events


# -------------------------------------------------
# PRODUCER
# -------------------------------------------------
def create_producer():
    return Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS
    })


# -------------------------------------------------
# SUMMARY
# -------------------------------------------------
def get_summary_content(summary_uri):

    bucket, path = parse_minio_uri(summary_uri)
    client = get_minio_client()

    obj = client.get_object(bucket, path)
    return obj.read().decode()


# -------------------------------------------------
# VIDEO DOWNLOAD
# -------------------------------------------------
def download_video(video_uri):

    bucket, path = parse_minio_uri(video_uri)
    client = get_minio_client()

    temp_path = os.path.join(
        tempfile.gettempdir(),
        os.path.basename(path)
    )

    response = client.get_object(bucket, path)

    with open(temp_path, "wb") as f:
        for chunk in response.stream(32 * 1024):
            f.write(chunk)

    return temp_path


# -------------------------------------------------
# SAVE VERIFICATION
# -------------------------------------------------
def save_verification(clip_event, status, reviewer,comment):

    producer = create_producer()

    record = {
        "clip_id": clip_event["clip_id"],
        "status": status,
        "reviewer": reviewer,
        "reviewer_comment": comment,
        "verified_at": datetime.utcnow().isoformat(),
        "clip_uri": clip_event["clip_uri"],
        "summary_uri": clip_event["summary_uri"]
    }

    producer.produce(
        VERIFICATION_TOPIC,
        json.dumps(record).encode()
    )
    producer.flush()

    store_verification_object(record)

    return True


# -------------------------------------------------
# STORE RESULT
# -------------------------------------------------


def store_verification_object(record):

    client = get_minio_client()
    bucket = "verification-results"

    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    clip_id = record["clip_id"]

    # -----------------------------
    # DELETE OLD STATE (approved & rejected)
    # -----------------------------
    for prefix in ["approved/", "rejected/"]:

        objects = client.list_objects(
            bucket,
            prefix=prefix,
            recursive=True
        )

        for obj in objects:
            if obj.object_name.endswith(f"{clip_id}.json"):
                print("Deleting old state:", obj.object_name)
                client.remove_object(bucket, obj.object_name)

    # -----------------------------
    # WRITE NEW STATE
    # -----------------------------
    status_folder = (
        "approved"
        if record["status"] == "approved"
        else "rejected"
    )

    now = datetime.utcnow()

    object_name = (
        f"{status_folder}/"
        f"{now.strftime('%Y/%m/%d/%H')}/"
        f"{clip_id}.json"
    )

    payload = json.dumps(record, indent=2).encode()
    data_stream = BytesIO(payload)

    client.put_object(
        bucket,
        object_name,
        data=data_stream,
        length=len(payload),
        content_type="application/json"
    )