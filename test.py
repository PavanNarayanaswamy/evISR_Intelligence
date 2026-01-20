# check_kafka_messages.py
from kafka import KafkaConsumer
import json
import kafka_consumer.consumer_config as config

def check_messages():
    print(f"Checking topic: {config.KAFKA_TOPIC_INPUT}")
    
    consumer = KafkaConsumer(
        config.KAFKA_TOPIC_INPUT,
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset='earliest',
        enable_auto_commit=False,
        consumer_timeout_ms=3000  # 3 second timeout
    )
    
    message_count = 0
    
    print("Polling for messages...")
    for message in consumer:
        print(f"\n=== Message found! ===")
        print(f"Partition: {message.partition}")
        print(f"Offset: {message.offset}")
        print(f"Key: {message.key}")
        
        # Try to decode the value
        try:
            # First try to decode as UTF-8 JSON
            value_str = message.value.decode('utf-8')
            print(f"Value (raw string): {value_str}")
            
            # Try to parse as JSON
            try:
                value_json = json.loads(value_str)
                print(f"Value (parsed JSON): {json.dumps(value_json, indent=2)}")
            except json.JSONDecodeError:
                print(f"Value is not valid JSON: {value_str}")
                
        except UnicodeDecodeError:
            print(f"Value (raw bytes, not UTF-8): {message.value}")
        
        message_count += 1
    
    consumer.close()
    
    if message_count == 0:
        print(f"\n❌ No messages found in topic: {config.KAFKA_TOPIC_INPUT}")
        print("You need to produce messages to the topic first.")
    else:
        print(f"\n✅ Found {message_count} message(s) in topic")

if __name__ == "__main__":
    check_messages()

# check_kafka_messages_with_klv.py

# from kafka import KafkaConsumer
# import json
# import subprocess
# import os
# import kafka_consumer.consumer_config as config


# def has_klv_metadata(ts_path: str) -> bool:
#     """
#     Returns True if the TS file contains a data (KLV) stream.
#     Uses ffprobe (safe, read-only).
#     """
#     if not os.path.exists(ts_path):
#         print(f"   ❌ TS file does not exist: {ts_path}")
#         return False

#     cmd = [
#         "ffprobe",
#         "-v", "error",
#         "-select_streams", "d",              # data streams only
#         "-show_entries", "stream=index,codec_type",
#         "-of", "json",
#         ts_path,
#     ]

#     result = subprocess.run(
#         cmd,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         text=True
#     )

#     if result.returncode != 0:
#         print(f"   ❌ ffprobe failed: {result.stderr.strip()}")
#         return False

#     try:
#         data = json.loads(result.stdout)
#         streams = data.get("streams", [])
#         return len(streams) > 0
#     except json.JSONDecodeError:
#         return False


# def check_messages():
#     print(f"\n📡 Checking Kafka topic: {config.KAFKA_TOPIC_INPUT}\n")

#     consumer = KafkaConsumer(
#         config.KAFKA_TOPIC_INPUT,
#         bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
#         auto_offset_reset="earliest",
#         enable_auto_commit=False,
#         value_deserializer=lambda m: json.loads(m.decode("utf-8")),
#         consumer_timeout_ms=5000,
#     )

#     message_count = 0

#     for message in consumer:
#         event = message.value
#         message_count += 1

#         print("\n" + "=" * 60)
#         print(f"📨 Message #{message_count}")
#         print(f"Partition: {message.partition}, Offset: {message.offset}")

#         clip_id = event.get("clip_id")
#         ts_path = event.get("ts_path") or event.get("local_ts_path")

#         print(f"Clip ID : {clip_id}")
#         print(f"TS Path : {ts_path}")

#         if not ts_path:
#             print("   ⚠️ No TS path in message — skipping")
#             continue

#         print("🔍 Checking for KLV metadata...")

#         if has_klv_metadata(ts_path):
#             print("   ✅ KLV METADATA FOUND")
#         else:
#             print("   ❌ NO KLV METADATA")

#     consumer.close()

#     if message_count == 0:
#         print("\n❌ No messages found in topic")
#     else:
#         print(f"\n✅ Processed {message_count} Kafka message(s)")


# if __name__ == "__main__":
#     check_messages()
