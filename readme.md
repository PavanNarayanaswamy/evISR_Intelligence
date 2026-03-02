# EVISR Intelligence Platform

## Project Overview

**EVISR Intelligence** is an end-to-end video intelligence pipeline designed to ingest live and offline video streams containing embedded KLV (ST0601) metadata, perform intelligent video segmentation, extract metadata, generate Kafka-based events, and run downstream analytics such as object detection, tracking, and future context fusion.

The system is built to simulate real-world ISR (Intelligence, Surveillance, Reconnaissance) workflows using open-source components and is fully deployable on a local development environment (WSL / Ubuntu).

The platform also includes an Agentic AI execution layer built using LangGraph. 
This layer wraps key processing stages (KLV extraction, object detection, fusion, and summarization) 
into structured state-driven execution graphs with strong validation and deterministic transitions.

Instead of executing linear function calls, each intelligence stage is implemented as a stateful 
graph-based agent, enabling modularity, traceability, lifecycle management, and future extensibility.

---

## Project Root

```
evISR_Intelligence/
```

All commands below assume you are inside this directory.

---


## STEP 1: Start Infrastructure Services

### 1.1 Start MinIO

```bash
sudo systemctl start minio
sudo systemctl status minio
```

Other useful commands:

```bash
sudo systemctl stop minio
sudo systemctl restart minio
```

MinIO Console:
- API: http://localhost:9000
- Console: http://localhost:9001

---

### 1.2 Start Kafka (KRaft Mode)

```bash
sudo systemctl start kafka
sudo systemctl status kafka
```

Other useful commands:

```bash
sudo systemctl stop kafka
sudo systemctl restart kafka
```

Kafka Broker:
- Bootstrap Server: `127.0.0.1:9092`

---

## STEP 2: Download and Run MediaMTX (Video Streaming Server)

MediaMTX is used to receive and relay MPEG-TS / UDP video streams.

```bash
wget https://github.com/bluenviron/mediamtx/releases/download/v1.4.0/mediamtx_v1.4.0_linux_amd64.tar.gz
tar -xzf mediamtx_v1.4.0_linux_amd64.tar.gz
```


---

## STEP 3: Python Environment Setup

### 3.1 Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3.2 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## STEP 4: Install Agentic AI Dependencies (Graphviz)

LangGraph requires Graphviz for graph compilation and visualization.

```bash
sudo apt update
sudo apt install graphviz graphviz-dev
```
This installs the Graphviz system libraries required for LangGraph DAG compilation.

---


## STEP 5: Start Video Streaming

This command streams a `.ts` file (with embedded KLV) over UDP using FFmpeg.

```bash
PYTHONPATH=. python3 stream_video.py
```

This simulates a live ISR video feed.

---

## STEP 6: Video Ingestion

### 6.1 Live Streaming Ingestion

Segments the video stream into 30-second clips and uploads them to MinIO.

```bash
PYTHONPATH=. python3 video_ingest_service/ingest_video_streaming.py
```

### 6.2 Offline Video Clip Ingestion

Splits a single video file into fixed-duration (30-second) clips and uploads them to MinIO.

```bash
PYTHONPATH=. python3 video_ingest_service/ingest_video_clip.py
```

---

## STEP 7: Kafka Eventing Service

Generates Kafka events for each ingested video clip and maintains ingestion state.

```bash
PYTHONPATH=. python3 eventing/main.py
```

Event metadata is written to:
- `events_log.json`
- `state.json`

Kafka Topic:
- `videoclips`

---
## STEP 8: ZenML Setup (Pipeline Orchestration)

```bash
zenml init
zenml login --local
```

ZenML is used for orchestrating of streaming pipelines and experimentation.

---

## STEP 9: Consume Kafka Events (Debug / Validation)

### 9.1 Kafka CLI Consumer

```bash
/opt/kafka/bin/kafka-console-consumer.sh   --bootstrap-server 127.0.0.1:9092   --topic videoclips   --partition 0   --offset latest
```

### 9.2 Python Kafka Consumer

```bash
PYTHONPATH=. python3 kafka_consumer/consumer.py --live
```

---


## Logging & Configuration

- Centralized configuration: `utils/config.py`
- Centralized logging: `utils/logger.py`
- Runtime logs are written to the `logs/` directory

---


