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

## Initial System Setup (First-Time Setup Only)

If you are running this project on a **new system**, you must first install and configure the required infrastructure services:

- **MinIO**
- **Kafka (KRaft Mode)**
- **Kafbat (Kafka UI)**
- **Langfuse**
- **Ollama (with required model pre-downloaded)**

> **Important (Ollama Model Requirement)**  
> This project depends on the following Ollama model being available locally: qwen3-vl:30b

👉 Please refer to the **Service Setup Documentation** for detailed step-by-step instructions on:
- Installing dependencies (Java, MinIO, Kafka, etc.)
- Creating system users
- Configuring service files
- Enabling and starting services

---

### For Existing Systems

If the system is already configured:
- Ensure services are installed
- Ensure systemd services are created
- You can directly proceed to **STEP 1**


## STEP 1: Start Infrastructure Services

Start all required services:

```bash
sudo systemctl start minio kafka kafka-ui

```

Check status:

```bash
sudo systemctl status minio kafka kafka-ui
```

Other useful commands:

```bash
sudo systemctl stop minio kafka kafka-ui
sudo systemctl restart minio kafka kafka-ui
```

Access Points:
- MinIO
    - API: http://localhost:9000
    - Console: http://localhost:9001
- kafka
    - Bootstrap Server: `127.0.0.1:9092`
- Kafka UI (Kafbat)
    - UI: http://localhost:8085

### Langfuse Setup (One-Time)

After hosting Langfuse:
- Open the UI: http://localhost:3000
- Create setup:
    - Create Organization
    - Create Project
    - Generate API Keys
- Add keys to .env file:
    - LANGFUSE_PUBLIC_KEY=your_public_key
    - LANGFUSE_SECRET_KEY=your_secret_key

---

## STEP 2: Python Environment Setup

### 2.1 Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2.2 Upgrade pip

```bash
pip install --upgrade pip
```

### 2.3 Install Project Dependencies

```bash
pip install --no-cache-dir --progress-bar off -r requirements.txt
```

### 2.4 Install openscenesense_ollama (Required)
This package is installed separately to avoid dependency conflicts.

```bash
pip install openscenesense_ollama --no-deps
```

---

## STEP 3: Install Agentic AI Dependencies (Graphviz)

LangGraph requires Graphviz for graph compilation and visualization.

```bash
sudo apt update
sudo apt install graphviz graphviz-dev
```
This installs the Graphviz system libraries required for LangGraph DAG compilation.

---

## STEP 4: Initialize ZenML

```bash
zenml init
zenml login --local
```

ZenML is used for orchestrating of streaming pipelines and experimentation.

---

## Alternative: Automated Setup & Execution (Using Shell Scripts)

In addition to the manual setup described above, the EVISR platform provides shell scripts to automate the entire setup and execution process.

---

### Automated Infrastructure & Environment Setup

Instead of manually performing STEP 1 → STEP 4, you can run:

```bash
./setup_evisr.sh
```
---


## STEP 5: Start Video Streaming

### 5.1 Run Stream Simulator

```bash
python stream_video.py --mode multi --count <number_of_streams>
```
- <number_of_streams> controls how many parallel video streams will run.
- Example:
    - --count 1 → Single stream

    - --count 2 → Two parallel streams

    - --count N → N parallel streams

This simulates live ISR video feeds dynamically based on the provided stream count.

### 5.2 Single Stream (Default Port)
```bash
python stream_video.py
```

### 5.3 Single Stream on a Specific Port
```bash
PYTHONPATH=. python3 stream_video.py --mode single --port <port_number>
```

This simulates a live ISR video feed.

---

## STEP 6: Video Ingestion

### 6.1 Offline Video Clip Ingestion

Splits a single video file into fixed-duration (30-second) clips and uploads them to MinIO.

```bash
PYTHONPATH=. python3 video_ingest_service/ingest_video_clip.py
```

### 6.2 Live Streaming Ingestion

Segments the video stream into 30-second clips and uploads them to MinIO.

```bash
PYTHONPATH=. python3 video_ingest_service/ingest_video_streaming.py
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

## STEP 8: MCP Server (Agent Layer)

Starts the agentic execution layer for downstream intelligence processing.

```bash
PYTHONPATH=. python3 agents/mcp_server.py
```
---

## STEP 9: Kafka Consumer Autoscaler

Automatically spawns multiple consumers based on Kafka partitions.

```bash
PYTHONPATH=. python3 kafka_consumer/consumer_autoscaler.py
```

---

## STEP 10: Frontend Application

Starts the frontend UI for interaction.

```bash
PYTHONPATH=. python3 frontend/app.py
```

---

## Alternative: Automated Application Startup (Using Shell Scripts)

Instead of manually running STEP 5 → STEP 10, you can start the full pipeline using:

```bash
./start_evisr.sh
```
This script will:
- Ask for number of video streams
- Start stream simulator
- Run ingestion pipelines (offline + streaming)
- Start Kafka eventing service
- Launch MCP server (agent layer)
- Start Kafka consumer autoscaler
- Launch frontend application
- Monitor all processes (auto-restart on failure)
---

## Stop the System (Using Shell Scripts)

Stop the System:

```bash
./stop_evisr.sh
```
This script will:
- Gracefully stop all running EVISR processes
- Kill any remaining background processes
- Clean up PID tracking
---

## Logging & Configuration

- Centralized configuration: `utils/config.py`
- Centralized logging: `utils/logger.py`
- Runtime logs are written to the `logs/` directory

---


