#!/bin/bash

set -e

LOG_DIR="logs"
SYSTEM_LOG="$LOG_DIR/multiprocessing_runner.log"
PID_FILE="evisr_multi.pids"

mkdir -p $LOG_DIR

# ==========================================
# Stream Count Argument
# ==========================================
echo ""
read -p "Enter number of streams to run (default: 2): " STREAM_COUNT

# If empty → default to 2
STREAM_COUNT=${STREAM_COUNT:-2}

# Validate numeric input
if ! [[ "$STREAM_COUNT" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid input. Please enter a numeric value."
    exit 1
fi

if [ "$STREAM_COUNT" -le 0 ]; then
    echo "[ERROR] Stream count must be greater than 0."
    exit 1
fi

# ==========================================
# Prevent Duplicate Execution
# ==========================================
if [ -f "$PID_FILE" ] && [ -s "$PID_FILE" ]; then
    echo "[ERROR] Multiprocessing EVISR already running."
    echo "Delete $PID_FILE if not."
    exit 1
fi

: > $PID_FILE

echo "=============================================" | tee -a $SYSTEM_LOG
echo "Starting EVISR MULTIPROCESSING System at $(date)" | tee -a $SYSTEM_LOG
echo "Stream Count: $STREAM_COUNT" | tee -a $SYSTEM_LOG
echo "=============================================" | tee -a $SYSTEM_LOG

export ZENML_AUTO_OPEN_DASHBOARD=false
export PYTHONPATH=.

# ==========================================
# Virtual Environment
# ==========================================
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..." | tee -a $SYSTEM_LOG
    python3 -m venv venv
fi

source venv/bin/activate

# ==========================================
# Install Python Dependencies
# ==========================================
if [ ! -f "venv/.deps_installed" ]; then
    echo "[INFO] Installing Python dependencies..." | tee -a $SYSTEM_LOG
    pip install -r requirements.txt >> $SYSTEM_LOG 2>&1
    touch venv/.deps_installed
else
    echo "[INFO] Python dependencies already installed." | tee -a $SYSTEM_LOG
fi

# ==========================================
# Install Graphviz
# ==========================================
if ! command -v dot >/dev/null 2>&1; then
    echo "[INFO] Installing Graphviz..." | tee -a $SYSTEM_LOG
    sudo apt update >> $SYSTEM_LOG 2>&1
    sudo apt install -y graphviz graphviz-dev >> $SYSTEM_LOG 2>&1
else
    echo "[INFO] Graphviz already installed." | tee -a $SYSTEM_LOG
fi

# ==========================================
# Start Infrastructure
# ==========================================
echo "[INFO] Starting MinIO..." | tee -a $SYSTEM_LOG
sudo systemctl start minio

echo "[INFO] Starting Kafka..." | tee -a $SYSTEM_LOG
sudo systemctl start kafka

sleep 5

if ! systemctl is-active --quiet minio; then
    echo "[ERROR] MinIO failed to start." | tee -a $SYSTEM_LOG
    exit 1
fi

if ! systemctl is-active --quiet kafka; then
    echo "[ERROR] Kafka failed to start." | tee -a $SYSTEM_LOG
    exit 1
fi

echo "[INFO] Infrastructure running." | tee -a $SYSTEM_LOG

# ==========================================
# ZenML Init
# ==========================================
if [ ! -d ".zen" ]; then
    echo "[INFO] Initializing ZenML..." | tee -a $SYSTEM_LOG
    zenml init >> $SYSTEM_LOG 2>&1
fi

zenml login --local >> $SYSTEM_LOG 2>&1

# ==========================================
# Graceful Shutdown
# ==========================================
cleanup() {
    echo ""
    echo "[INFO] Stopping Multiprocessing EVISR services..." | tee -a $SYSTEM_LOG

    if [ -f "$PID_FILE" ]; then
        while read pid; do
            if ps -p $pid > /dev/null 2>&1; then
                kill $pid 2>/dev/null || true
                sleep 1
                kill -9 $pid 2>/dev/null || true
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi

    echo "[INFO] Shutdown complete." | tee -a $SYSTEM_LOG
    exit 0
}

trap cleanup SIGINT SIGTERM

# ==========================================
# STEP 1: Offline Clip Ingestion
# ==========================================
echo "[INFO] Running ingest_video_clip..." | tee -a $SYSTEM_LOG
python3 video_ingest_service/ingest_video_clip.py > /dev/null 2>&1
echo "[INFO] Video clip ingestion completed." | tee -a $SYSTEM_LOG

# ==========================================
# STEP 2: Start MULTI Stream Simulator
# ==========================================
echo "[INFO] Starting MULTI stream simulator ($STREAM_COUNT streams)..." | tee -a $SYSTEM_LOG
python3 stream_video.py --mode multi --count $STREAM_COUNT > /dev/null 2>&1 &
STREAM_PID=$!
echo $STREAM_PID >> $PID_FILE

sleep 5

if ! ps -p $STREAM_PID > /dev/null; then
    echo "[ERROR] stream_video.py crashed on startup." | tee -a $SYSTEM_LOG
    exit 1
fi

# ==========================================
# STEP 3: Streaming Ingestion
# ==========================================
echo "[INFO] Starting ingest_video_streaming..." | tee -a $SYSTEM_LOG
python3 video_ingest_service/ingest_video_streaming.py > /dev/null 2>&1 &
INGEST_PID=$!
echo $INGEST_PID >> $PID_FILE

sleep 5

if ! ps -p $INGEST_PID > /dev/null; then
    echo "[ERROR] ingest_video_streaming crashed on startup." | tee -a $SYSTEM_LOG
    exit 1
fi

# ==========================================
# STEP 4: Start Eventing 
# ==========================================
echo "[INFO] Starting eventing service..." | tee -a $SYSTEM_LOG
python3 eventing/main.py >> $SYSTEM_LOG 2>&1 &
EVENT_PID=$!
echo $EVENT_PID >> $PID_FILE

sleep 3

if ! ps -p $EVENT_PID > /dev/null; then
    echo "[ERROR] Eventing service crashed on startup." | tee -a $SYSTEM_LOG
    exit 1
fi

# ==========================================
# STEP 5: Autoscaler Kafka Consumer
# ==========================================
echo "[INFO] Starting consumer_autoscaler..." | tee -a $SYSTEM_LOG
python3 kafka_consumer/consumer_autoscaler.py >> $SYSTEM_LOG 2>&1 &
CONSUMER_PID=$!
echo $CONSUMER_PID >> $PID_FILE

sleep 3

if ! ps -p $CONSUMER_PID > /dev/null; then
    echo "[ERROR] consumer_autoscaler crashed on startup." | tee -a $SYSTEM_LOG
    exit 1
fi

echo ""
echo "[INFO] EVISR MULTIPROCESSING System started successfully."
echo "Streams running: $STREAM_COUNT"
echo "Press Ctrl+C to stop everything."
echo ""

# ==========================================
# Health Monitor Loop
# ==========================================
while true; do
    for pid in $(cat $PID_FILE); do
        if ! ps -p $pid > /dev/null; then
            echo "[ERROR] Process $pid died unexpectedly." | tee -a $SYSTEM_LOG
            cleanup
        fi
    done
    sleep 5
done