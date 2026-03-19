#start_evisr.sh
#!/bin/bash

set -e

LOG_DIR="logs"
SYSTEM_LOG="$LOG_DIR/evisr_runner.log"
PID_FILE="evisr_multi.pids"

mkdir -p $LOG_DIR

# ==========================================
# Prevent Duplicate Run
# ==========================================
if [ -f "$PID_FILE" ] && [ -s "$PID_FILE" ]; then
    echo "[ERROR] EVISR already running."
    exit 1
fi

: > $PID_FILE

# ==========================================
# Ask Stream Count
# ==========================================
echo ""
read -p "Enter number of streams to run (default: 2): " STREAM_COUNT
STREAM_COUNT=${STREAM_COUNT:-2}

if ! [[ "$STREAM_COUNT" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid input."
    exit 1
fi

echo "=============================================" | tee -a $SYSTEM_LOG
echo "Starting EVISR at $(date)" | tee -a $SYSTEM_LOG
echo "Streams: $STREAM_COUNT" | tee -a $SYSTEM_LOG
echo "=============================================" | tee -a $SYSTEM_LOG

export ZENML_AUTO_OPEN_DASHBOARD=false
export PYTHONPATH=.

source venv/bin/activate

# ==========================================
# Ensure Infrastructure Running
# ==========================================
for service in minio kafka kafka-ui; do
    if ! systemctl is-active --quiet $service; then
        echo "[INFO] Starting $service..."
        sudo systemctl start $service
    fi
done

sleep 5

# ==========================================
# Graceful Shutdown
# ==========================================
cleanup() {
    echo ""
    echo "[INFO] Shutting down EVISR..."

    while read -r pid; do
        if ps -p $pid > /dev/null; then
            kill $pid 2>/dev/null || true
            sleep 1
            kill -9 $pid 2>/dev/null || true
        fi
    done < "$PID_FILE"

    rm -f "$PID_FILE"
    echo "[INFO] Shutdown complete."
    exit 0
}

trap cleanup SIGINT SIGTERM

# ==========================================
# STEP 1: Offline Clip Ingestion
# ==========================================
python3 video_ingest_service/ingest_video_clip.py >> $SYSTEM_LOG 2>&1

# ==========================================
# STEP 2: Stream Simulator
# ==========================================
python3 stream_video.py --mode multi --count $STREAM_COUNT >> /dev/null 2>&1 &
echo $! >> $PID_FILE

sleep 5

# ==========================================
# STEP 3: Streaming Ingestion
# ==========================================
python3 video_ingest_service/ingest_video_streaming.py >> $SYSTEM_LOG 2>&1 &
echo $! >> $PID_FILE

sleep 5

# ==========================================
# STEP 4: Eventing
# ==========================================
python3 eventing/main.py >> $SYSTEM_LOG 2>&1 &
echo $! >> $PID_FILE

sleep 3

# ==========================================
# STEP 5: MCP Server
# ==========================================
echo "[INFO] Starting MCP Server..." | tee -a $SYSTEM_LOG
PYTHONPATH=. python3 agents/mcp_server.py >> $SYSTEM_LOG 2>&1 &
echo $! >> $PID_FILE

sleep 3

# ==========================================
# STEP 6: Autoscaler
# ==========================================
python3 kafka_consumer/consumer_autoscaler.py >> $SYSTEM_LOG 2>&1 &
echo $! >> $PID_FILE

sleep 3

# ==========================================
# STEP 7: Frontend
# ==========================================
echo "[INFO] Starting Frontend..." | tee -a $SYSTEM_LOG
PYTHONPATH=. python3 frontend/app.py >> $SYSTEM_LOG 2>&1 &
echo $! >> $PID_FILE

sleep 3

echo ""
echo "============================================="
echo "EVISR STARTED SUCCESSFULLY"
echo "Streams running: $STREAM_COUNT"
echo "Press Ctrl+C to stop."
echo "============================================="
echo ""

# ==========================================
# Health Monitor
# ==========================================
while true; do
    while read -r pid; do
        if ! ps -p "$pid" > /dev/null; then
            echo "[ERROR] Process $pid crashed." | tee -a $SYSTEM_LOG
            cleanup
        fi
    done < "$PID_FILE"
    sleep 5
done