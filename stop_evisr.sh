#stop_evisr.sh
#!/bin/bash

PID_FILE="evisr_multi.pids"

echo "============================================="
echo "Stopping EVISR System..."
echo "============================================="

# ==========================================
# Stop Application Processes
# ==========================================
if [ -f "$PID_FILE" ] && [ -s "$PID_FILE" ]; then
    while read pid; do
        if ps -p $pid > /dev/null 2>&1; then
            echo "Stopping PID $pid ..."

            # Kill process group (if started with setsid)
            kill -TERM -$pid 2>/dev/null || kill $pid 2>/dev/null || true
            sleep 2

            # Force kill if still alive
            if ps -p $pid > /dev/null 2>&1; then
                echo "Force killing PID $pid ..."
                kill -9 $pid 2>/dev/null || true
            fi
        fi
    done < "$PID_FILE"

    rm -f "$PID_FILE"
    echo "[INFO] Application processes stopped."
else
    echo "[INFO] No PID file found."
fi

# ==========================================
# Kill Any Leftover Python Processes (Safety)
# ==========================================
echo "[INFO] Cleaning stray EVISR processes..."

pkill -f stream_video.py 2>/dev/null || true
pkill -f ingest_video_streaming.py 2>/dev/null || true
pkill -f ingest_video_clip.py 2>/dev/null || true
pkill -f eventing/main.py 2>/dev/null || true
pkill -f consumer_autoscaler.py 2>/dev/null || true

# ==========================================
# Stop ZenML Local Server (if running)
# ==========================================
zenml logout --local 2>/dev/null || true

# ==========================================
# Optional: Stop Infrastructure
# ==========================================
# sudo systemctl stop minio
# sudo systemctl stop kafka
# sudo systemctl stop kafka-ui

echo ""
echo "============================================="
echo "EVISR stopped successfully."
echo "============================================="