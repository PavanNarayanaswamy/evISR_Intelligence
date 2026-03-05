#setup_evisr.sh
#!/bin/bash

set -e

LOG_DIR="logs"
SETUP_LOG="$LOG_DIR/setup_evisr.log"

mkdir -p $LOG_DIR

echo "=============================================" | tee -a $SETUP_LOG
echo "EVISR SYSTEM SETUP STARTED" | tee -a $SETUP_LOG
echo "=============================================" | tee -a $SETUP_LOG

# ==========================================
# Virtual Environment Check
# ==========================================
if [ ! -d "venv" ]; then
    echo "[INFO] Virtual environment not found. Creating..." | tee -a $SETUP_LOG
    python3 -m venv venv >> $SETUP_LOG 2>&1
fi

echo "[INFO] Activating virtual environment..." | tee -a $SETUP_LOG
source venv/bin/activate

# ==========================================
# Install Python Dependencies (Smart Install)
# ==========================================
echo "[INFO] Checking Python requirements..." | tee -a $SETUP_LOG
pip install -r requirements.txt --quiet >> $SETUP_LOG 2>&1
echo "[INFO] Python dependencies verified." | tee -a $SETUP_LOG

# ==========================================
# Check Graphviz (Install Only If Missing)
# ==========================================
if ! command -v dot &> /dev/null; then
    echo "[INFO] Graphviz not found. Installing..." | tee -a $SETUP_LOG
    sudo apt install -y graphviz libgraphviz-dev >> $SETUP_LOG 2>&1
else
    echo "[INFO] Graphviz already installed. Skipping." | tee -a $SETUP_LOG
fi

# ==========================================
# Start Infrastructure Services (Only If Not Running)
# ==========================================

echo "[INFO] Checking MinIO..."
if systemctl is-active --quiet minio; then
    echo "[INFO] MinIO already running." | tee -a $SETUP_LOG
else
    echo "[INFO] Starting MinIO..." | tee -a $SETUP_LOG
    sudo systemctl start minio >> $SETUP_LOG 2>&1
fi

echo "[INFO] Checking Kafka..."
if systemctl is-active --quiet kafka; then
    echo "[INFO] Kafka already running." | tee -a $SETUP_LOG
else
    echo "[INFO] Starting Kafka..." | tee -a $SETUP_LOG
    sudo systemctl start kafka >> $SETUP_LOG 2>&1
fi

echo "[INFO] Checking Kafka UI..."
if systemctl is-active --quiet kafka-ui; then
    echo "[INFO] Kafka UI already running." | tee -a $SETUP_LOG
else
    echo "[INFO] Starting Kafka UI..." | tee -a $SETUP_LOG
    sudo systemctl start kafka-ui >> $SETUP_LOG 2>&1
fi

sleep 3

# ==========================================
# Validate Services
# ==========================================
for service in minio kafka kafka-ui; do
    if systemctl is-active --quiet $service; then
        echo "[INFO] $service is running." | tee -a $SETUP_LOG
    else
        echo "[ERROR] $service failed to start." | tee -a $SETUP_LOG
        exit 1
    fi
done

# ==========================================
# ZenML Initialization
# ==========================================
if [ ! -d ".zen" ]; then
    echo "[INFO] Initializing ZenML..." | tee -a $SETUP_LOG
    zenml init >> $SETUP_LOG 2>&1
fi

if ! pgrep -f "zenml" > /dev/null; then
    echo "[INFO] Starting ZenML server..." | tee -a $SETUP_LOG
    zenml login --local >> $SETUP_LOG 2>&1
else
    echo "[INFO] ZenML already running." | tee -a $SETUP_LOG
fi

echo ""
echo "============================================="
echo "EVISR INFRA STARTED SUCCESSFULLY"
echo "Now run: ./start_evisr.sh"
echo "============================================="
