#!/bin/bash

PID_FILE="evisr.pids"

if [ ! -f "$PID_FILE" ]; then
    echo "No running EVISR services found."
    exit 0
fi

echo "Stopping EVISR services..."

while read pid; do
    kill $pid 2>/dev/null || true
done < "$PID_FILE"

rm -f "$PID_FILE"

echo "All services stopped successfully."