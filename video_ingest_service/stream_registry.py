# stream_registry.py
import json
import os
import time
from typing import Dict

REGISTRY_FILE = "/tmp/evisr_stream_registry.json"
STREAM_TTL_SECONDS = 60  # configurable

def load_registry() -> Dict[str, dict]:
    if not os.path.exists(REGISTRY_FILE):
        return {}
    with open(REGISTRY_FILE, "r") as f:
        return json.load(f)

def save_registry(registry: Dict[str, dict]):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)

def register_stream(port: int, source_name: str):
    registry = load_registry()
    now = time.time()

    registry[str(port)] = {
        "port": port,
        "source": source_name,
        "created_at": registry.get(str(port), {}).get("created_at", now),
        "last_heartbeat": now,
    }

    save_registry(registry)

def heartbeat(port: int):
    registry = load_registry()
    if str(port) in registry:
        registry[str(port)]["last_heartbeat"] = time.time()
        save_registry(registry)

def cleanup_stale_streams():
    registry = load_registry()
    now = time.time()
    updated = False

    for port in list(registry.keys()):
        last_seen = registry[port].get("last_heartbeat", 0)
        if now - last_seen > STREAM_TTL_SECONDS:
            del registry[port]
            updated = True

    if updated:
        save_registry(registry)
