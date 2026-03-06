import os
import requests
import time

TILE_SERVER = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

MAX_ZOOM = 5
TILE_DIR = "tiles"

headers = {
    "User-Agent": "evISR-map-downloader/1.0 (offline tiles)"
}

for z in range(MAX_ZOOM + 1):

    for x in range(2**z):
        for y in range(2**z):

            path = f"{TILE_DIR}/{z}/{x}"
            os.makedirs(path, exist_ok=True)

            file = f"{path}/{y}.png"

            if os.path.exists(file):
                continue

            url = TILE_SERVER.format(z=z, x=x, y=y)

            try:
                r = requests.get(url, headers=headers, timeout=10)

                if r.status_code == 200:
                    with open(file, "wb") as f:
                        f.write(r.content)

                    print("Downloaded", z, x, y)

                else:
                    print("Failed", z, x, y, r.status_code)

            except Exception as e:
                print("Error", z, x, y)

            time.sleep(0.3)