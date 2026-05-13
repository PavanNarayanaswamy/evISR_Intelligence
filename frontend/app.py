import json
import os
import time
import subprocess
import math
from typing import Any

import folium
from flask import Flask, render_template, request, redirect, url_for, Response, flash
from minio.error import S3Error
from frontend import verification_utils

app = Flask(__name__)
app.secret_key = "secret"

BUCKET = "verification-results"

# ---------------- IN-MEMORY CACHE ----------------
CACHE = {
    "pending": [],
    "approved": [],
    "rejected": [],
    "verified_ts": 0
}

CACHE_TTL = 10
DELETED_IDS = set()

# ---------------- SEVERITY DERIVATION ----------------
def derive_severity(score: float):

    score_int = int(round(score))

    if score_int <= 1:
        return score_int, "LOW", "green"
    elif score_int <= 3:
        return score_int, "MODERATE", "orange"
    else:
        return score_int, "CRITICAL", "red"


# ---------------- HAVERSINE ----------------
def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2

    return 2 * R * math.asin(math.sqrt(a))

# ---------------- MINIO INIT (RUN ONCE) ----------------
@app.before_request
def init_minio():
    client = verification_utils.get_minio_client()
    try:
        if not client.bucket_exists(BUCKET):
            client.make_bucket(BUCKET)
    except S3Error as e:
        print("MinIO bucket error:", e)

def enrich_clip_geo(clip):

    start_lat = clip.get("start_latitude")
    start_lon = clip.get("start_longitude")
    end_lat = clip.get("end_latitude")
    end_lon = clip.get("end_longitude")

    if all(v is not None for v in [start_lat, start_lon, end_lat, end_lon]):

        clip["center_latitude"] = (start_lat + end_lat) / 2
        clip["center_longitude"] = (start_lon + end_lon) / 2

        distance_km = haversine(start_lat, start_lon, end_lat, end_lon)
        clip["radius_km"] = max(distance_km, 0.2)

    else:
        clip["center_latitude"] = 52.0
        clip["center_longitude"] = 19.0
        clip["radius_km"] = 0.5

    score = clip.get("severity_score", 0)
    numeric, label, color = derive_severity(score)

    clip["severity_numeric"] = numeric
    clip["severity_label"] = label
    clip["severity_color"] = color
    
# ---------------- KAFKA LOAD (BUTTON DRIVEN) ----------------
@app.route("/load_clips")
def load_clips():
    print("Loading new clips from Kafka...")

    clips = verification_utils.consume_pipeline_results()
    
    # remove previously deleted events
    clips = [c for c in clips if c["clip_id"] not in DELETED_IDS]   
    for clip in clips:
        enrich_clip_geo(clip)
        
#     for clip in clips:

#         start_lat = clip.get("start_latitude")
#         start_lon = clip.get("start_longitude")
#         end_lat = clip.get("end_latitude")
#         end_lon = clip.get("end_longitude")

#         if all(v is not None for v in [start_lat, start_lon, end_lat, end_lon]):

#             clip["center_latitude"] = (start_lat + end_lat) / 2
#             clip["center_longitude"] = (start_lon + end_lon) / 2

#             distance_km = haversine(start_lat, start_lon, end_lat, end_lon)
#             clip["radius_km"] = max(distance_km, 0.2)

#         else:
#             clip["center_latitude"] = 52.0
#             clip["center_longitude"] = 19.0
#             clip["radius_km"] = 0.5

#         score = clip.get("severity_score", 0)
#         numeric, label, color = derive_severity(score)

#         clip["severity_numeric"] = numeric
#         clip["severity_label"] = label
#         clip["severity_color"] = color

    CACHE["pending"] = clips

    return redirect(url_for("index"))

# ---------------- MINIO VERIFIED LOAD WITH CACHE ----------------
def load_verified_cached():

    if time.time() - CACHE["verified_ts"] < CACHE_TTL:
        return

    client = verification_utils.get_minio_client()

    approved = []
    rejected = []

    try:
        objects = client.list_objects(BUCKET, recursive=True)

        for obj in objects:
            data = client.get_object(BUCKET, obj.object_name).read().decode()
            record = json.loads(data)
            if record["clip_id"] in DELETED_IDS:
                continue
            enrich_clip_geo(record)

            if obj.object_name.startswith("approved/"):
                approved.append(record)

            elif obj.object_name.startswith("rejected/"):
                rejected.append(record)

    except S3Error as e:
        print("MinIO read error:", e)

    CACHE["approved"] = approved
    CACHE["rejected"] = rejected
    CACHE["verified_ts"] = time.time()

# ---------------- VIDEO STREAM (WITH LOCAL CACHE) ----------------
@app.route("/video")
def stream_video():

    clip_uri = request.args.get("uri")

    ts_path = verification_utils.download_video(clip_uri)

    mp4_path = ts_path.replace(".ts", ".mp4")

    # convert only once
    if not os.path.exists(mp4_path):

        subprocess.run([
            "ffmpeg",
            "-i", ts_path,
            "-c:v", "copy",
            "-c:a", "copy",
            "-movflags", "faststart",
            "-y",
            mp4_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return Response(
        open(mp4_path, "rb"),
        mimetype="video/mp4"
    )
    
# ---------------- MAP GENERATION ----------------
def create_event_map(clips: list[dict[str, Any]]):

    if not clips:
        return folium.Map(location=[52.0, 19.0], zoom_start=4)

    lats = [c.get("center_latitude", 52.0) for c in clips]
    lons = [c.get("center_longitude", 19.0) for c in clips]

    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')

    for clip in clips:

        popup_html = f"""
        <div style='min-width:220px'>
        <strong>{clip['clip_id']}</strong><br>
        Status: {clip.get('status')}<br>
        Severity: {clip.get('severity_label')}<br>
        Processed: {clip.get('processed_at')}
        </div>
        """
        tooltip_text = f"Event: {clip['clip_id']}"

        # choose marker color based on event status
        status = clip.get("status", "pending")

        if status == "approved":
            color = "green"
        elif status == "rejected":
            color = "red"
        else:
            color = "yellow"

        folium.Circle(
            location=[clip["center_latitude"], clip["center_longitude"]],
            radius=clip["radius_km"] * 1000,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            weight=3,
            popup=popup_html,
            tooltip=tooltip_text
        ).add_to(m)

    return m

# ---------------- MAIN UI ----------------
@app.route("/")
def index():

    filter_type = request.args.get("filter", "pending")
    selected_ids = request.args.getlist("clip_id")

    load_verified_cached()

    pending = [c for c in CACHE["pending"] if c["clip_id"] not in DELETED_IDS]
    approved = [c for c in CACHE["approved"] if c["clip_id"] not in DELETED_IDS]
    rejected = [c for c in CACHE["rejected"] if c["clip_id"] not in DELETED_IDS]

    if filter_type == "pending":
        clips = pending
    elif filter_type == "approved":
        clips = approved
    else:
        clips = rejected

    selected_clips = [c for c in clips if c["clip_id"] in selected_ids][:3]

    # LOAD LLM SUMMARY ONLY FOR SELECTED CLIPS
    for clip in selected_clips:
        try:
            if "summary_text" not in clip:
                clip["summary_text"] = verification_utils.get_summary_content(
                    clip["summary_uri"]
                )
        except Exception as e:
            print("Summary load error:", e)
            clip["summary_text"] = "Summary not available"

    # ---------------- MAP DATA SELECTION ----------------

    if selected_clips:
        map_data = selected_clips

    else:
        if filter_type == "pending":
            map_data = pending
        elif filter_type == "approved":
            map_data = approved
        elif filter_type == "rejected":
            map_data = rejected
        else:
            map_data = pending + approved + rejected


    # generate map
    m = create_event_map(map_data)
    event_map = m.get_root().render()
        
    stats = {
        "total": len(pending) + len(approved) + len(rejected),
        "pending": len(pending),
        "approved": len(approved),
        "rejected": len(rejected),
    }

    return render_template(
        "index.html",
        clips=clips,
        selected_clips=selected_clips,
        selected_ids=selected_ids,
        current_filter=filter_type,
        stats=stats,
        event_map=event_map
    )

# ---------------- APPROVE / REJECT ----------------
# @app.route("/verify", methods=["POST"])
# def verify():

#     clip = {
#         "clip_id": request.form["clip_id"],
#         "clip_uri": request.form["clip_uri"],
#         "summary_uri": request.form["summary_uri"],
#     }

#     status = request.form["status"]
#     reviewer = request.form.get("reviewer", "").strip()
#     comment = request.form.get("reviewer_comment", "").strip()

#     if not reviewer:
#         flash("Reviewer Name is required ❗")
#         return redirect(url_for("index"))

#     verification_utils.save_verification(clip, status, reviewer,comment)

#     flash(f"{status.upper()} ✅")

#     clip_id = clip["clip_id"]

#     # REMOVE FROM ALL CACHES
#     CACHE["pending"] = [c for c in CACHE["pending"] if c["clip_id"] != clip_id]
#     CACHE["approved"] = [c for c in CACHE["approved"] if c["clip_id"] != clip_id]
#     CACHE["rejected"] = [c for c in CACHE["rejected"] if c["clip_id"] != clip_id]

#     print(f"pending: {len(CACHE['pending'])}, approved: {len(CACHE['approved'])}, rejected: {len(CACHE['rejected'])}")
#     print(f"pending IDs: {[c['clip_id'] for c in CACHE['pending']]}")
#     print(f"approved IDs: {[c['clip_id'] for c in CACHE['approved']]}")
#     print(f"rejected IDs: {[c['clip_id'] for c in CACHE['rejected']]}")

#     # force reload from MinIO for correct tab
#     CACHE["verified_ts"] = 0
#     flash(f"{status.upper()} ✅")
#     return redirect(url_for("index"))


@app.route("/verify", methods=["POST"])
def verify():

    clip_id = request.form["clip_id"]

    all_clips = (
        CACHE["pending"] +
        CACHE["approved"] +
        CACHE["rejected"]
    )

    clip = next(
        (c for c in all_clips if c["clip_id"] == clip_id),
        None
    )

    if not clip:
        flash("Clip not found ❗")
        return redirect(url_for("index"))

    status = request.form["status"]
    reviewer = request.form.get("reviewer", "").strip()
    comment = request.form.get("reviewer_comment", "").strip()

    if not reviewer:
        flash("Reviewer Name is required ❗")
        return redirect(url_for("index"))

    verification_utils.save_verification(
        clip,
        status,
        reviewer,
        comment
    )

    flash(f"{status.upper()} ✅")

    CACHE["pending"] = [
        c for c in CACHE["pending"]
        if c["clip_id"] != clip_id
    ]

    CACHE["approved"] = [
        c for c in CACHE["approved"]
        if c["clip_id"] != clip_id
    ]

    CACHE["rejected"] = [
        c for c in CACHE["rejected"]
        if c["clip_id"] != clip_id
    ]

    CACHE["verified_ts"] = 0

    return redirect(url_for("index"))

# ---------------- DELETE EVENT ----------------
@app.route("/delete_event/<clip_id>", methods=["POST"])
def delete_event(clip_id):

    # persist deletion
    DELETED_IDS.add(clip_id)

    CACHE["pending"] = [c for c in CACHE["pending"] if c["clip_id"] != clip_id]
    CACHE["approved"] = [c for c in CACHE["approved"] if c["clip_id"] != clip_id]
    CACHE["rejected"] = [c for c in CACHE["rejected"] if c["clip_id"] != clip_id]

    return ("", 204)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True, port=5001)