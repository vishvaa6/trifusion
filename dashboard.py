"""
AI Waste Segregation Prototype - Real-Time Camera AI Scanner & 5-Door SCADA Dashboard
Detects real-world objects using webcam/Camo and YOLOv8, displays bounding boxes & labels,
triggers the 5 Simulation Doors in real-time, and streams live to the web dashboard.
(Conveyor diverter simulation removed for a focused camera + door routing experience).
"""

import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional
import cv2
from flask import Flask, Response, jsonify, render_template, request, send_file
import numpy as np

from config import (
    CAMERA_INDEX,
    CONFIDENCE_THRESHOLD,
    DOORS,
    HAZARD_LOG_FILE,
    HAZARD_SAFETY_PAUSE_SEC,
    WEBCAM_HEIGHT,
    WEBCAM_WIDTH,
)
from detector import WasteDetector
from door_system import DoorSystem
from logger import WasteLogger

app = Flask(__name__, template_folder="templates")


def discover_available_cameras() -> List[dict]:
    """
    Enumerate all physical and virtual video capture devices registered on the system
    using DirectShow FilterGraph via pygrabber.
    """
    device_names = []
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        device_names = graph.get_input_devices()
    except Exception as e:
        print(f"[Camera Discovery] DirectShow FilterGraph error: {e}")

    discovered = []
    if device_names:
        for idx, dev_name in enumerate(device_names):
            clean_name = dev_name.strip()
            if "camo" in clean_name.lower():
                display_label = f"{clean_name} (Phone HD 720p)"
            elif "hd webcam" in clean_name.lower() or "integrated" in clean_name.lower() or "built-in" in clean_name.lower():
                display_label = f"{clean_name} (Laptop Integrated)"
            elif "obs" in clean_name.lower():
                display_label = f"{clean_name} (Virtual Stream)"
            else:
                display_label = f"{clean_name} (Device {idx})"

            discovered.append({
                "id": idx,
                "name": display_label,
                "raw_name": clean_name
            })
    else:
        # Fallback probe for non-DirectShow environments
        for idx in range(4):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    discovered.append({
                        "id": idx,
                        "name": f"Camera {idx} [{w}x{h}]",
                        "raw_name": f"Camera {idx}"
                    })
                cap.release()

    return discovered


# Shared Thread-Safe State
class DashboardState:
    def __init__(self):
        self.lock = threading.Lock()
        self.available_cameras: List[dict] = discover_available_cameras()
        self.latest_cam_jpeg: Optional[bytes] = None
        self.fps: float = 60.0
        self.inference_ms: float = 0.0
        self.vram_mb: float = 0.0
        self.is_hazard_locked: bool = False
        self.last_hazard_class: str = "None"
        self.last_hazard_conf: float = 0.0
        self.scanner_paused: bool = False
        self.active_camera_id: int = CAMERA_INDEX
        
        # Determine camera friendly name
        active_name = f"Camera {CAMERA_INDEX}"
        for cam in self.available_cameras:
            if cam["id"] == CAMERA_INDEX:
                active_name = cam["name"]
                break
        self.active_camera_name: str = active_name
        self.requested_camera_id: Optional[int] = None
        self.current_detected_label: str = "Scanning..."
        self.recent_items = deque(maxlen=35)  # Chronological list of classified items
        self.last_detection_timestamps = {}   # Cooldown per class to avoid spamming feed
        self.conf_threshold: float = CONFIDENCE_THRESHOLD
        self.show_rectangle_labels: bool = True  # Toggle for drawing bounding box rectangles and labels

state = DashboardState()

# Initialize Core AI Subsystems
logger = WasteLogger()
door_system = DoorSystem()
detector = WasteDetector(conf_threshold=CONFIDENCE_THRESHOLD)


# -----------------------------------------------------------------------------
# Real-Time Camera Detection & Door Triggering Thread
# -----------------------------------------------------------------------------
def camera_worker():
    global state
    current_cam_idx = state.active_camera_id

    def get_cam_name(idx):
        for c_info in state.available_cameras:
            if c_info["id"] == idx:
                return c_info["name"]
        return f"Camera Index {idx}"

    def init_cap(idx):
        cam_name = get_cam_name(idx)
        print(f"[Camera Worker] Initializing {cam_name} (Index {idx})...", flush=True)

        # On Windows, try Media Foundation (MSMF) first (required for Camo), then DirectShow (DSHOW)
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW] if sys.platform == "win32" else [cv2.CAP_ANY]

        for backend in backends:
            b_name = "MSMF" if backend == cv2.CAP_MSMF else ("DSHOW" if backend == cv2.CAP_DSHOW else "ANY")
            try:
                c = cv2.VideoCapture(idx, backend)
                if c.isOpened():
                    ret, test = False, None
                    # Give camera up to 5 warmup frames to handshake and provide live video
                    for _ in range(5):
                        ret, test = c.read()
                        if ret and test is not None and test.mean() > 0.0:
                            break
                        time.sleep(0.05)

                    if ret and test is not None and (test.mean() > 0.0 or len(backends) == 1):
                        print(f"[Camera Worker] Successfully connected to {cam_name} via {b_name} (mean={test.mean():.1f}, shape={test.shape}).", flush=True)
                        return c
                    else:
                        print(f"[Camera Worker] Backend {b_name} returned black or empty frame for {cam_name}. Trying fallback...", flush=True)
                    c.release()
            except Exception as e:
                print(f"[Camera Worker] Backend {b_name} error: {e}", flush=True)

        print(f"[Camera Worker] Warning: Could not read active stream from {cam_name} (Index {idx}).", flush=True)
        return None

    cap = init_cap(current_cam_idx)
    # If initial camera (e.g. Camo Index 1) failed on start, fallback to built-in (Index 0)
    if cap is None and current_cam_idx != 0:
        print("[Camera Worker] Falling back to Built-in Camera (Index 0)...", flush=True)
        current_cam_idx = 0
        cap = init_cap(0)
        state.active_camera_id = 0
        state.active_camera_name = get_cam_name(0)

    prev_time = time.time()
    fps = 60.0

    while True:
        current_time = time.time()
        dt = max(0.001, min(0.1, current_time - prev_time))
        prev_time = current_time

        # Update FPS
        current_fps = 1.0 / dt
        fps = 0.9 * fps + 0.1 * current_fps
        logger.record_fps(fps)

        # Update 5 Simulation Doors animation & timers
        is_hazard_locked = door_system.update(dt)

        # Handle runtime camera switch requests
        if state.requested_camera_id is not None:
            new_idx = state.requested_camera_id
            state.requested_camera_id = None
            if cap is not None:
                cap.release()
                cap = None
                time.sleep(0.25)  # Allow Windows OS driver time to close handle
            current_cam_idx = new_idx
            cap = init_cap(current_cam_idx)
            state.active_camera_id = current_cam_idx
            state.active_camera_name = get_cam_name(current_cam_idx)

        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            # Overlay warning if camera sends all-black frames (e.g., Camo app paused on phone)
            if frame.mean() == 0.0:
                cv2.putText(frame, "CAMO PAUSED OR PHONE LOCKED", (40, WEBCAM_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
                cv2.putText(frame, "Please unlock phone & keep Camo active", (60, (WEBCAM_HEIGHT // 2) + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

            frame = cv2.resize(frame, (WEBCAM_WIDTH, WEBCAM_HEIGHT))
        else:
            # Fallback synthetic frame if camera disconnects
            frame = np.zeros((WEBCAM_HEIGHT, WEBCAM_WIDTH, 3), dtype=np.uint8)
            frame[:] = (24, 26, 32)
            cv2.putText(frame, f"{state.active_camera_name.upper()} OFFLINE / CONNECTING...", (90, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 145, 155), 2)
            time.sleep(0.04)

        inf_ms = 0.0
        detected_in_frame = []

        # Only process inference if scanner is not paused
        if not state.scanner_paused:
            # 1. Run YOLOv8 Inference on Realtime Camera Frame
            det_result = detector.detect(frame)
            inf_ms = det_result.inference_time_ms

            # 2. Draw Bounding Boxes and Classification Labels ONLY for selected framed objects
            if state.show_rectangle_labels:
                for item in det_result.items:
                    # Only show rectangular frame if item is selected ("other wise no")
                    if not item.has_frame:
                        continue

                    x1, y1, x2, y2 = item.bbox
                    color = item.color_bgr
                    door_id = item.door_id
                    cat_name = item.category

                    # Bounding box
                    box_thickness = 3 if item.is_hazard else 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

                    # Corner brackets for HUD aesthetic
                    corner_len = 16
                    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 2)
                    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 2)
                    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), 2)
                    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), 2)

                    # Label banner
                    conf_pct = int(item.confidence * 100)
                    tag_text = f"[{cat_name}] {item.class_name.upper()} {conf_pct}%"
                    (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

                    cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), (20, 20, 24), -1)
                    cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), color, 1)
                    cv2.putText(frame, tag_text, (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                    # Door routing hint under box
                    if door_id and door_id in DOORS:
                        dest_text = f"ROUTE -> DOOR {door_id} ({DOORS[door_id].category})"
                    else:
                        dest_text = "NON-WASTE OBJECT"
                    cv2.putText(frame, dest_text, (x1 + 2, min(WEBCAM_HEIGHT - 6, y2 + 16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

            # 3. Process items for door actuation, feeds, and HUD (only for framed objects)
            for item in det_result.items:
                if not item.has_frame:
                    continue

                door_id = item.door_id
                cat_name = item.category
                detected_in_frame.append(item.class_name.title())

                # Trigger Simulation Doors & Feed Update (1.5s cooldown per item class)
                last_seen = state.last_detection_timestamps.get(item.class_name, 0.0)
                if (current_time - last_seen) > 1.5:
                    state.last_detection_timestamps[item.class_name] = current_time

                    # Only actuate doors if mapped to an actual waste stream (Doors 1 - 5)
                    if door_id and door_id in DOORS:
                        door_system.get_door(door_id).trigger_open(item.class_name)
                        logger.record_sorted_item(door_id)

                        # If Hazard -> Trigger Lockout & Log
                        if item.is_hazard:
                            logger.log_hazard(item.class_name, item.confidence, item.bbox, door_id=1)
                            with state.lock:
                                state.last_hazard_class = item.class_name
                                state.last_hazard_conf = item.confidence

                    # Add to Classified Objects Feed
                    item_entry = {
                        "name": item.class_name.title(),
                        "category": cat_name,
                        "door_id": door_id if door_id else 0,
                        "confidence": round(item.confidence, 2),
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "source": "REAL CAMERA"
                    }
                    with state.lock:
                        state.recent_items.appendleft(item_entry)

        # Draw HUD status bar at bottom of camera frame
        status_bar_y = WEBCAM_HEIGHT - 28
        cv2.rectangle(frame, (0, status_bar_y), (WEBCAM_WIDTH, WEBCAM_HEIGHT), (15, 17, 21), -1)
        cv2.line(frame, (0, status_bar_y), (WEBCAM_WIDTH, status_bar_y), (45, 48, 56), 1)

        cam_display_name = state.active_camera_name
        cam_text = f"{cam_display_name.upper()} | AI: {inf_ms:.1f}ms | DETECTED: {', '.join(detected_in_frame) if detected_in_frame else 'None'}"
        cv2.putText(frame, cam_text, (10, status_bar_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 180), 1)

        # Encode Camera Frame for HTTP Streaming
        ret_enc, cam_jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret_enc:
            with state.lock:
                state.latest_cam_jpeg = cam_jpeg.tobytes()
                state.fps = fps
                state.inference_ms = inf_ms
                state.vram_mb = detector.get_vram_usage_mb()
                state.is_hazard_locked = is_hazard_locked
                if detected_in_frame:
                    state.current_detected_label = ", ".join(detected_in_frame)
                else:
                    state.current_detected_label = "Monitoring (Present object to camera)..."

        # Cap camera loop rate
        time.sleep(0.01)


# -----------------------------------------------------------------------------
# Flask Web Routes
# -----------------------------------------------------------------------------
@app.route('/')
def index():
    """Main Web SCADA Dashboard View."""
    return render_template('index.html')


def generate_camera_stream():
    """MJPEG streaming generator for live camera feed."""
    while True:
        with state.lock:
            frame_bytes = state.latest_cam_jpeg

        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 fps web stream


@app.route('/video_cam')
def video_cam():
    """Stream live camera feed with YOLOv8 bounding boxes and labels."""
    return Response(generate_camera_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/state')
def api_state():
    """JSON Telemetry & Doors State endpoint for frontend polling."""
    with state.lock:
        doors_data = {}
        for door_id, door in door_system.doors.items():
            doors_data[door_id] = {
                "name": door.info.name,
                "category": door.info.category,
                "count": door.item_count,
                "is_open": door.is_open,
                "open_progress": round(door.open_progress, 2),
                "is_hazard_locked": door.is_hazard_locked,
                "last_item": door.last_item_name
            }

        uptime_sec = int(time.time() - logger.session_start_time)
        m, s = divmod(uptime_sec, 60)
        h, m = divmod(m, 60)
        uptime_str = f"{h:02d}:{m:02d}:{s:02d}"

        return jsonify({
            "fps": state.fps,
            "inference_ms": state.inference_ms,
            "vram_mb": state.vram_mb,
            "uptime": uptime_str,
            "is_hazard_locked": state.is_hazard_locked,
            "last_hazard_class": state.last_hazard_class,
            "last_hazard_conf": state.last_hazard_conf,
            "scanner_paused": state.scanner_paused,
            "active_camera_id": state.active_camera_id,
            "active_camera_name": state.active_camera_name,
            "available_cameras": state.available_cameras,
            "current_detected_label": state.current_detected_label,
            "total_sorted": logger.total_scanned_count,
            "doors": doors_data,
            "recent_items": list(state.recent_items),
            "show_rectangle_labels": state.show_rectangle_labels,
            "framed_objects_count": len([c for c in detector.get_class_catalog() if c["is_enabled"]]),
            "total_objects_count": 80
        })


@app.route('/api/cameras')
def api_cameras():
    """Return all connected video capture devices."""
    with state.lock:
        return jsonify({
            "active_camera_id": state.active_camera_id,
            "active_camera_name": state.active_camera_name,
            "available_cameras": state.available_cameras
        })


@app.route('/api/rescan_cameras')
def api_rescan_cameras():
    """Re-scan system for connected cameras and update device list."""
    cams = discover_available_cameras()
    with state.lock:
        state.available_cameras = cams
    return jsonify({
        "success": True,
        "active_camera_id": state.active_camera_id,
        "available_cameras": cams
    })


@app.route('/api/object_collections')
def api_object_collections():
    """Return all object collections and their frame status."""
    collections = detector.get_collections_status()
    catalog = detector.get_class_catalog()
    enabled_count = len([c for c in catalog if c["is_enabled"]])
    return jsonify({
        "collections": collections,
        "enabled_count": enabled_count,
        "total_count": len(catalog)
    })


@app.route('/api/toggle_collection', methods=['POST'])
def api_toggle_collection():
    """Toggle rectangular frames on or off for all objects in a collection."""
    data = request.get_json(force=True, silent=True) or {}
    col_id = data.get('collection_id', '')
    enable = bool(data.get('enable', True))
    collections = detector.toggle_collection_frames(col_id, enable)
    catalog = detector.get_class_catalog()
    enabled_count = len([c for c in catalog if c["is_enabled"]])
    return jsonify({
        "success": True,
        "collection_id": col_id,
        "is_enabled": enable,
        "collections": collections,
        "enabled_count": enabled_count,
        "total_count": len(catalog)
    })

@app.route('/api/object_filters', methods=['GET', 'POST'])
def api_object_filters():
    """Retrieve or update active object detection filters."""
    if request.method == 'POST':
        data = request.get_json(force=True, silent=True) or {}
        enabled_list = data.get('enabled_classes', None)
        detector.set_enabled_classes(enabled_list)
        catalog = detector.get_class_catalog()
        enabled_count = len([c for c in catalog if c["is_enabled"]])
        return jsonify({
            "success": True,
            "enabled_count": enabled_count,
            "total_count": len(catalog),
            "collections": detector.get_collections_status()
        })

    catalog = detector.get_class_catalog()
    enabled_count = len([c for c in catalog if c["is_enabled"]])
    return jsonify({
        "catalog": catalog,
        "enabled_count": enabled_count,
        "total_count": len(catalog),
        "collections": detector.get_collections_status()
    })


@app.route('/api/filter_preset/<preset>')
def api_filter_preset(preset: str):
    """Apply preset object filters (all, waste_only, hazard_only, clear)."""
    preset_lower = preset.lower().strip()
    if preset_lower == 'all':
        detector.set_enabled_classes(None)
    elif preset_lower == 'waste_only':
        # Enable all classes mapped to Doors 1-5 (waste streams)
        waste_classes = []
        for door in DOORS.values():
            waste_classes.extend(door.target_classes)
        extra_waste = ["cell phone", "laptop", "mouse", "keyboard", "remote", "bottle", "wine glass",
                       "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                       "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "book", "scissors"]
        detector.set_enabled_classes(list(set(waste_classes + extra_waste)))
    elif preset_lower == 'hazard_only':
        detector.set_enabled_classes(list(DOORS[1].target_classes))
    elif preset_lower in ('stationery_only', 'stationery', 'stationary_only', 'stationary'):
        # Detect all stationary & desk collection items (books, scissors, backpack, electronics, clock, pens)
        stationery_items = [
            "book", "scissors", "backpack", "clock", "cell phone", "laptop", "mouse", "keyboard", "toothbrush"
        ]
        detector.set_enabled_classes(stationery_items)
    elif preset_lower == 'clear':
        detector.set_enabled_classes([])
    else:
        return jsonify({"error": f"Unknown preset '{preset}'"}), 400

    catalog = detector.get_class_catalog()
    enabled_count = len([c for c in catalog if c["is_enabled"]])
    return jsonify({
        "success": True,
        "preset": preset_lower,
        "enabled_count": enabled_count,
        "total_count": len(catalog),
        "collections": detector.get_collections_status()
    })


@app.route('/api/toggle_rectangle_labels', methods=['GET', 'POST'])
def api_toggle_rectangle_labels():
    """Toggle or set rectangle labels and bounding boxes visibility on camera feed."""
    if request.method == 'POST':
        data = request.get_json(force=True, silent=True) or {}
        with state.lock:
            if 'show' in data:
                state.show_rectangle_labels = bool(data['show'])
            else:
                state.show_rectangle_labels = not state.show_rectangle_labels
    elif request.method == 'GET':
        show_param = request.args.get('show', None)
        if show_param is not None:
            with state.lock:
                state.show_rectangle_labels = (show_param.lower() in ('true', '1', 'yes'))

    return jsonify({
        "success": True,
        "show_rectangle_labels": state.show_rectangle_labels
    })


@app.route('/api/switch_camera/<int:cam_id>')
def api_switch_camera(cam_id: int):
    """Switch active camera device."""
    with state.lock:
        state.requested_camera_id = cam_id
    return jsonify({"success": True, "target_camera": cam_id})


@app.route('/api/trigger_door/<int:door_id>')
def api_trigger_door(door_id: int):
    """Manual test trigger for any of the 5 doors."""
    if door_id in DOORS:
        test_item_names = {
            1: "Knife (Manual Test)",
            2: "Phone (Manual Test)",
            3: "Cardboard (Manual Test)",
            4: "Cup (Manual Test)",
            5: "Apple (Manual Test)"
        }
        item_name = test_item_names.get(door_id, "Test Item")
        door_system.get_door(door_id).trigger_open(item_name)
        logger.record_sorted_item(door_id)

        door_info = DOORS[door_id]
        if door_id == 1:
            logger.log_hazard("Knife", 0.99, (200, 200, 300, 300), door_id=1)
            with state.lock:
                state.last_hazard_class = "Knife"
                state.last_hazard_conf = 0.99

        item_entry = {
            "name": item_name,
            "category": door_info.category,
            "door_id": door_id,
            "confidence": 0.99,
            "time": datetime.now().strftime("%H:%M:%S"),
            "source": "MANUAL TRIGGER"
        }
        with state.lock:
            state.recent_items.appendleft(item_entry)

        return jsonify({"success": True, "triggered_door": door_id})
    return jsonify({"error": "Invalid door ID"}), 400


@app.route('/api/toggle_pause')
def api_toggle_pause():
    """Toggle scanner pause."""
    with state.lock:
        state.scanner_paused = not state.scanner_paused
        return jsonify({"paused": state.scanner_paused})


@app.route('/api/reset')
def api_reset():
    """Reset session item counts."""
    logger.total_scanned_count = 0
    for d in door_system.doors.values():
        d.item_count = 0
        d.last_item_name = "None"
    with state.lock:
        state.recent_items.clear()
        state.is_hazard_locked = False
    return jsonify({"success": True})


@app.route('/api/download_log')
def api_download_log():
    """Download hazard audit CSV."""
    if os.path.exists(HAZARD_LOG_FILE):
        return send_file(HAZARD_LOG_FILE, as_attachment=True, download_name="hazard_log.csv")
    return jsonify({"error": "No hazard events recorded yet"}), 404


def run_server(host="0.0.0.0", port=5000):
    cam_thread = threading.Thread(target=camera_worker, daemon=True)
    cam_thread.start()

    print("\n" + "=" * 65)
    print("   AI WASTE SEGREGATION - REALTIME CAMERA + 5-DOOR SCADA")
    print("=" * 65)
    print(f" Web Dashboard : http://localhost:{port}")
    print(f" Live Cam Feed : http://localhost:{port}/video_cam")
    print("=" * 65 + "\n")

    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    run_server()
