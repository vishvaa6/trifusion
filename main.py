"""
AI Waste Segregation Prototype - Main Application Entrypoint
Integrates YOLOv8 FP16 CUDA detection, 5-Door Smart Routing System,
Procedural Conveyor Simulation, SCADA HUD, and Hazard Intercept Protocol.
"""

import argparse
import os
import sys
import time
from datetime import datetime
import cv2
import numpy as np

from config import (
    CAMERA_INDEX,
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    CONFIDENCE_THRESHOLD,
    HAZARD_SAFETY_PAUSE_SEC,
    PALETTE,
    WEBCAM_HEIGHT,
    WEBCAM_WIDTH,
)
from detector import WasteDetector
from door_system import DoorSystem
from hud import SCADAVisualizer
from logger import WasteLogger
from scenario_engine import ScenarioEngine
from simulator import ConveyorSimulator


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Waste Segregation Prototype - 5-Door Smart Routing System"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Force Virtual Conveyor Simulation mode without webcam hardware."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help="Camera device index (default: 0)."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Confidence threshold for YOLOv8 detection (default: 0.40)."
    )
    parser.add_argument(
        "--no-pip",
        action="store_true",
        help="Disable webcam Picture-in-Picture window."
    )
    return parser.parse_args()


def check_and_init_camera(camera_idx: int, force_simulate: bool) -> tuple:
    """
    Attempts to initialize the webcam at low latency.
    If unavailable, prompts the user to switch on Virtual Conveyor Simulation mode.
    """
    if force_simulate:
        print("[System] Running in dedicated Virtual Conveyor Simulation mode (--simulate).")
        return None, True

    print(f"[System] Probing Webcam Index {camera_idx} at {WEBCAM_WIDTH}x{WEBCAM_HEIGHT}...")
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Zero buffer lag

    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            print(f"[System] Webcam index {camera_idx} initialized successfully.")
            return cap, False
        cap.release()

    # Camera failed or missing -> Ask to switch on Virtual Simulation mode
    print(f"\n[!] Camera (Index {camera_idx}) not detected, in use, or blocked.")
    try:
        user_choice = input("[?] Switch on Virtual Conveyor Simulation mode? (y/n) [default: y]: ").strip().lower()
    except (EOFError, OSError):
        print("[System] Non-interactive environment detected. Defaulting to Virtual Conveyor Simulation mode.")
        user_choice = "y"

    if user_choice in ("", "y", "yes"):
        print("[System] Switching on Virtual Conveyor Simulation mode...\n")
        return None, True
    else:
        print("[System] Simulation aborted by user. Exiting.")
        sys.exit(0)


def main():
    args = parse_arguments()

    print("=" * 65)
    print("      AI INDUSTRIAL WASTE SEGREGATION - 5-DOOR ROUTING SYSTEM")
    print("=" * 65)

    # 1. Initialize Camera or Simulation Fallback
    cap, is_simulation_mode = check_and_init_camera(args.camera, args.simulate)

    # 2. Initialize Core Subsystems
    logger = WasteLogger()
    door_system = DoorSystem()
    scenario_engine = ScenarioEngine(initial_scenario_id=1)
    simulator = ConveyorSimulator(scenario_engine, door_system, logger)
    hud = SCADAVisualizer(door_system, scenario_engine, logger)
    if args.no_pip:
        hud.show_pip = False

    detector = WasteDetector(conf_threshold=args.conf)

    window_name = "AI Waste Segregation System - 5-Door SCADA Control"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CANVAS_WIDTH, CANVAS_HEIGHT)

    # Performance Timers
    prev_time = time.time()
    fps = 60.0
    last_webcam_inject_time = 0.0

    print("\n[System] System loop active. Press 'q' to quit.")

    try:
        while True:
            current_time = time.time()
            dt = max(0.001, min(0.1, current_time - prev_time))
            prev_time = current_time

            # Compute smoothed FPS
            current_fps = 1.0 / dt
            fps = 0.9 * fps + 0.1 * current_fps
            logger.record_fps(fps)

            # Create Base Canvas
            canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
            canvas[:] = PALETTE["background"]

            webcam_frame_display = None
            inference_ms = 0.0

            # -------------------------------------------------------------
            # Physical Webcam Infeed Processing (if available)
            # -------------------------------------------------------------
            if cap is not None and cap.isOpened():
                ret, raw_cam = cap.read()
                if ret and raw_cam is not None:
                    webcam_frame = cv2.resize(raw_cam, (WEBCAM_WIDTH, WEBCAM_HEIGHT))
                    # Run YOLOv8 on camera feed
                    det_result = detector.detect(webcam_frame)
                    inference_ms = det_result.inference_time_ms

                    # Annotate detections on webcam frame
                    for item in det_result.items:
                        x1, y1, x2, y2 = item.bbox
                        cv2.rectangle(webcam_frame, (x1, y1), (x2, y2), item.color_bgr, 2)
                        tag = f"{item.category}: {item.class_name} ({item.confidence:.2f})"
                        cv2.putText(webcam_frame, tag, (x1, max(18, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, item.color_bgr, 1)

                        # Trigger Hazard Intercept if hazard detected via webcam
                        if item.is_hazard:
                            door_system.get_door(1).trigger_open(item.class_name)
                            simulator.pause_belt(HAZARD_SAFETY_PAUSE_SEC)
                            logger.log_hazard(item.class_name, item.confidence, item.bbox, door_id=1)
                            hud.trigger_hazard_alert(item.class_name, item.confidence)

                        # Hybrid spawn: inject real-world object onto virtual conveyor
                        if (current_time - last_webcam_inject_time) > 2.5:
                            simulator.inject_webcam_detection(item.class_name, item.door_id)
                            last_webcam_inject_time = current_time

                    webcam_frame_display = webcam_frame

            # -------------------------------------------------------------
            # Conveyor Belt Simulation & Inspection Processing
            # -------------------------------------------------------------
            newly_scanned_item = simulator.update(dt)
            if newly_scanned_item is not None:
                # Classify item through detector rules
                door_id = detector.classify_to_door(newly_scanned_item.name)
                newly_scanned_item.door_id = door_id

                # If Hazard item scanned in Zone B
                if door_id == 1:
                    conf = round(np.random.uniform(0.85, 0.98), 2)
                    door_system.get_door(1).trigger_open(newly_scanned_item.name)
                    simulator.pause_belt(HAZARD_SAFETY_PAUSE_SEC)
                    logger.log_hazard(
                        newly_scanned_item.name,
                        conf,
                        (int(newly_scanned_item.x - 20), int(newly_scanned_item.y - 20),
                         int(newly_scanned_item.x + 20), int(newly_scanned_item.y + 20)),
                        door_id=1
                    )
                    hud.trigger_hazard_alert(newly_scanned_item.name, conf)

            # Update 5-Door subsystem animations
            is_hazard_locked = door_system.update(dt)

            # -------------------------------------------------------------
            # Rendering Stage
            # -------------------------------------------------------------
            simulator.draw(canvas, current_time)
            door_system.draw(canvas, current_time)

            vram_mb = detector.get_vram_usage_mb()
            hud.draw(
                canvas,
                fps=fps,
                inference_ms=inference_ms,
                vram_mb=vram_mb,
                is_hazard_locked=is_hazard_locked,
                current_time=current_time,
                webcam_frame=webcam_frame_display
            )

            # Display composite canvas
            cv2.imshow(window_name, canvas)

            # -------------------------------------------------------------
            # Operator Keyboard Controls
            # -------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # 'q' or ESC
                print("\n[System] Shutdown initiated by operator.")
                break
            elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
                scenario_id = int(chr(key))
                scenario_engine.set_scenario(scenario_id)
            elif key in (ord('+'), ord('=')):
                simulator.base_belt_speed = min(220.0, simulator.base_belt_speed + 15.0)
                print(f"[Controls] Belt Speed: {simulator.base_belt_speed:.0f} px/s")
            elif key in (ord('-'), ord('_')):
                simulator.base_belt_speed = max(40.0, simulator.base_belt_speed - 15.0)
                print(f"[Controls] Belt Speed: {simulator.base_belt_speed:.0f} px/s")
            elif key in (ord('p'), ord('P')):
                if simulator.belt_paused:
                    simulator.resume_belt()
                    print("[Controls] Belt Resumed.")
                else:
                    simulator.pause_belt(999999.0)
                    print("[Controls] Belt Paused by operator.")
            elif key in (ord('h'), ord('H')):
                hud.show_hud = not hud.show_hud
            elif key in (ord('c'), ord('C')):
                hud.show_pip = not hud.show_pip
            elif key in (ord('s'), ord('S')):
                snap_name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(snap_name, canvas)
                print(f"[Controls] Snapshot saved: {os.path.abspath(snap_name)}")
            elif key in (ord('r'), ord('R')):
                logger.total_scanned_count = 0
                for d in door_system.doors.values():
                    d.item_count = 0
                print("[Controls] Session counters reset.")

    except KeyboardInterrupt:
        print("\n[System] Interrupted by keyboard.")

    finally:
        # Cleanup and write session report
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

        report_summary = logger.generate_session_report()
        print("\n" + report_summary)


if __name__ == "__main__":
    main()
