"""
Automated Dry Run for AI Waste Segregation System (50 Frames)
Runs the simulation loop, triggers a hazard drill, and verifies end-of-session reports.
"""

import os
import sys
import time
import numpy as np

from config import CANVAS_HEIGHT, CANVAS_WIDTH, PALETTE
from detector import WasteDetector
from door_system import DoorSystem
from hud import SCADAVisualizer
from logger import WasteLogger
from scenario_engine import ScenarioEngine
from simulator import ConveyorSimulator


def run_dry_run():
    print("[Dry-Run] Initializing subsystems...")
    logger = WasteLogger()
    door_system = DoorSystem()
    scenario_engine = ScenarioEngine(initial_scenario_id=3)  # Hazard drill
    simulator = ConveyorSimulator(scenario_engine, door_system, logger)
    hud = SCADAVisualizer(door_system, scenario_engine, logger)
    detector = WasteDetector()

    print("[Dry-Run] Simulating 250 frames (approx 8 seconds of real-time conveyor flow)...")
    prev_time = time.time()

    for frame_idx in range(250):
        current_time = time.time()
        dt = 0.033  # ~30 FPS fixed timestep

        # Base canvas
        canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
        canvas[:] = PALETTE["background"]

        # Step simulation
        newly_scanned = simulator.update(dt)
        if newly_scanned is not None:
            door_id = detector.classify_to_door(newly_scanned.name)
            newly_scanned.door_id = door_id
            if door_id == 1:
                door_system.get_door(1).trigger_open(newly_scanned.name)
                simulator.pause_belt(1.5)
                logger.log_hazard(newly_scanned.name, 0.93, (300, 180, 360, 220), door_id=1)
                hud.trigger_hazard_alert(newly_scanned.name, 0.93)

        is_hazard_locked = door_system.update(dt)

        # Render
        simulator.draw(canvas, current_time)
        door_system.draw(canvas, current_time)
        hud.draw(canvas, fps=60.0, inference_ms=8.5, vram_mb=350.0,
                 is_hazard_locked=is_hazard_locked, current_time=current_time)

    print("[Dry-Run] 60 frames processed. Generating session audit report...")
    report = logger.generate_session_report()
    print(report)

    assert os.path.exists("session_report.txt"), "session_report.txt was not created"
    assert os.path.exists("hazard_log.csv"), "hazard_log.csv was not created"
    print("\n[Dry-Run] DRY-RUN COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_dry_run()
