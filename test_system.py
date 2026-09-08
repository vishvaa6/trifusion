"""
Unit & Integration Verification Suite for AI Waste Segregation Prototype
Tests:
- Classification rules (Doors 1 to 5)
- Door opening & trajectory mechanics
- Scenario engine item generation
- Simulation physics & zone transitions
- Hazard Intercept Protocol (HIP) & CSV logging
- Headless OpenCV frame rendering
"""

import os
import sys
import numpy as np

from config import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DOORS,
    HAZARD_LOG_FILE,
    PALETTE,
    SESSION_REPORT_FILE,
)
from detector import WasteDetector
from door_system import DoorSystem
from hud import SCADAVisualizer
from logger import WasteLogger
from scenario_engine import ScenarioEngine
from simulator import ConveyorSimulator


def test_door_classification():
    print("[TEST 1/6] Verifying Door Classification Rules...")
    detector = WasteDetector()

    # Hazard items -> Door 1
    for item in ["knife", "scissors", "bottle", "syringe"]:
        door = detector.classify_to_door(item)
        assert door == 1, f"Expected Door 1 for '{item}', got Door {door}"

    # E-Waste -> Door 2
    for item in ["cell phone", "laptop", "keyboard", "mouse", "remote"]:
        door = detector.classify_to_door(item)
        assert door == 2, f"Expected Door 2 for '{item}', got Door {door}"

    # Paper -> Door 3
    for item in ["book", "newspaper", "cardboard"]:
        door = detector.classify_to_door(item)
        assert door == 3, f"Expected Door 3 for '{item}', got Door {door}"

    # Recyclable -> Door 4
    for item in ["cup", "fork", "spoon", "can"]:
        door = detector.classify_to_door(item)
        assert door == 4, f"Expected Door 4 for '{item}', got Door {door}"

    # Organic -> Door 5
    for item in ["banana", "apple", "sandwich", "pizza"]:
        door = detector.classify_to_door(item)
        assert door == 5, f"Expected Door 5 for '{item}', got Door {door}"

    print("  -> Passed! All categories mapped to correct doors.")


def test_door_system():
    print("[TEST 2/6] Verifying Door System Mechanics & Trajectories...")
    door_system = DoorSystem()
    assert len(door_system.doors) == 5, f"Expected 5 doors, found {len(door_system.doors)}"

    # Test Door 1 (Hazard) trigger
    door1 = door_system.get_door(1)
    assert not door1.is_open
    assert not door1.is_hazard_locked

    door_system.route_item_to_door((400, 200), 1, "utility knife")
    assert door1.is_open
    assert door1.is_hazard_locked
    assert door1.item_count == 1
    assert len(door_system.active_trajectories) == 1

    # Update dt
    hazard_locked = door_system.update(0.1)
    assert hazard_locked is True, "Expected hazard lockout to be active"
    assert door1.open_progress > 0.0

    print("  -> Passed! Door 1 opening, trajectory, and hazard lockout verified.")


def test_scenario_engine():
    print("[TEST 3/6] Verifying Scenario Engine & Campaigns...")
    engine = ScenarioEngine(1)
    assert engine.active_scenario.id == 1

    # Test switching
    for scen_id in [1, 2, 3, 4, 5]:
        assert engine.set_scenario(scen_id) is True
        assert engine.active_scenario.id == scen_id
        # Sample items
        item = engine.sample_random_item()
        assert "name" in item and "door_id" in item

    print("  -> Passed! All 5 scenarios switch and generate catalog items correctly.")


def test_conveyor_simulator():
    print("[TEST 4/6] Verifying Conveyor Belt Simulator...")
    engine = ScenarioEngine(1)
    door_sys = DoorSystem()
    logger = WasteLogger("test_hazard.csv", "test_report.txt")
    sim = ConveyorSimulator(engine, door_sys, logger)

    # Step simulation 30 times (1.0 second total)
    for _ in range(30):
        sim.update(0.033)

    assert sim.stripe_offset >= 0.0
    print("  -> Passed! Conveyor motion and step simulation executed cleanly.")


def test_hazard_logger():
    print("[TEST 5/6] Verifying Hazard CSV Logger & Session Report...")
    test_csv = "test_hazard.csv"
    test_rep = "test_report.txt"

    if os.path.exists(test_csv):
        os.remove(test_csv)
    if os.path.exists(test_rep):
        os.remove(test_rep)

    logger = WasteLogger(test_csv, test_rep)
    logger.log_hazard("knife", 0.94, (100, 100, 200, 200), door_id=1)
    logger.record_sorted_item(1)
    logger.record_sorted_item(2)
    logger.record_fps(85.0)

    report = logger.generate_session_report()

    assert os.path.exists(test_csv), "Hazard log file was not created"
    assert os.path.exists(test_rep), "Session report file was not created"
    assert "knife" in open(test_csv).read()
    assert "DOOR 1" in report

    # Cleanup test files
    os.remove(test_csv)
    os.remove(test_rep)
    print("  -> Passed! Hazard audit row and session report generated.")


def test_headless_render():
    print("[TEST 6/6] Verifying Headless OpenCV Canvas Rendering...")
    engine = ScenarioEngine(1)
    door_sys = DoorSystem()
    logger = WasteLogger("test_hazard.csv", "test_report.txt")
    sim = ConveyorSimulator(engine, door_sys, logger)
    hud = SCADAVisualizer(door_sys, engine, logger)

    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    canvas[:] = PALETTE["background"]

    # Render all layers
    sim.draw(canvas, 1.0)
    door_sys.draw(canvas, 1.0)
    dummy_cam = np.zeros((480, 640, 3), dtype=np.uint8)
    hud.draw(canvas, fps=85.2, inference_ms=9.4, vram_mb=450.0,
             is_hazard_locked=False, current_time=1.0, webcam_frame=dummy_cam)

    # Test hazard strobe state
    hud.trigger_hazard_alert("knife", 0.96)
    hud.draw(canvas, fps=85.2, inference_ms=9.4, vram_mb=450.0,
             is_hazard_locked=True, current_time=1.0, webcam_frame=None)

    assert canvas.shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 3)
    # Check that canvas has drawn elements (not completely uniform)
    assert np.std(canvas) > 5.0, "Canvas should have rich drawn elements"

    # Cleanup
    if os.path.exists("test_hazard.csv"):
        os.remove("test_hazard.csv")
    if os.path.exists("test_report.txt"):
        os.remove("test_report.txt")

    print("  -> Passed! Composite SCADA canvas rendered without OpenCV errors.")


def test_transparent_object_enhancement():
    print("[TEST 7/7] Verifying Transparent Object Enhancement & Sensitivity Pipeline...")
    from detector import enhance_transparent_clarity, OBJECT_COLLECTIONS, TRANSPARENT_CLASSES

    # 1. Test LAB-CLAHE edge enhancement
    dummy = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    enhanced = enhance_transparent_clarity(dummy)
    assert enhanced.shape == (480, 640, 3), f"Expected shape (480, 640, 3), got {enhanced.shape}"
    assert enhanced.dtype == np.uint8, f"Expected dtype uint8, got {enhanced.dtype}"

    # 2. Test WasteDetector transparent parameters
    detector = WasteDetector()
    assert detector.transparent_conf_threshold <= 0.25, f"Expected transparent threshold <= 0.25, got {detector.transparent_conf_threshold}"
    for cls in ["bottle", "wine glass", "cup", "bowl", "vase"]:
        assert cls in detector.transparent_classes, f"Expected '{cls}' in transparent classes"

    # 3. Test transparent label refinements
    b_label, b_cat, b_door = detector.refine_label("bottle", (100, 100, 200, 450))
    assert "Bottle" in b_label and b_door == 1, f"Bottle refinement failed: {b_label}"

    c_label, c_cat, c_door = detector.refine_label("cup", (100, 100, 250, 250))
    assert ("Glass" in c_label or "Cup" in c_label) and c_door == 4, f"Cup refinement failed: {c_label}"

    w_label, w_cat, w_door = detector.refine_label("wine glass", (100, 100, 200, 400))
    assert "Wine Glass" in w_label and w_door == 4, f"Wine glass refinement failed: {w_label}"

    v_label, v_cat, v_door = detector.refine_label("vase", (100, 100, 200, 400))
    assert "Vase" in v_label and v_door == 4, f"Vase refinement failed: {v_label}"

    # 4. Test transparency collection in catalog
    assert "transparency" in OBJECT_COLLECTIONS
    assert "bottle" in OBJECT_COLLECTIONS["transparency"]["classes"]

    print("  -> Passed! Transparent CLAHE enhancement, sensitivity floor, and glassware labels verified.")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("    RUNNING AI WASTE SEGREGATION INTEGRATION TESTS")
    print("=" * 60)

    test_door_classification()
    test_door_system()
    test_scenario_engine()
    test_conveyor_simulator()
    test_hazard_logger()
    test_headless_render()
    test_transparent_object_enhancement()

    print("\n" + "=" * 60)
    print("    ALL INTEGRATION TESTS PASSED SUCCESSFULLY! [7/7]")
    print("=" * 60 + "\n")
