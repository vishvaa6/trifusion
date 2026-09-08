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
from detector import DualStreamDetector, WasteDetector
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


def test_error_matrix_and_mse():
    print("[TEST 8/8] Verifying MES / MSE Error Matrix & Classification Metrics...")
    from error_matrix import ErrorMatrixTracker, DOOR_NAMES

    test_file = "test_error_matrix.json"
    if os.path.exists(test_file):
        os.remove(test_file)

    tracker = ErrorMatrixTracker(persistence_file=test_file)
    # Reset to blank for deterministic unit test
    tracker.reset()
    assert tracker.total_samples == 0
    assert tracker.matrix.shape == (5, 5)
    assert tracker.multivariate_mse_matrix.shape == (5, 5)

    # 1. Test Perfect Diagonal Classification
    for door_id in range(1, 6):
        tracker.record_evaluation(
            actual_door=door_id,
            predicted_door=door_id,
            confidence=1.0,
            class_name=f"test_item_{door_id}",
            source="UNIT_TEST"
        )

    summary = tracker.get_summary()
    assert summary["overall_accuracy_pct"] == 100.0, f"Expected 100%, got {summary['overall_accuracy_pct']}"
    assert summary["chute_distance_mse"] == 0.0
    assert summary["mes_kpis"]["hazard_safety_index_pct"] == 100.0

    # 2. Test Misroute & Hazard Intercept Metric
    # Actual Hazard (Door 1) misrouted to Recyclable (Door 4)
    tracker.record_evaluation(
        actual_door=1,
        predicted_door=4,
        confidence=0.85,
        class_name="missed_knife",
        source="UNIT_TEST"
    )

    summary = tracker.get_summary()
    assert tracker.matrix[0, 3] == 1, "Expected cell (0, 3) to record misroute from Door 1 to 4"
    assert summary["overall_mse"] > 0.0, "MSE should be positive after misclassification"
    assert summary["chute_distance_mse"] > 0.0, "Chute distance MSE should be > 0"
    assert summary["mes_kpis"]["hazard_safety_index_pct"] < 100.0, "Hazard intercept safety index must reflect missed hazard"

    # 3. Test Benchmark Battery Generation
    initial_samples = tracker.total_samples
    benchmark_res = tracker.run_benchmark_battery(num_samples=50)
    assert benchmark_res["total_evaluations"] == initial_samples + 50
    assert len(benchmark_res["multivariate_mse_matrix"]) == 5
    assert len(benchmark_res["matrix_rows"]) == 5

    # 4. Test CSV Export
    csv_text = tracker.export_csv()
    assert "MES / MSE Error Matrix Audit Report" in csv_text
    assert "CONFUSION / ERROR MATRIX" in csv_text
    assert "MULTIVARIATE MSE ERROR COVARIANCE MATRIX" in csv_text

    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)

    print("  -> Passed! 5x5 Error Matrix, MSE covariance, chute distance, and MES KPIs validated.")


def test_gpu_safe_rollback():
    print("[TEST 9/9] Verifying GPU Execution & Zero-Drop Safe CPU Rollback Engine...")
    from detector import WasteDetector
    import numpy as np

    detector = WasteDetector()
    assert detector.is_ready is True
    assert hasattr(detector, "rollback_to_cpu")
    assert hasattr(detector, "retry_gpu")
    assert hasattr(detector, "get_device_telemetry")

    # 1. Test Telemetry Format
    telem = detector.get_device_telemetry()
    assert "device" in telem
    assert "active_device" in telem
    assert "device_mode" in telem
    assert "rollback_triggered" in telem
    assert "rollback_reason" in telem

    # 2. Test Dynamic Safe Rollback Trigger
    initial_rollback_count = detector.rollback_count
    detector.rollback_to_cpu("Simulated CUDA Out Of Memory fault during belt inspection")

    assert detector.device == "cpu"
    assert detector.active_device == "cpu"
    assert detector.device_mode == "CPU"
    assert detector.use_half is False
    assert detector.rollback_triggered is True
    assert "Simulated CUDA Out Of Memory fault" in detector.rollback_reason
    assert detector.rollback_count == initial_rollback_count + 1

    # 3. Test Zero-Drop Frame Inference Following Rollback
    dummy_frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    res = detector.detect(dummy_frame)
    assert res is not None
    assert isinstance(res.items, list)
    assert res.inference_time_ms >= 0.0

    # 4. Test Operator GPU Re-engagement
    ok, msg = detector.retry_gpu()
    assert isinstance(ok, bool)
    assert isinstance(msg, str)
    # The system must remain healthy regardless of whether physical CUDA is linked or CPU fallback
    assert detector.is_ready is True

    print("  -> Passed! GPU telemetry, simulated OOM fallback, and zero-drop CPU inference verified.")


def test_dual_stream_detector():
    print("[TEST 10/10] Verifying DualStreamDetector — dual-model comparison and safe rollback...")

    # 1. Primary detector initializes cleanly
    det_a = WasteDetector()
    assert det_a.is_ready is True, "Primary WasteDetector must be ready"

    # 2. DualStreamDetector wraps primary without loading secondary model
    dual = DualStreamDetector(detector_a=det_a)
    assert dual.detector_a is det_a
    assert dual.detector_b is None, "Model B must NOT be loaded until enable() is called"
    assert dual.enabled is False

    # 3. get_status() works before enabling
    status = dual.get_status()
    assert "enabled" in status
    assert "model_a" in status
    assert "model_b" in status
    assert "total_conflicts" in status
    assert "agreement_pct" in status
    assert "conflict_log" in status

    # 4. dual_detect() in disabled mode returns (result_a, None, [])
    dummy_frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    r_a, r_b, conflicts = dual.dual_detect(dummy_frame)
    assert r_a is not None, "result_a must always be returned"
    assert r_b is None, "result_b must be None when dual is disabled"
    assert conflicts == [], "Conflicts must be empty when dual is disabled"

    # 5. Conflict detection logic (unit test with mocked items)
    from detector import DetectedItem, DetectionResult
    def make_result(door_id):
        item = DetectedItem(
            class_name="bottle", confidence=0.85,
            bbox=(10, 10, 100, 200), door_id=door_id,
            category="TEST", color_bgr=(0, 200, 200),
            is_hazard=(door_id == 1), center_pos=(55, 105),
            has_frame=True, display_name="Test Bottle"
        )
        return DetectionResult(items=[item], has_hazard=(door_id == 1), inference_time_ms=50.0)

    result_a_mock = make_result(door_id=1)  # Model A → Hazard Door 1
    result_b_mock = make_result(door_id=4)  # Model B → Recyclable Door 4

    # Temporarily enable dual to test conflict detection
    dual.enabled = True
    dual.model_b_online = True
    conflicts_found = dual._find_conflicts(result_a_mock, result_b_mock)
    assert len(conflicts_found) == 1, f"Expected 1 conflict, got {len(conflicts_found)}"
    c = conflicts_found[0]
    assert c.class_name == "bottle"
    assert c.door_a == 1
    assert c.door_b == 4

    # 6. disable() cleans up
    dual.disable()
    assert dual.enabled is False
    assert dual.model_b_online is False
    assert dual.detector_b is None

    print("  -> Passed! DualStreamDetector init, dual_detect, conflict logic, and safe rollback verified.")


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
    test_error_matrix_and_mse()
    test_gpu_safe_rollback()
    test_dual_stream_detector()

    print("\n" + "=" * 60)
    print("    ALL INTEGRATION TESTS PASSED SUCCESSFULLY! [10/10]")
    print("=" * 60 + "\n")


