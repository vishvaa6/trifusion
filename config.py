"""
AI Waste Segregation Prototype - Configuration
Defines camera parameters, YOLO model settings, the 5-Door Waste Routing taxonomy,
colors, simulation scenarios, and industrial HUD styling.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

# Video & Canvas Settings
# Index 1 = Camo Camera (HD 720p), Index 0 = Built-in Webcam
CAMO_CAMERA_INDEX: int = 1
BUILTIN_CAMERA_INDEX: int = 0
CAMERA_INDEX: int = 1  # Default to Camo Camera
WEBCAM_WIDTH: int = 640
WEBCAM_HEIGHT: int = 480
CANVAS_WIDTH: int = 1100
CANVAS_HEIGHT: int = 650
TARGET_FPS: int = 60

# Model Settings
MODEL_NAME: str = "yolo11n.pt"  # Latest Ultralytics YOLO11 Nano model
CONFIDENCE_THRESHOLD: float = 0.40
TRANSPARENT_CONFIDENCE_THRESHOLD: float = 0.20  # Heightened sensitivity floor for transparent glass & PET plastic
TRANSPARENT_CLASSES: Set[str] = {
    "bottle", "wine glass", "cup", "bowl", "vase", "tie"
}
USE_HALF_PRECISION: bool = True  # FP16 inference for RTX 3050 Tensor Cores

# 5-Door Waste Routing Taxonomy
# Maps COCO-80 classes and waste items to specific doors
DOOR_1_HAZARD: Set[str] = {
    "knife", "scissors", "bottle", "syringe", "broken glass", "battery"
}

DOOR_2_EWASTE: Set[str] = {
    "cell phone", "laptop", "mouse", "keyboard", "remote",
    "tv", "microwave", "toaster", "electronics"
}

DOOR_3_PAPER: Set[str] = {
    "book", "newspaper", "paper", "cardboard", "box", "notebook", "stationery", "binder", "envelope", "backpack"
}

DOOR_4_RECYCLABLE: Set[str] = {
    "cup", "fork", "spoon", "bowl", "wine glass", "can", "tin"
}

DOOR_5_ORGANIC: Set[str] = {
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "pizza", "donut", "cake", "food"
}

@dataclass
class DoorInfo:
    id: int
    name: str
    category: str
    color_bgr: Tuple[int, int, int]
    target_classes: Set[str]
    description: str

DOORS: Dict[int, DoorInfo] = {
    1: DoorInfo(
        id=1,
        name="DOOR 1",
        category="HAZARD",
        color_bgr=(0, 0, 255),      # Red
        target_classes=DOOR_1_HAZARD,
        description="Hazardous Materials & Sharps"
    ),
    2: DoorInfo(
        id=2,
        name="DOOR 2",
        category="E-WASTE",
        color_bgr=(255, 255, 0),    # Cyan
        target_classes=DOOR_2_EWASTE,
        description="Electronic Waste & Valuables"
    ),
    3: DoorInfo(
        id=3,
        name="DOOR 3",
        category="PAPER & STATIONERY",
        color_bgr=(0, 220, 255),    # Yellow-Amber
        target_classes=DOOR_3_PAPER,
        description="Paper, Books, Stationery & Fiber"
    ),
    4: DoorInfo(
        id=4,
        name="DOOR 4",
        category="RECYCLABLE",
        color_bgr=(0, 255, 0),      # Pure Green
        target_classes=DOOR_4_RECYCLABLE,
        description="Plastics, Cans & Tableware"
    ),
    5: DoorInfo(
        id=5,
        name="DOOR 5",
        category="ORGANIC",
        color_bgr=(0, 140, 255),    # Industrial Orange
        target_classes=DOOR_5_ORGANIC,
        description="Compost, Scraps & Organics"
    ),
}

# UI & HUD Styling Palette
PALETTE = {
    "background": (24, 26, 30),        # Dark industrial charcoal
    "panel_bg": (35, 38, 45),          # Slightly lighter panel
    "panel_border": (60, 65, 75),      # Border
    "text_primary": (245, 245, 245),   # White
    "text_secondary": (160, 165, 175), # Muted gray
    "accent_blue": (235, 150, 40),     # Cyan-blue highlight
    "hazard_red": (0, 0, 255),         # Emergency Red
    "hazard_bg": (20, 20, 180),        # Dark red alert fill
    "belt_rubber": (40, 42, 46),       # Conveyor rubber
    "belt_stripe": (65, 68, 75),       # Conveyor moving stripe
    "belt_guard": (0, 190, 240),       # Conveyor yellow guardrails
    "zone_scan": (255, 180, 50),       # AI Scan zone border
}

# Conveyor Belt Geometry on Canvas
CONVEYOR_X: int = 30
CONVEYOR_Y: int = 140
CONVEYOR_WIDTH: int = 760
CONVEYOR_HEIGHT: int = 220

# Inspection Zones (relative to conveyor X)
ZONE_A_END: int = 180   # Staging / Entry Zone
ZONE_B_START: int = 200 # AI Inspection / Detection Zone Start
ZONE_B_END: int = 560   # AI Inspection / Detection Zone End
ZONE_C_START: int = 580 # Routing / Door Diversion Zone Start

# Safety & Intercept Protocol Timers
HAZARD_SAFETY_PAUSE_SEC: float = 2.0  # Belt pause duration on hazard lock
DOOR_OPEN_DURATION_SEC: float = 1.4   # Time a door remains open to accept item
STROBE_FREQUENCY_HZ: float = 4.0      # Alert flashing rate

# Audit Log Paths
HAZARD_LOG_FILE: str = "hazard_log.csv"
SESSION_REPORT_FILE: str = "session_report.txt"
