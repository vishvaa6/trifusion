# AI Industrial Waste Segregation Prototype
### 5-Door Smart Routing System with Hazard Intercept Protocol (HIP) & Virtual Conveyor Simulation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-orange.svg)](https://ultralytics.com/)
[![CUDA](https://img.shields.io/badge/NVIDIA-CUDA_FP16-76B900.svg)](https://developer.nvidia.com/cuda-zone)

---

## Executive Overview
The **AI Industrial Waste Segregation System** is a real-time computer vision prototype designed for modern Materials Recovery Facilities (MRFs). It leverages **Ultralytics YOLOv8** accelerated with **CUDA FP16 (Half-Precision)** to automatically identify, classify, and route incoming municipal and commercial waste into **5 dedicated sorting doors**.

Because physical conveyor belts and pneumatic sorting gates are not always accessible during development and evaluation, this system features a **full-fidelity procedural software simulation engine**: animated conveyor belts, 3-zone scanning architecture, physics-based item travel, smooth curved Bezier routing trajectories, animated sliding door shutters, and a SCADA telemetry dashboard.

---

## 5-Door Smart Routing Taxonomy

```
                   ← PROCEDURAL CONVEYOR BELT ←
┌────────────────────────────────────────────────────────────────────────┐
│ [STAGING ZONE A]  →  [🔍 AI INSPECTION ZONE B]  →  [ROUTING ZONE C]    │
│                                                                        │
│                      ↓ AI CLASSIFIER ROUTING ↓                         │
│                                                                        │
│  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐              │
│  │DOOR 1│    │DOOR 2│    │DOOR 3│    │DOOR 4│    │DOOR 5│              │
│  │  🔴  │    │  🟣  │    │  🟡  │    │  🟢  │    │  🟠  │              │
│  │HAZARD│    │E-WSTE│    │PAPER │    │RECYCL│    │ORGNIC│              │
│  └──────┘    └──────┘    └──────┘    └──────┘    └──────┘              │
│     🪣           🪣           🪣           🪣           🪣              │
│  (BIN 1)     (BIN 2)     (BIN 3)     (BIN 4)     (BIN 5)              │
└────────────────────────────────────────────────────────────────────────┘
```

| Door | Stream Name | Color | Target Waste Items | Sorting Behavior |
|:---:|:---:|:---:|:---|:---|
| **DOOR 1** | 🔴 **HAZARD** | Red | `knife`, `scissors`, `bottle` (glass/chemicals), `syringe`, `battery` | **EMERGENCY INTERCEPT**: Belt halts (2s), Door 1 locks, flashing strobe banner, logged to `hazard_log.csv`. |
| **DOOR 2** | 🟣 **E-WASTE** | Cyan | `cell phone`, `laptop`, `mouse`, `keyboard`, `remote`, `tv`, `toaster` | Door 2 shutter animates open; routes e-waste to recover rare-earths and isolate Li-ion fire risks. |
| **DOOR 3** | 🟡 **PAPER** | Yellow | `book`, `newspaper`, `cardboard`, `paper`, `box` | Door 3 shutter animates open; channels clean cellulose directly into fiber recycling chutes. |
| **DOOR 4** | 🟢 **RECYCLABLE** | Green | `cup`, `fork`, `spoon`, `bowl`, `wine glass`, `can`, `tin` | Door 4 shutter animates open; routes plastics, tableware, and aluminum cans. |
| **DOOR 5** | 🟠 **ORGANIC** | Orange | `banana`, `apple`, `sandwich`, `orange`, `broccoli`, `carrot`, `pizza`, `donut` | Door 5 shutter animates open; routes food scraps and organics to composting bins. |

---

## Key Innovations

1. **Hardware-Agnostic Software Simulation**:
   - Zero physical hardware required. The conveyor belt, moving items, laser scanning beam, and pneumatic sorting doors are procedurally rendered in real time using OpenCV and NumPy geometry.
2. **Hazard Intercept Protocol (HIP)**:
   - Instantaneous lock-on when a hazard enters Zone B. Halts the virtual conveyor, activates a 4Hz perimeter strobe border, displays `!!! HAZARD INTERCEPT ACTIVATED - DOOR 1 LOCKED !!!`, and writes an audit row to `hazard_log.csv`.
3. **5 Prebuilt Operational Scenarios**:
   - Test campaigns selectable on-the-fly with number keys `1–5`:
     - `[1] NORMAL OPERATIONS`: Standard municipal waste stream.
     - `[2] HIGH-TRAFFIC RUSH`: Elevated belt speed and dense stream.
     - `[3] HAZARD DRILL`: High-frequency hazard injection testing Door 1 lockout.
     - `[4] E-WASTE SURGE`: Electronics drop-off campaign load.
     - `[5] STRESS TEST`: Maximum throughput benchmark firing all 5 doors.
4. **RTX 3050 CUDA FP16 Acceleration**:
   - Runs YOLOv8 with Tensor Core FP16 (`half=True`) for ultra-fast frame rates (60–120+ FPS) and low latency.
   - Live VRAM meter on the HUD showing memory allocation against the 4096 MiB budget.
5. **Interactive Hardware Fallback ("Ask to Switch On")**:
   - If a physical webcam is missing or occupied, the system gracefully queries:
     `[?] Switch on Virtual Conveyor Simulation mode? (y/n) [default: y]: `
     Pressing Enter immediately boots the simulation.
6. **Webcam Hybrid Mode (PiP Infeed)**:
   - If a webcam is connected, you can hold up real objects (a bottle, scissors, phone, cup) in front of the lens. The AI detects them, highlights them with category bounding boxes, and **dynamically injects a virtual counterpart onto the simulation belt**!
7. **Comprehensive Audit Trail**:
   - Every hazard interception is recorded to `hazard_log.csv` with timestamp, class, confidence, and bounding box.
   - On exit (`q`), a shift audit report is generated and saved as `session_report.txt`.

---

## Project Structure

```
trifusion/
├── main.py              # Application entrypoint & event loop
├── config.py            # 5-Door definitions, categories, colors, geometry
├── detector.py          # YOLOv8 FP16 CUDA inference engine
├── simulator.py         # Procedural conveyor belt, item spawner & physics
├── door_system.py       # 5 animated doors, slide shutters & Bezier routing
├── hud.py               # SCADA dashboard, Door Status Monitor & alerts
├── scenario_engine.py   # 5 test campaign scenarios & item catalog
├── logger.py            # Hazard CSV audit logger & session report writer
├── requirements.txt     # Dependency definitions
├── README.md            # This documentation
├── hazard_log.csv       # Auto-generated runtime hazard log
└── session_report.txt   # Auto-generated runtime session report
```

---

## Installation & Setup

### 1. Prerequisites
- **Python 3.10+** (64-bit)
- **NVIDIA GPU** (RTX 3050 or compatible, CUDA driver 12.x+)

### 2. Virtual Environment Setup
```powershell
# Navigate to project root
cd d:\programming\trifusion

# Activate virtual environment
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
# Install PyTorch with CUDA support (for RTX 3050)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install application dependencies
pip install -r requirements.txt
```

---

## Running the Application

### Option A: Dedicated Virtual Simulation (No Webcam Needed)
```powershell
.venv\Scripts\python.exe main.py --simulate
```

### Option B: Live Webcam Mode (With Automatic Fallback)
```powershell
.venv\Scripts\python.exe main.py
```
*(If no webcam is detected, it will ask to switch on the simulation mode automatically.)*

### Optional CLI Arguments
```powershell
--simulate       # Force simulation mode directly without probing webcam
--camera <idx>   # Specify webcam device index (default: 0)
--conf <float>   # Confidence threshold (default: 0.40)
--no-pip         # Disable webcam Picture-in-Picture window
```

### Option C: Interactive Web SCADA Dashboard (Recommended)
```powershell
python dashboard.py
# Or double-click run_dashboard.bat
```
Open your browser at: **[http://localhost:5000](http://localhost:5000)** to view the live video stream, real-time classified objects feed with confidence scores, animated 5-door monitor cards, and in-browser scenario controls.

---

## Operator Keyboard Controls

| Key | Function |
|:---:|:---|
| `1` – `5` | Switch active test scenario (`Normal`, `High-Traffic`, `Hazard`, `E-Waste`, `Stress`) |
| `+` / `=` | Increase conveyor belt speed |
| `-` / `_` | Decrease conveyor belt speed |
| `P` | Pause / resume conveyor belt motion |
| `H` | Toggle SCADA Door Status Monitor on/off |
| `C` | Toggle Webcam Picture-in-Picture window |
| `S` | Capture and save a timestamped PNG screenshot |
| `R` | Reset session item counters and door tallies |
| `Q` / `ESC` | Safe shutdown: print and save `session_report.txt` |

---

## Compliance & Audit Output

### `hazard_log.csv` Example
```csv
timestamp,class_name,confidence,door_id,action,bbox_x1,bbox_y1,bbox_x2,bbox_y2
2026-09-08 12:35:10.412,knife,0.912,1,DOOR_LOCKED,320,180,380,210
2026-09-08 12:35:14.891,scissors,0.884,1,DOOR_LOCKED,310,175,370,225
```

### `session_report.txt` Example
```text
============================================================
     AI WASTE SEGREGATION PROTOTYPE - SESSION AUDIT REPORT
============================================================
Start Time       : 2026-09-08 12:30:00
End Time         : 2026-09-08 12:38:45
Session Duration : 00:08:45
Average FPS      : 74.2 FPS
Total Items      : 148
Hazard Alerts    : 11
------------------------------------------------------------
DOOR ROUTING BREAKDOWN:
  [DOOR 1] HAZARD       :   11 items (  7.4%) -> Hazardous Materials & Sharps
  [DOOR 2] E-WASTE      :   28 items ( 18.9%) -> Electronic Waste & Valuables
  [DOOR 3] PAPER        :   42 items ( 28.4%) -> Dry Paper, Cardboard & Fiber
  [DOOR 4] RECYCLABLE   :   45 items ( 30.4%) -> Plastics, Cans & Tableware
  [DOOR 5] ORGANIC      :   22 items ( 14.9%) -> Compost, Scraps & Organics
============================================================
```
