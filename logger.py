"""
AI Waste Segregation Prototype - Audit & Reporting Logger
Handles real-time hazard event logging to CSV and generates a comprehensive
industrial session report upon shutdown.
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional
from config import HAZARD_LOG_FILE, SESSION_REPORT_FILE, DOORS


class WasteLogger:
    """Manages compliance audit logs and end-of-shift session reports."""

    def __init__(self, hazard_file: str = HAZARD_LOG_FILE, report_file: str = SESSION_REPORT_FILE):
        self.hazard_file = hazard_file
        self.report_file = report_file
        self.session_start_time = time.time()
        self.session_start_dt = datetime.now()
        self.total_scanned_count: int = 0
        self.door_counts: Dict[int, int] = {door_id: 0 for door_id in DOORS.keys()}
        self.hazard_intercept_count: int = 0
        self.fps_samples: List[float] = []

        self._initialize_hazard_log()

    def _initialize_hazard_log(self) -> None:
        """Create or ensure header exists in the CSV audit log."""
        if not os.path.exists(self.hazard_file):
            with open(self.hazard_file, "w", encoding="utf-8") as f:
                f.write("timestamp,class_name,confidence,door_id,action,bbox_x1,bbox_y1,bbox_x2,bbox_y2\n")

    def log_hazard(self, class_name: str, confidence: float, bbox: tuple, door_id: int = 1) -> None:
        """Record a single hazard interception event into the audit CSV."""
        self.hazard_intercept_count += 1
        x1, y1, x2, y2 = bbox if bbox else (0, 0, 0, 0)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        row = f"{timestamp},{class_name},{confidence:.3f},{door_id},DOOR_LOCKED,{int(x1)},{int(y1)},{int(x2)},{int(y2)}\n"
        with open(self.hazard_file, "a", encoding="utf-8") as f:
            f.write(row)

    def record_sorted_item(self, door_id: int) -> None:
        """Increment counters when an item successfully passes through a door."""
        self.total_scanned_count += 1
        if door_id in self.door_counts:
            self.door_counts[door_id] += 1

    def record_fps(self, fps: float) -> None:
        """Sample FPS for the session average calculation."""
        if fps > 0:
            self.fps_samples.append(fps)
            if len(self.fps_samples) > 2000:
                self.fps_samples = self.fps_samples[-1000:]

    def generate_session_report(self) -> str:
        """Generate formatted summary report and write to file."""
        duration_sec = time.time() - self.session_start_time
        mins, secs = divmod(int(duration_sec), 60)
        hours, mins = divmod(mins, 60)
        duration_str = f"{hours:02d}:{mins:02d}:{secs:02d}"

        avg_fps = sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0.0

        lines = [
            "=" * 60,
            "     AI WASTE SEGREGATION PROTOTYPE - SESSION AUDIT REPORT",
            "=" * 60,
            f"Start Time       : {self.session_start_dt.strftime('%Y-%m-%d %H:%M:%S')}",
            f"End Time         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Session Duration : {duration_str}",
            f"Average FPS      : {avg_fps:.1f} FPS",
            f"Total Items      : {self.total_scanned_count}",
            f"Hazard Alerts    : {self.hazard_intercept_count}",
            "-" * 60,
            "DOOR ROUTING BREAKDOWN:",
        ]

        for door_id, door in DOORS.items():
            count = self.door_counts.get(door_id, 0)
            pct = (count / self.total_scanned_count * 100) if self.total_scanned_count > 0 else 0.0
            lines.append(f"  [{door.name}] {door.category:<12} : {count:4d} items ({pct:5.1f}%) -> {door.description}")

        lines.extend([
            "-" * 60,
            f"Hazard Log File  : {os.path.abspath(self.hazard_file)}",
            f"Session Report   : {os.path.abspath(self.report_file)}",
            "=" * 60,
        ])

        report_text = "\n".join(lines)
        with open(self.report_file, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")

        return report_text
