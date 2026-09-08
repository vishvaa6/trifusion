"""
AI Waste Segregation Prototype - SCADA HUD & Telemetry Visualizer
Renders the industrial control room interface, Door Status Monitor,
VRAM meter, throughput statistics, and the Hazard Intercept emergency alert.
"""

import math
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from config import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DOORS,
    PALETTE,
    STROBE_FREQUENCY_HZ,
)
from door_system import DoorSystem
from logger import WasteLogger
from scenario_engine import ScenarioEngine


class SCADAVisualizer:
    """Industrial SCADA HUD Visualizer with telemetry, live door monitoring, and alerts."""

    def __init__(self, door_system: DoorSystem, scenario_engine: ScenarioEngine, logger: WasteLogger):
        self.door_system = door_system
        self.scenario_engine = scenario_engine
        self.logger = logger
        self.show_hud: bool = True
        self.show_pip: bool = True

        self.last_hazard_class: str = ""
        self.last_hazard_conf: float = 0.0

    def trigger_hazard_alert(self, class_name: str, confidence: float) -> None:
        """Cache latest hazard detection details for the alert banner."""
        self.last_hazard_class = class_name
        self.last_hazard_conf = confidence

    def draw(
        self,
        canvas: np.ndarray,
        fps: float,
        inference_ms: float,
        vram_mb: float,
        is_hazard_locked: bool,
        current_time: float,
        webcam_frame: Optional[np.ndarray] = None
    ) -> None:
        """Render complete SCADA interface over canvas."""
        # 1. Top Industrial Navigation Header Bar
        self._draw_header(canvas, fps, inference_ms, vram_mb, current_time)

        # 2. Right Side Door Status Monitor Panel
        if self.show_hud:
            self._draw_door_monitor_panel(canvas, current_time)

        # 3. Bottom Operator Key Controls Bar
        self._draw_bottom_bar(canvas)

        # 4. Webcam Picture-in-Picture (PiP) Inset
        if self.show_pip and webcam_frame is not None:
            self._draw_pip_webcam(canvas, webcam_frame)

        # 5. Emergency Hazard Intercept Protocol Banner & Strobe
        if is_hazard_locked:
            self._draw_hazard_strobe(canvas, current_time)

    def _draw_header(self, canvas: np.ndarray, fps: float, inf_ms: float, vram_mb: float, cur_time: float) -> None:
        """Render top telemetry header."""
        header_h = 55
        cv2.rectangle(canvas, (0, 0), (CANVAS_WIDTH, header_h), (26, 28, 34), -1)
        cv2.line(canvas, (0, header_h), (CANVAS_WIDTH, header_h), (50, 55, 65), 2)

        # Title and system protocol
        cv2.putText(canvas, "AI INDUSTRIAL WASTE SEGREGATION", (24, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2)
        cv2.putText(canvas, "5-DOOR SMART ROUTING PROTOCOL | SCADA V4.2", (24, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 145, 155), 1)

        # Telemetry pills on right side of header
        # FPS Pill
        self._draw_pill(canvas, (CANVAS_WIDTH - 460, 12, 100, 32), f"FPS: {fps:4.1f}", (0, 255, 180))
        # Inference Pill
        self._draw_pill(canvas, (CANVAS_WIDTH - 345, 12, 110, 32), f"INF: {inf_ms:4.1f}ms", (255, 200, 50))
        # VRAM Pill (4096MB limit on RTX 3050)
        vram_color = (0, 255, 0) if vram_mb < 2500 else (0, 200, 255)
        self._draw_pill(canvas, (CANVAS_WIDTH - 220, 12, 125, 32), f"VRAM: {int(vram_mb)}MB", vram_color)
        # Uptime
        uptime_sec = int(cur_time - self.logger.session_start_time)
        m, s = divmod(uptime_sec, 60)
        h, m = divmod(m, 60)
        self._draw_pill(canvas, (CANVAS_WIDTH - 85, 12, 75, 32), f"{h:02d}:{m:02d}:{s:02d}", (200, 200, 200))

    def _draw_pill(self, canvas: np.ndarray, rect: Tuple[int, int, int, int], text: str, color: Tuple[int, int, int]) -> None:
        """Helper to draw modern rounded telemetry pill."""
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (38, 41, 50), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 65, 78), 1)
        cv2.putText(canvas, text, (x + 8, y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    def _draw_door_monitor_panel(self, canvas: np.ndarray, current_time: float) -> None:
        """Render right-side SCADA panel with live door statuses and metrics."""
        px = 810
        py = 70
        pw = 265
        ph = 490

        # Background panel
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), PALETTE["panel_bg"], -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), PALETTE["panel_border"], 2)

        # Header
        cv2.rectangle(canvas, (px, py), (px + pw, py + 34), (28, 30, 36), -1)
        cv2.line(canvas, (px, py + 34), (px + pw, py + 34), PALETTE["panel_border"], 1)
        cv2.putText(canvas, "DOOR STATUS MONITOR", (px + 14, py + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 235, 240), 1)

        card_y = py + 44
        card_h = 58

        for door_id in sorted(DOORS.keys()):
            door = self.door_system.get_door(door_id)
            if not door:
                continue

            door_info = door.info
            color = door_info.color_bgr

            # Door Card Box
            card_border = color if (door.is_open or door.is_hazard_locked) else (50, 54, 64)
            cv2.rectangle(canvas, (px + 10, card_y), (px + pw - 10, card_y + card_h), (28, 30, 36), -1)
            cv2.rectangle(canvas, (px + 10, card_y), (px + pw - 10, card_y + card_h), card_border, 1)

            # Left color bar
            cv2.rectangle(canvas, (px + 10, card_y), (px + 16, card_y + card_h), color, -1)

            # Door Title & Category
            title_text = f"{door_info.name}: {door_info.category}"
            cv2.putText(canvas, title_text, (px + 22, card_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1)

            # Status indicator
            if door.is_hazard_locked:
                st = "LOCKED"
                st_color = (0, 0, 255)
            elif door.is_open:
                st = "OPEN"
                st_color = color
            else:
                st = "SHUT"
                st_color = (120, 125, 135)

            cv2.putText(canvas, f"[{st}]", (px + pw - 68, card_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, st_color, 1)

            # Metrics
            count_str = f"Items: {door.item_count:<3d} | Last: {door.last_item_name[:10]}"
            cv2.putText(canvas, count_str, (px + 22, card_y + 43),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 165, 175), 1)

            card_y += card_h + 10

        # Total Metrics Section at bottom of panel
        metric_y = card_y + 5
        cv2.line(canvas, (px + 10, metric_y), (px + pw - 10, metric_y), (60, 65, 78), 1)

        total_items = self.logger.total_scanned_count
        duration_min = max(0.01, (current_time - self.logger.session_start_time) / 60.0)
        items_per_min = total_items / duration_min

        scen = self.scenario_engine.active_scenario
        cv2.putText(canvas, f"TOTAL SORTED: {total_items}", (px + 14, metric_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1)
        cv2.putText(canvas, f"RATE: {items_per_min:4.1f} items/min", (px + 14, metric_y + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 180), 1)
        cv2.putText(canvas, f"SCENARIO [{scen.id}]: {scen.name}", (px + 14, metric_y + 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 200, 50), 1)

    def _draw_bottom_bar(self, canvas: np.ndarray) -> None:
        """Render operator hotkey legend at bottom."""
        by = CANVAS_HEIGHT - 38
        cv2.rectangle(canvas, (0, by), (CANVAS_WIDTH, CANVAS_HEIGHT), (20, 22, 26), -1)
        cv2.line(canvas, (0, by), (CANVAS_WIDTH, by), (45, 48, 56), 1)

        hints = (
            "[1-5] Scenarios  |  [+/-] Speed  |  [P] Pause  |  [H] Toggle HUD  |  "
            "[S] Snapshot  |  [C] Cam PiP  |  [R] Reset  |  [Q] Quit"
        )
        cv2.putText(canvas, hints, (25, by + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 185, 195), 1)

    def _draw_pip_webcam(self, canvas: np.ndarray, webcam_frame: np.ndarray) -> None:
        """Render live webcam in top-center PiP inset with scanner frame."""
        pip_w, pip_h = 190, 130
        pip_x = CANVAS_WIDTH - 265 - pip_w - 20
        pip_y = 65

        # Resize webcam frame
        resized = cv2.resize(webcam_frame, (pip_w, pip_h))

        # Border and header
        cv2.rectangle(canvas, (pip_x - 3, pip_y - 22), (pip_x + pip_w + 3, pip_y + pip_h + 3), (25, 28, 34), -1)
        cv2.rectangle(canvas, (pip_x - 3, pip_y - 22), (pip_x + pip_w + 3, pip_y + pip_h + 3), (0, 200, 255), 1)
        cv2.putText(canvas, "LIVE CAM INFEED", (pip_x + 6, pip_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)

        # Place frame
        canvas[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = resized

    def _draw_hazard_strobe(self, canvas: np.ndarray, current_time: float) -> None:
        """Render high-priority Hazard Intercept banner and strobe border."""
        # 4Hz Strobe oscillation
        flash_on = (int(current_time * STROBE_FREQUENCY_HZ * 2) % 2) == 0

        # Full perimeter strobe border
        border_color = (0, 0, 255) if flash_on else (20, 20, 180)
        cv2.rectangle(canvas, (0, 0), (CANVAS_WIDTH, CANVAS_HEIGHT), border_color, 8)

        # Top Emergency Banner
        banner_h = 60
        banner_y = 56
        banner_color = (0, 0, 220) if flash_on else (0, 0, 140)
        cv2.rectangle(canvas, (0, banner_y), (800, banner_y + banner_h), banner_color, -1)
        cv2.rectangle(canvas, (0, banner_y), (800, banner_y + banner_h), (255, 255, 255), 2)

        main_alert = "!!! HAZARD INTERCEPT ACTIVATED - DOOR 1 LOCKED !!!"
        cv2.putText(canvas, main_alert, (30, banner_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        sub_detail = f"DANGEROUS ITEM: {self.last_hazard_class.upper()} ({self.last_hazard_conf:.1%}) | CONVEYOR PAUSED"
        cv2.putText(canvas, sub_detail, (30, banner_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 230, 100), 1)
