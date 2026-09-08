"""
AI Waste Segregation Prototype - 5-Door Routing System
Manages door states, slide/pivot opening animations, item diversion trajectories,
bin counters, and Door 1 Hazard Lockout protocol.
"""

import math
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from config import (
    DOORS,
    DoorInfo,
    DOOR_OPEN_DURATION_SEC,
    HAZARD_SAFETY_PAUSE_SEC,
    PALETTE,
)


class WasteDoor:
    """Represents an individual sorting door and its associated collection bin."""

    def __init__(self, info: DoorInfo, rect: Tuple[int, int, int, int]):
        self.info = info
        self.x, self.y, self.w, self.h = rect
        self.center_x = self.x + self.w // 2
        self.top_y = self.y

        # Animation & State
        self.is_open: bool = False
        self.open_progress: float = 0.0  # 0.0 (closed) to 1.0 (fully open)
        self.open_timer: float = 0.0
        self.item_count: int = 0
        self.last_item_name: str = "None"
        self.last_item_time: float = 0.0

        # Door 1 specific hazard state
        self.is_hazard_locked: bool = False
        self.hazard_timer: float = 0.0

    def trigger_open(self, item_name: str) -> None:
        """Trigger door opening animation and accept item."""
        self.is_open = True
        self.open_timer = DOOR_OPEN_DURATION_SEC
        self.last_item_name = item_name
        self.last_item_time = time.time()
        self.item_count += 1

        if self.info.id == 1:
            self.is_hazard_locked = True
            self.hazard_timer = HAZARD_SAFETY_PAUSE_SEC

    def update(self, dt: float) -> None:
        """Update door animation progress and timers."""
        # Update door open timer
        if self.open_timer > 0:
            self.open_timer -= dt
            if self.open_timer <= 0:
                self.is_open = False

        # Smooth opening animation
        target_progress = 1.0 if self.is_open else 0.0
        speed = 5.0 * dt
        if self.open_progress < target_progress:
            self.open_progress = min(target_progress, self.open_progress + speed)
        elif self.open_progress > target_progress:
            self.open_progress = max(target_progress, self.open_progress - speed)

        # Update hazard lock timer
        if self.hazard_timer > 0:
            self.hazard_timer -= dt
            if self.hazard_timer <= 0:
                self.is_hazard_locked = False

    def draw(self, canvas: np.ndarray, current_time: float) -> None:
        """Render the door, chute, animated shutter, and collection bin."""
        color = self.info.color_bgr
        x, y, w, h = self.x, self.y, self.w, self.h

        # 1. Door Frame & Chute
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (45, 48, 55), -1)
        border_color = color if (self.is_open or self.is_hazard_locked) else (70, 75, 85)
        border_thickness = 3 if (self.is_open or self.is_hazard_locked) else 1
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, border_thickness)

        # 2. Door Header & Label
        header_h = 24
        cv2.rectangle(canvas, (x, y), (x + w, y + header_h), (30, 32, 38), -1)
        cv2.line(canvas, (x, y + header_h), (x + w, y + header_h), border_color, 1)

        door_title = f"{self.info.name}"
        cv2.putText(canvas, door_title, (x + 8, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 225, 230), 1)

        # Status badge [CLOSED] or [ OPEN ]
        if self.is_hazard_locked:
            badge_text = "LOCKED"
            badge_color = (0, 0, 255)
        elif self.is_open:
            badge_text = "OPEN"
            badge_color = color
        else:
            badge_text = "SHUT"
            badge_color = (120, 125, 135)

        badge_w = 42
        badge_x = x + w - badge_w - 6
        cv2.rectangle(canvas, (badge_x, y + 4), (badge_x + badge_w, y + header_h - 4), (20, 20, 24), -1)
        cv2.rectangle(canvas, (badge_x, y + 4), (badge_x + badge_w, y + header_h - 4), badge_color, 1)
        cv2.putText(canvas, badge_text, (badge_x + 5, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, badge_color, 1)

        # 3. Category Banner Inside Door
        cat_y = y + header_h + 18
        cv2.putText(canvas, self.info.category, (x + 8, cat_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # 4. Animated Sliding Shutter
        shutter_y = y + header_h + 24
        shutter_h = h - header_h - 52
        shutter_w = w - 16
        shutter_x = x + 8

        # Interior dark chute
        cv2.rectangle(canvas, (shutter_x, shutter_y), (shutter_x + shutter_w, shutter_y + shutter_h), (18, 18, 22), -1)

        # Sliding panel moves UP as open_progress increases
        panel_offset = int(shutter_h * self.open_progress)
        current_shutter_y = shutter_y + panel_offset
        current_shutter_h = shutter_h - panel_offset

        if current_shutter_h > 0:
            # Render ribbed metal shutter texture
            cv2.rectangle(canvas, (shutter_x, current_shutter_y),
                          (shutter_x + shutter_w, current_shutter_y + current_shutter_h), (55, 60, 70), -1)
            # Horizontal shutter slats
            for slat_y in range(current_shutter_y, current_shutter_y + current_shutter_h, 6):
                cv2.line(canvas, (shutter_x, slat_y), (shutter_x + shutter_w, slat_y), (40, 44, 52), 1)
            cv2.rectangle(canvas, (shutter_x, current_shutter_y),
                          (shutter_x + shutter_w, current_shutter_y + current_shutter_h), (80, 85, 95), 1)

        # 5. Collection Bin at Bottom of Door
        bin_y = y + h - 26
        cv2.rectangle(canvas, (x + 6, bin_y), (x + w - 6, y + h - 6), (28, 30, 36), -1)
        cv2.rectangle(canvas, (x + 6, bin_y), (x + w - 6, y + h - 6), border_color, 1)

        count_text = f"BIN: {self.item_count}"
        cv2.putText(canvas, count_text, (x + 12, bin_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1)


class DoorSystem:
    """Manages all 5 sorting doors, routing trajectories, and diverter animations."""

    def __init__(self, start_x: int = 40, y: int = 390, total_width: int = 740, height: int = 155):
        self.doors: Dict[int, WasteDoor] = {}
        num_doors = len(DOORS)
        gap = 14
        door_width = (total_width - (num_doors - 1) * gap) // num_doors

        current_x = start_x
        for door_id in sorted(DOORS.keys()):
            rect = (current_x, y, door_width, height)
            self.doors[door_id] = WasteDoor(DOORS[door_id], rect)
            current_x += door_width + gap

        # Active Routing Trajectory Paths
        # List of dicts: {"start": (x,y), "end": (x,y), "color": bgr, "progress": float, "item_name": str}
        self.active_trajectories: List[dict] = []

    def get_door(self, door_id: int) -> Optional[WasteDoor]:
        return self.doors.get(door_id)

    def route_item_to_door(self, item_pos: Tuple[int, int], door_id: int, item_name: str) -> None:
        """Initiate animated trajectory from item conveyor position to assigned door."""
        door = self.doors.get(door_id)
        if not door:
            return

        door.trigger_open(item_name)

        # Record trajectory path for drawing
        self.active_trajectories.append({
            "start": item_pos,
            "end": (door.center_x, door.top_y),
            "color": door.info.color_bgr,
            "progress": 0.0,
            "item_name": item_name,
            "door_id": door_id
        })

    def update(self, dt: float) -> bool:
        """
        Updates all doors and active trajectories.
        Returns True if any door has triggered a Hazard Intercept Lock.
        """
        hazard_active = False
        for door in self.doors.values():
            door.update(dt)
            if door.is_hazard_locked:
                hazard_active = True

        # Update trajectories
        remaining_trajectories = []
        for traj in self.active_trajectories:
            traj["progress"] += 2.0 * dt  # Complete trajectory in ~0.5s
            if traj["progress"] < 1.0:
                remaining_trajectories.append(traj)
        self.active_trajectories = remaining_trajectories

        return hazard_active

    def draw(self, canvas: np.ndarray, current_time: float) -> None:
        """Render all 5 doors and active curved routing arrows."""
        # 1. Draw Door units
        for door in self.doors.values():
            door.draw(canvas, current_time)

        # 2. Draw active curved routing trajectories
        for traj in self.active_trajectories:
            self._draw_curved_routing_arrow(canvas, traj)

    def _draw_curved_routing_arrow(self, canvas: np.ndarray, traj: dict) -> None:
        """Draw a smooth curved directional trajectory from conveyor to door."""
        sx, sy = traj["start"]
        ex, ey = traj["end"]
        color = traj["color"]
        p = min(1.0, traj["progress"])

        # Quadratic Bezier control point (curves outward and down)
        cx = (sx + ex) // 2 + (20 if ex >= sx else -20)
        cy = (sy + ey) // 2 - 15

        points = []
        steps = 16
        end_step = max(2, int(steps * p))

        for i in range(end_step + 1):
            t = i / float(steps)
            # Quadratic Bezier formula: (1-t)^2 * S + 2*(1-t)*t * C + t^2 * E
            px = int((1 - t) ** 2 * sx + 2 * (1 - t) * t * cx + t ** 2 * ex)
            py = int((1 - t) ** 2 * sy + 2 * (1 - t) * t * cy + t ** 2 * ey)
            points.append((px, py))

        # Draw glow line
        if len(points) >= 2:
            pts_arr = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_arr], False, color, 2, cv2.LINE_AA)
            # Arrow tip at front
            tip_x, tip_y = points[-1]
            cv2.circle(canvas, (tip_x, tip_y), 4, color, -1)
