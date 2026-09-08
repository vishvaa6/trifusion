"""
AI Waste Segregation Prototype - Virtual Conveyor Simulation
Procedural conveyor belt rendering, item spawner with realistic physics,
3-Zone inspection architecture, and diversion routing.
"""

import math
import random
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from config import (
    CONVEYOR_HEIGHT,
    CONVEYOR_WIDTH,
    CONVEYOR_X,
    CONVEYOR_Y,
    DOORS,
    PALETTE,
    ZONE_A_END,
    ZONE_B_END,
    ZONE_B_START,
    ZONE_C_START,
)
from door_system import DoorSystem
from logger import WasteLogger
from scenario_engine import ScenarioEngine


class ConveyorItem:
    """A procedural waste item traveling on the conveyor belt."""

    def __init__(self, data: dict, start_x: float, y: float, base_speed: float):
        self.name: str = data["name"]
        self.label: str = data.get("label", self.name)
        self.w: int = data.get("width", 50)
        self.h: int = data.get("height", 35)
        self.door_id: int = data.get("door_id", 4)
        self.color: Tuple[int, int, int] = data.get("color", (100, 100, 100))

        self.x: float = start_x
        self.y: float = y
        self.base_y: float = y
        self.speed: float = base_speed
        self.angle: float = random.uniform(-15.0, 15.0)
        self.jitter_phase: float = random.uniform(0.0, 6.28)

        # Lifecycle flags
        self.scanned: bool = False
        self.diverted: bool = False
        self.completed: bool = False

        # Trajectory towards door
        self.target_door_pos: Optional[Tuple[int, int]] = None
        self.trajectory_t: float = 0.0

    def update(self, dt: float, belt_running: bool) -> None:
        """Update item physics and trajectory."""
        if self.diverted and self.target_door_pos:
            # Traveling towards door chute
            self.trajectory_t += 1.8 * dt
            if self.trajectory_t >= 1.0:
                self.completed = True
            else:
                tx, ty = self.target_door_pos
                t = self.trajectory_t
                # Smooth curve down into door
                self.x = (1 - t) * self.x + t * tx
                self.y = (1 - t) * self.y + t * ty
        elif belt_running:
            # Moving along conveyor
            self.x += self.speed * dt
            self.jitter_phase += 8.0 * dt
            self.y = self.base_y + math.sin(self.jitter_phase) * 1.5

    def draw(self, canvas: np.ndarray) -> None:
        """Draw item shape, category border, and label on canvas."""
        ix, iy = int(self.x), int(self.y)
        hw, hh = self.w // 2, self.h // 2
        door_info = DOORS[self.door_id]

        # Draw rotated rounded body
        rect = ((float(ix), float(iy)), (float(self.w), float(self.h)), self.angle)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # Main item body
        cv2.fillPoly(canvas, [box], self.color)
        # Category-coded outline
        outline_color = door_info.color_bgr if self.scanned else (120, 125, 135)
        outline_thick = 2 if self.scanned else 1
        cv2.polylines(canvas, [box], True, outline_color, outline_thick, cv2.LINE_AA)

        # Center highlight dot
        cv2.circle(canvas, (ix, iy), 3, (255, 255, 255), -1)

        # Floating label
        label_text = self.name.upper()
        font_scale = 0.35
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        lx = ix - tw // 2
        ly = iy - hh - 6
        cv2.rectangle(canvas, (lx - 3, ly - th - 2), (lx + tw + 3, ly + 2), (20, 20, 24), -1)
        cv2.rectangle(canvas, (lx - 3, ly - th - 2), (lx + tw + 3, ly + 2), outline_color, 1)
        cv2.putText(canvas, label_text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (240, 240, 240), 1)


class ConveyorSimulator:
    """Manages procedural conveyor belt animation, zones, and item life cycle."""

    def __init__(self, scenario_engine: ScenarioEngine, door_system: DoorSystem, logger: WasteLogger):
        self.scenario_engine = scenario_engine
        self.door_system = door_system
        self.logger = logger

        self.belt_x = CONVEYOR_X
        self.belt_y = CONVEYOR_Y
        self.belt_w = CONVEYOR_WIDTH
        self.belt_h = CONVEYOR_HEIGHT

        self.items: List[ConveyorItem] = []
        self.stripe_offset: float = 0.0
        self.base_belt_speed: float = 95.0  # pixels per second

        # Pause / Safety state
        self.belt_paused: bool = False
        self.pause_timer: float = 0.0

        # Scan line animation
        self.scan_beam_x: float = float(CONVEYOR_X + ZONE_B_START)
        self.scan_beam_dir: float = 1.0

    def pause_belt(self, duration: float) -> None:
        """Pause conveyor motion for safety lockout."""
        self.belt_paused = True
        self.pause_timer = max(self.pause_timer, duration)

    def resume_belt(self) -> None:
        self.belt_paused = False
        self.pause_timer = 0.0

    def inject_webcam_detection(self, class_name: str, door_id: int) -> None:
        """Spawn a virtual counterpart when an object is detected on physical webcam."""
        catalog_items = [item for item in [
            {"name": class_name, "label": class_name.title(), "width": 55, "height": 40, "color": (160, 160, 160)}
        ]]
        y_pos = random.uniform(self.belt_y + 40, self.belt_y + self.belt_h - 40)
        new_item = ConveyorItem(catalog_items[0], self.belt_x + 10, y_pos, self.base_belt_speed)
        new_item.door_id = door_id
        self.items.append(new_item)

    def update(self, dt: float) -> Optional[ConveyorItem]:
        """
        Updates conveyor motion, spawns items, handles zone inspection,
        and triggers door routing.
        Returns newly scanned item in Zone B, if any.
        """
        # Handle safety pause
        if self.pause_timer > 0:
            self.pause_timer -= dt
            if self.pause_timer <= 0:
                self.belt_paused = False

        is_running = not self.belt_paused
        current_speed = self.base_belt_speed * self.scenario_engine.active_scenario.speed_multiplier

        # Animate belt stripes
        if is_running:
            self.stripe_offset = (self.stripe_offset + current_speed * dt) % 36.0

        # Animate AI scan laser beam in Zone B
        zone_b_left = float(self.belt_x + ZONE_B_START)
        zone_b_right = float(self.belt_x + ZONE_B_END)
        self.scan_beam_x += self.scan_beam_dir * 180.0 * dt
        if self.scan_beam_x >= zone_b_right:
            self.scan_beam_x = zone_b_right
            self.scan_beam_dir = -1.0
        elif self.scan_beam_x <= zone_b_left:
            self.scan_beam_x = zone_b_left
            self.scan_beam_dir = 1.0

        # Spawn new items from active scenario
        if is_running and self.scenario_engine.should_spawn_item(dt):
            item_data = self.scenario_engine.sample_random_item()
            spawn_x = float(self.belt_x + 15)
            spawn_y = random.uniform(self.belt_y + 45, self.belt_y + self.belt_h - 45)
            self.items.append(ConveyorItem(item_data, spawn_x, spawn_y, current_speed))

        newly_scanned_item = None

        # Update items
        active_items = []
        for item in self.items:
            item.speed = current_speed
            item.update(dt, is_running)

            # Zone B: AI Inspection Trigger
            item_center_x = item.x
            if not item.scanned and (self.belt_x + ZONE_B_START) <= item_center_x <= (self.belt_x + ZONE_B_END):
                item.scanned = True
                newly_scanned_item = item

            # Zone C: Routing / Diversion to Door
            if not item.diverted and item_center_x >= (self.belt_x + ZONE_C_START):
                item.diverted = True
                door = self.door_system.get_door(item.door_id)
                if door:
                    item.target_door_pos = (door.center_x, door.top_y + 10)
                    self.door_system.route_item_to_door((int(item.x), int(item.y)), item.door_id, item.name)
                    self.logger.record_sorted_item(item.door_id)

            if not item.completed and item.x < (self.belt_x + self.belt_w + 50):
                active_items.append(item)

        self.items = active_items
        return newly_scanned_item

    def draw(self, canvas: np.ndarray, current_time: float) -> None:
        """Render conveyor frame, rubber belt, zones, scan laser, and items."""
        bx, by, bw, bh = self.belt_x, self.belt_y, self.belt_w, self.belt_h

        # 1. Outer Belt Steel Housing & Shadow
        cv2.rectangle(canvas, (bx - 8, by - 8), (bx + bw + 8, by + bh + 8), (28, 30, 36), -1)
        cv2.rectangle(canvas, (bx - 8, by - 8), (bx + bw + 8, by + bh + 8), (55, 60, 70), 2)

        # 2. Rubber Conveyor Surface
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), PALETTE["belt_rubber"], -1)

        # 3. Moving Belt Diagonal Traction Ribs
        stripe_spacing = 36
        for offset_x in range(-stripe_spacing * 2, bw + stripe_spacing * 2, stripe_spacing):
            x_start = bx + offset_x + int(self.stripe_offset)
            if bx - 20 <= x_start <= bx + bw + 20:
                p1 = (x_start, by + 4)
                p2 = (x_start + 18, by + bh - 4)
                # Clip line to belt rectangle
                cv2.line(canvas, p1, p2, PALETTE["belt_stripe"], 2, cv2.LINE_AA)

        # 4. Top and Bottom Guardrails with Hazard Stripes
        guard_h = 10
        # Top guard
        cv2.rectangle(canvas, (bx, by - guard_h), (bx + bw, by), (35, 38, 45), -1)
        cv2.rectangle(canvas, (bx, by - guard_h), (bx + bw, by), (0, 190, 240), 1)
        # Bottom guard
        cv2.rectangle(canvas, (bx, by + bh), (bx + bw, by + bh + guard_h), (35, 38, 45), -1)
        cv2.rectangle(canvas, (bx, by + bh), (bx + bw, by + bh + guard_h), (0, 190, 240), 1)

        # 5. Zone Demarcations & Header Indicators
        # Zone A (Staging)
        za_end = bx + ZONE_A_END
        cv2.line(canvas, (za_end, by), (za_end, by + bh), (90, 95, 105), 1, cv2.LINE_AA)
        cv2.putText(canvas, "[ZONE A: STAGING]", (bx + 12, by + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 145, 155), 1)

        # Zone B (AI Inspection Kill-Zone)
        zb_start = bx + ZONE_B_START
        zb_end = bx + ZONE_B_END
        # Draw Zone B boundary brackets
        zone_b_color = (255, 180, 50) if not self.belt_paused else (0, 0, 255)
        cv2.rectangle(canvas, (zb_start, by + 2), (zb_end, by + bh - 2), zone_b_color, 1)
        cv2.putText(canvas, "[ZONE B: AI INSPECTION KILL-ZONE]", (zb_start + 12, by + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, zone_b_color, 1)

        # Animated AI scanning vertical laser beam
        beam_x = int(self.scan_beam_x)
        beam_color = (255, 230, 120) if not self.belt_paused else (80, 80, 255)
        cv2.line(canvas, (beam_x, by + 4), (beam_x, by + bh - 4), beam_color, 2, cv2.LINE_AA)

        # Zone C (Routing & Diversion)
        zc_start = bx + ZONE_C_START
        cv2.line(canvas, (zc_start, by), (zc_start, by + bh), (90, 95, 105), 1, cv2.LINE_AA)
        cv2.putText(canvas, "[ZONE C: ROUTING]", (zc_start + 12, by + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 145, 155), 1)

        # 6. Render Items on Belt
        for item in self.items:
            item.draw(canvas)
