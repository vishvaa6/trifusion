"""
AI Waste Segregation Prototype - Scenario Engine
Provides 5 prebuilt industrial test campaigns for simulating normal operations,
surge conditions, hazard drills, and system stress testing.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Scenario:
    id: int
    name: str
    description: str
    speed_multiplier: float
    spawn_interval: float  # seconds between item spawns
    door_weights: Dict[int, float]  # Probability distribution across Doors 1..5


SCENARIOS: Dict[int, Scenario] = {
    1: Scenario(
        id=1,
        name="NORMAL OPERATIONS",
        description="Balanced industrial municipal waste flow.",
        speed_multiplier=1.0,
        spawn_interval=2.4,
        door_weights={
            1: 0.05,  # 5% Hazard
            2: 0.15,  # 15% E-Waste
            3: 0.30,  # 30% Paper
            4: 0.35,  # 35% Recyclable
            5: 0.15   # 15% Organic
        }
    ),
    2: Scenario(
        id=2,
        name="HIGH-TRAFFIC RUSH",
        description="Peak morning intake stream at elevated conveyor speed.",
        speed_multiplier=1.5,
        spawn_interval=1.3,
        door_weights={
            1: 0.08,
            2: 0.20,
            3: 0.30,
            4: 0.25,
            5: 0.17
        }
    ),
    3: Scenario(
        id=3,
        name="HAZARD DRILL",
        description="Emergency drill testing Door 1 Lockout & Intercept Protocol.",
        speed_multiplier=0.9,
        spawn_interval=2.0,
        door_weights={
            1: 0.65,  # 65% Hazard items
            2: 0.10,
            3: 0.10,
            4: 0.10,
            5: 0.05
        }
    ),
    4: Scenario(
        id=4,
        name="E-WASTE SURGE",
        description="Commercial electronics drop-off campaign load.",
        speed_multiplier=1.1,
        spawn_interval=1.8,
        door_weights={
            1: 0.05,
            2: 0.65,  # 65% E-Waste items
            3: 0.10,
            4: 0.10,
            5: 0.10
        }
    ),
    5: Scenario(
        id=5,
        name="STRESS TEST",
        description="Maximum throughput benchmark with rapid multi-door routing.",
        speed_multiplier=2.0,
        spawn_interval=0.8,
        door_weights={
            1: 0.20,
            2: 0.20,
            3: 0.20,
            4: 0.20,
            5: 0.20
        }
    )
}

# Catalog of simulated waste item prototypes for each door
ITEM_CATALOG: Dict[int, List[dict]] = {
    1: [
        {"name": "knife", "label": "Utility Knife", "width": 64, "height": 26, "color": (40, 40, 200)},
        {"name": "scissors", "label": "Industrial Shears", "width": 58, "height": 34, "color": (50, 50, 190)},
        {"name": "bottle", "label": "Chemical Glass", "width": 46, "height": 62, "color": (30, 30, 220)},
        {"name": "syringe", "label": "Medical Syringe", "width": 60, "height": 20, "color": (20, 20, 240)},
    ],
    2: [
        {"name": "cell phone", "label": "Smart Phone", "width": 38, "height": 68, "color": (210, 210, 40)},
        {"name": "laptop", "label": "Laptop Chassis", "width": 80, "height": 55, "color": (190, 190, 50)},
        {"name": "keyboard", "label": "PCB Keyboard", "width": 84, "height": 38, "color": (180, 200, 30)},
        {"name": "mouse", "label": "Optical Mouse", "width": 34, "height": 48, "color": (200, 220, 40)},
        {"name": "remote", "label": "Remote Unit", "width": 30, "height": 65, "color": (190, 180, 50)},
    ],
    3: [
        {"name": "book", "label": "Bound Paper", "width": 55, "height": 68, "color": (20, 180, 230)},
        {"name": "newspaper", "label": "Newsprint Pack", "width": 65, "height": 52, "color": (30, 190, 220)},
        {"name": "cardboard", "label": "Corrugated Box", "width": 72, "height": 58, "color": (40, 170, 210)},
    ],
    4: [
        {"name": "cup", "label": "PET Drink Cup", "width": 40, "height": 52, "color": (40, 210, 60)},
        {"name": "can", "label": "Aluminum Can", "width": 36, "height": 58, "color": (50, 190, 70)},
        {"name": "fork", "label": "Metal Utensils", "width": 60, "height": 22, "color": (60, 200, 80)},
        {"name": "bowl", "label": "Plastic Container", "width": 58, "height": 44, "color": (40, 220, 90)},
    ],
    5: [
        {"name": "banana", "label": "Organic Banana", "width": 58, "height": 30, "color": (30, 140, 240)},
        {"name": "apple", "label": "Food Scraps", "width": 44, "height": 44, "color": (40, 120, 230)},
        {"name": "sandwich", "label": "Bakery Waste", "width": 52, "height": 46, "color": (50, 130, 220)},
    ]
}


class ScenarioEngine:
    """Controls the active campaign, timing, and procedural item generation."""

    def __init__(self, initial_scenario_id: int = 1):
        self.active_scenario = SCENARIOS.get(initial_scenario_id, SCENARIOS[1])
        self.spawn_timer: float = 0.0

    def set_scenario(self, scenario_id: int) -> bool:
        """Switch to a new scenario by ID (1..5)."""
        if scenario_id in SCENARIOS:
            self.active_scenario = SCENARIOS[scenario_id]
            self.spawn_timer = 0.0
            print(f"[Scenario] Switched to {self.active_scenario.name}: {self.active_scenario.description}")
            return True
        return False

    def should_spawn_item(self, dt: float) -> bool:
        """Check if interval has elapsed to spawn a new item on the belt."""
        self.spawn_timer += dt
        if self.spawn_timer >= self.active_scenario.spawn_interval:
            self.spawn_timer = 0.0
            return True
        return False

    def sample_random_item(self) -> dict:
        """Sample an item based on the active scenario's door probability weights."""
        doors = list(self.active_scenario.door_weights.keys())
        weights = list(self.active_scenario.door_weights.values())
        chosen_door = random.choices(doors, weights=weights, k=1)[0]

        catalog = ITEM_CATALOG.get(chosen_door, ITEM_CATALOG[4])
        base_item = random.choice(catalog).copy()
        base_item["door_id"] = chosen_door
        return base_item
