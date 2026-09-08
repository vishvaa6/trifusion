"""
AI Waste Segregation Prototype - AI Detection Engine
Wraps Ultralytics YOLOv8 with FP16 CUDA acceleration for RTX 3050,
5-Door category mapping, and Hazard Intercept tagging.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from config import (
    CONFIDENCE_THRESHOLD,
    DOORS,
    MODEL_NAME,
    USE_HALF_PRECISION,
)


@dataclass
class DetectedItem:
    """Structured representation of a single detected waste item."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    door_id: int
    category: str
    color_bgr: Tuple[int, int, int]
    is_hazard: bool
    center_pos: Tuple[int, int]
    has_frame: bool = True  # Whether to display the rectangular frame and HUD labels


@dataclass
class DetectionResult:
    """Result of inference on a single frame."""
    items: List[DetectedItem]
    has_hazard: bool
    inference_time_ms: float
    gpu_memory_used_mb: float = 0.0


# -----------------------------------------------------------------------------
# Object Collections Taxonomy for Frame Visibility Toggles
# -----------------------------------------------------------------------------
OBJECT_COLLECTIONS: Dict[str, dict] = {
    "stationery": {
        "id": "stationery",
        "name": "Stationery & Desk",
        "icon": "fa-pen-ruler",
        "color": "amber",
        "description": "Notebooks, books, scissors, bags & desk tools",
        "classes": ["book", "scissors", "backpack", "clock", "cell phone", "laptop", "mouse", "keyboard", "toothbrush"]
    },
    "hazard": {
        "id": "hazard",
        "name": "Hazardous & Sharps",
        "icon": "fa-triangle-exclamation",
        "color": "red",
        "description": "Blades, shears, glassware & hazardous items",
        "classes": ["knife", "scissors", "bottle"]
    },
    "ewaste": {
        "id": "ewaste",
        "name": "Electronics & E-Waste",
        "icon": "fa-microchip",
        "color": "cyan",
        "description": "Computers, phones, remotes & peripherals",
        "classes": ["laptop", "cell phone", "tv", "mouse", "keyboard", "remote", "microwave", "toaster", "clock"]
    },
    "paper": {
        "id": "paper",
        "name": "Paper & Cardboard",
        "icon": "fa-book-open",
        "color": "yellow",
        "description": "Books, boxes, fiber & paper materials",
        "classes": ["book"]
    },
    "recyclable": {
        "id": "recyclable",
        "name": "Plastics & Tableware",
        "icon": "fa-bottle-water",
        "color": "emerald",
        "description": "Bottles, cups, bowls, forks & spoons",
        "classes": ["cup", "fork", "spoon", "bowl", "wine glass", "bottle", "vase"]
    },
    "organic": {
        "id": "organic",
        "name": "Food & Organics",
        "icon": "fa-apple-whole",
        "color": "orange",
        "description": "Fruits, vegetables, bread & food scraps",
        "classes": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]
    },
    "furniture": {
        "id": "furniture",
        "name": "Furniture & Home",
        "icon": "fa-couch",
        "color": "slate",
        "description": "Chairs, sofas, beds, tables & fixtures",
        "classes": ["chair", "couch", "potted plant", "bed", "dining table", "toilet", "sink", "refrigerator", "oven"]
    },
    "people": {
        "id": "people",
        "name": "People",
        "icon": "fa-user",
        "color": "blue",
        "description": "Ambient human presence",
        "classes": ["person"]
    },
    "accessories": {
        "id": "accessories",
        "name": "Accessories & Bags",
        "icon": "fa-briefcase",
        "color": "purple",
        "description": "Backpacks, handbags, ties & luggage",
        "classes": ["backpack", "umbrella", "handbag", "tie", "suitcase", "hair drier"]
    },
    "vehicles": {
        "id": "vehicles",
        "name": "Vehicles & Transit",
        "icon": "fa-car",
        "color": "teal",
        "description": "Cars, bicycles, motorcycles, trucks & signals",
        "classes": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter"]
    },
    "animals": {
        "id": "animals",
        "name": "Animals & Pets",
        "icon": "fa-paw",
        "color": "amber",
        "description": "Domestic pets, wildlife & toys",
        "classes": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "teddy bear"]
    },
    "sports": {
        "id": "sports",
        "name": "Sports & Leisure",
        "icon": "fa-futbol",
        "color": "indigo",
        "description": "Balls, bats, skateboards & sporting gear",
        "classes": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bench"]
    }
}

# Stationery and Office Desk items recognized in COCO
STATIONERY_CLASSES: Set[str] = {
    "book", "scissors", "backpack", "clock", "cell phone", "laptop", "mouse", "keyboard", "toothbrush"
}


class WasteDetector:
    """YOLOv8 Edge Inference Engine with GPU optimization and waste stream routing."""

    def __init__(self, model_path: str = MODEL_NAME, conf_threshold: float = CONFIDENCE_THRESHOLD):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = "cpu"
        self.use_half = False
        self.model = None
        self.is_ready = False
        self.enabled_classes: Optional[Set[str]] = None  # None means all 80 classes enabled

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Load YOLOv8 model, select CUDA/CPU device, and perform warmup."""
        try:
            import torch
            from ultralytics import YOLO

            # Check CUDA capability
            if torch.cuda.is_available():
                self.device = "cuda:0"
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[Detector] GPU Detected: {gpu_name} (CUDA Active)")
                self.use_half = USE_HALF_PRECISION
            else:
                self.device = "cpu"
                print("[Detector] CUDA not available. Running on CPU.")
                self.use_half = False

            print(f"[Detector] Loading model '{self.model_path}' on {self.device}...")
            self.model = YOLO(self.model_path)

            # Warmup pass to pre-compile CUDA kernels and allocate VRAM
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            warmup_kwargs = {"conf": self.conf_threshold, "device": self.device, "verbose": False}
            if self.use_half:
                warmup_kwargs["half"] = True
            _ = self.model(dummy_img, **warmup_kwargs)
            print(f"[Detector] CUDA Warmup complete. FP16={self.use_half}. System Ready.")
            self.is_ready = True

        except Exception as e:
            print(f"[Detector] Warning during initialization: {e}")
            print("[Detector] Will operate in simulation heuristic mode if YOLO is unavailable.")
            self.is_ready = False

    def get_vram_usage_mb(self) -> float:
        """Retrieve current PyTorch allocated GPU memory in MB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated(0) / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def get_category_info(self, class_name: str) -> Tuple[Optional[int], str, Tuple[int, int, int], bool]:
        """
        Return (door_id, category_name, color_bgr, is_hazard).
        If non-waste/ambient (like person, chair, dog), returns (None, 'NON-WASTE', (140, 145, 155), False).
        """
        name_lower = class_name.lower().strip()

        # Check Door 1 (Hazard) first
        for target in DOORS[1].target_classes:
            if target in name_lower:
                return 1, DOORS[1].category, DOORS[1].color_bgr, True

        # Check Door 2 (E-Waste)
        for target in DOORS[2].target_classes:
            if target in name_lower:
                return 2, DOORS[2].category, DOORS[2].color_bgr, False

        # Check Door 3 (Paper)
        for target in DOORS[3].target_classes:
            if target in name_lower:
                return 3, DOORS[3].category, DOORS[3].color_bgr, False

        # Check Door 5 (Organic)
        for target in DOORS[5].target_classes:
            if target in name_lower:
                return 5, DOORS[5].category, DOORS[5].color_bgr, False

        # Check Door 4 (Recyclable)
        for target in DOORS[4].target_classes:
            if target in name_lower:
                return 4, DOORS[4].category, DOORS[4].color_bgr, False

        return None, "NON-WASTE", (140, 145, 155), False

    def classify_to_door(self, class_name: str) -> int:
        """Map object class name to assigned Door ID (1 to 5). Fallback to 4 for backwards compatibility."""
        door_id, _, _, _ = self.get_category_info(class_name)
        return door_id if door_id is not None else 4

    def set_enabled_classes(self, classes: Optional[List[str]]) -> None:
        """Update allowed classes. Pass None to enable all classes."""
        if classes is None:
            self.enabled_classes = None
        else:
            self.enabled_classes = {c.lower().strip() for c in classes}

    def get_class_catalog(self) -> List[dict]:
        """
        Returns full list of 80 classes with metadata:
        id, name, door_id, category, is_hazard, is_waste, is_enabled
        """
        if not self.is_ready or self.model is None:
            return []
        catalog = []
        for cls_id, name in self.model.names.items():
            name_lower = name.lower().strip()
            door_id, cat, color, is_haz = self.get_category_info(name_lower)
            is_enabled = True if self.enabled_classes is None else (name_lower in self.enabled_classes)
            catalog.append({
                "id": int(cls_id),
                "name": name,
                "door_id": door_id,
                "category": cat,
                "is_hazard": is_haz,
                "is_waste": (door_id is not None),
                "is_stationery": (name_lower in STATIONERY_CLASSES),
                "is_enabled": is_enabled,
                "has_frame": is_enabled,
                "collections": [cid for cid, cdata in OBJECT_COLLECTIONS.items() if name_lower in cdata["classes"]]
            })
        return catalog

    def is_frame_enabled(self, class_name: str) -> bool:
        """Check if an object class is selected to display a rectangular frame."""
        name_lower = class_name.lower().strip()
        if self.enabled_classes is None:
            return True
        return name_lower in self.enabled_classes

    def toggle_collection_frames(self, collection_id: str, enable: bool) -> List[dict]:
        """Enable or disable rectangular frames for all items in a given collection."""
        col = OBJECT_COLLECTIONS.get(collection_id)
        if not col:
            return self.get_collections_status()

        # If currently all enabled (None), instantiate full 80-set first
        if self.enabled_classes is None:
            if self.is_ready and self.model:
                self.enabled_classes = {name.lower().strip() for name in self.model.names.values()}
            else:
                self.enabled_classes = set()

        col_classes = {c.lower().strip() for c in col["classes"]}
        if enable:
            self.enabled_classes.update(col_classes)
        else:
            self.enabled_classes.difference_update(col_classes)

        return self.get_collections_status()

    def get_collections_status(self) -> List[dict]:
        """Returns live status of all collections with framed item counts."""
        catalog = self.get_class_catalog()
        catalog_map = {c["name"].lower().strip(): c["is_enabled"] for c in catalog}

        status_list = []
        for col_id, col in OBJECT_COLLECTIONS.items():
            classes = col["classes"]
            total = len(classes)
            framed_count = sum(1 for c in classes if catalog_map.get(c.lower().strip(), False))
            status_list.append({
                "id": col_id,
                "name": col["name"],
                "icon": col["icon"],
                "color": col["color"],
                "description": col["description"],
                "classes": classes,
                "total_count": total,
                "framed_count": framed_count,
                "is_enabled": (framed_count == total and total > 0),
                "is_partial": (0 < framed_count < total)
            })
        return status_list


    def detect(self, frame: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> DetectionResult:
        """
        Run inference on frame, filter confidence, assign doors, and format items.
        offset_x, offset_y are added to bounding boxes when scanning a sub-region.
        """
        t0 = time.time()
        items: List[DetectedItem] = []
        has_hazard = False

        if not self.is_ready or self.model is None:
            # Fallback if model failed to load
            return DetectionResult(items=[], has_hazard=False, inference_time_ms=0.0)

        try:
            predict_kwargs = {
                "conf": self.conf_threshold,
                "device": self.device,
                "verbose": False
            }
            if self.use_half:
                predict_kwargs["half"] = True

            results = self.model(frame, **predict_kwargs)

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < self.conf_threshold:
                        continue

                    cls_id = int(box.cls[0])
                    class_name = self.model.names.get(cls_id, f"class_{cls_id}")
                    name_lower = class_name.lower().strip()

                    # Determine whether this object is selected to have a rectangular frame
                    has_frame = self.is_frame_enabled(name_lower)

                    # Coordinate bounding box
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1 = int(xyxy[0]) + offset_x
                    y1 = int(xyxy[1]) + offset_y
                    x2 = int(xyxy[2]) + offset_x
                    y2 = int(xyxy[3]) + offset_y

                    # Determine Door & Category
                    door_id, category, color_bgr, is_hazard = self.get_category_info(name_lower)
                    if is_hazard and has_frame:
                        has_hazard = True

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    items.append(DetectedItem(
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        door_id=door_id if door_id is not None else 0,
                        category=category,
                        color_bgr=color_bgr,
                        is_hazard=is_hazard,
                        center_pos=(center_x, center_y),
                        has_frame=has_frame
                    ))

        except Exception as e:
            print(f"[Detector] Inference error: {e}")

        inference_time_ms = (time.time() - t0) * 1000.0
        vram_used = self.get_vram_usage_mb()

        return DetectionResult(
            items=items,
            has_hazard=has_hazard,
            inference_time_ms=inference_time_ms,
            gpu_memory_used_mb=vram_used
        )
