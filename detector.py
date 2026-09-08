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
    TRANSPARENT_CLASSES,
    TRANSPARENT_CONFIDENCE_THRESHOLD,
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
    display_name: str = ""  # Refined proper label name (e.g. Pen / Marker, Wrist Watch)


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
# -----------------------------------------------------------------------------
# Fine-Grained Object Disambiguation & Proper Label Dictionary
# Refines coarse COCO classes into precise everyday accessory and stationery names
# -----------------------------------------------------------------------------
REFINED_OBJECT_INFO: Dict[str, dict] = {
    "toothbrush": {
        "proper_name": "Pen / Marker / Pencil",
        "category": "STATIONERY",
        "door_id": 3,
        "aliases": ["pen", "pencil", "marker", "stylus", "highlighter", "ballpoint", "toothbrush"],
        "description": "Pens, pencils, markers and slender writing tools",
        "icon": "fa-pen"
    },
    "clock": {
        "proper_name": "Wrist Watch / Smartwatch",
        "category": "ACCESSORIES",
        "door_id": 2,
        "aliases": ["watch", "wrist watch", "smartwatch", "fitness tracker", "clock", "timer"],
        "description": "Wrist watches, smartwatches, timers and desk clocks",
        "icon": "fa-clock"
    },
    "tie": {
        "proper_name": "Eyeglasses / Shades / Lanyard",
        "category": "ACCESSORIES",
        "door_id": None,
        "aliases": ["glasses", "eyeglasses", "sunglasses", "spectacles", "shades", "clear glasses", "reading glasses", "lanyard", "tie", "necktie"],
        "description": "Clear eyeglasses, reading glasses, sunglasses, lanyards and neckties",
        "icon": "fa-glasses"
    },
    "handbag": {
        "proper_name": "Wallet / Purse / Pouch",
        "category": "ACCESSORIES",
        "door_id": None,
        "aliases": ["wallet", "purse", "pouch", "card holder", "clutch", "handbag", "coin purse"],
        "description": "Wallets, coin purses, card holders and small pouches",
        "icon": "fa-wallet"
    },
    "backpack": {
        "proper_name": "Backpack / School Bag",
        "category": "ACCESSORIES",
        "door_id": 3,
        "aliases": ["backpack", "bag", "school bag", "rucksack", "knapsack", "daypack"],
        "description": "School bags, daypacks, rucksacks and carry bags",
        "icon": "fa-backpack"
    },
    "book": {
        "proper_name": "Notebook / Diary / Book",
        "category": "PAPER & STATIONERY",
        "door_id": 3,
        "aliases": ["notebook", "notepad", "diary", "journal", "book", "textbook", "novel", "pad"],
        "description": "Notebooks, journals, spiral pads, textbooks and documents",
        "icon": "fa-book"
    },
    "scissors": {
        "proper_name": "Scissors / Craft Shears",
        "category": "HAZARD",
        "door_id": 1,
        "aliases": ["scissors", "shears", "clippers", "paper cutter", "craft scissors"],
        "description": "Cutting shears, craft scissors and paper cutters",
        "icon": "fa-scissors"
    },
    "knife": {
        "proper_name": "Utility Knife / Box Cutter",
        "category": "HAZARD",
        "door_id": 1,
        "aliases": ["knife", "box cutter", "utility knife", "blade", "cutter", "pocket knife", "scalpel"],
        "description": "Box cutters, craft blades, utility knives and sharps",
        "icon": "fa-utensils"
    },
    "bottle": {
        "proper_name": "Clear Bottle / Flask / Can",
        "category": "HAZARD",
        "door_id": 1,
        "aliases": ["bottle", "water bottle", "plastic bottle", "pet bottle", "glass bottle", "transparent bottle", "clear bottle", "thermos", "flask", "beverage bottle", "tumbler", "soda can", "jar"],
        "description": "Clear plastic water bottles, glass bottles, metal flasks and drink containers",
        "icon": "fa-bottle-water"
    },
    "cup": {
        "proper_name": "Glass Tumbler / Cup / Mug",
        "category": "RECYCLABLE",
        "door_id": 4,
        "aliases": ["cup", "glass", "tumbler", "clear cup", "plastic cup", "glass cup", "coffee mug", "tea cup", "drinkware", "mug"],
        "description": "Glass tumblers, transparent plastic cups, coffee mugs, tea cups and drinkware",
        "icon": "fa-mug-hot"
    },
    "cell phone": {
        "proper_name": "Smartphone / Mobile",
        "category": "E-WASTE",
        "door_id": 2,
        "aliases": ["cell phone", "phone", "smartphone", "mobile", "iphone", "android", "device"],
        "description": "Smartphones, cellular phones and handheld mobile devices",
        "icon": "fa-mobile-screen"
    },
    "laptop": {
        "proper_name": "Laptop / Notebook PC",
        "category": "E-WASTE",
        "door_id": 2,
        "aliases": ["laptop", "computer", "notebook pc", "macbook", "pc", "chromebook"],
        "description": "Laptops, notebook computers and portable PCs",
        "icon": "fa-laptop"
    },
    "mouse": {
        "proper_name": "Mouse / Charger / Earbuds",
        "category": "E-WASTE",
        "door_id": 2,
        "aliases": ["mouse", "computer mouse", "charger", "earbuds", "airpods", "adapter", "usb"],
        "description": "Computer mice, wireless earbuds cases, charging bricks and adapters",
        "icon": "fa-mouse"
    },
    "keyboard": {
        "proper_name": "Keyboard / Keypad",
        "category": "E-WASTE",
        "door_id": 2,
        "aliases": ["keyboard", "keypad", "mechanical keyboard", "typing keyboard"],
        "description": "Computer keyboards, number pads and input boards",
        "icon": "fa-keyboard"
    },
    "remote": {
        "proper_name": "Stapler / Remote / Calculator",
        "category": "E-WASTE",
        "door_id": 2,
        "aliases": ["remote", "stapler", "calculator", "controller", "tv remote"],
        "description": "Remote controls, desktop staplers and calculators",
        "icon": "fa-calculator"
    },
    "umbrella": {
        "proper_name": "Umbrella / Rain Gear",
        "category": "ACCESSORIES",
        "door_id": None,
        "aliases": ["umbrella", "parasol", "rain gear"],
        "description": "Folding umbrellas, rain umbrellas and parasols",
        "icon": "fa-umbrella"
    },
    "suitcase": {
        "proper_name": "Suitcase / Luggage",
        "category": "ACCESSORIES",
        "door_id": None,
        "aliases": ["suitcase", "luggage", "trolley", "briefcase", "travel bag"],
        "description": "Suitcases, trolley bags, briefcases and travel luggage",
        "icon": "fa-suitcase-rolling"
    },
    "hair drier": {
        "proper_name": "Hair Drier / Tool",
        "category": "ACCESSORIES",
        "door_id": 2,
        "aliases": ["hair drier", "hairdryer", "blower", "styling tool"],
        "description": "Electric hair driers, blowers and styling appliances",
        "icon": "fa-wind"
    },
    "wine glass": {
        "proper_name": "Wine Glass / Glassware",
        "category": "RECYCLABLE",
        "door_id": 4,
        "aliases": ["wine glass", "goblet", "stemware", "glass cup", "glassware", "champagne glass", "clear glass"],
        "description": "Wine glasses, goblets, champagne flutes and clear glassware",
        "icon": "fa-wine-glass"
    },
    "bowl": {
        "proper_name": "Glass / Tableware Bowl",
        "category": "RECYCLABLE",
        "door_id": 4,
        "aliases": ["bowl", "glass bowl", "salad bowl", "soup bowl", "dish", "clear bowl", "tableware"],
        "description": "Glass bowls, clear salad bowls, food dishes and reusable tableware",
        "icon": "fa-bowl-food"
    },
    "vase": {
        "proper_name": "Glass Vase / Vessel",
        "category": "RECYCLABLE",
        "door_id": 4,
        "aliases": ["vase", "glass vase", "flower vase", "glass vessel", "carafe", "decanter", "clear vase"],
        "description": "Glass vases, decorative carafes, vessels and clear glassware",
        "icon": "fa-flask"
    }
}

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
    "transparency": {
        "id": "transparency",
        "name": "Glass & Transparent Items",
        "icon": "fa-glasses",
        "color": "cyan",
        "description": "Clear plastic bottles, glassware, tumblers, cups & vases",
        "classes": ["bottle", "wine glass", "cup", "bowl", "vase", "tie"]
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
        "name": "Personal Accessories & Wearables",
        "icon": "fa-glasses",
        "color": "purple",
        "description": "Watches, glasses, wallets, jewelry, bags & accessories",
        "classes": ["backpack", "umbrella", "handbag", "tie", "suitcase", "hair drier", "clock", "toothbrush", "mouse"]
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


def enhance_transparent_clarity(img: np.ndarray) -> np.ndarray:
    """
    Enhance specular reflections and refractive boundaries of transparent objects
    (clear PET bottles, glass tumblers, glassware) using LAB-space CLAHE.
    Blends 75% CLAHE luminance with 25% original luminance to sharpen faint glass
    rims and fluid menisci without color cast distortion.
    """
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        l_blend = cv2.addWeighted(cl, 0.75, l, 0.25, 0)
        enhanced_lab = cv2.merge((l_blend, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return img


class WasteDetector:
    """YOLOv8 Edge Inference Engine with GPU optimization and waste stream routing."""

    def __init__(
        self,
        model_path: str = MODEL_NAME,
        conf_threshold: float = CONFIDENCE_THRESHOLD,
        transparent_conf_threshold: float = TRANSPARENT_CONFIDENCE_THRESHOLD,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.transparent_conf_threshold = transparent_conf_threshold
        self.transparent_classes = set(TRANSPARENT_CLASSES)
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

    def refine_label(self, class_name: str, bbox: Tuple[int, int, int, int], frame_w: int = 640, frame_h: int = 480) -> Tuple[str, str, Optional[int]]:
        """
        Dynamically refines a generic COCO class name into a proper descriptive label name
        using bounding box aspect ratio, normalized area, and accessory heuristics.
        Returns: (proper_display_name, category, door_id)
        """
        raw_lower = class_name.lower().strip()
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        aspect_ratio = max(w, h) / max(1, min(w, h))
        area_ratio = (w * h) / (frame_w * frame_h)

        # 1. Toothbrush -> Pen / Pencil / Marker
        if raw_lower == "toothbrush":
            if aspect_ratio >= 2.5:
                return "Pen / Marker", "STATIONERY", 3
            return "Pen / Stylus", "STATIONERY", 3

        # 2. Clock -> Wrist Watch / Smartwatch vs Wall/Desk Clock
        if raw_lower == "clock":
            if area_ratio < 0.08 or max(w, h) < 180:
                return "Wrist Watch", "ACCESSORIES", None
            return "Desk / Wall Clock", "ACCESSORIES", None

        # 3. Handbag -> Wallet / Card Case vs Handbag / Tote
        if raw_lower == "handbag":
            if area_ratio < 0.09 or max(w, h) < 190:
                return "Wallet / Pouch", "ACCESSORIES", None
            return "Handbag / Purse", "ACCESSORIES", None

        # 4. Tie -> Eyeglasses / Spectacles vs Necktie / Lanyard
        if raw_lower == "tie":
            if w > h or aspect_ratio < 2.4:
                return "Eyeglasses / Shades", "ACCESSORIES", None
            return "Necktie / Lanyard", "ACCESSORIES", None

        # 5. Book -> Notebook / Diary vs Hardcover Book
        if raw_lower == "book":
            if aspect_ratio > 1.3:
                return "Notebook / Diary", "PAPER & STATIONERY", 3
            return "Book / Document", "PAPER & STATIONERY", 3

        # 6. Remote -> Stapler / Remote / Calculator
        if raw_lower == "remote":
            if aspect_ratio > 2.0 and area_ratio < 0.08:
                return "Stapler / Remote", "STATIONERY", 3
            return "Remote / Controller", "E-WASTE", 2

        # 7. Mouse -> Mouse / Charger / Earbuds Case
        if raw_lower == "mouse":
            if area_ratio < 0.04:
                return "Earbuds / Charger", "ACCESSORIES", 2
            return "Computer Mouse", "E-WASTE", 2

        # 8. Cell Phone -> Smartphone
        if raw_lower == "cell phone":
            return "Smartphone", "E-WASTE", 2

        # 9. Bottle -> Clear Water Bottle / Glass Bottle / Flask
        if raw_lower == "bottle":
            if aspect_ratio > 1.8:
                return "Water Bottle / Flask", "HAZARD", 1
            return "Clear Bottle / Glass Jar", "HAZARD", 1

        # 10. Cup -> Glass Tumbler / Cup / Mug
        if raw_lower == "cup":
            return "Glass / Cup / Tumbler", "RECYCLABLE", 4

        # 10b. Wine Glass -> Wine Glass / Glassware
        if raw_lower == "wine glass":
            return "Wine Glass / Glassware", "RECYCLABLE", 4

        # 10c. Vase -> Glass Vase / Vessel
        if raw_lower == "vase":
            return "Glass Vase / Vessel", "RECYCLABLE", 4

        # 10d. Bowl -> Glass / Tableware Bowl
        if raw_lower == "bowl":
            return "Glass / Tableware Bowl", "RECYCLABLE", 4

        # 11. Knife -> Utility Knife / Box Cutter
        if raw_lower == "knife":
            return "Box Cutter / Knife", "HAZARD", 1

        # 12. Scissors -> Scissors / Shears
        if raw_lower == "scissors":
            return "Scissors / Shears", "STATIONERY", 3

        # 13. Backpack -> Backpack / School Bag
        if raw_lower == "backpack":
            return "Backpack / Daypack", "ACCESSORIES", 3

        # 14. Laptop -> Laptop PC
        if raw_lower == "laptop":
            return "Laptop Computer", "E-WASTE", 2

        # Check pre-defined refined dictionary
        if raw_lower in REFINED_OBJECT_INFO:
            info = REFINED_OBJECT_INFO[raw_lower]
            return info["proper_name"], info["category"], info["door_id"]

        # Default fallback
        door_id, cat, _, _ = self.get_category_info(raw_lower)
        return class_name.title(), cat, door_id

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
            # Check refined dictionary for proper naming & aliases
            refined = REFINED_OBJECT_INFO.get(name_lower, {})
            proper_name = refined.get("proper_name", name.title())
            aliases = refined.get("aliases", [name_lower, name.title()])
            icon = refined.get("icon", "fa-tag")
            display_cat = refined.get("category", cat)

            catalog.append({
                "id": int(cls_id),
                "name": name,
                "proper_name": proper_name,
                "aliases": aliases,
                "icon": icon,
                "door_id": door_id,
                "category": display_cat,
                "is_hazard": is_haz,
                "is_waste": (door_id is not None),
                "is_stationery": (name_lower in STATIONERY_CLASSES or "stationery" in [cid for cid, cdata in OBJECT_COLLECTIONS.items() if name_lower in cdata["classes"]]),
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
            # 1. Enhance specular rims and refraction boundaries for transparent glass & PET bottles
            enhanced_frame = enhance_transparent_clarity(frame)

            # 2. Lower prediction threshold so YOLO NMS does not prematurely drop transparent objects
            inference_conf = min(self.conf_threshold, self.transparent_conf_threshold)

            predict_kwargs = {
                "conf": inference_conf,
                "device": self.device,
                "verbose": False
            }
            if self.use_half:
                predict_kwargs["half"] = True

            results = self.model(enhanced_frame, **predict_kwargs)

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = self.model.names.get(cls_id, f"class_{cls_id}")
                    name_lower = class_name.lower().strip()

                    # Class-specific sensitivity filtering:
                    # Transparent objects use heightened sensitivity floor (0.20), standard items use threshold (0.40)
                    min_conf = self.transparent_conf_threshold if (name_lower in self.transparent_classes) else self.conf_threshold
                    if conf < min_conf:
                        continue

                    # Determine whether this object is selected to have a rectangular frame
                    has_frame = self.is_frame_enabled(name_lower)

                    # Coordinate bounding box
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1 = int(xyxy[0]) + offset_x
                    y1 = int(xyxy[1]) + offset_y
                    x2 = int(xyxy[2]) + offset_x
                    y2 = int(xyxy[3]) + offset_y

                    # Refine generic COCO label to proper descriptive accessory/item name
                    proper_name, ref_cat, ref_door = self.refine_label(name_lower, (x1, y1, x2, y2))

                    # Determine Door & Category
                    door_id, category, color_bgr, is_hazard = self.get_category_info(name_lower)
                    if ref_cat:
                        category = ref_cat
                    if ref_door is not None:
                        door_id = ref_door

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
                        has_frame=has_frame,
                        display_name=proper_name
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
