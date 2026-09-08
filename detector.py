"""
AI Waste Segregation Prototype - AI Detection Engine
Wraps Ultralytics YOLOv8 with FP16 CUDA acceleration for RTX 3050,
5-Door category mapping, and Hazard Intercept tagging.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from config import (
    AUTO_ROLLBACK_ON_ERROR,
    CONFIDENCE_THRESHOLD,
    DOORS,
    FALLBACK_DEVICE,
    MODEL_NAME,
    PREFER_GPU,
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
    """YOLO11 Edge Inference Engine with GPU acceleration and safe zero-drop CPU rollback."""

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
        
        # Hardware execution and fail-safe state
        self.device = "cpu"
        self.active_device = "cpu"
        self.device_mode = "CPU"
        self.gpu_name = "None"
        self.gpu_available = False
        self.use_half = False
        self.rollback_triggered = False
        self.rollback_reason: Optional[str] = None
        self.rollback_count = 0
        
        self.model = None
        self.is_ready = False
        self.enabled_classes: Optional[Set[str]] = None  # None means all 80 classes enabled

        self._initialize_model()

    def rollback_to_cpu(self, reason: str) -> None:
        """
        Safely and dynamically roll back inference from GPU to CPU.
        Frees VRAM, sets fallback flags, and migrates or reloads the model on CPU.
        """
        self.device = FALLBACK_DEVICE
        self.active_device = FALLBACK_DEVICE
        self.device_mode = "CPU"
        self.use_half = False
        self.rollback_triggered = True
        self.rollback_reason = str(reason)
        self.rollback_count += 1

        print(f"[Detector] ⚠️ SAFE ROLLBACK ENGAGED: Diverting compute to CPU. Reason: {reason}")

        # Safely release allocated PyTorch VRAM
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Attempt to shift model in-memory or reload on CPU
        try:
            if self.model is not None:
                if hasattr(self.model, "to"):
                    self.model.to("cpu")
        except Exception as e:
            print(f"[Detector] Model transfer to CPU warning: {e}. Reinitializing on CPU...")
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
            except Exception as re_err:
                print(f"[Detector] CPU reload error: {re_err}")

    def retry_gpu(self) -> Tuple[bool, str]:
        """
        Operator action to attempt re-engaging the GPU after a rollback or resource recovery.
        """
        try:
            import torch
            from ultralytics import YOLO

            if not torch.cuda.is_available():
                msg = "CUDA is not available in current PyTorch build (CPU-only distribution)."
                print(f"[Detector] {msg}")
                return False, msg

            gpu_name = torch.cuda.get_device_name(0)
            print(f"[Detector] Attempting GPU re-engagement on {gpu_name}...")

            test_device = "cuda:0"
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Reload fresh YOLO instance on CUDA
            new_model = YOLO(self.model_path)
            warmup_kwargs = {"conf": self.conf_threshold, "device": test_device, "verbose": False}
            if USE_HALF_PRECISION:
                warmup_kwargs["half"] = True

            _ = new_model(dummy_img, **warmup_kwargs)

            # Successfully validated on GPU
            self.model = new_model
            self.device = test_device
            self.active_device = test_device
            self.device_mode = "GPU"
            self.gpu_name = gpu_name
            self.gpu_available = True
            self.use_half = USE_HALF_PRECISION
            self.rollback_triggered = False
            self.rollback_reason = None
            self.is_ready = True

            msg = f"GPU {gpu_name} successfully re-engaged with FP16={self.use_half}."
            print(f"[Detector] {msg}")
            return True, msg

        except Exception as e:
            err_msg = f"GPU re-engagement failed: {e}"
            self.rollback_to_cpu(err_msg)
            return False, err_msg

    def get_device_telemetry(self) -> Dict[str, Any]:
        """Returns real-time compute hardware and safe rollback diagnostics."""
        return {
            "device": self.device,
            "active_device": self.active_device,
            "device_mode": self.device_mode,
            "gpu_name": self.gpu_name,
            "gpu_available": self.gpu_available,
            "use_half": self.use_half,
            "rollback_triggered": self.rollback_triggered,
            "rollback_reason": self.rollback_reason,
            "rollback_count": self.rollback_count,
            "vram_mb": round(self.get_vram_usage_mb(), 1)
        }

    def _initialize_model(self) -> None:
        """Load YOLO11 model with automatic GPU selection and fallback to CPU."""
        try:
            import torch
            from ultralytics import YOLO

            # 1. Determine compute device
            cuda_avail = torch.cuda.is_available()
            if PREFER_GPU and cuda_avail:
                self.device = "cuda:0"
                self.active_device = "cuda:0"
                self.device_mode = "GPU"
                self.gpu_available = True
                self.gpu_name = torch.cuda.get_device_name(0)
                self.use_half = USE_HALF_PRECISION
                print(f"[Detector] GPU Hardware Detected: {self.gpu_name} (CUDA Active)")
            else:
                self.device = "cpu"
                self.active_device = "cpu"
                self.device_mode = "CPU"
                self.gpu_available = False
                self.use_half = False
                if not cuda_avail:
                    self.rollback_reason = "PyTorch CUDA build not active in environment (native CPU)"
                    print("[Detector] CUDA not available in PyTorch build. Running on CPU.")
                else:
                    self.rollback_reason = "GPU disabled by configuration (PREFER_GPU=False)"

            # 2. Attempt model loading
            print(f"[Detector] Loading model '{self.model_path}' on {self.device}...")
            self.model = YOLO(self.model_path)

            # 3. Perform hardware warmup
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            warmup_kwargs = {"conf": self.conf_threshold, "device": self.device, "verbose": False}
            if self.use_half:
                warmup_kwargs["half"] = True

            try:
                _ = self.model(dummy_img, **warmup_kwargs)
                print(f"[Detector] {self.device.upper()} Warmup complete. FP16={self.use_half}. System Ready.")
                self.is_ready = True
            except Exception as warmup_err:
                if self.device != "cpu":
                    print(f"[Detector] Warmup failed on GPU: {warmup_err}. Triggering safe rollback to CPU...")
                    self.rollback_to_cpu(f"Warmup failure: {warmup_err}")
                    # Re-warmup on CPU
                    _ = self.model(dummy_img, conf=self.conf_threshold, device="cpu", verbose=False)
                    print("[Detector] CPU Fallback Warmup complete. System Ready.")
                    self.is_ready = True
                else:
                    raise warmup_err

        except Exception as e:
            print(f"[Detector] Warning during initialization: {e}")
            self.rollback_to_cpu(f"Initialization failure: {e}")
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

            # 3. Dynamic Safe Inference (GPU with automatic zero-drop CPU retry)
            results = None
            if self.device != "cpu":
                try:
                    predict_kwargs = {
                        "conf": inference_conf,
                        "device": self.device,
                        "verbose": False
                    }
                    if self.use_half:
                        predict_kwargs["half"] = True
                    results = self.model(enhanced_frame, **predict_kwargs)
                except Exception as gpu_runtime_err:
                    # Catch CUDA OOM, driver crashes, or kernel errors
                    print(f"[Detector] ⚠️ GPU Runtime Exception: {gpu_runtime_err}. Engaging safe zero-drop rollback to CPU...")
                    self.rollback_to_cpu(f"GPU Runtime Exception: {gpu_runtime_err}")
                    # Zero-drop retry on CPU for this exact frame
                    results = self.model(enhanced_frame, conf=inference_conf, device="cpu", verbose=False)
            else:
                results = self.model(enhanced_frame, conf=inference_conf, device="cpu", verbose=False)

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


# =============================================================================
# Dual-Stream AI Comparison Engine
# Runs YOLO11n (Nano) + YOLO11s (Small) on the same frame simultaneously,
# producing side-by-side annotated frames and conflict detection alerts.
# =============================================================================

class DualConflict:
    """Represents a routing conflict between the two models on the same object region."""
    def __init__(self, class_name: str, display_name: str,
                 door_a: int, door_b: int,
                 conf_a: float, conf_b: float,
                 bbox: Tuple[int, int, int, int]):
        self.class_name = class_name
        self.display_name = display_name
        self.door_a = door_a    # YOLO11n decision
        self.door_b = door_b    # YOLO11s decision
        self.conf_a = conf_a
        self.conf_b = conf_b
        self.bbox = bbox

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "display_name": self.display_name,
            "door_a": self.door_a,
            "door_b": self.door_b,
            "conf_a": round(self.conf_a, 2),
            "conf_b": round(self.conf_b, 2),
        }


class DualStreamDetector:
    """
    Dual-model AI comparison engine.

    Model A: YOLO11n (Nano)  — fast, lightweight  — cyan boxes on left half
    Model B: YOLO11s (Small) — accurate, heavier   — violet boxes on right half

    Safe rollback per model:
    - If Model B fails to load (download error, memory), dual mode is disabled cleanly.
    - If Model B crashes at runtime 3x in a row, it is marked OFFLINE; Model A continues.
    - Disabling dual mode immediately frees Model B from memory.
    """

    # Colors for left (Nano) and right (Small) streams
    COLOR_A = (0, 240, 200)     # Cyan-teal for YOLO11n
    COLOR_B = (200, 50, 255)    # Violet for YOLO11s

    MODEL_A_NAME = "yolo11n.pt"
    MODEL_B_NAME = "yolo11s.pt"

    def __init__(self, detector_a: "WasteDetector"):
        """
        Args:
            detector_a: The already-initialized primary WasteDetector (YOLO11n).
                        DualStreamDetector borrows it; it does NOT own it.
        """
        self.detector_a = detector_a
        self.detector_b: Optional["WasteDetector"] = None

        self.enabled = False
        self.model_b_online = False
        self.model_b_status = "NOT LOADED"   # Human-readable status string
        self.model_b_error_count = 0
        self.MAX_B_ERRORS = 3

        # Runtime stats
        self.total_conflicts = 0
        self.conflict_log: "deque[DualConflict]" = deque(maxlen=30)
        self.last_ms_a = 0.0
        self.last_ms_b = 0.0
        self.last_det_a = 0
        self.last_det_b = 0
        self.last_conf_a = 0.0
        self.last_conf_b = 0.0
        self.agreement_pct = 100.0

    # ------------------------------------------------------------------
    # Enable / Disable
    # ------------------------------------------------------------------

    def enable(self) -> Tuple[bool, str]:
        """
        Lazy-load YOLO11s (Model B) and enable dual-stream mode.
        Returns (success, message).
        """
        if self.enabled and self.model_b_online:
            return True, "Dual-stream already active."

        print("[DualStream] Loading Model B (YOLO11s)...")
        try:
            self.detector_b = WasteDetector(
                model_path=self.MODEL_B_NAME,
                conf_threshold=self.detector_a.conf_threshold,
                transparent_conf_threshold=self.detector_a.transparent_conf_threshold,
            )
            if not self.detector_b.is_ready:
                raise RuntimeError("YOLO11s failed to initialize (is_ready=False)")

            self.model_b_online = True
            self.model_b_status = "ONLINE"
            self.model_b_error_count = 0
            self.enabled = True
            print("[DualStream] Model B (YOLO11s) loaded. Dual-stream ACTIVE.")
            return True, "Dual-stream enabled. YOLO11s loaded."

        except Exception as e:
            self.model_b_online = False
            self.model_b_status = f"LOAD ERROR: {e}"
            self.enabled = False
            self.detector_b = None
            print(f"[DualStream] Model B load failed: {e}. Single-stream continues.")
            return False, f"Model B (yolo11s.pt) failed to load: {e}"

    def disable(self) -> None:
        """Disable dual-stream and free Model B memory."""
        self.enabled = False
        self.model_b_online = False
        self.model_b_status = "UNLOADED"
        self.detector_b = None
        print("[DualStream] Dual-stream disabled. Model B freed from memory.")

    # ------------------------------------------------------------------
    # Dual Detection
    # ------------------------------------------------------------------

    def dual_detect(self, frame: np.ndarray) -> Tuple[
        "DetectionResult", Optional["DetectionResult"], List[DualConflict]
    ]:
        """
        Run both models on the same frame and return (result_a, result_b, conflicts).
        If dual is disabled or Model B is offline, result_b = None, conflicts = [].
        """
        result_a = self.detector_a.detect(frame)
        self.last_ms_a = result_a.inference_time_ms
        self.last_det_a = len(result_a.items)
        if result_a.items:
            self.last_conf_a = sum(i.confidence for i in result_a.items) / len(result_a.items)

        if not self.enabled or not self.model_b_online or self.detector_b is None:
            return result_a, None, []

        # Run Model B with fault tolerance
        result_b = None
        try:
            result_b = self.detector_b.detect(frame)
            self.last_ms_b = result_b.inference_time_ms
            self.last_det_b = len(result_b.items)
            if result_b.items:
                self.last_conf_b = sum(i.confidence for i in result_b.items) / len(result_b.items)
            self.model_b_error_count = 0  # reset on success
        except Exception as e:
            self.model_b_error_count += 1
            print(f"[DualStream] Model B runtime error #{self.model_b_error_count}: {e}")
            if self.model_b_error_count >= self.MAX_B_ERRORS:
                self.model_b_online = False
                self.model_b_status = f"OFFLINE ({self.MAX_B_ERRORS} consecutive errors)"
                print("[DualStream] Model B marked OFFLINE after repeated failures.")
            return result_a, None, []

        # Conflict detection: match items by overlapping class name
        conflicts = self._find_conflicts(result_a, result_b)
        for c in conflicts:
            self.conflict_log.appendleft(c)
        self.total_conflicts += len(conflicts)

        # Agreement %: fraction of classes detected by both that agree on door
        self._update_agreement(result_a, result_b)

        return result_a, result_b, conflicts

    def _find_conflicts(
        self,
        result_a: "DetectionResult",
        result_b: "DetectionResult"
    ) -> List[DualConflict]:
        """Match items by class_name and flag when door routing differs."""
        conflicts: List[DualConflict] = []
        map_b: Dict[str, "DetectedItem"] = {}
        for item in result_b.items:
            map_b[item.class_name.lower()] = item

        for item_a in result_a.items:
            key = item_a.class_name.lower()
            item_b = map_b.get(key)
            if item_b is None:
                continue
            if item_a.door_id != item_b.door_id:
                conflicts.append(DualConflict(
                    class_name=item_a.class_name,
                    display_name=item_a.display_name or item_a.class_name.title(),
                    door_a=item_a.door_id,
                    door_b=item_b.door_id,
                    conf_a=item_a.confidence,
                    conf_b=item_b.confidence,
                    bbox=item_a.bbox,
                ))
        return conflicts

    def _update_agreement(
        self,
        result_a: "DetectionResult",
        result_b: "DetectionResult"
    ) -> None:
        """Compute running agreement % between both models."""
        names_a = {i.class_name.lower(): i.door_id for i in result_a.items}
        names_b = {i.class_name.lower(): i.door_id for i in result_b.items}
        shared = set(names_a) & set(names_b)
        if not shared:
            return
        agreed = sum(1 for k in shared if names_a[k] == names_b[k])
        self.agreement_pct = (agreed / len(shared)) * 100.0

    # ------------------------------------------------------------------
    # Split Frame Rendering
    # ------------------------------------------------------------------

    def build_split_frame(
        self,
        frame: np.ndarray,
        result_a: "DetectionResult",
        result_b: Optional["DetectionResult"],
        conflicts: List[DualConflict],
        show_boxes: bool = True,
    ) -> np.ndarray:
        """
        Render a side-by-side comparison frame:
        - Left half: YOLO11n Nano detections (cyan boxes)
        - Right half: YOLO11s Small detections (violet boxes)
        - Center divider with model labels
        - Conflict items: red flash border on both sides
        """
        h, w = frame.shape[:2]
        mid = w // 2

        # Build conflict class set for quick lookup
        conflict_classes = {c.class_name.lower() for c in conflicts}

        # --- LEFT HALF: Nano ---
        out = frame.copy()

        if show_boxes:
            for item in result_a.items:
                if not item.has_frame:
                    continue
                x1, y1, x2, y2 = item.bbox
                # Clip to left half
                x2_clip = min(x2, mid - 2)
                if x1 >= mid:
                    continue

                is_conflict = item.class_name.lower() in conflict_classes
                box_color = (0, 60, 200) if is_conflict else self.COLOR_A
                thickness = 3 if is_conflict else 2

                cv2.rectangle(out, (x1, y1), (x2_clip, y2), box_color, thickness)
                _draw_label(out, item, self.COLOR_A, "A", is_conflict)

        # --- RIGHT HALF: Small ---
        if result_b is not None and show_boxes:
            for item in result_b.items:
                if not item.has_frame:
                    continue
                x1, y1, x2, y2 = item.bbox
                # Clip to right half
                x1_clip = max(x1, mid + 2)
                if x2 <= mid:
                    continue

                is_conflict = item.class_name.lower() in conflict_classes
                box_color = (180, 20, 240) if is_conflict else self.COLOR_B
                thickness = 3 if is_conflict else 2

                cv2.rectangle(out, (x1_clip, y1), (x2, y2), box_color, thickness)
                _draw_label_b(out, item, self.COLOR_B, "B", is_conflict, mid)

        # --- Center Divider ---
        cv2.line(out, (mid, 0), (mid, h), (200, 200, 200), 2)

        # --- Model Labels (top-center each half) ---
        # Left label: Nano
        _draw_model_label(out, "  NANO  ", 8, 6, self.COLOR_A,
                          f"{self.last_ms_a:.0f}ms", "A")
        # Right label: Small
        if result_b is not None:
            _draw_model_label(out, "  SMALL  ", mid + 8, 6, self.COLOR_B,
                              f"{self.last_ms_b:.0f}ms", "B")
        else:
            _draw_model_label(out, "  SMALL  ", mid + 8, 6, (80, 80, 80),
                              "OFFLINE", "B")

        # --- Conflict count at bottom ---
        if conflicts:
            msg = f"  CONFLICT: {len(conflicts)} item(s) routed to different doors  "
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cx = (w - tw) // 2
            cv2.rectangle(out, (cx - 4, h - th - 18), (cx + tw + 4, h - 4), (20, 0, 60), -1)
            cv2.rectangle(out, (cx - 4, h - th - 18), (cx + tw + 4, h - 4), (160, 20, 240), 1)
            cv2.putText(out, msg, (cx, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 140, 255), 1)

        return out

    # ------------------------------------------------------------------
    # Status / Telemetry
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model_a": {
                "name": self.MODEL_A_NAME,
                "status": "ONLINE",
                "device": self.detector_a.device_mode,
                "inf_ms": round(self.last_ms_a, 1),
                "detections": self.last_det_a,
                "avg_conf_pct": round(self.last_conf_a * 100, 1),
            },
            "model_b": {
                "name": self.MODEL_B_NAME,
                "status": self.model_b_status,
                "device": (self.detector_b.device_mode if self.detector_b else "N/A"),
                "inf_ms": round(self.last_ms_b, 1),
                "detections": self.last_det_b,
                "avg_conf_pct": round(self.last_conf_b * 100, 1),
            },
            "total_conflicts": self.total_conflicts,
            "agreement_pct": round(self.agreement_pct, 1),
            "conflict_log": [c.to_dict() for c in list(self.conflict_log)[:10]],
        }


# ------------------------------------------------------------------
# Helper drawing functions for DualStreamDetector
# ------------------------------------------------------------------

def _draw_label(frame: np.ndarray, item: "DetectedItem",
                color: Tuple[int, int, int], side: str, is_conflict: bool) -> None:
    """Draw label banner on left-side (Model A) detection."""
    x1, y1 = item.bbox[0], item.bbox[1]
    conf_pct = int(item.confidence * 100)
    dname = (item.display_name or item.class_name).upper()
    label = f"[{side}] {dname} {conf_pct}%"
    if is_conflict:
        label = f"⚠ {label}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1),
                  (10, 10, 16), -1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, 1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)


def _draw_label_b(frame: np.ndarray, item: "DetectedItem",
                  color: Tuple[int, int, int], side: str,
                  is_conflict: bool, mid: int) -> None:
    """Draw label banner on right-side (Model B) detection."""
    x1, y1 = max(item.bbox[0], mid + 2), item.bbox[1]
    conf_pct = int(item.confidence * 100)
    dname = (item.display_name or item.class_name).upper()
    label = f"[{side}] {dname} {conf_pct}%"
    if is_conflict:
        label = f"⚠ {label}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1),
                  (10, 10, 16), -1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, 1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)


def _draw_model_label(frame: np.ndarray, text: str, x: int, y: int,
                      color: Tuple[int, int, int], speed: str, side: str) -> None:
    """Draw model identity pill at top-left or top-right of each half."""
    full = f"{text} {speed}"
    (tw, th), _ = cv2.getTextSize(full, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.rectangle(frame, (x, y), (x + tw + 10, y + th + 10),
                  (15, 15, 20), -1)
    cv2.rectangle(frame, (x, y), (x + tw + 10, y + th + 10), color, 1)
    cv2.putText(frame, full, (x + 5, y + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
