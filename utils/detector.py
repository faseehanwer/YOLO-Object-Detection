"""
utils/detector.py
=================
YOLOv8 Object Detector – Core Inference Engine

Wraps the Ultralytics YOLOv8 API with a clean, reusable interface.
Handles model loading, image inference, and structured result parsing.

Key design decisions:
- Models are loaded once and reused (Streamlit caches the ObjectDetector instance)
- Results are returned as plain Python dicts for easy serialization
- Visualization is delegated to utils/visualization.py (separation of concerns)
- All errors are caught and logged; empty results are returned on failure
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from utils.visualization import draw_detections

# ── Module-level logger ────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    YOLOv8-based real-time object detector.

    Wraps Ultralytics YOLO with a clean interface for:
    - Single-image detection  → detect_image()
    - Model metadata queries  → get_model_info()

    Attributes:
        model_name  (str):          YOLOv8 variant identifier (e.g. 'yolov8n').
        model       (YOLO):         Loaded Ultralytics YOLO model instance.
        class_names (dict[int,str]):COCO class index → human-readable name.
    """

    def __init__(self, model_name: str = "yolov8n") -> None:
        """
        Initialise the detector and load the specified YOLOv8 variant.

        Args:
            model_name: One of yolov8n / yolov8s / yolov8m / yolov8l / yolov8x.
                        'n' (nano) is fastest; 'x' (extra-large) is most accurate.

        Raises:
            RuntimeError: If the model cannot be loaded or downloaded.
        """
        self.model_name = model_name
        self.model = self._load_model(model_name)
        # COCO class dict: {0: 'person', 1: 'bicycle', ..., 79: 'toothbrush'}
        self.class_names: Dict[int, str] = self.model.names
        logger.info(
            "ObjectDetector ready | model=%s | classes=%d",
            model_name,
            len(self.class_names),
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_model(self, model_name: str) -> YOLO:
        """
        Load a YOLOv8 model, using a local cache when available.

        Ultralytics auto-downloads the .pt weights on first use and caches
        them in ~/.config/Ultralytics/ (Linux) or equivalent on other OSes.
        We additionally try to cache the file inside the project's models/
        directory for offline / air-gapped deployments.

        Args:
            model_name: Model variant name without extension (e.g. 'yolov8n').

        Returns:
            Loaded YOLO instance ready for inference.
        """
        local_path = Path("models") / f"{model_name}.pt"

        try:
            if local_path.exists():
                logger.info("Loading model from local cache: %s", local_path)
                model = YOLO(str(local_path))
            else:
                logger.info(
                    "Downloading model '%s.pt' from Ultralytics hub …", model_name
                )
                model = YOLO(f"{model_name}.pt")
                # Attempt to save locally for future runs (fails gracefully on
                # read-only filesystems such as some HF Spaces configurations)
                try:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    model.save(str(local_path))
                    logger.info("Model cached at %s", local_path)
                except (OSError, PermissionError) as cache_err:
                    logger.warning(
                        "Could not write model cache (%s) – will re-download next time.",
                        cache_err,
                    )
            return model

        except Exception as exc:
            logger.error("Model loading failed for '%s': %s", model_name, exc)
            raise RuntimeError(
                f"Failed to load YOLOv8 model '{model_name}'. "
                "Check your internet connection or verify the model name."
            ) from exc

    def _parse_results(self, results) -> List[Dict[str, Any]]:
        """
        Convert raw Ultralytics prediction output into structured Python dicts.

        Each detection becomes a dict with keys:
            class_id   (int)   : COCO class index
            class_name (str)   : Human-readable class label
            confidence (float) : Detection confidence [0, 1]
            bbox       (list)  : [x1, y1, x2, y2] pixel coordinates

        Args:
            results: Raw list returned by YOLO.predict().

        Returns:
            List of detection dicts, sorted by confidence descending.
        """
        detections: List[Dict[str, Any]] = []

        for result in results:
            boxes = result.boxes

            # Guard against frames with zero detections
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # xyxy bounding box → integer pixel coords
                x1, y1, x2, y2 = (
                    box.xyxy[0].cpu().numpy().astype(int).tolist()
                )

                conf: float = float(box.conf[0].cpu().numpy())
                cls_id: int = int(box.cls[0].cpu().numpy())
                cls_name: str = self.class_names.get(cls_id, f"class_{cls_id}")

                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        # Sort highest-confidence detections first for consistent display order
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect_image(
        self,
        image: np.ndarray,
        confidence: float = 0.50,
        iou: float = 0.45,
        show_labels: bool = True,
        show_confidence: bool = True,
        show_bbox: bool = True,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Run YOLOv8 inference on a single BGR image array.

        Args:
            image:           Input image as a BGR numpy array (OpenCV format).
            confidence:      Minimum score to keep a detection (0–1).
            iou:             Non-Maximum Suppression IoU threshold (0–1).
            show_labels:     Draw class-name labels on the output image.
            show_confidence: Append confidence % to each label.
            show_bbox:       Draw bounding boxes on the output image.

        Returns:
            Tuple of:
            ├── detections     : List of detection dicts (may be empty).
            └── annotated_image: BGR numpy array with annotations drawn.
        """
        try:
            # ── Run YOLOv8 inference ──────────────────────────────────────────
            results = self.model.predict(
                source=image,
                conf=confidence,
                iou=iou,
                verbose=False,          # Suppress per-frame console spam
                stream=False,
            )

            # ── Parse raw results into structured dicts ───────────────────────
            detections = self._parse_results(results)

            # ── Annotate a copy of the input image ────────────────────────────
            annotated_image = draw_detections(
                image.copy(),
                detections,
                self.class_names,
                show_labels=show_labels,
                show_confidence=show_confidence,
                show_bbox=show_bbox,
            )

            logger.debug("detect_image → %d objects found", len(detections))
            return detections, annotated_image

        except Exception as exc:
            logger.error("Inference error: %s", exc, exc_info=True)
            # Return empty results instead of crashing the Streamlit app
            return [], image.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Return a metadata summary about the loaded model.

        Returns:
            Dict with model_name, num_classes, class_names, framework, dataset.
        """
        return {
            "model_name": self.model_name,
            "num_classes": len(self.class_names),
            "class_names": list(self.class_names.values()),
            "framework": "Ultralytics YOLOv8",
            "dataset": "COCO (80 classes)",
            "input_size": "640 × 640 (auto-scaled)",
        }
