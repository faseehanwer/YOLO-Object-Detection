"""
utils/visualization.py
======================
Detection Visualization Engine

Handles all visual annotation applied to images/frames after inference:
  - Colour-coded bounding boxes (unique colour per COCO class)
  - Rounded-corner box drawing for a modern look
  - Class-name + confidence badge labels
  - Corner-accent tick marks (military/targeting aesthetic)
  - Semi-transparent fill for detected regions
  - Watermark count text overlay
  - Detection summary DataFrame builder

Design decisions:
  - Colours are generated in HSV space and evenly spaced across the hue wheel
    so every class gets a visually distinct, high-saturation colour.
  - Text badges compute background luminance to automatically choose black or
    white foreground text for maximum readability.
  - The drawing API works exclusively on BGR numpy arrays (OpenCV convention)
    so there is no format conversion overhead in the hot inference loop.
"""

import colorsys
import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Colour generation ──────────────────────────────────────────────────────────

def _generate_class_colors(num_classes: int = 80) -> List[Tuple[int, int, int]]:
    """
    Pre-compute one BGR colour per COCO class using evenly spaced HSV hues.

    Args:
        num_classes: Number of distinct colours to generate.

    Returns:
        List of (B, G, R) tuples with length == num_classes.
    """
    colors: List[Tuple[int, int, int]] = []
    for i in range(num_classes):
        hue = i / num_classes          # Evenly spaced around the colour wheel
        r, g, b = colorsys.hsv_to_rgb(hue, 0.88, 0.95)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR
    return colors


# Module-level pre-generated palette — created once, reused for every frame
CLASS_COLORS: List[Tuple[int, int, int]] = _generate_class_colors(80)


def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """Return the BGR colour assigned to COCO class `class_id`."""
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


# ── Low-level drawing primitives ───────────────────────────────────────────────

def _draw_rounded_rectangle(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    radius: int = 8,
) -> None:
    """
    Draw a rectangle with rounded corners directly on `image` (in-place).

    Rounded corners are achieved by drawing four straight edges (excluding
    corner areas) and four quarter-circle arcs with cv2.ellipse().

    Args:
        image:     BGR numpy array — modified in place.
        pt1:       (x, y) top-left corner.
        pt2:       (x, y) bottom-right corner.
        color:     BGR colour tuple.
        thickness: Line width in pixels.
        radius:    Corner arc radius in pixels.
    """
    x1, y1 = pt1
    x2, y2 = pt2

    # Clamp radius so it never exceeds a quarter of the shorter side
    r = min(radius, (x2 - x1) // 4, (y2 - y1) // 4, 1)

    # ── Four straight edges (stopping short of corners) ──
    cv2.line(image, (x1 + r, y1), (x2 - r, y1), color, thickness)  # top
    cv2.line(image, (x1 + r, y2), (x2 - r, y2), color, thickness)  # bottom
    cv2.line(image, (x1, y1 + r), (x1, y2 - r), color, thickness)  # left
    cv2.line(image, (x2, y1 + r), (x2, y2 - r), color, thickness)  # right

    # ── Four quarter-circle corner arcs ──
    # cv2.ellipse args: center, axes, angle, startAngle, endAngle, color, thickness
    cv2.ellipse(image, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)  # TL
    cv2.ellipse(image, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)  # TR
    cv2.ellipse(image, (x1 + r, y2 - r), (r, r), 90,  0, 90, color, thickness)  # BL
    cv2.ellipse(image, (x2 - r, y2 - r), (r, r), 0,   0, 90, color, thickness)  # BR


def _draw_corner_accents(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    length: int = 14,
    thickness: int = 3,
) -> None:
    """
    Draw military-style corner tick marks at the four corners of a bounding box.

    These short L-shaped accents reinforce the corners of the rounded rectangle
    and give the UI a technical / HUD aesthetic.

    Args:
        image:     BGR image — modified in place.
        x1, y1:   Top-left pixel coordinates.
        x2, y2:   Bottom-right pixel coordinates.
        color:     BGR colour.
        length:    Length of each tick arm in pixels.
        thickness: Line width.
    """
    # Clamp tick length to 20 % of the shorter box dimension
    ln = min(length, (x2 - x1) // 5, (y2 - y1) // 5)
    if ln < 3:
        return

    # Top-left
    cv2.line(image, (x1, y1), (x1 + ln, y1), color, thickness)
    cv2.line(image, (x1, y1), (x1, y1 + ln), color, thickness)
    # Top-right
    cv2.line(image, (x2, y1), (x2 - ln, y1), color, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + ln), color, thickness)
    # Bottom-left
    cv2.line(image, (x1, y2), (x1 + ln, y2), color, thickness)
    cv2.line(image, (x1, y2), (x1, y2 - ln), color, thickness)
    # Bottom-right
    cv2.line(image, (x2, y2), (x2 - ln, y2), color, thickness)
    cv2.line(image, (x2, y2), (x2, y2 - ln), color, thickness)


def _draw_label_badge(
    image: np.ndarray,
    text: str,
    anchor: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float = 0.52,
    thickness: int = 1,
    padding: int = 5,
) -> None:
    """
    Draw a filled pill-shaped label badge above a bounding box.

    The badge background uses the class colour; foreground text colour is
    chosen automatically (black or white) for maximum contrast.

    Args:
        image:      BGR image — modified in place.
        text:       Label string (class name + confidence).
        anchor:     (x, y) top-left corner of the bounding box.
        color:      BGR background colour for the badge.
        font_scale: OpenCV font scale.
        thickness:  Text stroke weight.
        padding:    Inner whitespace around the text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    bx, by = anchor
    img_h, img_w = image.shape[:2]

    # Clamp badge so it stays within image bounds
    bx = max(0, min(bx, img_w - tw - padding * 2 - 1))
    by_top = max(0, by - th - baseline - padding * 2)

    # Background rectangle
    cv2.rectangle(
        image,
        (bx, by_top),
        (bx + tw + padding * 2, by_top + th + baseline + padding * 2),
        color,
        cv2.FILLED,
    )

    # Auto-contrast: luminance check → white or black text
    b_c, g_c, r_c = color
    lum = 0.299 * r_c + 0.587 * g_c + 0.114 * b_c
    text_color = (0, 0, 0) if lum > 140 else (255, 255, 255)

    cv2.putText(
        image,
        text,
        (bx + padding, by_top + th + padding),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


# ── Public drawing API ─────────────────────────────────────────────────────────

def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    class_names: Dict[int, str],
    show_labels: bool = True,
    show_confidence: bool = True,
    show_bbox: bool = True,
) -> np.ndarray:
    """
    Annotate an image with all detection results (bounding boxes + labels).

    For each detection the function draws:
      1. A semi-transparent region fill (alpha-blended).
      2. A rounded-corner bounding box outline.
      3. Corner accent tick marks.
      4. A filled label badge with class name and optional confidence.
    Finally a small watermark is stamped in the bottom-left corner.

    Args:
        image:           BGR numpy array — modified in place (pass image.copy()
                         if you need to preserve the original).
        detections:      List of detection dicts from ObjectDetector.
        class_names:     COCO class index → name mapping.
        show_labels:     Draw class-name label badges.
        show_confidence: Append confidence percentage to label text.
        show_bbox:       Draw bounding boxes and corner marks.

    Returns:
        Annotated BGR numpy array.
    """
    if not detections:
        return image

    # Overlay for alpha-blended fill effects
    overlay = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id: int = det["class_id"]
        cls_name: str = det["class_name"]
        conf: float = det["confidence"]

        color = get_class_color(cls_id)

        if show_bbox:
            # ── Semi-transparent interior fill (8 % opacity) ──
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)
            cv2.addWeighted(overlay, 0.08, image, 0.92, 0, image)
            overlay = image.copy()   # Reset overlay for the next detection

            # ── Rounded bounding box ──
            _draw_rounded_rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

            # ── Corner accent marks ──
            _draw_corner_accents(image, x1, y1, x2, y2, color, length=14, thickness=3)

        if show_labels:
            label = cls_name.capitalize()
            if show_confidence:
                label = f"{label}  {conf:.0%}"
            _draw_label_badge(image, label, (x1, y1), color)

    # ── Watermark ────────────────────────────────────────────────────────────
    wm_text = f"YOLOv8  |  {len(detections)} object{'s' if len(detections) != 1 else ''} detected"
    cv2.putText(
        image,
        wm_text,
        (10, image.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    return image


# ── DataFrame builder ─────────────────────────────────────────────────────────

def get_detection_summary(detections: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Aggregate detection results into a tidy pandas DataFrame for table display.

    Columns:
        Object Class   | Count | Confidence (mean) | Max Confidence

    Args:
        detections: List of detection dicts.

    Returns:
        DataFrame sorted by Count descending (most frequent class first).
        Returns an empty DataFrame with correct columns when detections is empty.
    """
    _EMPTY = pd.DataFrame(
        columns=["Object Class", "Count", "Confidence", "Max Confidence"]
    )

    if not detections:
        return _EMPTY

    # Group confidences by class name
    class_data: Dict[str, Dict] = {}
    for det in detections:
        name = det["class_name"].capitalize()
        if name not in class_data:
            class_data[name] = {"count": 0, "confidences": []}
        class_data[name]["count"] += 1
        class_data[name]["confidences"].append(det["confidence"])

    rows = []
    for cls, data in sorted(class_data.items(), key=lambda x: -x[1]["count"]):
        confs = data["confidences"]
        rows.append(
            {
                "Object Class": cls,
                "Count": data["count"],
                "Confidence": round(float(np.mean(confs)), 3),
                "Max Confidence": round(float(max(confs)), 3),
            }
        )

    return pd.DataFrame(rows)
