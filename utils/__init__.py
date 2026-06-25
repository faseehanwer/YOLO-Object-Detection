"""
utils/__init__.py
=================
Utility package for the YOLOv8 Real-Time Object Detection System.

Exposes the core classes and functions for clean top-level imports
across the Streamlit application.

Modules:
    detector     - YOLOv8 model loading and inference engine
    visualization - Bounding box drawing and annotation
    analytics     - Statistics computation and Plotly chart generation
"""

from utils.detector import ObjectDetector
from utils.visualization import draw_detections, get_detection_summary
from utils.analytics import (
    compute_statistics,
    plot_class_distribution,
    plot_confidence_distribution,
    plot_detection_timeline,
)

__all__ = [
    "ObjectDetector",
    "draw_detections",
    "get_detection_summary",
    "compute_statistics",
    "plot_class_distribution",
    "plot_confidence_distribution",
    "plot_detection_timeline",
]
