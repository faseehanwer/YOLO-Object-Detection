"""
app.py
======
YOLOv8 Real-Time Object Detection System
=========================================
A production-ready Streamlit dashboard for real-time object detection
powered by Ultralytics YOLOv8 and OpenCV.

Detection Modes:
    📷 Webcam Snapshot  – capture a frame from the browser camera
    🖼️  Image Upload     – detect objects in any uploaded image
    🎬 Video Upload     – process a video file frame by frame

Public deployment targets:
    • Hugging Face Spaces (primary)
    • Streamlit Community Cloud (secondary)

Author : Mehmet (AI/ML Engineer)
License: MIT
"""

# ── Standard library ────────────────────────────────────────────────────────
import logging
import os
import tempfile
import time
from pathlib import Path

# ── Third-party ─────────────────────────────────────────────────────────────
import cv2
import numpy as np
import streamlit as st

# ── Internal utilities ───────────────────────────────────────────────────────
from utils.analytics import (
    compute_statistics,
    plot_class_distribution,
    plot_confidence_distribution,
    plot_detection_timeline,
)
from utils.detector import ObjectDetector
from utils.visualization import draw_detections, get_detection_summary

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s › %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("yolo_app")

# ═══════════════════════════════════════════════════════════════════════════════
# DIRECTORY BOOTSTRAP  – create required dirs if they don't exist
# ═══════════════════════════════════════════════════════════════════════════════
for _d in ("models", "uploads", "outputs", "assets"):
    Path(_d).mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  – must be the very first Streamlit call
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/YOLO-Object-Detection",
        "Report a bug": "https://github.com/yourusername/YOLO-Object-Detection/issues",
        "About": (
            "## 🎯 YOLOv8 Object Detection System\n"
            "Real-time object detection powered by Ultralytics YOLOv8.\n\n"
            "Built as an AI/ML portfolio project."
        ),
    },
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
/* ── Google Font import ───────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Root variables ───────────────────────────────────────── */
:root {
  --indigo:   #818cf8;
  --violet:   #a78bfa;
  --cyan:     #22d3ee;
  --surface0: #0f172a;
  --surface1: #1e293b;
  --surface2: #334155;
  --text-muted: #94a3b8;
  --border: rgba(129,140,248,0.20);
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero header ──────────────────────────────────────────── */
.hero-title {
  font-size: clamp(1.8rem, 4vw, 2.6rem);
  font-weight: 700;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, var(--indigo) 0%, var(--violet) 50%, var(--cyan) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-align: center;
  line-height: 1.2;
  padding: 0.5rem 0 0.25rem;
}
.hero-sub {
  text-align: center;
  color: var(--text-muted);
  font-size: 0.92rem;
  margin-bottom: 1.6rem;
  letter-spacing: 0.01em;
}

/* ── KPI metric cards ─────────────────────────────────────── */
.kpi-grid { display: flex; gap: 12px; flex-wrap: wrap; margin: 1rem 0; }
.kpi-card {
  flex: 1 1 140px;
  background: linear-gradient(145deg, #1e293b, #0f172a);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.1rem 1.2rem;
  text-align: center;
  box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.kpi-value {
  font-size: 2rem;
  font-weight: 700;
  color: var(--indigo);
  line-height: 1;
}
.kpi-label {
  font-size: 0.78rem;
  color: var(--text-muted);
  margin-top: 0.4rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

/* ── Class badges ─────────────────────────────────────────── */
.badge-wrap { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.badge {
  display: inline-block;
  padding: 3px 12px;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 600;
  background: rgba(129,140,248,0.12);
  border: 1px solid rgba(129,140,248,0.35);
  color: var(--indigo);
  letter-spacing: 0.03em;
}

/* ── Section labels ───────────────────────────────────────── */
.section-label {
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.10em;
  margin: 1.4rem 0 0.5rem;
  border-left: 3px solid var(--indigo);
  padding-left: 8px;
}

/* ── Sidebar overrides ────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: #0f172a !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; }

/* ── Primary button ───────────────────────────────────────── */
div.stButton > button[kind="primary"],
div.stButton > button {
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
  padding: 0.55rem 1.4rem !important;
  transition: opacity 0.18s ease !important;
  width: 100%;
}
div.stButton > button:hover { opacity: 0.88 !important; }

/* ── Download button ──────────────────────────────────────── */
div.stDownloadButton > button {
  background: linear-gradient(135deg, #0ea5e9 0%, #22d3ee 100%) !important;
  color: #0f172a !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
  width: 100%;
}

/* ── Footer ───────────────────────────────────────────────── */
.footer {
  text-align: center;
  color: #475569;
  font-size: 0.78rem;
  padding: 2rem 0 1rem;
  border-top: 1px solid var(--border);
  margin-top: 2rem;
}
.footer a { color: var(--indigo); text-decoration: none; }

/* ── Hide default Streamlit footer & hamburger ────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_detector(model_size: str) -> ObjectDetector:
    """
    Load (or reuse a cached) ObjectDetector for the given model variant.

    `@st.cache_resource` ensures the model weights are loaded from disk only
    once per model_size per Streamlit server session, avoiding expensive
    repeated downloads/loads on every UI interaction.

    Args:
        model_size: YOLOv8 variant name (e.g. 'yolov8n', 'yolov8s', …).

    Returns:
        Initialised ObjectDetector ready for inference.
    """
    logger.info("Cache MISS → loading model: %s", model_size)
    return ObjectDetector(model_name=model_size)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
def render_sidebar() -> dict:
    """
    Render the left sidebar and return user configuration as a dict.

    Configuration keys:
        mode          (str)  : Selected detection mode label
        model_size    (str)  : YOLOv8 variant name
        confidence    (float): Min confidence threshold [0.1 – 1.0]
        iou           (float): NMS IoU threshold [0.1 – 1.0]
        show_labels   (bool) : Draw class-name labels
        show_conf     (bool) : Append confidence % to labels
        show_bbox     (bool) : Draw bounding boxes

    Returns:
        Configuration dict consumed by the main render functions.
    """
    with st.sidebar:

        # ── Branding ──────────────────────────────────────────────────────
        st.markdown(
            "<div style='text-align:center; padding:0.8rem 0 0.4rem;'>"
            "<span style='font-size:2.2rem;'>🎯</span><br>"
            "<span style='color:#818cf8; font-weight:700; font-size:1.1rem;'>"
            "YOLOv8 Detection</span><br>"
            "<span style='color:#64748b; font-size:0.74rem;'>AI Portfolio Project</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # ── Detection mode ────────────────────────────────────────────────
        st.markdown(
            "<p class='section-label'>📡 Detection Mode</p>", unsafe_allow_html=True
        )
        mode = st.radio(
            "mode",
            options=["📷  Webcam Snapshot", "🖼️  Image Upload", "🎬  Video Upload"],
            index=1,                     # Default: image upload (safest for all platforms)
            label_visibility="collapsed",
        )

        st.markdown("---")

        # ── Model selection ───────────────────────────────────────────────
        st.markdown(
            "<p class='section-label'>🤖 Model</p>", unsafe_allow_html=True
        )
        model_size = st.selectbox(
            "YOLOv8 Variant",
            options=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
            index=0,
            help=(
                "**n** = Nano (fastest, ~6 MB)\n\n"
                "**s** = Small\n\n"
                "**m** = Medium\n\n"
                "**l** = Large\n\n"
                "**x** = Extra-Large (most accurate, ~136 MB)"
            ),
        )

        # Size & speed tooltip
        _speed = {"n": "⚡ Fastest", "s": "🔵 Fast", "m": "🟡 Balanced",
                  "l": "🟠 Accurate", "x": "🔴 Most Accurate"}
        _suffix = model_size[-1]
        st.caption(f"{_speed.get(_suffix, '')}  |  COCO 80-class")

        st.markdown("---")

        # ── Inference parameters ──────────────────────────────────────────
        st.markdown(
            "<p class='section-label'>⚙️ Inference Parameters</p>",
            unsafe_allow_html=True,
        )
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=1.00,
            value=0.50,
            step=0.05,
            help="Detections below this score are discarded.",
        )
        iou = st.slider(
            "IoU Threshold (NMS)",
            min_value=0.10,
            max_value=1.00,
            value=0.45,
            step=0.05,
            help=(
                "Non-Maximum Suppression threshold.\n\n"
                "Lower → fewer overlapping boxes."
            ),
        )

        st.markdown("---")

        # ── Display toggles ───────────────────────────────────────────────
        st.markdown(
            "<p class='section-label'>🎨 Overlay Options</p>", unsafe_allow_html=True
        )
        col_a, col_b = st.columns(2)
        with col_a:
            show_bbox   = st.toggle("Boxes",      value=True)
            show_labels = st.toggle("Labels",     value=True)
        with col_b:
            show_conf   = st.toggle("Confidence", value=True)

        st.markdown("---")

        # ── Model info card ───────────────────────────────────────────────
        st.markdown(
            "<p class='section-label'>📋 Model Info</p>", unsafe_allow_html=True
        )
        st.info(
            f"**Model :** {model_size.upper()}\n\n"
            f"**Dataset :** COCO (80 classes)\n\n"
            f"**Input :** 640 × 640 px\n\n"
            f"**Framework :** Ultralytics"
        )

    return {
        "mode":        mode,
        "model_size":  model_size,
        "confidence":  confidence,
        "iou":         iou,
        "show_labels": show_labels,
        "show_conf":   show_conf,
        "show_bbox":   show_bbox,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS PANEL  – shared by all detection modes
# ═══════════════════════════════════════════════════════════════════════════════
def render_analytics(detections: list, inference_ms: float) -> None:
    """
    Render the full analytics section below the detection output.

    Sections:
        1. Four KPI cards  (objects, classes, avg confidence, inference time)
        2. Class distribution bar chart  +  Confidence histogram (side by side)
        3. Sortable detection summary table (pandas DataFrame)
        4. Detected class badges

    Args:
        detections:   List of detection dicts from ObjectDetector.
        inference_ms: Inference wall-clock time in milliseconds.
    """
    if not detections:
        st.warning(
            "⚠️  No objects detected above the current confidence threshold.  "
            "Try lowering the threshold in the sidebar."
        )
        return

    stats = compute_statistics(detections, inference_ms)

    # ── 1. KPI cards ──────────────────────────────────────────────────────
    st.markdown(
        "<p class='section-label'>📊 Detection Statistics</p>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)

    def _kpi(col, value: str, label: str) -> None:
        col.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-value'>{value}</div>"
            f"<div class='kpi-label'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _kpi(c1, str(stats["total_objects"]),          "Objects Found")
    _kpi(c2, str(stats["unique_classes"]),          "Unique Classes")
    _kpi(c3, f"{stats['avg_confidence']:.0%}",      "Avg Confidence")
    _kpi(c4, f"{stats['inference_time']:.0f} ms",   "Inference Time")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 2. Charts ──────────────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown(
            "<p class='section-label'>📈 Objects by Class</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            plot_class_distribution(detections), use_container_width=True
        )
    with ch2:
        st.markdown(
            "<p class='section-label'>📉 Confidence Distribution</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            plot_confidence_distribution(detections), use_container_width=True
        )

    # ── 3. Summary table ───────────────────────────────────────────────────
    st.markdown(
        "<p class='section-label'>📋 Detection Summary Table</p>",
        unsafe_allow_html=True,
    )
    df = get_detection_summary(detections)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Object Class": st.column_config.TextColumn(
                "Object Class", width="medium"
            ),
            "Count": st.column_config.NumberColumn("Count", format="%d 🔢"),
            "Confidence": st.column_config.ProgressColumn(
                "Avg Confidence",
                format="%.3f",
                min_value=0.0,
                max_value=1.0,
            ),
            "Max Confidence": st.column_config.ProgressColumn(
                "Max Confidence",
                format="%.3f",
                min_value=0.0,
                max_value=1.0,
            ),
        },
    )

    # ── 4. Class badges ────────────────────────────────────────────────────
    st.markdown(
        "<p class='section-label'>🏷️ Detected Classes</p>",
        unsafe_allow_html=True,
    )
    badges = "".join(
        f"<span class='badge'>✓ {cls.capitalize()}</span>"
        for cls in stats["class_names"]
    )
    st.markdown(
        f"<div class='badge-wrap'>{badges}</div>", unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: IMAGE UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
def run_image_mode(detector: ObjectDetector, cfg: dict) -> None:
    """
    Image Upload detection mode.

    Flow:
        1. User uploads an image file (JPG / PNG / BMP / WEBP).
        2. Original and annotated images are shown side by side.
        3. Analytics panel renders below.
        4. Download button for the annotated image is provided.

    Args:
        detector: Loaded ObjectDetector instance (cached).
        cfg:      User configuration dict from render_sidebar().
    """
    st.markdown(
        "<p class='section-label'>🖼️ Upload an image</p>", unsafe_allow_html=True
    )
    uploaded = st.file_uploader(
        "Drop an image here or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supported: JPG · JPEG · PNG · BMP · WEBP",
        label_visibility="collapsed",
    )

    if uploaded is None:
        # ── Placeholder state ──────────────────────────────────────────────
        st.markdown(
            "<div style='text-align:center; padding: 3rem 0; color:#475569;'>"
            "<div style='font-size:3rem;'>🖼️</div>"
            "<div style='font-size:1rem; margin-top:0.5rem;'>"
            "Upload an image to start detection</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Decode uploaded bytes → BGR numpy array ────────────────────────────
    raw_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("❌  Could not decode the uploaded file.  Please try another image.")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ── Side-by-side display ───────────────────────────────────────────────
    col_orig, col_result = st.columns(2, gap="medium")

    with col_orig:
        st.markdown(
            "<p class='section-label'>📥 Original</p>", unsafe_allow_html=True
        )
        st.image(image_rgb, use_container_width=True, caption=uploaded.name)

    # ── Run inference ──────────────────────────────────────────────────────
    with st.spinner("🔍  Running YOLOv8 inference …"):
        t0 = time.perf_counter()
        detections, annotated_bgr = detector.detect_image(
            image_bgr,
            confidence=cfg["confidence"],
            iou=cfg["iou"],
            show_labels=cfg["show_labels"],
            show_confidence=cfg["show_conf"],
            show_bbox=cfg["show_bbox"],
        )
        inference_ms = (time.perf_counter() - t0) * 1000

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    with col_result:
        st.markdown(
            "<p class='section-label'>📤 Detection Result</p>",
            unsafe_allow_html=True,
        )
        st.image(
            annotated_rgb,
            use_container_width=True,
            caption=f"{len(detections)} object(s) detected  ·  {inference_ms:.0f} ms",
        )

    # ── Download button ────────────────────────────────────────────────────
    out_path = Path("outputs") / f"detected_{uploaded.name}"
    cv2.imwrite(str(out_path), annotated_bgr)
    with open(out_path, "rb") as fh:
        st.download_button(
            label="💾  Download Annotated Image",
            data=fh,
            file_name=f"yolo_{uploaded.name}",
            mime="image/jpeg",
            use_container_width=True,
        )

    st.markdown("---")

    # ── Analytics panel ────────────────────────────────────────────────────
    render_analytics(detections, inference_ms)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: WEBCAM SNAPSHOT
# ═══════════════════════════════════════════════════════════════════════════════
def run_webcam_mode(detector: ObjectDetector, cfg: dict) -> None:
    """
    Webcam Snapshot detection mode.

    Uses Streamlit's `st.camera_input()` which accesses the browser's
    MediaDevices API — this works seamlessly on Hugging Face Spaces,
    Streamlit Cloud, and local machines without requiring OpenCV camera access.

    For true real-time streaming at full frame-rate, upgrade to
    `streamlit-webrtc` (see README for instructions).

    Args:
        detector: Loaded ObjectDetector instance.
        cfg:      User configuration dict.
    """
    st.markdown(
        "<p class='section-label'>📷 Webcam Capture</p>", unsafe_allow_html=True
    )
    st.info(
        "📸  Click **Take Photo** to capture a frame, then YOLOv8 will run "
        "inference on it automatically.  Capture again for continuous detection.",
        icon="ℹ️",
    )

    frame = st.camera_input(
        "Capture a frame", label_visibility="collapsed"
    )

    if frame is None:
        return

    # ── Decode captured JPEG → BGR numpy array ────────────────────────────
    raw_bytes = np.frombuffer(frame.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("❌  Could not process the captured frame.")
        return

    # ── Run inference ──────────────────────────────────────────────────────
    with st.spinner("🔍  Detecting objects …"):
        t0 = time.perf_counter()
        detections, annotated_bgr = detector.detect_image(
            image_bgr,
            confidence=cfg["confidence"],
            iou=cfg["iou"],
            show_labels=cfg["show_labels"],
            show_confidence=cfg["show_conf"],
            show_bbox=cfg["show_bbox"],
        )
        inference_ms = (time.perf_counter() - t0) * 1000

    # ── Result display ──────────────────────────────────────────────────────
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    st.markdown(
        "<p class='section-label'>📤 Detection Result</p>", unsafe_allow_html=True
    )
    st.image(
        annotated_rgb,
        use_container_width=True,
        caption=f"{len(detections)} object(s) found  ·  {inference_ms:.0f} ms",
    )

    # ── Download ───────────────────────────────────────────────────────────
    out_path = Path("outputs") / "webcam_detection.jpg"
    cv2.imwrite(str(out_path), annotated_bgr)
    with open(out_path, "rb") as fh:
        st.download_button(
            "💾  Download Result",
            data=fh,
            file_name="yolo_webcam_detection.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )

    st.markdown("---")
    render_analytics(detections, inference_ms)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: VIDEO UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
def run_video_mode(detector: ObjectDetector, cfg: dict) -> None:
    """
    Video Upload detection mode.

    Flow:
        1. User uploads a video file.
        2. App shows video metadata (frames, FPS, resolution).
        3. User picks a frame-sampling rate (every N frames) to control speed.
        4. Detection runs frame by frame with a live preview and progress bar.
        5. Annotated video is written to disk and offered as a download.
        6. Aggregate analytics rendered from all processed frames.

    Args:
        detector: Loaded ObjectDetector instance.
        cfg:      User configuration dict.
    """
    st.markdown(
        "<p class='section-label'>🎬 Upload a video</p>", unsafe_allow_html=True
    )
    uploaded_video = st.file_uploader(
        "Drop a video here or click to browse",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Supported: MP4 · AVI · MOV · MKV · WEBM",
        label_visibility="collapsed",
    )

    if uploaded_video is None:
        st.markdown(
            "<div style='text-align:center; padding: 3rem 0; color:#475569;'>"
            "<div style='font-size:3rem;'>🎬</div>"
            "<div style='margin-top:0.5rem;'>Upload a video file to start</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Save upload to temp file so cv2.VideoCapture can open it ──────────
    suffix = Path(uploaded_video.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_video.read())
    tmp.flush()
    video_path = tmp.name
    tmp.close()

    # ── Read metadata ──────────────────────────────────────────────────────
    cap_meta = cv2.VideoCapture(video_path)
    total_frames = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap_meta.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s   = total_frames / fps
    cap_meta.release()

    # Metadata display
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Frames", total_frames)
    m2.metric("FPS",          f"{fps:.1f}")
    m3.metric("Resolution",   f"{width}×{height}")
    m4.metric("Duration",     f"{duration_s:.1f}s")

    st.markdown("---")

    # ── Sampling control ───────────────────────────────────────────────────
    st.markdown(
        "<p class='section-label'>⚡ Processing Speed</p>", unsafe_allow_html=True
    )
    every_n = st.slider(
        "Process every N frames  (higher = faster, lower = more detail)",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
        help="e.g. 5 → detect on frames 1, 6, 11, …",
    )
    est_frames = max(1, total_frames // every_n)
    st.caption(f"Estimated frames to process: **{est_frames}**")

    if not st.button("🚀  Start Video Detection", use_container_width=True):
        return

    # ── Processing loop ────────────────────────────────────────────────────
    out_path = Path("outputs") / "detected_video.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    col_preview, col_live = st.columns([3, 1], gap="medium")
    with col_preview:
        frame_ph  = st.empty()          # Live frame placeholder
        prog_bar  = st.progress(0.0)
        status_ph = st.empty()
    with col_live:
        live_ph = st.empty()            # Live stats placeholder

    cap = cv2.VideoCapture(video_path)
    all_detections: list = []
    frame_timeline: list = []
    last_annotated  = None
    frame_idx       = 0
    processed       = 0
    total_infer_ms  = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every_n == 0:
                # ── Run inference on this frame ────────────────────────────
                t0 = time.perf_counter()
                dets, annotated = detector.detect_image(
                    frame,
                    confidence=cfg["confidence"],
                    iou=cfg["iou"],
                    show_labels=cfg["show_labels"],
                    show_confidence=cfg["show_conf"],
                    show_bbox=cfg["show_bbox"],
                )
                infer_ms = (time.perf_counter() - t0) * 1000
                total_infer_ms += infer_ms

                all_detections.extend(dets)
                frame_timeline.append({"frame": frame_idx, "count": len(dets)})
                last_annotated = annotated
                processed += 1

                # Live preview update (every processed frame)
                frame_ph.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                    caption=f"Frame {frame_idx}/{total_frames}  ·  {len(dets)} objects",
                )

                # Live stats panel
                with live_ph.container():
                    st.metric("Frames Processed", processed)
                    st.metric("Objects (this frame)", len(dets))
                    st.metric("Total Detections", len(all_detections))
                    st.metric("Avg Inference", f"{total_infer_ms/processed:.0f}ms")

            # Write original (or last annotated) frame to output video
            writer.write(last_annotated if last_annotated is not None else frame)

            # Progress bar
            frame_idx += 1
            prog_bar.progress(min(frame_idx / max(total_frames, 1), 1.0))
            status_ph.caption(
                f"⏳  Frame {frame_idx} / {total_frames}   "
                f"|   Processed: {processed}"
            )

    finally:
        cap.release()
        writer.release()
        # Clean up temp input file
        try:
            os.unlink(video_path)
        except OSError:
            pass

    status_ph.success(
        f"✅  Finished!  Processed {processed} frames  |  "
        f"{len(all_detections)} total detections."
    )

    # ── Download processed video ───────────────────────────────────────────
    with open(str(out_path), "rb") as fh:
        st.download_button(
            "💾  Download Processed Video",
            data=fh,
            file_name="yolo_detected_video.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

    st.markdown("---")

    # ── Aggregate analytics ────────────────────────────────────────────────
    avg_infer = total_infer_ms / max(processed, 1)
    render_analytics(all_detections, avg_infer)

    # ── Detection timeline chart ───────────────────────────────────────────
    if frame_timeline:
        st.markdown(
            "<p class='section-label'>📈 Detection Timeline</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            plot_detection_timeline(frame_timeline, fps=fps),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    """
    Streamlit application entry point.

    Render order:
        1. Hero header
        2. Sidebar (configuration)
        3. Model loader (cached)
        4. Mode-specific detection UI
        5. Footer
    """

    # ── Hero ───────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 class='hero-title'>🎯 YOLOv8 Object Detection System</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='hero-sub'>"
        "Real-time object detection · Ultralytics YOLOv8 · COCO 80 classes · "
        "OpenCV · Streamlit"
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Sidebar ─────────────────────────────────────────────────────────────
    cfg = render_sidebar()

    # ── Model loader ────────────────────────────────────────────────────────
    with st.spinner(f"⚙️  Loading {cfg['model_size'].upper()} …"):
        try:
            detector = load_detector(cfg["model_size"])
        except RuntimeError as exc:
            st.error(f"❌  {exc}")
            logger.error("Failed to load model: %s", exc)
            st.stop()

    # Subtle model-ready badge (non-intrusive; replaces spinner)
    st.success(
        f"🤖  **{cfg['model_size'].upper()}** ready  ·  "
        f"{len(detector.class_names)} COCO classes",
        icon="✅",
    )

    st.markdown("---")

    # ── Route to mode ────────────────────────────────────────────────────────
    mode = cfg["mode"]

    if "Webcam" in mode:
        run_webcam_mode(detector, cfg)
    elif "Image" in mode:
        run_image_mode(detector, cfg)
    elif "Video" in mode:
        run_video_mode(detector, cfg)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='footer'>"
        "🎯 YOLOv8 Object Detection System &nbsp;|&nbsp; "
        "Built with Ultralytics, OpenCV &amp; Streamlit &nbsp;|&nbsp; "
        "<a href='https://github.com/yourusername/YOLO-Object-Detection'>GitHub</a>"
        " &nbsp;·&nbsp; "
        "<a href='https://huggingface.co/spaces/yourusername/yolo-detection'>HF Spaces</a>"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
