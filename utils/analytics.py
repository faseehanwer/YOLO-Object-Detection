"""
utils/analytics.py
==================
Analytics & Statistics Engine

Computes summary statistics from detection results and generates
interactive Plotly figures for the Streamlit dashboard:

    compute_statistics()        → dict of aggregated metrics
    plot_class_distribution()   → horizontal bar chart (objects per class)
    plot_confidence_distribution() → histogram (confidence score spread)
    plot_detection_timeline()   → area line chart (detections over time)

Design notes:
  - All charts use transparent backgrounds so they float naturally inside
    Streamlit's dark or light themed containers.
  - The colour palette cycles through a curated 12-colour set that matches
    the application's indigo-violet-cyan identity.
  - Charts are compact (≤ 300 px tall) to avoid overwhelming the page.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ── Brand colour palette (12 colours, repeating) ──────────────────────────────
_PALETTE = [
    "#818cf8",  # indigo-400
    "#a78bfa",  # violet-400
    "#22d3ee",  # cyan-400
    "#34d399",  # emerald-400
    "#fbbf24",  # amber-400
    "#f87171",  # red-400
    "#c084fc",  # purple-400
    "#fb7185",  # rose-400
    "#2dd4bf",  # teal-400
    "#fb923c",  # orange-400
    "#60a5fa",  # blue-400
    "#a3e635",  # lime-400
]

# Shared layout defaults (applied to every figure)
_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
    margin=dict(l=8, r=8, t=24, b=8),
    hoverlabel=dict(bgcolor="#1e1b4b", font_size=12, font_color="#e2e8f0"),
    showlegend=False,
)

_GRID = dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", gridwidth=1)


# ── Statistics computation ────────────────────────────────────────────────────

def compute_statistics(
    detections: List[Dict[str, Any]],
    inference_time_ms: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute a flat statistics dict from a list of detection results.

    Args:
        detections:        List of detection dicts produced by ObjectDetector.
        inference_time_ms: Model inference wall-clock time in milliseconds.

    Returns:
        Dict with the following keys:
            total_objects    (int)
            unique_classes   (int)
            avg_confidence   (float)
            max_confidence   (float)
            min_confidence   (float)
            class_counts     (dict[str, int])
            class_names      (list[str])   – sorted alphabetically
            inference_time   (float)       – milliseconds
    """
    if not detections:
        return {
            "total_objects": 0,
            "unique_classes": 0,
            "avg_confidence": 0.0,
            "max_confidence": 0.0,
            "min_confidence": 0.0,
            "class_counts": {},
            "class_names": [],
            "inference_time": inference_time_ms,
        }

    confidences = [d["confidence"] for d in detections]
    class_counts: Dict[str, int] = {}
    for d in detections:
        name = d["class_name"]
        class_counts[name] = class_counts.get(name, 0) + 1

    return {
        "total_objects": len(detections),
        "unique_classes": len(class_counts),
        "avg_confidence": float(np.mean(confidences)),
        "max_confidence": float(np.max(confidences)),
        "min_confidence": float(np.min(confidences)),
        "class_counts": class_counts,
        "class_names": sorted(class_counts.keys()),
        "inference_time": inference_time_ms,
    }


# ── Chart generators ──────────────────────────────────────────────────────────

def plot_class_distribution(detections: List[Dict[str, Any]]) -> go.Figure:
    """
    Horizontal bar chart: number of detected objects per class.

    Uses the brand palette (cycling) so each bar has a distinct colour.
    Bars are sorted descending by count for easy reading.

    Args:
        detections: List of detection dicts.

    Returns:
        Plotly Figure. Returns an empty annotated figure when detections is [].
    """
    fig = go.Figure()

    if not detections:
        fig.add_annotation(
            text="No objects detected",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color="#64748b"),
        )
        fig.update_layout(**_LAYOUT_BASE, height=200)
        return fig

    # Aggregate counts
    counts: Dict[str, int] = {}
    for d in detections:
        name = d["class_name"].capitalize()
        counts[name] = counts.get(name, 0) + 1

    # Sort descending
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    names = [it[0] for it in sorted_items]
    values = [it[1] for it in sorted_items]
    bar_colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(names))]

    fig.add_trace(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(color="rgba(0,0,0,0.15)", width=0.8),
                opacity=0.9,
            ),
            text=values,
            textposition="outside",
            textfont=dict(size=11, color="#cbd5e1"),
            hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>",
        )
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        height=max(220, len(names) * 38 + 60),
        xaxis=dict(title="Count", **_GRID, title_font=dict(color="#64748b")),
        yaxis=dict(showgrid=False, tickfont=dict(size=11)),
    )

    return fig


def plot_confidence_distribution(detections: List[Dict[str, Any]]) -> go.Figure:
    """
    Histogram of confidence scores across all detections.

    Includes a dashed vertical line marking the mean confidence.

    Args:
        detections: List of detection dicts.

    Returns:
        Plotly Figure. Returns an annotated empty figure when detections is [].
    """
    fig = go.Figure()

    if not detections:
        fig.add_annotation(
            text="No detections",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color="#64748b"),
        )
        fig.update_layout(**_LAYOUT_BASE, height=260)
        return fig

    confidences = [d["confidence"] for d in detections]
    mean_conf = float(np.mean(confidences))

    fig.add_trace(
        go.Histogram(
            x=confidences,
            nbinsx=20,
            marker=dict(
                color="#818cf8",
                line=dict(color="rgba(0,0,0,0.2)", width=0.8),
                opacity=0.85,
            ),
            hovertemplate="Range: %{x:.2f}<br>Count: %{y}<extra></extra>",
        )
    )

    # Mean line annotation
    fig.add_vline(
        x=mean_conf,
        line_dash="dash",
        line_color="#fbbf24",
        line_width=1.5,
        annotation_text=f"  Mean {mean_conf:.0%}",
        annotation_position="top right",
        annotation_font=dict(color="#fbbf24", size=11),
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        height=270,
        xaxis=dict(
            title="Confidence Score",
            tickformat=".0%",
            range=[0, 1.05],
            title_font=dict(color="#64748b"),
            **_GRID,
        ),
        yaxis=dict(title="Count", title_font=dict(color="#64748b"), **_GRID),
    )

    return fig


def plot_detection_timeline(
    frame_stats: List[Dict[str, Any]],
    fps: float = 30.0,
) -> go.Figure:
    """
    Area line chart: number of detections per processed video frame over time.

    Useful for video-mode summaries showing detection density across the clip.

    Args:
        frame_stats: List of dicts with at least {'frame': int, 'count': int}.
        fps:         Video frames-per-second (for the time-axis labels).

    Returns:
        Plotly Figure. Returns an annotated empty figure when frame_stats is [].
    """
    fig = go.Figure()

    if not frame_stats:
        fig.add_annotation(
            text="No frame data available",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color="#64748b"),
        )
        fig.update_layout(**_LAYOUT_BASE, height=220)
        return fig

    frames = [d.get("frame", i) for i, d in enumerate(frame_stats)]
    counts = [d.get("count", 0) for d in frame_stats]
    timestamps = [f / max(fps, 1) for f in frames]

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=counts,
            mode="lines",
            line=dict(color="#818cf8", width=2, shape="spline", smoothing=0.8),
            fill="tozeroy",
            fillcolor="rgba(129,140,248,0.12)",
            hovertemplate="Time: %{x:.1f}s<br>Objects: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        height=230,
        xaxis=dict(
            title="Time (seconds)",
            title_font=dict(color="#64748b"),
            **_GRID,
        ),
        yaxis=dict(
            title="Objects Detected",
            title_font=dict(color="#64748b"),
            **_GRID,
        ),
    )

    return fig
