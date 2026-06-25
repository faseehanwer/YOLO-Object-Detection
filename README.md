---
title: YOLO Object Detection
emoji: U+1F3AF
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

---
title: YOLO Object Detection
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# ðŸŽ¯ YOLOv8 Real-Time Object Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=pytorch&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22D3EE?style=for-the-badge)

**A production-ready, publicly deployable AI web application for real-time object detection.**  
Powered by Ultralytics YOLOv8 Â· OpenCV Â· Streamlit Â· COCO Dataset (80 classes)

[ðŸ¤— Live Demo on HF Spaces](#) &nbsp;Â·&nbsp; [ðŸ“‚ GitHub Repository](#) &nbsp;Â·&nbsp; [ðŸ› Report a Bug](#)

</div>

---

## ðŸ“¸ Screenshots

| Image Detection | Video Processing | Analytics Dashboard |
|:---:|:---:|:---:|
| ![Image Detection](assets/screenshot_image.png) | ![Video Detection](assets/screenshot_video.png) | ![Analytics](assets/screenshot_analytics.png) |

> **Add screenshots:** Run the app, capture the three views above, and save them to the `assets/` folder.

---

## ðŸ“Œ Project Overview

### What It Does

This project is an end-to-end AI web application that performs **real-time object detection** using **YOLOv8**, one of the most powerful and widely adopted computer-vision models in production today. Users can detect objects across three input modes â€” webcam snapshots, uploaded images, and uploaded videos â€” all through a polished browser-based dashboard with live analytics.

### Why It Was Built

This project was designed to demonstrate **production-level computer vision engineering** skills:

- Translating a research model (YOLOv8) into a usable, deployed product
- Engineering modular, maintainable Python architecture
- Deploying a GPU-optional AI system to a public URL (Hugging Face Spaces)
- Visualising results with interactive analytics for non-technical audiences

### Technologies Used

| Layer | Technology | Purpose |
|---|---|---|
| Detection Model | **Ultralytics YOLOv8** | State-of-the-art object detection |
| Image Processing | **OpenCV 4.9** | Frame I/O, annotation, video I/O |
| Numerical Computing | **NumPy** | Array operations on image tensors |
| Web Framework | **Streamlit 1.35** | Interactive browser-based UI |
| Visualisation | **Plotly** | Interactive analytics charts |
| Data Handling | **Pandas** | Detection result tables |
| Image Utils | **Pillow** | Image format conversion |
| Language | **Python 3.11+** | Core application |

---

## âœ¨ Features

### Core Detection Capabilities

- **Real-Time Object Detection** â€” YOLOv8 inference with sub-100ms latency on CPU for the Nano model
- **Webcam Snapshot Mode** â€” Capture frames directly from any browser-accessible camera (desktop or mobile)
- **Image Upload Mode** â€” Detect objects in any JPG / PNG / BMP / WEBP image; original vs result shown side by side
- **Video Upload Mode** â€” Frame-by-frame video processing with live preview and progress bar (MP4, AVI, MOV, MKV)
- **5 Model Variants** â€” YOLOv8n / s / m / l / x â€” trade speed for accuracy via a single dropdown
- **80 COCO Classes** â€” people, vehicles, animals, furniture, food, electronics, and more
- **Adjustable Thresholds** â€” Confidence and IoU (NMS) sliders for precision tuning

### Visualisation & Annotation

- **Colour-Coded Bounding Boxes** â€” Each COCO class gets a unique HSV-derived colour, consistent across all frames
- **Rounded-Corner Boxes** â€” Modern UI aesthetic instead of standard rectangle outlines
- **Corner Accent Marks** â€” Military HUDâ€“style tick marks at box corners
- **Semi-Transparent Fill** â€” Subtle alpha-blended interior highlight for each detection
- **Class + Confidence Labels** â€” Auto-contrast badge (white/black text) over the class colour background
- **Watermark Overlay** â€” YOLOv8 + detection count stamped on every output frame

### Analytics Dashboard

- **KPI Cards** â€” Total objects, unique classes, average confidence, inference time
- **Interactive Bar Chart** â€” Objects detected per class (Plotly, dark theme)
- **Confidence Histogram** â€” Score distribution with mean-confidence marker
- **Detection Timeline** â€” Detections per frame over time (video mode only)
- **Sortable Summary Table** â€” Class, count, avg confidence, max confidence with progress bars
- **Class Badges** â€” Visual pill-badges listing all detected object categories

### Output & Export

- **Download Annotated Image** â€” One-click JPEG download of the processed result
- **Download Processed Video** â€” Download the fully annotated MP4 output file
- **Variable Frame Sampling** â€” Control how many frames to process in video mode (speed vs detail)

---

## ðŸ—‚ï¸ Project Structure

```
YOLO-Object-Detection/
â”‚
â”œâ”€â”€ app.py                  # â† Main Streamlit application (entry point)
â”œâ”€â”€ requirements.txt        # Python dependencies
â”œâ”€â”€ packages.txt            # System-level APT packages (Hugging Face Spaces)
â”œâ”€â”€ README.md               # This file
â”œâ”€â”€ .gitignore              # Git exclusions
â”‚
â”œâ”€â”€ utils/
â”‚   â”œâ”€â”€ __init__.py         # Package init; public exports
â”‚   â”œâ”€â”€ detector.py         # YOLOv8 model loader + inference engine
â”‚   â”œâ”€â”€ visualization.py    # Bounding box drawing + detection summary
â”‚   â””â”€â”€ analytics.py        # Statistics computation + Plotly chart builders
â”‚
â”œâ”€â”€ models/                 # Auto-downloaded YOLOv8 .pt weights (git-ignored)
â”œâ”€â”€ uploads/                # Temporary input files (git-ignored)
â”œâ”€â”€ outputs/                # Processed images/videos for download (git-ignored)
â””â”€â”€ assets/                 # Static files: screenshots, demo GIFs
```

---

## âš¡ Quick Start (Local)

### Prerequisites

- Python **3.11** or higher
- pip / conda
- Webcam (optional, for webcam mode)
- GPU (optional â€” CUDA accelerates inference but is not required)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/YOLO-Object-Detection.git
cd YOLO-Object-Detection

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run app.py
```

The app opens at **http://localhost:8501** in your default browser.

> **First run:** Ultralytics will automatically download the selected YOLOv8 model
> weights (~6 MB for Nano). Subsequent runs load from cache instantly.

---

## ðŸš€ Deployment

### GitHub

```bash
# 1. Create a new GitHub repository (via github.com or CLI)
gh repo create YOLO-Object-Detection --public --description \
  "Real-time object detection system using YOLOv8, OpenCV & Streamlit"

# 2. Initialise git and push
cd YOLO-Object-Detection
git init
git add .
git commit -m "feat: initial production-ready YOLOv8 detection system"
git branch -M main
git remote add origin https://github.com/yourusername/YOLO-Object-Detection.git
git push -u origin main
```

**Adding screenshots (strongly recommended for portfolio impact):**

```bash
# Run the app locally, capture screenshots, then:
git add assets/screenshot_image.png assets/screenshot_video.png assets/screenshot_analytics.png
git commit -m "docs: add app screenshots to README"
git push
```

---

### Hugging Face Spaces

Hugging Face Spaces provides **free GPU-optional Streamlit hosting** with a public URL â€” ideal for portfolio demos.

#### Step 1 â€” Create a Space

1. Log in at [huggingface.co](https://huggingface.co) and go to **Spaces â†’ Create new Space**
2. Set **Space name** â†’ `yolo-object-detection`
3. **SDK** â†’ `Streamlit`
4. **Visibility** â†’ `Public`
5. Click **Create Space**

#### Step 2 â€” Upload your project

**Option A â€” Git push (recommended):**

```bash
# Add HF Spaces as a second remote
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/yolo-object-detection

# Push to Spaces (triggers automatic build)
git push hf main
```

**Option B â€” Web UI upload:**

Upload all project files via the **Files** tab in your Space settings.  
Ensure `app.py`, `requirements.txt`, and `packages.txt` are at the **root** of the Space.

#### Step 3 â€” Configure the Space

Hugging Face will automatically:
1. Detect `packages.txt` â†’ install APT system packages
2. Detect `requirements.txt` â†’ install Python packages
3. Start the app with `streamlit run app.py`

#### Step 4 â€” Get your public URL

Once the build completes (3â€“5 minutes), your app is live at:
```
https://YOUR_HF_USERNAME-yolo-object-detection.hf.space
```

#### Optional â€” Hardware upgrade

For faster inference, upgrade to a **GPU Space** in Space Settings â†’ Hardware:
- Free T4 GPU â†’ 10Ã— faster inference vs CPU
- YOLOv8n on T4 â†’ ~5ms/frame (200 FPS capable)

---

## ðŸ¤– Model Comparison

| Model | Size | mAP50-95 | Speed (CPU) | Best For |
|:---:|:---:|:---:|:---:|:---:|
| **YOLOv8n** | 6.2 MB | 37.3 | ~80ms | Real-time, edge, demos |
| **YOLOv8s** | 21.5 MB | 44.9 | ~120ms | Balanced speed/accuracy |
| **YOLOv8m** | 49.7 MB | 50.2 | ~234ms | Production, moderate HW |
| **YOLOv8l** | 83.7 MB | 52.9 | ~375ms | High-accuracy tasks |
| **YOLOv8x** | 136.7 MB | 53.9 | ~479ms | Maximum accuracy |

> mAP values on COCO val2017. CPU times measured on Intel Core i7-12th gen.

---

## ðŸ·ï¸ COCO Classes (80)

The model detects all 80 standard COCO object categories:

**People & Accessories:** person, backpack, umbrella, handbag, tie, suitcase  
**Vehicles:** bicycle, car, motorcycle, airplane, bus, train, truck, boat  
**Animals:** bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe  
**Sports:** frisbee, skis, snowboard, sports ball, kite, baseball bat, skateboard, surfboard  
**Kitchen:** bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, pizza  
**Electronics:** laptop, mouse, remote, keyboard, cell phone, TV, microwave  
**Furniture:** chair, couch, potted plant, bed, dining table, toilet  
**And more:** book, clock, vase, scissors, teddy bear, hair drier, toothbrush

---

## ðŸ”® Future Improvements

- **Multi-Camera Support** â€” Switch between multiple connected cameras at runtime using `streamlit-webrtc`
- **Object Tracking** â€” Cross-frame tracking with ByteTrack / DeepSORT (unique IDs per object)
- **Custom Model Training** â€” UI for uploading labelled datasets and fine-tuning on custom classes
- **Live RTSP Stream** â€” Connect to IP cameras and NVR systems via RTSP URL input
- **REST API Backend** â€” FastAPI endpoint for programmatic inference integration
- **Batch Processing** â€” Upload multiple images as a ZIP for batch detection
- **Export to ONNX / TensorRT** â€” One-click model export for edge deployment optimisation
- **Cloud Storage Integration** â€” Auto-save results to AWS S3 / Google Cloud Storage
- **Detection Alerts** â€” Email / webhook notifications when specific classes are detected
- **Performance Dashboard** â€” Historical detection logs with trend analytics

---

## ðŸ“„ License

Distributed under the **MIT License**. See `LICENSE` for details.

YOLOv8 is released under the [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) by Ultralytics.

---

## ðŸ™ Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) â€” Model architecture and training pipeline
- [Streamlit](https://streamlit.io) â€” Web application framework
- [COCO Dataset](https://cocodataset.org) â€” Training dataset for object detection
- [Plotly](https://plotly.com) â€” Interactive charting library

---

<div align="center">

**â­ If you found this project useful, please star the repository!**

Made with â¤ï¸ as an AI/ML portfolio project

</div>

---

## ðŸ’¼ Portfolio Content

### âœ… Resume Bullet Points (ATS-Optimised)

```
â€¢ Engineered a production-ready real-time object detection system using
  YOLOv8 and OpenCV, achieving sub-100ms inference latency on CPU across
  80 COCO object classes; deployed publicly on Hugging Face Spaces with a
  live demo URL.

â€¢ Built a modular Python application (Streamlit dashboard) with three
  detection modes â€” webcam snapshots, image uploads, and video processing â€”
  featuring bounding-box annotation, confidence score display, and an
  interactive analytics panel with Plotly visualisations.

â€¢ Implemented an optimised inference pipeline using Ultralytics YOLOv8 with
  configurable confidence and IoU thresholds, frame-sampling for video
  throughput control, and Streamlit resource caching to eliminate redundant
  model reloads in a multi-user cloud deployment.
```

---

### ðŸ’¼ LinkedIn Project Description

```
ðŸŽ¯ YOLOv8 Real-Time Object Detection System

Built and publicly deployed a full-stack AI computer vision application
powered by Ultralytics YOLOv8 and Streamlit. The system detects 80 object
categories in real time across webcam snapshots, uploaded images, and video
files â€” achieving sub-100ms CPU inference with the Nano model variant.

Key highlights:
â–ª Modular Python architecture: detector, visualisation, and analytics
  engines built as separate, reusable utility modules
â–ª Professional UI with colour-coded bounding boxes, rounded-corner
  overlays, confidence badges, and a dark-themed analytics dashboard
â–ª Interactive analytics: KPI cards, class distribution charts, confidence
  histograms, and detection timeline (video mode)
â–ª Cloud-deployed to Hugging Face Spaces with zero configuration required
  from end users â€” just open the URL and detect

Tech: Python 3.11 Â· Ultralytics YOLOv8 Â· OpenCV Â· Streamlit Â· Plotly Â·
      NumPy Â· Pandas Â· Hugging Face Spaces

ðŸ”— Live Demo: https://huggingface.co/spaces/yourusername/yolo-detection
ðŸ“‚ Source: https://github.com/yourusername/YOLO-Object-Detection
```

---

### ðŸ“‹ GitHub Repository Description (One-Liner)

```
Real-time object detection web app using YOLOv8 + OpenCV + Streamlit â€”
supports webcam, image & video inputs with analytics dashboard.
Deployed on Hugging Face Spaces.
```


