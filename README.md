# HoloTrack

**Simple visual tracker + annotation UI for microscopy / video sources**

HoloTrack is a small Python/OpenCV project that combines an easy-to-use Tkinter frontend with a robust, practical tracking pipeline. It’s designed for annotating regions in a video (e.g., microscopy) and keeping those annotations roughly locked to the object as it moves using a mixture of template-matching, Lucas–Kanade optical flow, and periodic feature replenishment.

This README explains what the project does, how to run it, and the most important knobs you may want to tweak.

---

## What it does (in plain words)
- Launch a compact GUI to pick a video file (or use the default `Microscopy1.mp4`).
- Drag the mouse to draw contour regions you want to track over time.
- The tracker uses local features + optical flow to estimate object motion and warps your drawn contours accordingly.
- If the tracker loses the object it attempts to re-acquire it using template matching inside a search window.

---

## Key features
- Lightweight Tkinter GUI for quick experiments.
- Click-to-anchor & draw-to-annotate workflow.
- Robustness measures:
  - Replenishes feature points when they get sparse.
  - Detects when the tracker is lost and tries to re-find the object with template matching.
  - Applies an estimated affine transform to keep drawn contours aligned with motion.

---

## Requirements
- Python 3.8+
- `opencv-python`
- `numpy`

Install dependencies with pip:
```bash
pip install opencv-python numpy
```

---

## Short Demo
[Click here to access the demo](https://www.youtube.com/watch?v=4Kaoa3i0Pko)
