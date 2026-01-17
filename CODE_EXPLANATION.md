# HoloRay Motion-Tracked Annotation System - Complete Code Explanation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [File-by-File Breakdown](#file-by-file-breakdown)
5. [Data Flow](#data-flow)
6. [Key Algorithms](#key-algorithms)

---

## Project Overview

**HoloRay** is a real-time motion tracking system designed for medical imaging applications. It allows users to place annotations (points or regions) on video frames, and these annotations automatically track anatomical features as they move through the video sequence.

### Key Features:
- **Point Tracking**: Click to place point annotations that follow features using optical flow
- **Region Tracking**: Draw bounding boxes that track larger areas using CSRT/KCF/MOSSE trackers
- **Real-time Performance**: Optimized for 30+ FPS processing
- **Robust Tracking**: Handles occlusions, drift, and feature loss with automatic recovery
- **Web Interface**: Browser-based UI with real-time video streaming
- **CLI Demo**: Standalone command-line version for testing

---

## System Architecture

The system consists of three main layers:

```
┌─────────────────────────────────────┐
│   Frontend (Browser)                 │
│   - HTML/CSS/JavaScript              │
│   - Canvas rendering                  │
│   - WebSocket client                  │
└──────────────┬────────────────────────┘
               │ WebSocket (Socket.IO)
               │ HTTP REST API
┌──────────────▼────────────────────────┐
│   Backend (Flask Server)              │
│   - Flask web framework                │
│   - SocketIO for real-time comm        │
│   - Video processing loop              │
│   - State management                   │
└──────────────┬────────────────────────┘
               │
┌──────────────▼────────────────────────┐
│   Tracking Engine (Core)               │
│   - HybridMotionTracker                │
│   - Optical flow algorithms             │
│   - Kalman filtering                   │
│   - Template matching                  │
└───────────────────────────────────────┘
```

---

## Core Components

### 1. Tracking Engine (`src/tracker.py`)

The heart of the system - implements the motion tracking algorithms.

#### **TrackingState Enum**
Defines the possible states of a tracked object:
- `TRACKING`: Successfully tracking
- `LOST`: Track lost, cannot recover
- `OCCLUDED`: Temporarily occluded, attempting recovery
- `REINITIALIZING`: Recovering after being lost

#### **TrackedPoint DataClass**
Stores information about a tracked point annotation:
- Position (x, y) and original position
- Current state (tracking/lost/occluded)
- Confidence score (0-1)
- Template image for re-detection
- Kalman filter instance (for smoothing)
- History of positions

#### **TrackedRegion DataClass**
Stores information about a tracked bounding box:
- Bounding box coordinates (x, y, width, height)
- OpenCV tracker instance (CSRT/KCF/MOSSE)
- Template for re-detection
- State and confidence

#### **KalmanPointTracker Class**
Implements a Kalman filter for trajectory smoothing:
- **State Variables**: 4D (x, y, velocity_x, velocity_y)
- **Measurements**: 2D (x, y positions)
- **Purpose**: Reduces jitter in tracked positions by predicting motion
- **Model**: Constant velocity model (assumes objects move at constant speed)

**How it works:**
1. Predicts next position based on current velocity
2. Updates prediction with actual measured position
3. Blends prediction and measurement for smooth result

#### **HybridMotionTracker Class** (Main Tracker)

This is the core tracking system that combines multiple techniques:

**Initialization Parameters:**
- `use_kalman`: Enable/disable Kalman filtering
- `drift_correction_interval`: How often to correct for drift (every N frames)
- `max_lost_frames`: Frames before declaring track lost
- `template_size`: Size of template for re-detection
- `optical_flow_quality`: Quality threshold for feature detection

**Key Methods:**

1. **`add_point(point_id, x, y, frame)`**
   - Creates a new tracked point
   - Extracts template around the point
   - Initializes Kalman filter if enabled
   - Stores in `tracked_points` dictionary

2. **`add_region(region_id, bbox, frame, tracker_type)`**
   - Creates a new tracked region (bounding box)
   - Initializes OpenCV tracker (CSRT/KCF/MOSSE)
   - Extracts template from region
   - Stores in `tracked_regions` dictionary

3. **`update(frame)`** - Main tracking update
   - Converts frame to grayscale
   - Updates all points using optical flow
   - Updates all regions using OpenCV trackers
   - Performs periodic drift correction
   - Calculates performance metrics
   - Returns tracking results dictionary

4. **`_update_points_optical_flow()`** - Point tracking
   - Uses Lucas-Kanade pyramidal optical flow
   - Tracks points from previous frame to current frame
   - Applies Kalman filter smoothing
   - Detects occlusions and out-of-bounds
   - Attempts recovery using template matching

5. **`_try_recover_point()`** - Recovery mechanism
   - Searches for lost point using template matching
   - Searches in region around last known position
   - If found with high confidence, reinitializes tracking

6. **`_correct_drift()`** - Drift correction
   - Runs every N frames (drift_correction_interval)
   - Uses template matching to find true position
   - Blends current position with template match
   - Prevents long-term accumulation of errors

7. **`_update_regions()`** - Region tracking
   - Uses OpenCV's built-in trackers
   - CSRT: Most accurate, slower
   - KCF: Balanced accuracy/speed
   - MOSSE: Fastest, less accurate

**Optical Flow Algorithm (Lucas-Kanade):**
- Detects features in previous frame
- Tracks them to current frame using gradient information
- Uses image pyramids for handling large motions
- Window size: 21x21 pixels
- Max pyramid levels: 3

**Template Matching:**
- Extracts small image patch around point (50x50 pixels)
- Uses normalized cross-correlation
- Searches in region around expected position
- Confidence threshold: 0.5-0.7

---

### 2. Web Application (`src/app.py`)

Flask-based web server that provides the user interface.

#### **AppState Class**
Global application state:
- `tracker`: HybridMotionTracker instance
- `video_capture`: OpenCV VideoCapture object
- `current_frame`: Current video frame (numpy array)
- `is_playing`: Playback state
- `annotations`: Dictionary of all annotations
- `frame_lock`: Thread lock for thread-safe access

#### **Key Functions:**

1. **`encode_frame(frame)`**
   - Converts numpy array (BGR image) to JPEG
   - Encodes to base64 string
   - Used for sending frames over WebSocket

2. **`draw_annotations(frame, tracking_results)`**
   - Draws all tracked points and regions on frame
   - Color coding:
     - Green: Tracking successfully
     - Orange: Occluded
     - Red: Lost
   - Draws labels and confidence indicators

3. **`video_processing_loop()`** - Background thread
   - Continuously processes video frames
   - When playing: reads next frame, updates tracking
   - When paused: uses current frame, still updates tracking for visualization
   - Draws annotations and HUD
   - Encodes and sends frames via WebSocket
   - Controls frame rate (video FPS when playing, 10 FPS when paused)

#### **REST API Endpoints:**

1. **`GET /api/videos`**
   - Lists all video files in `videos/` directory
   - Recursively searches subdirectories
   - Returns JSON array of video paths

2. **`POST /api/load_video`**
   - Loads a video file
   - Opens with OpenCV VideoCapture
   - Reads first frame
   - Returns video properties (width, height, FPS, total frames)

#### **WebSocket Events:**

**Client → Server:**
- `play`: Start video playback
- `pause`: Pause video playback
- `add_point`: Add point annotation
- `add_region`: Add region annotation
- `remove_annotation`: Remove annotation
- `clear_annotations`: Clear all annotations
- `seek`: Jump to specific frame
- `get_first_frame`: Request first frame

**Server → Client:**
- `frame`: New video frame with tracking data
- `first_frame`: First frame of loaded video
- `status`: Status messages and playback state
- `annotation_added`: Notification of new annotation
- `annotation_removed`: Notification of removed annotation
- `annotations_cleared`: Notification of cleared annotations

---

### 3. Frontend (`static/js/app.js`)

JavaScript client that handles user interaction and rendering.

#### **HoloRayApp Class**

**State Variables:**
- `socket`: Socket.IO connection
- `videoCanvas`: Canvas for video frames
- `annotationCanvas`: Canvas for annotation overlays
- `currentTool`: 'point' or 'region'
- `isPlaying`: Playback state
- `videoLoaded`: Whether video is loaded
- `annotations`: Dictionary of annotations
- `trackingData`: Current tracking results

**Key Methods:**

1. **`loadVideoList()`**
   - Fetches list of videos from `/api/videos`
   - Populates dropdown menu

2. **`loadSelectedVideo()`**
   - Sends video path to server
   - Receives video properties
   - Resizes canvas to fit video
   - Enables controls
   - Requests first frame

3. **`resizeCanvas(width, height)`**
   - Calculates scale to fit container
   - Sets canvas internal size (actual pixels)
   - Sets canvas display size (CSS pixels)
   - Stores scale for coordinate conversion

4. **`getCanvasCoordinates(event)`**
   - Converts mouse click coordinates to canvas coordinates
   - Accounts for canvas scaling
   - Clamps to canvas bounds
   - Returns pixel coordinates in video space

5. **`handleCanvasMouseDown(event)`**
   - Point tool: Immediately adds point annotation
   - Region tool: Starts region drawing

6. **`handleCanvasMouseMove(event)`**
   - Draws preview rectangle while dragging region

7. **`handleCanvasMouseUp(event)`**
   - Finalizes region annotation
   - Calculates bounding box
   - Sends to server

8. **`displayFrame(base64Image)`**
   - Decodes base64 JPEG
   - Draws on video canvas

9. **`updateStats(tracking)`**
   - Updates FPS display
   - Updates tracking count
   - Updates latency display

10. **`updateAnnotationsList()`**
    - Renders list of active annotations
    - Shows tracking status (green/red dot)
    - Adds remove buttons

**Socket Event Handlers:**
- `connect`: Updates connection status
- `frame`: Displays new frame, updates UI
- `first_frame`: Displays first frame
- `annotation_added`: Updates annotation list
- `status`: Updates status message and playback state

---

### 4. Entry Points

#### **`run.py`** - Web Server Entry Point
- Sets up Python path
- Parses command-line arguments (host, port, debug)
- Calls `run_server()` from `app.py`
- Starts Flask + SocketIO server

#### **`demo_cli.py`** - CLI Demo Entry Point
- Standalone demo without web interface
- Uses OpenCV's GUI for display
- Mouse callbacks for annotation placement
- Keyboard controls (P: play/pause, Q: quit, C: clear, R: region mode, S: save frame)
- Can save output video with `--output` flag

**DemoApp Class:**
- Manages video playback
- Handles mouse/keyboard input
- Draws annotations and HUD
- Integrates with HybridMotionTracker

---

## Data Flow

### Web Application Flow:

1. **User loads video:**
   ```
   Browser → POST /api/load_video → Server opens video → Returns properties
   ```

2. **User adds annotation:**
   ```
   Browser (click) → getCanvasCoordinates() → emit('add_point') 
   → Server adds to tracker → Server sends updated frame → Browser displays
   ```

3. **Video playback:**
   ```
   Browser → emit('play') → Server sets is_playing=True 
   → video_processing_loop() reads frames → Updates tracking 
   → Draws annotations → Encodes frame → emit('frame') 
   → Browser receives → Displays frame
   ```

4. **Frame update cycle:**
   ```
   Every frame:
   - Read frame from video
   - tracker.update(frame) → Optical flow + region tracking
   - draw_annotations() → Draw on frame
   - encode_frame() → JPEG + base64
   - socketio.emit('frame') → Send to all clients
   - Browser receives → displayFrame() → Update UI
   ```

---

## Key Algorithms

### 1. Lucas-Kanade Optical Flow

**Purpose:** Track points between consecutive frames

**Algorithm:**
1. Extract grayscale images
2. Detect features in previous frame (Shi-Tomasi corners)
3. For each feature point:
   - Calculate image gradients in 21x21 window
   - Solve for displacement using least squares
   - Use image pyramids for large motions
4. Filter out bad matches (status)
5. Update point positions

**Parameters:**
- Window size: 21x21 pixels
- Pyramid levels: 3
- Quality threshold: 0.01

### 2. Kalman Filtering

**Purpose:** Smooth trajectories and reduce jitter

**State Model:**
- State: [x, y, vx, vy] (position + velocity)
- Prediction: x_new = x_old + vx, y_new = y_old + vy
- Measurement: [x, y] from optical flow
- Blends prediction and measurement

**Benefits:**
- Reduces noise
- Handles temporary tracking failures
- Predicts motion during occlusions

### 3. Template Matching

**Purpose:** Re-detect lost features

**Algorithm:**
1. Extract template (50x50 pixels) when point is added
2. When point is lost:
   - Search in region around last known position (±100 pixels)
   - Use normalized cross-correlation
   - Find best match above confidence threshold
3. If found, reinitialize tracking

**Search Strategy:**
- Local search (not full frame) for efficiency
- Confidence threshold: 0.5-0.7
- Blends with current position to avoid jumps

### 4. Drift Correction

**Purpose:** Prevent accumulation of small errors

**Algorithm:**
1. Every N frames (default: 30):
   - For each tracked point:
     - Search template in small region (±30 pixels)
     - If high-confidence match found:
       - Blend current position with template match
       - Blend factor: 0.3 (30% template, 70% current)

**Why needed:**
- Optical flow has small errors per frame
- Errors accumulate over time
- Template matching provides absolute reference

### 5. Region Tracking (CSRT/KCF/MOSSE)

**Purpose:** Track bounding boxes

**CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability):**
- Most accurate
- Uses color and spatial information
- Slower but robust

**KCF (Kernelized Correlation Filter):**
- Balanced accuracy/speed
- Uses circular correlation
- Good for most cases

**MOSSE (Minimum Output Sum of Squared Error):**
- Fastest
- Basic correlation filter
- Less accurate but real-time

---

## Threading and Concurrency

### Thread Safety:
- `frame_lock`: Protects video capture and frame access
- Background thread: `video_processing_loop()` runs continuously
- Main thread: Flask handles HTTP/WebSocket requests
- SocketIO: Handles async communication

### Synchronization:
- All video operations use `frame_lock`
- Annotations are added/removed with lock held
- Frame updates are atomic

---

## Performance Optimizations

1. **Grayscale Processing:** Optical flow uses grayscale (1 channel vs 3)
2. **Template Caching:** Templates extracted once, reused
3. **Local Search:** Template matching searches small regions, not full frame
4. **Frame Rate Control:** Updates at video FPS when playing, slower when paused
5. **JPEG Compression:** Frames compressed to 85% quality before transmission
6. **Bounded History:** Point position history limited to 100 entries

---

## Error Handling

1. **Lost Tracks:**
   - Detected when optical flow fails
   - Attempts recovery via template matching
   - Declared lost after max_lost_frames

2. **Out of Bounds:**
   - Points outside frame marked as occluded
   - Attempts recovery when back in frame

3. **Video End:**
   - Loops back to start
   - Reinitializes tracker with existing annotations

4. **Connection Loss:**
   - SocketIO handles reconnection
   - State maintained on server

---

## File Structure

```
McHacks-1/
├── run.py                 # Web server entry point
├── demo_cli.py           # CLI demo entry point
├── requirements.txt      # Python dependencies
├── README.md            # Documentation
├── src/
│   ├── __init__.py
│   ├── tracker.py       # Core tracking engine
│   └── app.py           # Flask web application
├── templates/
│   └── index.html       # Web UI HTML
├── static/
│   ├── css/
│   │   └── style.css    # Styling
│   └── js/
│       └── app.js        # Frontend JavaScript
└── videos/
    └── Lapchole/        # Sample videos
        ├── Lapchole1.mp4
        ├── Lapchole2.mp4
        ├── Lapchole3.mp4
        └── Lapchole4.mp4
```

---

## Usage Examples

### Web Application:
1. Start server: `python run.py`
2. Open browser: `http://localhost:5000`
3. Select video from dropdown
4. Click "Load Video"
5. Click on video to add points
6. Select Region tool, drag to add boxes
7. Click Play to start tracking

### CLI Demo:
1. Run: `python demo_cli.py videos/Lapchole/Lapchole1.mp4`
2. Click on video to add points
3. Press 'R' then drag for regions
4. Press 'P' to play/pause
5. Press 'Q' to quit

---

## Summary

The HoloRay system combines:
- **Computer Vision**: Optical flow, template matching, Kalman filtering
- **Web Technologies**: Flask, SocketIO, HTML5 Canvas
- **Real-time Processing**: Optimized for 30+ FPS
- **Robust Tracking**: Handles occlusions, drift, and feature loss

The architecture separates concerns:
- **Tracking Engine**: Pure computer vision algorithms
- **Web Server**: Handles communication and state
- **Frontend**: User interface and rendering

This design allows the tracking engine to be used independently (CLI demo) or integrated into web applications.
