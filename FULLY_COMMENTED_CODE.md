# Fully Commented Code - HoloRay Motion Tracking System

This document contains all code files with comprehensive inline comments explaining every section, function, and algorithm.

---

## File 1: src/tracker.py - Core Tracking Engine

```python
"""
HoloRay Motion Tracker - Hybrid Tracking System

This module implements a robust motion tracking system that combines:
1. Sparse Optical Flow (Lucas-Kanade) for fast point tracking
2. Dense Optical Flow for global motion estimation  
3. CSRT/KCF trackers for region-based backup
4. Kalman filtering for trajectory smoothing
5. Template matching for re-detection after occlusion

Designed for medical imaging: ultrasound, echocardiography, laparoscopy, IVUS
"""

import cv2  # OpenCV for computer vision operations
import numpy as np  # NumPy for numerical operations
from typing import Optional, Tuple, List, Dict, Any  # Type hints for better code documentation
from dataclasses import dataclass, field  # Data classes for structured data
from enum import Enum  # Enumerations for tracking states
import time  # For performance measurement


# ============================================================================
# TRACKING STATE ENUMERATION
# ============================================================================
class TrackingState(Enum):
    """
    Enumeration of possible states for a tracked object.
    
    States:
    - TRACKING: Object is being successfully tracked
    - LOST: Track has been lost and cannot be recovered
    - OCCLUDED: Object is temporarily occluded, attempting recovery
    - REINITIALIZING: Object was lost but is being re-detected
    """
    TRACKING = "tracking"  # Successfully tracking the feature
    LOST = "lost"  # Track lost permanently
    OCCLUDED = "occluded"  # Temporarily hidden/blocked
    REINITIALIZING = "reinitializing"  # Being re-detected after loss


# ============================================================================
# DATA STRUCTURES FOR TRACKED OBJECTS
# ============================================================================

@dataclass
class TrackedPoint:
    """
    Data structure representing a single tracked point annotation.
    
    Attributes:
        id: Unique identifier for this point
        x, y: Current position coordinates (float for sub-pixel accuracy)
        original_x, original_y: Initial position when annotation was created
        state: Current tracking state (TRACKING, LOST, OCCLUDED, REINITIALIZING)
        confidence: Tracking confidence score (0.0 to 1.0)
        frames_lost: Number of consecutive frames where tracking failed
        template: Small image patch around point for re-detection (50x50 pixels)
        kalman: Kalman filter instance for trajectory smoothing (optional)
        history: List of past positions for trajectory visualization
    """
    id: str  # Unique identifier
    x: float  # Current X coordinate (sub-pixel precision)
    y: float  # Current Y coordinate (sub-pixel precision)
    original_x: float  # Original X when annotation was placed
    original_y: float  # Original Y when annotation was placed
    state: TrackingState = TrackingState.TRACKING  # Current tracking state
    confidence: float = 1.0  # Confidence score 0.0-1.0
    frames_lost: int = 0  # Consecutive frames where tracking failed
    template: Optional[np.ndarray] = None  # Image template for re-detection
    kalman: Optional[cv2.KalmanFilter] = None  # Kalman filter for smoothing
    history: List[Tuple[float, float]] = field(default_factory=list)  # Position history


@dataclass
class TrackedRegion:
    """
    Data structure representing a tracked bounding box region.
    
    Attributes:
        id: Unique identifier for this region
        bbox: Current bounding box (x, y, width, height)
        original_bbox: Initial bounding box when annotation was created
        state: Current tracking state
        confidence: Tracking confidence score
        tracker: OpenCV tracker instance (CSRT, KCF, or MOSSE)
        template: Image patch of the region for re-detection
    """
    id: str  # Unique identifier
    bbox: Tuple[int, int, int, int]  # (x, y, width, height) bounding box
    original_bbox: Tuple[int, int, int, int]  # Original bounding box
    state: TrackingState = TrackingState.TRACKING  # Current tracking state
    confidence: float = 1.0  # Confidence score
    tracker: Any = None  # OpenCV tracker instance
    template: Optional[np.ndarray] = None  # Region image template


# ============================================================================
# KALMAN FILTER FOR TRAJECTORY SMOOTHING
# ============================================================================

class KalmanPointTracker:
    """
    Kalman filter implementation for smoothing point trajectories.
    
    Purpose: Reduces jitter in tracked positions by predicting motion
    and blending predictions with actual measurements.
    
    Model: Constant velocity model - assumes objects move at constant speed
    State: [x, y, velocity_x, velocity_y] - 4D state vector
    Measurement: [x, y] - 2D position measurement from optical flow
    """
    
    def __init__(self, initial_x: float, initial_y: float):
        """
        Initialize Kalman filter with starting position.
        
        Args:
            initial_x: Starting X coordinate
            initial_y: Starting Y coordinate
        """
        # Create Kalman filter: 4 state variables, 2 measurements
        # State: [x, y, vx, vy] where vx, vy are velocities
        # Measurement: [x, y] position from optical flow
        self.kf = cv2.KalmanFilter(4, 2)
        
        # State transition matrix: predicts next state from current state
        # [x_new]   [1  0  1  0] [x_old]
        # [y_new] = [0  1  0  1] [y_old]
        # [vx_new]  [0  0  1  0] [vx_old]
        # [vy_new]  [0  0  0  1] [vy_old]
        # This implements: x_new = x_old + vx, y_new = y_old + vy
        #                  vx_new = vx_old, vy_new = vy_old (constant velocity)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x_new = x_old + vx
            [0, 1, 0, 1],  # y_new = y_old + vy
            [0, 0, 1, 0],  # vx_new = vx_old
            [0, 0, 0, 1]   # vy_new = vy_old
        ], dtype=np.float32)
        
        # Measurement matrix: extracts position from state
        # [x_measured]   [1  0  0  0] [x]
        # [y_measured] = [0  1  0  0] [y]
        #                            [vx]
        #                            [vy]
        # We only measure position, not velocity
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],  # Measure x position
            [0, 1, 0, 0]   # Measure y position
        ], dtype=np.float32)
        
        # Process noise covariance: uncertainty in motion model
        # Higher values = more uncertainty in predictions
        # 0.03 means we're fairly confident in constant velocity assumption
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Measurement noise covariance: uncertainty in measurements
        # Higher values = less trust in optical flow measurements
        # 0.5 means moderate trust in optical flow accuracy
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        
        # Initial state: position at (initial_x, initial_y), zero velocity
        self.kf.statePost = np.array([[initial_x], [initial_y], [0], [0]], dtype=np.float32)
        
        # Initial error covariance: uncertainty in initial state
        # Identity matrix = equal uncertainty in all dimensions
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
    
    def predict(self) -> Tuple[float, float]:
        """
        Predict next position based on current state and motion model.
        
        Returns:
            (predicted_x, predicted_y) tuple
        """
        # Kalman filter prediction step
        # Uses state transition matrix to predict next position
        prediction = self.kf.predict()
        return float(prediction[0, 0]), float(prediction[1, 0])
    
    def update(self, x: float, y: float) -> Tuple[float, float]:
        """
        Update filter with new measurement and return corrected position.
        
        Args:
            x: Measured X position from optical flow
            y: Measured Y position from optical flow
            
        Returns:
            (corrected_x, corrected_y) - Kalman-filtered position
        """
        # Create measurement vector
        measurement = np.array([[x], [y]], dtype=np.float32)
        
        # Kalman filter correction step
        # Blends prediction with measurement based on uncertainties
        corrected = self.kf.correct(measurement)
        
        # Return corrected position
        return float(corrected[0, 0]), float(corrected[1, 0])


# ============================================================================
# MAIN HYBRID MOTION TRACKER
# ============================================================================

class HybridMotionTracker:
    """
    Hybrid motion tracking system optimized for medical imaging.
    
    Combines multiple tracking techniques:
    1. Lucas-Kanade Optical Flow: Fast, accurate point tracking
    2. Template Matching: Re-detection after occlusion
    3. Kalman Filtering: Smooth trajectories, reduce jitter
    4. Drift Correction: Periodic correction to prevent error accumulation
    5. OpenCV Trackers: CSRT/KCF/MOSSE for region tracking
    
    Features:
    - Multi-point optical flow tracking
    - Automatic drift correction every N frames
    - Occlusion detection and recovery
    - Kalman filter smoothing (optional)
    - Template-based re-detection
    """
    
    def __init__(self,
                 use_kalman: bool = True,
                 drift_correction_interval: int = 30,
                 max_lost_frames: int = 15,
                 template_size: int = 50,
                 optical_flow_quality: float = 0.01):
        """
        Initialize the tracker with configuration parameters.
        
        Args:
            use_kalman: Enable Kalman filter smoothing (reduces jitter)
            drift_correction_interval: Frames between drift corrections (30 = every 30 frames)
            max_lost_frames: Max frames before declaring track lost (15 frames)
            template_size: Size of template for re-detection (50x50 pixels)
            optical_flow_quality: Quality threshold for feature detection (0.01 = 1%)
        """
        # Store configuration
        self.use_kalman = use_kalman
        self.drift_correction_interval = drift_correction_interval
        self.max_lost_frames = max_lost_frames
        self.template_size = template_size
        
        # Lucas-Kanade Optical Flow parameters
        # winSize: Search window size (21x21 pixels) - larger = more robust but slower
        # maxLevel: Pyramid levels (3) - handles large motions by tracking at multiple scales
        # criteria: Termination criteria - stop when error < 0.01 or after 30 iterations
        self.lk_params = dict(
            winSize=(21, 21),  # 21x21 pixel search window
            maxLevel=3,  # 3-level image pyramid for large motions
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            # Stop when: error < 0.01 OR 30 iterations reached
        )
        
        # Feature detection parameters (Shi-Tomasi corner detector)
        # Used for detecting good tracking points
        self.feature_params = dict(
            maxCorners=100,  # Maximum corners to detect
            qualityLevel=optical_flow_quality,  # Minimum quality (0.01 = top 1%)
            minDistance=10,  # Minimum distance between corners (pixels)
            blockSize=7  # Neighborhood size for corner detection
        )
        
        # Storage for tracked objects
        self.tracked_points: Dict[str, TrackedPoint] = {}  # Dictionary of tracked points
        self.tracked_regions: Dict[str, TrackedRegion] = {}  # Dictionary of tracked regions
        
        # Frame management
        self.prev_gray: Optional[np.ndarray] = None  # Previous frame (grayscale) for optical flow
        self.frame_count: int = 0  # Total frames processed
        self.fps_history: List[float] = []  # History of FPS for averaging
        
        # Performance metrics
        self.last_process_time: float = 0  # Time to process last frame (seconds)
        self.avg_fps: float = 0  # Average frames per second
    
    def add_point(self, point_id: str, x: float, y: float, frame: np.ndarray) -> TrackedPoint:
        """
        Add a new point annotation to track.
        
        Process:
        1. Convert frame to grayscale (optical flow needs grayscale)
        2. Extract template around point (for re-detection)
        3. Initialize Kalman filter (if enabled)
        4. Create TrackedPoint object
        5. Store in tracked_points dictionary
        
        Args:
            point_id: Unique identifier for this point
            x: X coordinate of point
            y: Y coordinate of point
            frame: Current video frame (BGR or grayscale)
            
        Returns:
            TrackedPoint object that was created
        """
        # Convert to grayscale if needed (optical flow requires grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Extract template: small image patch around point for re-detection
        # Template is used when tracking fails to search for the point
        template = self._extract_template(gray, x, y)
        
        # Initialize Kalman filter if enabled
        # Kalman filter smooths trajectories and reduces jitter
        kalman = KalmanPointTracker(x, y) if self.use_kalman else None
        
        # Create TrackedPoint data structure
        point = TrackedPoint(
            id=point_id,
            x=x, y=y,  # Current position
            original_x=x, original_y=y,  # Store original position
            template=template,  # Template for re-detection
            kalman=kalman,  # Kalman filter instance
            history=[(x, y)]  # Initialize history with starting position
        )
        
        # Store in dictionary for tracking
        self.tracked_points[point_id] = point
        return point
    
    def add_region(self, region_id: str, bbox: Tuple[int, int, int, int],
                   frame: np.ndarray, tracker_type: str = "CSRT") -> TrackedRegion:
        """
        Add a new bounding box region to track.
        
        Process:
        1. Create OpenCV tracker (CSRT/KCF/MOSSE)
        2. Initialize tracker with bounding box
        3. Extract template from region
        4. Create TrackedRegion object
        5. Store in tracked_regions dictionary
        
        Args:
            region_id: Unique identifier for this region
            bbox: Bounding box (x, y, width, height)
            frame: Current video frame
            tracker_type: "CSRT" (accurate), "KCF" (balanced), or "MOSSE" (fast)
            
        Returns:
            TrackedRegion object that was created
        """
        # Create appropriate OpenCV tracker based on type
        if tracker_type == "CSRT":
            # CSRT: Most accurate, uses spatial and color reliability
            tracker = cv2.TrackerCSRT_create()
        elif tracker_type == "KCF":
            # KCF: Balanced accuracy and speed, uses kernelized correlation
            tracker = cv2.TrackerKCF_create()
        else:  # MOSSE
            # MOSSE: Fastest, basic correlation filter
            tracker = cv2.legacy.TrackerMOSSE_create()
        
        # Initialize tracker with bounding box on first frame
        tracker.init(frame, bbox)
        
        # Extract template: full region image for re-detection
        x, y, w, h = bbox
        template = frame[y:y+h, x:x+w].copy()  # Copy region pixels
        
        # Create TrackedRegion data structure
        region = TrackedRegion(
            id=region_id,
            bbox=bbox,  # Current bounding box
            original_bbox=bbox,  # Store original bounding box
            tracker=tracker,  # OpenCV tracker instance
            template=template  # Region template
        )
        
        # Store in dictionary for tracking
        self.tracked_regions[region_id] = region
        return region
    
    def _extract_template(self, gray: np.ndarray, x: float, y: float) -> Optional[np.ndarray]:
        """
        Extract a template image patch around a point for re-detection.
        
        Template is a small square region (template_size x template_size) centered
        on the point. Used later to search for the point when tracking fails.
        
        Args:
            gray: Grayscale frame
            x: X coordinate of point
            y: Y coordinate of point
            
        Returns:
            Template image patch or None if too close to edge
        """
        h, w = gray.shape[:2]  # Frame dimensions
        half = self.template_size // 2  # Half template size (25 pixels)
        
        # Calculate template bounds, clamping to frame edges
        x1 = max(0, int(x) - half)  # Left edge
        y1 = max(0, int(y) - half)  # Top edge
        x2 = min(w, int(x) + half)  # Right edge
        y2 = min(h, int(y) + half)  # Bottom edge
        
        # Only return template if it's large enough (at least 10x10)
        if x2 - x1 > 10 and y2 - y1 > 10:
            return gray[y1:y2, x1:x2].copy()  # Extract and copy template
        return None  # Too close to edge
    
    def _find_template(self, gray: np.ndarray, template: np.ndarray,
                       search_region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[float, float, float]]:
        """
        Find template in frame using template matching.
        
        Uses normalized cross-correlation to find best match.
        Searches in specified region (or entire frame if None).
        
        Args:
            gray: Grayscale frame to search in
            template: Template image to find
            search_region: Optional (x1, y1, x2, y2) region to search, or None for full frame
            
        Returns:
            (x, y, confidence) tuple if found, or None if not found
            confidence is 0.0 to 1.0 (1.0 = perfect match)
        """
        if template is None:
            return None
        
        # Extract search area from frame
        if search_region:
            # Search in specified region (more efficient)
            x1, y1, x2, y2 = search_region
            search_area = gray[y1:y2, x1:x2]
            offset = (x1, y1)  # Offset to convert back to full frame coordinates
        else:
            # Search entire frame
            search_area = gray
            offset = (0, 0)
        
        # Check if search area is large enough
        if search_area.shape[0] < template.shape[0] or search_area.shape[1] < template.shape[1]:
            return None
        
        # Template matching using normalized cross-correlation
        # TM_CCOEFF_NORMED: Best match = highest value (0.0 to 1.0)
        result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
        
        # Find location of best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Only accept matches above confidence threshold (0.5 = 50% match)
        if max_val > 0.5:
            # Calculate center of matched region
            center_x = max_loc[0] + template.shape[1] // 2 + offset[0]
            center_y = max_loc[1] + template.shape[0] // 2 + offset[1]
            return (center_x, center_y, max_val)  # Return position and confidence
        
        return None  # No good match found
    
    def update(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Main update function: process new frame and update all tracked objects.
        
        This is called once per frame. It:
        1. Converts frame to grayscale
        2. Updates all points using optical flow
        3. Updates all regions using OpenCV trackers
        4. Performs periodic drift correction
        5. Calculates performance metrics
        
        Args:
            frame: New video frame (BGR or grayscale)
            
        Returns:
            Dictionary containing:
            - "points": Dict of point tracking results
            - "regions": Dict of region tracking results
            - "frame_count": Current frame number
            - "fps": Average processing FPS
            - "process_time_ms": Time to process this frame (milliseconds)
        """
        start_time = time.time()  # Start performance timer
        
        # Convert to grayscale if needed (optical flow requires grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h, w = gray.shape[:2]  # Frame dimensions
        
        # Initialize results dictionary
        results = {
            "points": {},  # Will contain point tracking results
            "regions": {},  # Will contain region tracking results
            "frame_count": self.frame_count,
            "fps": 0,
            "process_time_ms": 0
        }
        
        # Only update tracking if we have a previous frame
        # (Need previous frame for optical flow)
        if self.prev_gray is not None:
            # Update all tracked points using optical flow
            self._update_points_optical_flow(gray, h, w, results)
            
            # Update all tracked regions using OpenCV trackers
            self._update_regions(frame, results)
        
        # Periodic drift correction: every N frames, correct for accumulated errors
        # This prevents long-term drift by using template matching as absolute reference
        if self.frame_count > 0 and self.frame_count % self.drift_correction_interval == 0:
            self._correct_drift(gray)
        
        # Store current frame as previous for next iteration
        self.prev_gray = gray.copy()
        self.frame_count += 1
        
        # Calculate performance metrics
        process_time = time.time() - start_time  # Time to process frame
        self.last_process_time = process_time
        fps = 1.0 / process_time if process_time > 0 else 0  # FPS = 1 / time
        
        # Maintain rolling average of FPS (last 30 frames)
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)  # Remove oldest
        self.avg_fps = sum(self.fps_history) / len(self.fps_history)  # Average
        
        # Add metrics to results
        results["fps"] = self.avg_fps
        results["process_time_ms"] = process_time * 1000  # Convert to milliseconds
        
        return results
    
    def _update_points_optical_flow(self, gray: np.ndarray, h: int, w: int,
                                    results: Dict[str, Any]) -> None:
        """
        Update all tracked points using Lucas-Kanade optical flow.
        
        Optical flow tracks points by finding where they moved in the image.
        It uses image gradients to estimate motion between consecutive frames.
        
        Process:
        1. Collect all active points (TRACKING or REINITIALIZING state)
        2. Calculate optical flow from previous frame to current frame
        3. For each point:
           - If tracking successful: update position, apply Kalman filter
           - If tracking failed: mark as occluded, try recovery
           - If out of bounds: mark as occluded
        4. Update results dictionary
        
        Args:
            gray: Current grayscale frame
            h: Frame height
            w: Frame width
            results: Results dictionary to update
        """
        if not self.tracked_points:
            return  # No points to track
        
        # Collect all active points (only track points that are currently tracking)
        active_points = []  # List of [x, y] coordinates
        active_ids = []  # List of point IDs (parallel array)
        
        for pid, point in self.tracked_points.items():
            # Only track points that are actively tracking or reinitializing
            if point.state in [TrackingState.TRACKING, TrackingState.REINITIALIZING]:
                active_points.append([[point.x, point.y]])  # Current position
                active_ids.append(pid)  # Store ID for later
        
        if not active_points:
            return  # No active points to track
        
        # Convert to numpy array for OpenCV
        prev_pts = np.array(active_points, dtype=np.float32)
        
        # Calculate optical flow using Lucas-Kanade method
        # This finds where each point moved from previous frame to current frame
        # Returns:
        #   next_pts: New positions of points
        #   status: 1 if tracking successful, 0 if failed
        #   err: Tracking error (lower = better)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,  # Previous frame
            gray,  # Current frame
            prev_pts,  # Points to track
            None,  # No initial guess for new positions
            **self.lk_params  # Lucas-Kanade parameters
        )
        
        # Update each point based on optical flow results
        for i, (pid, st) in enumerate(zip(active_ids, status)):
            point = self.tracked_points[pid]
            
            if st[0] == 1:  # Tracking successful (status = 1)
                new_x, new_y = next_pts[i][0]  # Get new position from optical flow
                
                # Check if point is within frame bounds
                if 0 <= new_x < w and 0 <= new_y < h:
                    # Apply Kalman filter if enabled
                    # Kalman filter smooths the trajectory and reduces jitter
                    if self.use_kalman and point.kalman:
                        point.kalman.predict()  # Predict next position
                        new_x, new_y = point.kalman.update(new_x, new_y)  # Correct with measurement
                    
                    # Update point position
                    point.x = new_x
                    point.y = new_y
                    point.state = TrackingState.TRACKING  # Mark as successfully tracking
                    
                    # Calculate confidence from tracking error
                    # Lower error = higher confidence
                    # Error is normalized to 0-1 range (error/50, capped at 1.0)
                    point.confidence = 1.0 - (err[i][0] / 50.0 if err[i][0] < 50 else 1.0)
                    point.frames_lost = 0  # Reset lost frame counter
                    point.history.append((new_x, new_y))  # Add to position history
                    
                    # Keep history bounded (max 100 positions)
                    if len(point.history) > 100:
                        point.history.pop(0)  # Remove oldest
                else:
                    # Point moved out of frame bounds
                    point.state = TrackingState.OCCLUDED
                    point.frames_lost += 1
            else:
                # Optical flow tracking failed
                point.frames_lost += 1
                
                # Check if we've lost track for too long
                if point.frames_lost > self.max_lost_frames:
                    point.state = TrackingState.LOST  # Declare permanently lost
                else:
                    point.state = TrackingState.OCCLUDED  # Temporarily occluded
                    # Try to recover using template matching
                    self._try_recover_point(gray, point, h, w)
            
            # Add point result to output dictionary
            results["points"][pid] = {
                "x": point.x,
                "y": point.y,
                "state": point.state.value,  # Convert enum to string
                "confidence": point.confidence
            }
    
    def _try_recover_point(self, gray: np.ndarray, point: TrackedPoint,
                           h: int, w: int) -> bool:
        """
        Attempt to recover a lost point using template matching.
        
        When optical flow fails, we search for the point's template
        in a region around its last known position. If found with high
        confidence, we reinitialize tracking.
        
        Args:
            gray: Current grayscale frame
            point: Point to recover
            h: Frame height
            w: Frame width
            
        Returns:
            True if recovery successful, False otherwise
        """
        if point.template is None:
            return False  # No template available for matching
        
        # Define search region around last known position
        # Search margin: 100 pixels in each direction
        search_margin = 100
        x1 = max(0, int(point.x) - search_margin)  # Left bound
        y1 = max(0, int(point.y) - search_margin)  # Top bound
        x2 = min(w, int(point.x) + search_margin)  # Right bound
        y2 = min(h, int(point.y) + search_margin)  # Bottom bound
        
        # Search for template in this region
        result = self._find_template(gray, point.template, (x1, y1, x2, y2))
        
        if result:
            # Template found! Reinitialize tracking
            point.x, point.y, conf = result  # Update position and confidence
            point.confidence = conf
            point.state = TrackingState.REINITIALIZING  # Mark as recovering
            point.frames_lost = 0  # Reset lost counter
            return True
        
        return False  # Recovery failed
    
    def _update_regions(self, frame: np.ndarray, results: Dict[str, Any]) -> None:
        """
        Update all tracked regions using OpenCV trackers.
        
        Regions use dedicated OpenCV trackers (CSRT/KCF/MOSSE) which
        are optimized for bounding box tracking. These trackers use
        correlation filters and are more robust than optical flow
        for larger regions.
        
        Args:
            frame: Current video frame
            results: Results dictionary to update
        """
        for rid, region in self.tracked_regions.items():
            if region.tracker is None:
                continue  # Skip if no tracker initialized
            
            # Update OpenCV tracker with new frame
            # Returns success flag and new bounding box
            success, bbox = region.tracker.update(frame)
            
            if success:
                # Tracking successful
                region.bbox = tuple(map(int, bbox))  # Convert to integers
                region.state = TrackingState.TRACKING
                region.confidence = 1.0
            else:
                # Tracking failed
                region.state = TrackingState.LOST
                region.confidence = 0.0
            
            # Add region result to output dictionary
            results["regions"][rid] = {
                "bbox": region.bbox,
                "state": region.state.value,
                "confidence": region.confidence
            }
    
    def _correct_drift(self, gray: np.ndarray) -> None:
        """
        Periodic drift correction using template matching.
        
        Optical flow has small errors that accumulate over time (drift).
        This function corrects drift by periodically using template
        matching as an absolute reference. Runs every N frames.
        
        Process:
        1. For each actively tracking point
        2. Search for template in small region around current position
        3. If high-confidence match found, blend current position with match
        4. This prevents long-term error accumulation
        
        Args:
            gray: Current grayscale frame
        """
        h, w = gray.shape[:2]
        
        for point in self.tracked_points.values():
            # Only correct points that are currently tracking
            if point.state == TrackingState.TRACKING and point.template is not None:
                # Search in small region around current position (±30 pixels)
                # Small region because we expect only small drift
                search_margin = 30
                x1 = max(0, int(point.x) - search_margin)
                y1 = max(0, int(point.y) - search_margin)
                x2 = min(w, int(point.x) + search_margin)
                y2 = min(h, int(point.y) + search_margin)
                
                # Search for template
                result = self._find_template(gray, point.template, (x1, y1, x2, y2))
                
                if result and result[2] > 0.7:  # High confidence match (>70%)
                    # Blend current position with template match
                    # 70% current position, 30% template match
                    # This gradually corrects drift without sudden jumps
                    blend_factor = 0.3
                    point.x = point.x * (1 - blend_factor) + result[0] * blend_factor
                    point.y = point.y * (1 - blend_factor) + result[1] * blend_factor
    
    def remove_point(self, point_id: str) -> bool:
        """Remove a tracked point."""
        if point_id in self.tracked_points:
            del self.tracked_points[point_id]
            return True
        return False
    
    def remove_region(self, region_id: str) -> bool:
        """Remove a tracked region."""
        if region_id in self.tracked_regions:
            del self.tracked_regions[region_id]
            return True
        return False
    
    def clear_all(self) -> None:
        """Clear all tracked objects and reset state."""
        self.tracked_points.clear()
        self.tracked_regions.clear()
        self.prev_gray = None  # Reset frame history
        self.frame_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get tracking statistics.
        
        Returns:
            Dictionary with:
            - tracked_points: Number of tracked points
            - tracked_regions: Number of tracked regions
            - frame_count: Total frames processed
            - avg_fps: Average processing FPS
            - last_process_time_ms: Last frame processing time
            - points_by_state: Count of points in each state
        """
        return {
            "tracked_points": len(self.tracked_points),
            "tracked_regions": len(self.tracked_regions),
            "frame_count": self.frame_count,
            "avg_fps": self.avg_fps,
            "last_process_time_ms": self.last_process_time * 1000,
            "points_by_state": {
                state.value: sum(1 for p in self.tracked_points.values() if p.state == state)
                for state in TrackingState
            }
        }
```

---

*[Due to length limits, I'll continue with the remaining files in the next response. The pattern continues with similar detailed commenting for app.py, app.js, and demo_cli.py]*
