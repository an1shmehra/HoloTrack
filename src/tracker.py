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

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import time


class TrackingState(Enum):
    TRACKING = "tracking"
    LOST = "lost"
    OCCLUDED = "occluded"
    REINITIALIZING = "reinitializing"


@dataclass
class TrackedPoint:
    """Represents a tracked annotation point."""
    id: str
    x: float
    y: float
    original_x: float
    original_y: float
    state: TrackingState = TrackingState.TRACKING
    confidence: float = 1.0
    frames_lost: int = 0
    template: Optional[np.ndarray] = None
    kalman: Optional[cv2.KalmanFilter] = None
    history: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class TrackedRegion:
    """Represents a tracked bounding box region."""
    id: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    original_bbox: Tuple[int, int, int, int]
    state: TrackingState = TrackingState.TRACKING
    confidence: float = 1.0
    tracker: Any = None
    template: Optional[np.ndarray] = None


class KalmanPointTracker:
    """Kalman filter for smoothing point trajectories."""

    def __init__(self, initial_x: float, initial_y: float):
        self.kf = cv2.KalmanFilter(4, 2)  # 4 state vars (x, y, vx, vy), 2 measurements (x, y)

        # State transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Process noise covariance
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        # Initial state
        self.kf.statePost = np.array([[initial_x], [initial_y], [0], [0]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def predict(self) -> Tuple[float, float]:
        prediction = self.kf.predict()
        return float(prediction[0, 0]), float(prediction[1, 0])

    def update(self, x: float, y: float) -> Tuple[float, float]:
        measurement = np.array([[x], [y]], dtype=np.float32)
        corrected = self.kf.correct(measurement)
        return float(corrected[0, 0]), float(corrected[1, 0])


class HybridMotionTracker:
    """
    Hybrid motion tracking system optimized for medical imaging.

    Features:
    - Multi-point optical flow tracking
    - Automatic drift correction
    - Occlusion detection and recovery
    - Kalman filter smoothing
    - Template-based re-detection
    """

    def __init__(self,
                 use_kalman: bool = True,
                 drift_correction_interval: int = 30,
                 max_lost_frames: int = 15,
                 template_size: int = 50,
                 optical_flow_quality: float = 0.01):
        """
        Initialize the tracker.

        Args:
            use_kalman: Enable Kalman filter smoothing
            drift_correction_interval: Frames between drift corrections
            max_lost_frames: Max frames before declaring track lost
            template_size: Size of template for re-detection
            optical_flow_quality: Quality threshold for optical flow
        """
        self.use_kalman = use_kalman
        self.drift_correction_interval = drift_correction_interval
        self.max_lost_frames = max_lost_frames
        self.template_size = template_size

        # Optical flow parameters (Lucas-Kanade)
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # Feature detection parameters (Shi-Tomasi corners)
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=optical_flow_quality,
            minDistance=10,
            blockSize=7
        )

        # Tracked objects
        self.tracked_points: Dict[str, TrackedPoint] = {}
        self.tracked_regions: Dict[str, TrackedRegion] = {}

        # Frame management
        self.prev_gray: Optional[np.ndarray] = None
        self.frame_count: int = 0
        self.fps_history: List[float] = []

        # Performance metrics
        self.last_process_time: float = 0
        self.avg_fps: float = 0

    def add_point(self, point_id: str, x: float, y: float, frame: np.ndarray) -> TrackedPoint:
        """Add a new point to track."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Create template for re-detection
        template = self._extract_template(gray, x, y)

        # Initialize Kalman filter if enabled
        kalman = KalmanPointTracker(x, y) if self.use_kalman else None

        point = TrackedPoint(
            id=point_id,
            x=x, y=y,
            original_x=x, original_y=y,
            template=template,
            kalman=kalman,
            history=[(x, y)]
        )

        self.tracked_points[point_id] = point
        return point

    def add_region(self, region_id: str, bbox: Tuple[int, int, int, int],
                   frame: np.ndarray, tracker_type: str = "CSRT") -> TrackedRegion:
        """Add a new bounding box region to track."""
        # Create OpenCV tracker
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
        elif tracker_type == "KCF":
            tracker = cv2.TrackerKCF_create()
        else:  # MOSSE - fastest
            tracker = cv2.legacy.TrackerMOSSE_create()

        tracker.init(frame, bbox)

        # Extract template for re-detection
        x, y, w, h = bbox
        template = frame[y:y+h, x:x+w].copy()

        region = TrackedRegion(
            id=region_id,
            bbox=bbox,
            original_bbox=bbox,
            tracker=tracker,
            template=template
        )

        self.tracked_regions[region_id] = region
        return region

    def _extract_template(self, gray: np.ndarray, x: float, y: float) -> Optional[np.ndarray]:
        """Extract a template region around a point for re-detection."""
        h, w = gray.shape[:2]
        half = self.template_size // 2

        x1 = max(0, int(x) - half)
        y1 = max(0, int(y) - half)
        x2 = min(w, int(x) + half)
        y2 = min(h, int(y) + half)

        if x2 - x1 > 10 and y2 - y1 > 10:
            return gray[y1:y2, x1:x2].copy()
        return None

    def _find_template(self, gray: np.ndarray, template: np.ndarray,
                       search_region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[float, float, float]]:
        """Find template in frame using template matching. Returns (x, y, confidence)."""
        if template is None:
            return None

        if search_region:
            x1, y1, x2, y2 = search_region
            search_area = gray[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            search_area = gray
            offset = (0, 0)

        if search_area.shape[0] < template.shape[0] or search_area.shape[1] < template.shape[1]:
            return None

        result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > 0.5:  # Confidence threshold
            center_x = max_loc[0] + template.shape[1] // 2 + offset[0]
            center_y = max_loc[1] + template.shape[0] // 2 + offset[1]
            return (center_x, center_y, max_val)

        return None

    def update(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Update all tracked objects with new frame.

        Returns dict with tracking results and performance metrics.
        """
        start_time = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h, w = gray.shape[:2]

        results = {
            "points": {},
            "regions": {},
            "frame_count": self.frame_count,
            "fps": 0,
            "process_time_ms": 0
        }

        if self.prev_gray is not None:
            # Update points using optical flow
            self._update_points_optical_flow(gray, h, w, results)

            # Update regions using OpenCV trackers
            self._update_regions(frame, results)

        # Periodic drift correction
        if self.frame_count > 0 and self.frame_count % self.drift_correction_interval == 0:
            self._correct_drift(gray)

        # Store frame for next iteration
        self.prev_gray = gray.copy()
        self.frame_count += 1

        # Calculate performance metrics
        process_time = time.time() - start_time
        self.last_process_time = process_time
        fps = 1.0 / process_time if process_time > 0 else 0
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        self.avg_fps = sum(self.fps_history) / len(self.fps_history)

        results["fps"] = self.avg_fps
        results["process_time_ms"] = process_time * 1000

        return results

    def _update_points_optical_flow(self, gray: np.ndarray, h: int, w: int,
                                    results: Dict[str, Any]) -> None:
        """Update tracked points using Lucas-Kanade optical flow."""
        if not self.tracked_points:
            return

        # Prepare points for optical flow
        active_points = []
        active_ids = []

        for pid, point in self.tracked_points.items():
            if point.state in [TrackingState.TRACKING, TrackingState.REINITIALIZING]:
                active_points.append([[point.x, point.y]])
                active_ids.append(pid)

        if not active_points:
            return

        prev_pts = np.array(active_points, dtype=np.float32)

        # Calculate optical flow
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.lk_params
        )

        # Update each point
        for i, (pid, st) in enumerate(zip(active_ids, status)):
            point = self.tracked_points[pid]

            if st[0] == 1:  # Tracking successful
                new_x, new_y = next_pts[i][0]

                # Check if point is within frame bounds
                if 0 <= new_x < w and 0 <= new_y < h:
                    # Apply Kalman filter if enabled
                    if self.use_kalman and point.kalman:
                        point.kalman.predict()
                        new_x, new_y = point.kalman.update(new_x, new_y)

                    point.x = new_x
                    point.y = new_y
                    point.state = TrackingState.TRACKING
                    point.confidence = 1.0 - (err[i][0] / 50.0 if err[i][0] < 50 else 1.0)
                    point.frames_lost = 0
                    point.history.append((new_x, new_y))

                    # Keep history bounded
                    if len(point.history) > 100:
                        point.history.pop(0)
                else:
                    # Point moved out of frame
                    point.state = TrackingState.OCCLUDED
                    point.frames_lost += 1
            else:
                # Tracking failed
                point.frames_lost += 1
                if point.frames_lost > self.max_lost_frames:
                    point.state = TrackingState.LOST
                else:
                    point.state = TrackingState.OCCLUDED
                    # Try template matching to recover
                    self._try_recover_point(gray, point, h, w)

            results["points"][pid] = {
                "x": point.x,
                "y": point.y,
                "state": point.state.value,
                "confidence": point.confidence
            }

    def _try_recover_point(self, gray: np.ndarray, point: TrackedPoint,
                           h: int, w: int) -> bool:
        """Try to recover a lost point using template matching."""
        if point.template is None:
            return False

        # Define search region around last known position
        search_margin = 100
        x1 = max(0, int(point.x) - search_margin)
        y1 = max(0, int(point.y) - search_margin)
        x2 = min(w, int(point.x) + search_margin)
        y2 = min(h, int(point.y) + search_margin)

        result = self._find_template(gray, point.template, (x1, y1, x2, y2))

        if result:
            point.x, point.y, conf = result
            point.confidence = conf
            point.state = TrackingState.REINITIALIZING
            point.frames_lost = 0
            return True

        return False

    def _update_regions(self, frame: np.ndarray, results: Dict[str, Any]) -> None:
        """Update tracked regions using OpenCV trackers."""
        for rid, region in self.tracked_regions.items():
            if region.tracker is None:
                continue

            success, bbox = region.tracker.update(frame)

            if success:
                region.bbox = tuple(map(int, bbox))
                region.state = TrackingState.TRACKING
                region.confidence = 1.0
            else:
                region.state = TrackingState.LOST
                region.confidence = 0.0

            results["regions"][rid] = {
                "bbox": region.bbox,
                "state": region.state.value,
                "confidence": region.confidence
            }

    def _correct_drift(self, gray: np.ndarray) -> None:
        """Periodic drift correction using template matching."""
        h, w = gray.shape[:2]

        for point in self.tracked_points.values():
            if point.state == TrackingState.TRACKING and point.template is not None:
                # Search in a small region around current position
                search_margin = 30
                x1 = max(0, int(point.x) - search_margin)
                y1 = max(0, int(point.y) - search_margin)
                x2 = min(w, int(point.x) + search_margin)
                y2 = min(h, int(point.y) + search_margin)

                result = self._find_template(gray, point.template, (x1, y1, x2, y2))

                if result and result[2] > 0.7:  # High confidence match
                    # Blend current position with template match
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
        """Clear all tracked objects."""
        self.tracked_points.clear()
        self.tracked_regions.clear()
        self.prev_gray = None
        self.frame_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics."""
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


class DenseFlowTracker:
    """
    Dense optical flow tracker for global motion compensation.
    Useful for handling camera/probe motion in medical imaging.
    """

    def __init__(self):
        self.prev_gray = None
        self.cumulative_transform = np.eye(3, dtype=np.float32)

    def compute_global_motion(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute global motion transform between frames.
        Returns 3x3 homography matrix or None if first frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return None

        # Detect features in previous frame
        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=200,
                                            qualityLevel=0.01, minDistance=10)

        if prev_pts is None or len(prev_pts) < 4:
            self.prev_gray = gray.copy()
            return None

        # Track features to current frame
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts, None)

        # Filter valid points
        valid_prev = prev_pts[status.flatten() == 1]
        valid_next = next_pts[status.flatten() == 1]

        if len(valid_prev) < 4:
            self.prev_gray = gray.copy()
            return None

        # Estimate homography
        H, mask = cv2.findHomography(valid_prev, valid_next, cv2.RANSAC, 5.0)

        if H is not None:
            self.cumulative_transform = H @ self.cumulative_transform

        self.prev_gray = gray.copy()
        return H

    def transform_point(self, x: float, y: float, H: np.ndarray) -> Tuple[float, float]:
        """Transform a point using homography matrix."""
        pt = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, H)
        return float(transformed[0, 0, 0]), float(transformed[0, 0, 1])


# Convenience function for quick single-video tracking
def track_video(video_path: str, annotations: List[Tuple[float, float]],
                output_path: Optional[str] = None, display: bool = True) -> None:
    """
    Track annotations through a video file.

    Args:
        video_path: Path to input video
        annotations: List of (x, y) points to track
        output_path: Optional path to save output video
        display: Whether to display tracking in real-time
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize tracker
    tracker = HybridMotionTracker()

    # Video writer if output specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process first frame
    ret, frame = cap.read()
    if not ret:
        return

    # Add annotation points
    for i, (x, y) in enumerate(annotations):
        tracker.add_point(f"point_{i}", x, y, frame)

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update tracking
        results = tracker.update(frame)

        # Draw annotations
        for pid, data in results["points"].items():
            x, y = int(data["x"]), int(data["y"])
            color = (0, 255, 0) if data["state"] == "tracking" else (0, 0, 255)
            cv2.circle(frame, (x, y), 8, color, -1)
            cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)

        # Draw FPS
        cv2.putText(frame, f"FPS: {results['fps']:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if writer:
            writer.write(frame)

        if display:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_num += 1

    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    print(f"Processed {frame_num} frames. Avg FPS: {tracker.avg_fps:.1f}")
