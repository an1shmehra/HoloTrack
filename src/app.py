"""
HoloRay Motion-Tracked Annotation - Web Application

Real-time web interface for:
- Video playback with tracked annotations
- Drawing annotations (points, boxes, freehand)
- Live tracking visualization
- Multi-user collaboration via WebSocket
"""

import cv2
import numpy as np
import base64
import time
import os
import json
from threading import Thread, Lock
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
from .tracker import HybridMotionTracker, TrackingState

app = Flask(__name__,
            template_folder='../templates',
            static_folder='../static')
app.config['SECRET_KEY'] = 'holoray-hackathon-2024'

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global state
class AppState:
    def __init__(self):
        self.tracker = HybridMotionTracker()
        self.video_capture = None
        self.current_frame = None
        self.is_playing = False
        self.video_path = None
        self.frame_lock = Lock()
        self.annotations = {}  # id -> annotation data
        self.video_fps = 30
        self.frame_count = 0
        self.total_frames = 0

state = AppState()


def encode_frame(frame: np.ndarray) -> str:
    """Encode frame to base64 JPEG."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def draw_annotations(frame: np.ndarray, tracking_results: dict) -> np.ndarray:
    """Draw all annotations on frame."""
    frame = frame.copy()

    # Draw tracked points
    for pid, data in tracking_results.get("points", {}).items():
        x, y = int(data["x"]), int(data["y"])
        confidence = data.get("confidence", 1.0)
        is_tracking = data["state"] == "tracking"

        # Color based on state
        if is_tracking:
            # Green with confidence-based alpha
            color = (0, int(255 * confidence), 0)
        elif data["state"] == "occluded":
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red

        # Draw annotation marker
        cv2.circle(frame, (x, y), 12, color, -1)
        cv2.circle(frame, (x, y), 14, (255, 255, 255), 2)

        # Draw label
        label = state.annotations.get(pid, {}).get("label", pid)
        cv2.putText(frame, label, (x + 15, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, label, (x + 15, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw tracked regions
    for rid, data in tracking_results.get("regions", {}).items():
        bbox = data["bbox"]
        is_tracking = data["state"] == "tracking"
        color = (0, 255, 0) if is_tracking else (0, 0, 255)

        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        label = state.annotations.get(rid, {}).get("label", rid)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


def video_processing_loop():
    """Main video processing loop running in background thread."""
    while True:
        if state.video_capture is None:
            socketio.sleep(0.05)
            continue

        # If playing, read next frame
        if state.is_playing:
            with state.frame_lock:
                ret, frame = state.video_capture.read()

                if not ret:
                    # Loop video
                    state.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    state.tracker.clear_all()
                    # Re-add annotations from stored state
                    ret, frame = state.video_capture.read()
                    if ret:
                        for aid, ann in state.annotations.items():
                            if ann["type"] == "point":
                                state.tracker.add_point(aid, ann["x"], ann["y"], frame)
                            elif ann["type"] == "region":
                                state.tracker.add_region(aid, tuple(ann["bbox"]), frame)
                    state.frame_count = 0
                    if not ret:
                        socketio.sleep(0.05)
                        continue

                state.current_frame = frame.copy()
                state.frame_count += 1

            # Update tracking
            tracking_results = state.tracker.update(frame)
        else:
            # When paused, use current frame and still update tracking if we have annotations
            with state.frame_lock:
                if state.current_frame is None:
                    socketio.sleep(0.05)
                    continue
                frame = state.current_frame.copy()

            # Update tracking on current frame (for visualization)
            if len(state.annotations) > 0:
                tracking_results = state.tracker.update(frame)
            else:
                tracking_results = {"points": {}, "regions": {}, "fps": 0, "process_time_ms": 0}

        # Draw annotations
        annotated_frame = draw_annotations(frame, tracking_results)

        # Add HUD
        h, w = annotated_frame.shape[:2]

        # FPS counter
        fps_text = f"FPS: {tracking_results.get('fps', 0):.1f}"
        cv2.putText(annotated_frame, fps_text, (w - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Frame counter
        progress = f"Frame: {state.frame_count}/{state.total_frames}"
        cv2.putText(annotated_frame, progress, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Tracking stats
        stats = state.tracker.get_stats()
        points_tracking = stats["points_by_state"].get("tracking", 0)
        points_total = stats["tracked_points"]
        status_text = f"Tracking: {points_tracking}/{points_total}"
        cv2.putText(annotated_frame, status_text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Encode and emit
        frame_data = encode_frame(annotated_frame)

        socketio.emit('frame', {
            'image': frame_data,
            'tracking': tracking_results,
            'frame_num': state.frame_count,
            'total_frames': state.total_frames
        })

        # Control frame rate - slower when paused
        if state.is_playing:
            socketio.sleep(1.0 / state.video_fps)
        else:
            socketio.sleep(0.1)  # Update paused frame every 100ms


@app.route('/')
def index():
    """Serve main page."""
    return render_template('index.html')


@app.route('/api/videos')
def list_videos():
    """List available videos in the videos directory (recursively)."""
    videos_dir = os.path.join(os.path.dirname(__file__), '..', 'videos')
    videos = []

    if os.path.exists(videos_dir):
        # Walk through all subdirectories
        for root, dirs, files in os.walk(videos_dir):
            for f in files:
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    # Get relative path from videos_dir
                    rel_path = os.path.relpath(os.path.join(root, f), videos_dir)
                    # Use forward slashes for web compatibility
                    rel_path = rel_path.replace('\\', '/')
                    videos.append(rel_path)

    return jsonify(videos)


@app.route('/api/load_video', methods=['POST'])
def load_video():
    """Load a video file."""
    data = request.json
    video_name = data.get('video')

    if not video_name:
        return jsonify({'error': 'No video specified'}), 400

    videos_dir = os.path.join(os.path.dirname(__file__), '..', 'videos')
    # Normalize path separators (handle both / and \)
    video_name = video_name.replace('/', os.sep).replace('\\', os.sep)
    video_path = os.path.join(videos_dir, video_name)

    if not os.path.exists(video_path):
        return jsonify({'error': 'Video not found'}), 404

    with state.frame_lock:
        if state.video_capture:
            state.video_capture.release()

        state.video_capture = cv2.VideoCapture(video_path)
        state.video_path = video_path
        state.video_fps = state.video_capture.get(cv2.CAP_PROP_FPS) or 30
        state.total_frames = int(state.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        state.frame_count = 0
        state.tracker.clear_all()
        state.annotations.clear()

        # Read first frame
        ret, frame = state.video_capture.read()
        if ret:
            state.current_frame = frame
            state.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return jsonify({
        'success': True,
        'fps': state.video_fps,
        'total_frames': state.total_frames,
        'width': int(state.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(state.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    })


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print(f"Client connected: {request.sid}")
    emit('status', {'message': 'Connected to HoloRay Tracker'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print(f"Client disconnected: {request.sid}")


@socketio.on('play')
def handle_play():
    """Start video playback."""
    state.is_playing = True
    emit('status', {'playing': True})


@socketio.on('pause')
def handle_pause():
    """Pause video playback."""
    state.is_playing = False
    emit('status', {'playing': False})


@socketio.on('add_point')
def handle_add_point(data):
    """Add a new point annotation."""
    point_id = data.get('id', f"point_{len(state.annotations)}")
    x = int(data['x'])
    y = int(data['y'])
    label = data.get('label', point_id)

    with state.frame_lock:
        if state.current_frame is not None:
            state.tracker.add_point(point_id, x, y, state.current_frame)
            state.annotations[point_id] = {
                'type': 'point',
                'x': x,
                'y': y,
                'label': label
            }
            # Send updated frame immediately
            tracking_results = state.tracker.update(state.current_frame)
            annotated_frame = draw_annotations(state.current_frame, tracking_results)
            frame_data = encode_frame(annotated_frame)
            emit('frame', {
                'image': frame_data,
                'tracking': tracking_results,
                'frame_num': state.frame_count,
                'total_frames': state.total_frames
            }, broadcast=True)

    emit('annotation_added', {
        'id': point_id,
        'type': 'point',
        'x': x,
        'y': y,
        'label': label
    }, broadcast=True)


@socketio.on('add_region')
def handle_add_region(data):
    """Add a new region annotation."""
    region_id = data.get('id', f"region_{len(state.annotations)}")
    bbox = (int(data['x']), int(data['y']), int(data['width']), int(data['height']))
    label = data.get('label', region_id)
    tracker_type = data.get('tracker', 'CSRT')

    with state.frame_lock:
        if state.current_frame is not None:
            state.tracker.add_region(region_id, bbox, state.current_frame, tracker_type)
            state.annotations[region_id] = {
                'type': 'region',
                'bbox': list(bbox),
                'label': label
            }
            # Send updated frame immediately
            tracking_results = state.tracker.update(state.current_frame)
            annotated_frame = draw_annotations(state.current_frame, tracking_results)
            frame_data = encode_frame(annotated_frame)
            emit('frame', {
                'image': frame_data,
                'tracking': tracking_results,
                'frame_num': state.frame_count,
                'total_frames': state.total_frames
            }, broadcast=True)

    emit('annotation_added', {
        'id': region_id,
        'type': 'region',
        'bbox': list(bbox),
        'label': label
    }, broadcast=True)


@socketio.on('remove_annotation')
def handle_remove_annotation(data):
    """Remove an annotation."""
    ann_id = data['id']

    with state.frame_lock:
        if ann_id in state.annotations:
            ann_type = state.annotations[ann_id]['type']
            if ann_type == 'point':
                state.tracker.remove_point(ann_id)
            else:
                state.tracker.remove_region(ann_id)
            del state.annotations[ann_id]
            # Send updated frame
            if state.current_frame is not None:
                if len(state.annotations) > 0:
                    tracking_results = state.tracker.update(state.current_frame)
                else:
                    tracking_results = {"points": {}, "regions": {}, "fps": 0, "process_time_ms": 0}
                annotated_frame = draw_annotations(state.current_frame, tracking_results)
                frame_data = encode_frame(annotated_frame)
                emit('frame', {
                    'image': frame_data,
                    'tracking': tracking_results,
                    'frame_num': state.frame_count,
                    'total_frames': state.total_frames
                }, broadcast=True)

    emit('annotation_removed', {'id': ann_id}, broadcast=True)


@socketio.on('clear_annotations')
def handle_clear_annotations():
    """Clear all annotations."""
    with state.frame_lock:
        state.tracker.clear_all()
        state.annotations.clear()
        # Send updated frame
        if state.current_frame is not None:
            tracking_results = {"points": {}, "regions": {}, "fps": 0, "process_time_ms": 0}
            annotated_frame = draw_annotations(state.current_frame, tracking_results)
            frame_data = encode_frame(annotated_frame)
            emit('frame', {
                'image': frame_data,
                'tracking': tracking_results,
                'frame_num': state.frame_count,
                'total_frames': state.total_frames
            }, broadcast=True)

    emit('annotations_cleared', broadcast=True)


@socketio.on('get_first_frame')
def handle_get_first_frame():
    """Get the first frame of the loaded video."""
    with state.frame_lock:
        if state.current_frame is not None:
            frame_data = encode_frame(state.current_frame)
            emit('first_frame', {'image': frame_data})


@socketio.on('seek')
def handle_seek(data):
    """Seek to a specific frame."""
    frame_num = data.get('frame', 0)

    with state.frame_lock:
        if state.video_capture:
            state.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            state.frame_count = frame_num
            state.tracker.clear_all()

            ret, frame = state.video_capture.read()
            if ret:
                state.current_frame = frame
                # Re-add annotations
                for aid, ann in state.annotations.items():
                    if ann["type"] == "point":
                        state.tracker.add_point(aid, ann["x"], ann["y"], frame)
                    elif ann["type"] == "region":
                        state.tracker.add_region(aid, tuple(ann["bbox"]), frame)

                frame_data = encode_frame(frame)
                emit('first_frame', {'image': frame_data})

            # Reset position for playback
            state.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)


def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the web server."""
    # Start video processing thread
    video_thread = Thread(target=video_processing_loop, daemon=True)
    video_thread.start()

    print(f"\n{'='*50}")
    print(f"  HoloRay Motion-Tracked Annotation System")
    print(f"  Open http://localhost:{port} in your browser")
    print(f"{'='*50}\n")

    socketio.run(app, host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
