"""
HoloRay Motion-Tracked Annotation - Web Application
Uses the robust multi-ring tracker with template matching recovery
"""

import eventlet
eventlet.monkey_patch()

import cv2
import numpy as np
import base64
import time
import os
from threading import Lock
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

app = Flask(__name__,
            template_folder='../templates',
            static_folder='../static')
app.config['SECRET_KEY'] = 'holoray-hackathon-2024'

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# =========================
# CONFIGURATION
# =========================
SEARCH_WINDOW_SIZE = 80
CONFIDENCE_THRESHOLD = 0.65
REPLENISH_THRESHOLD = 30

CENTER_RADIUS = 25
MID_RADIUS = 45
OUTER_RADIUS = 70

MAX_FEATURES_TOTAL = 120
OUTLIER_THRESHOLD = 18


class TrackerState:
    """Encapsulates all tracking state"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.annotation_point = None
        self.template_patch = None
        self.annotation_trackers = []
        self.prev_gray = None
        self.is_lost = False
        self.last_known_pos = None

        # Strokes stored as offsets from anchor
        self.strokes = []
        self.current_stroke = []
        self.drawing = False


class AppState:
    """Global application state"""
    def __init__(self):
        self.tracker = TrackerState()
        self.video_capture = None
        self.current_frame = None
        self.is_playing = False
        self.video_path = None
        self.frame_lock = Lock()
        self.video_fps = 30
        self.frame_count = 0
        self.total_frames = 0
        self.playback_speed = 1.0

        # Performance metrics
        self.last_time = time.time()
        self.fps_counter = 0
        self.actual_fps = 0


state = AppState()


# =========================
# TRACKER FUNCTIONS
# =========================
def replenish_trackers(gray, pos):
    """Multi-ring dense feature sampling around anchor"""
    x, y = int(pos[0]), int(pos[1])
    h, w = gray.shape[:2]

    def sample(radius, count):
        mask = np.zeros_like(gray)
        cv2.circle(mask, (x, y), radius, 255, -1)
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=count,
            qualityLevel=0.01,
            minDistance=4,
            mask=mask
        )

    pts = []
    for r, c in [(CENTER_RADIUS, 30), (MID_RADIUS, 40), (OUTER_RADIUS, 50)]:
        p = sample(r, c)
        if p is not None:
            pts.extend([pt.reshape(1, 2) for pt in p])

    state.tracker.annotation_trackers = pts[:MAX_FEATURES_TOTAL]


def set_anchor(x, y, gray):
    """Set anchor point for tracking"""
    state.tracker.annotation_point = (float(x), float(y))
    state.tracker.last_known_pos = state.tracker.annotation_point
    state.tracker.is_lost = False

    patch = 25
    h, w = gray.shape[:2]
    y1, y2 = max(0, y - patch), min(h, y + patch)
    x1, x2 = max(0, x - patch), min(w, x + patch)
    state.tracker.template_patch = gray[y1:y2, x1:x2].copy()

    replenish_trackers(gray, state.tracker.annotation_point)


def update_tracking(gray):
    """Update tracker with new frame"""
    tracker = state.tracker

    if tracker.annotation_point is None or tracker.template_patch is None:
        tracker.prev_gray = gray.copy()
        return

    h, w = gray.shape[:2]

    if tracker.is_lost:
        # Template matching recovery
        lx, ly = tracker.last_known_pos
        r = SEARCH_WINDOW_SIZE
        y1, y2 = int(max(0, ly - r)), int(min(h, ly + r))
        x1, x2 = int(max(0, lx - r)), int(min(w, lx + r))
        search = gray[y1:y2, x1:x2]

        if search.shape[0] > tracker.template_patch.shape[0] and \
           search.shape[1] > tracker.template_patch.shape[1]:
            res = cv2.matchTemplate(search, tracker.template_patch, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > CONFIDENCE_THRESHOLD:
                nx = x1 + max_loc[0] + tracker.template_patch.shape[1] // 2
                ny = y1 + max_loc[1] + tracker.template_patch.shape[0] // 2
                tracker.annotation_point = (float(nx), float(ny))
                tracker.last_known_pos = tracker.annotation_point
                replenish_trackers(gray, tracker.annotation_point)
                tracker.is_lost = False

    else:
        dx_list, dy_list = [], []

        if tracker.annotation_trackers and tracker.prev_gray is not None:
            pts = np.array(tracker.annotation_trackers, dtype=np.float32)
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(tracker.prev_gray, gray, pts, None)

            good_new = new_pts[status == 1]
            good_old = pts[status == 1]

            if len(good_new) > 0:
                movement = good_new - good_old
                dist = np.linalg.norm(movement, axis=1)
                valid = dist < OUTLIER_THRESHOLD
                movement = movement[valid]
                good_new = good_new[valid]

                tracker.annotation_trackers = [p.reshape(1, 2) for p in good_new]

                if len(movement) > 0:
                    dx_list = movement[:, 0]
                    dy_list = movement[:, 1]

        if len(dx_list) > 0:
            dx = np.clip(np.median(dx_list), -8, 8)
            dy = np.clip(np.median(dy_list), -8, 8)
            tracker.annotation_point = (
                tracker.annotation_point[0] + dx,
                tracker.annotation_point[1] + dy
            )

        x, y = tracker.annotation_point

        # Check if lost
        if len(tracker.annotation_trackers) < 3 or not (10 < x < w - 10 and 10 < y < h - 10):
            tracker.is_lost = True
        else:
            tracker.last_known_pos = tracker.annotation_point

            # Drift correction every 5 frames
            if state.frame_count % 5 == 0:
                r = 40
                y1, y2 = int(max(0, y - r)), int(min(h, y + r))
                x1, x2 = int(max(0, x - r)), int(min(w, x + r))
                patch = gray[y1:y2, x1:x2]

                if patch.shape[0] > tracker.template_patch.shape[0] and \
                   patch.shape[1] > tracker.template_patch.shape[1]:
                    res = cv2.matchTemplate(patch, tracker.template_patch, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val > CONFIDENCE_THRESHOLD:
                        tracker.annotation_point = (
                            x1 + max_loc[0] + tracker.template_patch.shape[1] // 2,
                            y1 + max_loc[1] + tracker.template_patch.shape[0] // 2
                        )

            # Replenish trackers if needed
            if len(tracker.annotation_trackers) < REPLENISH_THRESHOLD:
                replenish_trackers(gray, tracker.annotation_point)

    tracker.prev_gray = gray.copy()


def draw_annotations(frame):
    """Draw all annotations on frame"""
    tracker = state.tracker
    display = frame.copy()

    if tracker.annotation_point and not tracker.is_lost:
        ax, ay = int(tracker.annotation_point[0]), int(tracker.annotation_point[1])

        # Draw anchor point
        cv2.circle(display, (ax, ay), 8, (0, 255, 0), -1)
        cv2.circle(display, (ax, ay), 10, (255, 255, 255), 2)

        # Draw strokes
        for stroke in tracker.strokes + ([tracker.current_stroke] if tracker.drawing else []):
            if len(stroke) < 2:
                continue
            pts = np.array(stroke) + np.array(tracker.annotation_point)
            cv2.polylines(display, [pts.astype(np.int32)], False, (0, 0, 255), 2)

    if tracker.is_lost and tracker.last_known_pos:
        lx, ly = int(tracker.last_known_pos[0]), int(tracker.last_known_pos[1])
        cv2.putText(display, "SEARCHING", (lx - 50, ly - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return display


def encode_frame(frame):
    """Encode frame to base64 JPEG"""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


# =========================
# VIDEO PROCESSING LOOP
# =========================
def video_processing_loop():
    """Main video processing loop running in background greenlet"""
    while True:
        if not state.is_playing or state.video_capture is None:
            eventlet.sleep(0.05)
            continue

        with state.frame_lock:
            ret, frame = state.video_capture.read()

            if not ret:
                # Loop video
                state.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                state.tracker.reset()
                state.frame_count = 0
                ret, frame = state.video_capture.read()
                if not ret:
                    continue

            state.current_frame = frame.copy()
            state.frame_count += 1

            # Convert to grayscale for tracking
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Update tracking
            update_tracking(gray)

        # Draw annotations
        display = draw_annotations(frame)

        # Calculate FPS
        state.fps_counter += 1
        current_time = time.time()
        if current_time - state.last_time >= 1.0:
            state.actual_fps = state.fps_counter
            state.fps_counter = 0
            state.last_time = current_time

        # Add HUD
        h, w = display.shape[:2]
        tracker = state.tracker

        # Status badge
        status = "LOST" if tracker.is_lost else f"TRACKING ({len(tracker.annotation_trackers)} pts)"
        color = (0, 0, 255) if tracker.is_lost else (0, 255, 0)
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # FPS
        cv2.putText(display, f"FPS: {state.actual_fps}", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Frame counter
        cv2.putText(display, f"Frame: {state.frame_count}/{state.total_frames}",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Encode and emit
        frame_data = encode_frame(display)

        socketio.emit('frame', {
            'image': frame_data,
            'tracking': {
                'is_lost': tracker.is_lost,
                'tracking_points': len(tracker.annotation_trackers),
                'fps': state.actual_fps
            },
            'frame_num': state.frame_count,
            'total_frames': state.total_frames
        })

        # Control frame rate
        delay = (1.0 / state.video_fps) / state.playback_speed
        eventlet.sleep(delay)


# =========================
# ROUTES
# =========================
@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')


@app.route('/api/videos')
def list_videos():
    """List available videos"""
    videos = []
    search_dirs = [
        os.path.join(os.path.dirname(__file__), '..', 'videos'),
        os.path.join(os.path.dirname(__file__), '..', 'data')
    ]

    for videos_dir in search_dirs:
        if os.path.exists(videos_dir):
            for root, dirs, files in os.walk(videos_dir):
                for f in files:
                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        rel_path = os.path.relpath(os.path.join(root, f),
                                                    os.path.dirname(__file__) + '/..')
                        videos.append(rel_path)

    return jsonify(videos)


@app.route('/api/load_video', methods=['POST'])
def load_video():
    """Load a video file"""
    data = request.json
    video_name = data.get('video')

    if not video_name:
        return jsonify({'error': 'No video specified'}), 400

    base_dir = os.path.join(os.path.dirname(__file__), '..')
    video_path = os.path.join(base_dir, video_name)

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
        state.tracker.reset()

        # Read first frame
        ret, frame = state.video_capture.read()
        if ret:
            state.current_frame = frame
            state.tracker.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            state.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return jsonify({
        'success': True,
        'fps': state.video_fps,
        'total_frames': state.total_frames,
        'width': int(state.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(state.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    })


# =========================
# SOCKET EVENTS
# =========================
@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    emit('status', {'message': 'Connected to HoloRay'})


@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")


@socketio.on('play')
def handle_play():
    state.is_playing = True
    emit('status', {'playing': True})


@socketio.on('pause')
def handle_pause():
    state.is_playing = False
    emit('status', {'playing': False})


@socketio.on('add_point')
def handle_add_point(data):
    """Add anchor point"""
    x = int(data['x'])
    y = int(data['y'])

    with state.frame_lock:
        if state.current_frame is not None:
            gray = cv2.cvtColor(state.current_frame, cv2.COLOR_BGR2GRAY)
            set_anchor(x, y, gray)

    emit('annotation_added', {
        'id': 'anchor',
        'type': 'point',
        'x': x,
        'y': y,
        'label': data.get('label', 'Anchor')
    }, broadcast=True)

    # Show anchor immediately
    if not state.is_playing:
        send_current_frame()


def send_current_frame():
    """Send the current frame with annotations to client"""
    if state.current_frame is None:
        return
    display = draw_annotations(state.current_frame)
    frame_data = encode_frame(display)
    emit('first_frame', {'image': frame_data})


@socketio.on('start_drawing')
def handle_start_drawing():
    """Start a new drawing stroke"""
    tracker = state.tracker
    if tracker.annotation_point is not None:
        tracker.drawing = True
        tracker.current_stroke = []


@socketio.on('draw_point')
def handle_draw_point(data):
    """Add point to current stroke"""
    tracker = state.tracker

    if tracker.annotation_point is None:
        return

    x, y = data['x'], data['y']
    dx = x - tracker.annotation_point[0]
    dy = y - tracker.annotation_point[1]

    if not tracker.drawing:
        tracker.drawing = True
        tracker.current_stroke = []

    tracker.current_stroke.append((dx, dy))

    # Inject trackers along stroke
    with state.frame_lock:
        if state.current_frame is not None:
            gray = cv2.cvtColor(state.current_frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), 12, 255, -1)

            new_pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=5,
                qualityLevel=0.01,
                minDistance=3,
                mask=mask
            )

            if new_pts is not None:
                for p in new_pts:
                    tracker.annotation_trackers.append(p.reshape(1, 2))

    # Send updated frame while paused
    if not state.is_playing:
        send_current_frame()


@socketio.on('end_stroke')
def handle_end_stroke():
    """End current stroke"""
    tracker = state.tracker
    if tracker.current_stroke:
        tracker.strokes.append(tracker.current_stroke)
    tracker.current_stroke = []
    tracker.drawing = False


@socketio.on('clear_annotations')
def handle_clear_annotations():
    """Clear all annotations"""
    with state.frame_lock:
        state.tracker.reset()
        if state.current_frame is not None:
            state.tracker.prev_gray = cv2.cvtColor(state.current_frame, cv2.COLOR_BGR2GRAY)

    emit('annotations_cleared', broadcast=True)


@socketio.on('get_first_frame')
def handle_get_first_frame():
    """Get first frame of video"""
    with state.frame_lock:
        if state.current_frame is not None:
            display = draw_annotations(state.current_frame)
            frame_data = encode_frame(display)
            emit('first_frame', {'image': frame_data})


@socketio.on('seek')
def handle_seek(data):
    """Seek to specific frame"""
    frame_num = data.get('frame', 0)

    with state.frame_lock:
        if state.video_capture:
            state.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            state.frame_count = frame_num

            ret, frame = state.video_capture.read()
            if ret:
                state.current_frame = frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                state.tracker.prev_gray = gray

                # Re-initialize tracking if we have anchor
                if state.tracker.annotation_point:
                    replenish_trackers(gray, state.tracker.annotation_point)
                    state.tracker.is_lost = False

                display = draw_annotations(frame)
                frame_data = encode_frame(display)
                emit('first_frame', {'image': frame_data})

            state.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)


@socketio.on('set_speed')
def handle_set_speed(data):
    """Set playback speed"""
    state.playback_speed = float(data.get('speed', 1.0))


# =========================
# RUN SERVER
# =========================
def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the web server"""
    # Start video processing with eventlet
    eventlet.spawn(video_processing_loop)

    print(f"\n{'='*50}")
    print(f"  HoloRay Motion-Tracked Annotation System")
    print(f"  Open http://localhost:{port} in your browser")
    print(f"{'='*50}\n")

    socketio.run(app, host=host, port=port, debug=debug, use_reloader=False)


if __name__ == '__main__':
    run_server(debug=True)
