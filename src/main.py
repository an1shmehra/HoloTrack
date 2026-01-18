import cv2
import numpy as np

# =========================
# CONFIGURATION
# =========================
VIDEO_PATH = "Lapchole1.mp4"

SEARCH_WINDOW_SIZE = 80
CONFIDENCE_THRESHOLD = 0.65
REPLENISH_THRESHOLD = 30

CENTER_RADIUS = 25
MID_RADIUS = 45
OUTER_RADIUS = 70

MAX_FEATURES_TOTAL = 120
OUTLIER_THRESHOLD = 18

# =========================
# STATE
# =========================
annotations = []
current_stroke = []
drawing = False

annotation_trackers = []
annotation_point = None
template_patch = None
prev_gray = None

paused = False
frame_count = 0

is_lost = False
last_known_pos = None

# =========================
# TRACKER UTILITIES
# =========================
def replenish_trackers(gray, pos):
    """Multi-ring dense feature sampling around anchor"""
    global annotation_trackers
    x, y = int(pos[0]), int(pos[1])

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

    annotation_trackers = pts[:MAX_FEATURES_TOTAL]

def set_anchor(x, y, gray):
    global annotation_point, template_patch, is_lost, last_known_pos
    annotation_point = (float(x), float(y))
    last_known_pos = annotation_point
    is_lost = False

    patch = 25
    y1, y2 = max(0, y - patch), min(gray.shape[0], y + patch)
    x1, x2 = max(0, x - patch), min(gray.shape[1], x + patch)
    template_patch = gray[y1:y2, x1:x2].copy()

    replenish_trackers(gray, annotation_point)
    print("Anchor set")

# =========================
# MOUSE CALLBACK
# =========================
def mouse_callback(event, x, y, flags, param):
    global drawing, current_stroke, annotations, annotation_point, prev_gray

    if prev_gray is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        if annotation_point is None:
            set_anchor(x, y, prev_gray)
        drawing = True
        current_stroke = []

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        dx = x - annotation_point[0]
        dy = y - annotation_point[1]
        current_stroke.append((dx, dy))

        # Inject trackers along stroke (CRITICAL)
        mask = np.zeros_like(prev_gray)
        cv2.circle(mask, (x, y), 12, 255, -1)

        new_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=5,
            qualityLevel=0.01,
            minDistance=3,
            mask=mask
        )

        if new_pts is not None:
            for p in new_pts:
                annotation_trackers.append(p.reshape(1, 2))

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        if current_stroke:
            annotations.append(current_stroke)
            print("Stroke saved")

# =========================
# MAIN LOOP
# =========================
def main():
    global prev_gray, frame_count, paused
    global annotation_point, annotation_trackers
    global is_lost, last_known_pos

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error opening video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 33

    cv2.namedWindow("Tracking")
    cv2.setMouseCallback("Tracking", mouse_callback)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_count += 1

            # =========================
            # TRACKING
            # =========================
            if annotation_point and template_patch is not None:

                if is_lost:
                    lx, ly = last_known_pos
                    r = SEARCH_WINDOW_SIZE
                    y1, y2 = int(max(0, ly - r)), int(min(gray.shape[0], ly + r))
                    x1, x2 = int(max(0, lx - r)), int(min(gray.shape[1], lx + r))
                    search = gray[y1:y2, x1:x2]

                    if search.shape[0] > template_patch.shape[0]:
                        res = cv2.matchTemplate(search, template_patch, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)

                        if max_val > CONFIDENCE_THRESHOLD:
                            nx = x1 + max_loc[0] + template_patch.shape[1] // 2
                            ny = y1 + max_loc[1] + template_patch.shape[0] // 2
                            annotation_point = (float(nx), float(ny))
                            last_known_pos = annotation_point
                            replenish_trackers(gray, annotation_point)
                            is_lost = False
                            print("Re-acquired")

                else:
                    dx_list, dy_list = [], []

                    if annotation_trackers and prev_gray is not None:
                        pts = np.array(annotation_trackers, dtype=np.float32)
                        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)

                        good_new = new_pts[status == 1]
                        good_old = pts[status == 1]

                        movement = good_new - good_old
                        dist = np.linalg.norm(movement, axis=1)
                        valid = dist < OUTLIER_THRESHOLD
                        movement = movement[valid]
                        good_new = good_new[valid]

                        annotation_trackers = [p.reshape(1, 2) for p in good_new]

                        if len(movement) > 0:
                            dx_list = movement[:, 0]
                            dy_list = movement[:, 1]

                    if len(dx_list) > 0:
                        dx = np.clip(np.median(dx_list), -8, 8)
                        dy = np.clip(np.median(dy_list), -8, 8)
                        annotation_point = (
                            annotation_point[0] + dx,
                            annotation_point[1] + dy
                        )

                    x, y = annotation_point
                    h, w = gray.shape
                    if len(annotation_trackers) < 3 or not (10 < x < w - 10 and 10 < y < h - 10):
                        is_lost = True
                        print("Target lost")
                    else:
                        last_known_pos = annotation_point

                        # Drift correction
                        if frame_count % 5 == 0:
                            r = 40
                            y1, y2 = int(max(0, y - r)), int(min(h, y + r))
                            x1, x2 = int(max(0, x - r)), int(min(w, x + r))
                            patch = gray[y1:y2, x1:x2]

                            if patch.shape[0] > template_patch.shape[0]:
                                res = cv2.matchTemplate(patch, template_patch, cv2.TM_CCOEFF_NORMED)
                                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                                if max_val > CONFIDENCE_THRESHOLD:
                                    annotation_point = (
                                        x1 + max_loc[0] + template_patch.shape[1] // 2,
                                        y1 + max_loc[1] + template_patch.shape[0] // 2
                                    )

                        if len(annotation_trackers) < REPLENISH_THRESHOLD:
                            replenish_trackers(gray, annotation_point)

            prev_gray = gray.copy()

        # =========================
        # RENDERING
        # =========================
        display = frame.copy()

        if annotation_point and not is_lost:
            cv2.circle(display, (int(annotation_point[0]), int(annotation_point[1])), 5, (0, 255, 0), -1)

            for stroke in annotations + ([current_stroke] if drawing else []):
                if len(stroke) < 2:
                    continue
                pts = np.array(stroke) + np.array(annotation_point)
                cv2.polylines(display, [pts.astype(np.int32)], False, (0, 0, 255), 2)

        if is_lost and last_known_pos:
            lx, ly = int(last_known_pos[0]), int(last_known_pos[1])
            # cv2.rectangle(display,
            #               (lx - SEARCH_WINDOW_SIZE, ly - SEARCH_WINDOW_SIZE),
            #               (lx + SEARCH_WINDOW_SIZE, ly + SEARCH_WINDOW_SIZE),
            #               (0, 255, 255), 2)
            cv2.putText(display, "SEARCHING", (lx - 40, ly - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        status = "LOST" if is_lost else f"TRACKING ({len(annotation_trackers)} pts)"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if is_lost else (0, 255, 0), 2)

        cv2.imshow("Tracking", display)

        key = cv2.waitKey(delay) & 0xFF
        if key == 27:
            break
        elif key == 32:
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
