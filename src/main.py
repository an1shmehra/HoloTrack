import cv2
import numpy as np
import os

# Configuration
VIDEO_PATH = os.path.join("data", "echo1.mp4")
MAX_JUMP_DISTANCE = 50
SEARCH_WINDOW_SIZE = 80  # Size of search area when object is lost
REPLENISH_THRESHOLD = 30 # Min points before we add more
CONFIDENCE_THRESHOLD = 0.65 # Template match score to accept a re-entry

# State Management
annotations = []         # List of strokes (relative to anchor)
current_stroke = []
drawing = False
annotation_trackers = [] # List of feature points

prev_gray = None
annotation_point = None
template_patch = None
frame_count = 0
paused = False

# New State Variables for Re-ID
is_lost = False
last_known_pos = None

def set_anchor(x, y, gray_frame):
    """Sets the anchor point and initial tracking features."""
    global annotation_point, template_patch, last_known_pos, is_lost
    
    annotation_point = (x, y)
    last_known_pos = (x, y)
    is_lost = False

    # Create a template patch for re-identification
    patch_size = 25 # Slightly larger for better uniqueness
    y1, y2 = max(0, y - patch_size), min(gray_frame.shape[0], y + patch_size)
    x1, x2 = max(0, x - patch_size), min(gray_frame.shape[1], x + patch_size)
    template_patch = gray_frame[y1:y2, x1:x2].copy()
    
    replenish_trackers(gray_frame, (x, y))
    print(f"Anchor set at ({x}, {y})")

def replenish_trackers(gray_frame, pos):
    """Finds new strong corners around the specific position to keep tracking dense."""
    global annotation_trackers
    
    x, y = pos
    mask = np.zeros_like(gray_frame)
    # Only look for points in a circle around the anchor
    cv2.circle(mask, (int(x), int(y)), 30, 255, -1)
    
    # Get good features
    new_pts = cv2.goodFeaturesToTrack(gray_frame, maxCorners=50, qualityLevel=0.01, minDistance=5, mask=mask)
    
    if new_pts is not None:
        # Flatten and add to list
        existing_pts = []
        if len(annotation_trackers) > 0:
            existing_pts = [pt.flatten() for pt in annotation_trackers]
        
        for p in new_pts:
            p_flat = p.flatten()
            # Simple check to avoid duplicates (optional but good for stability)
            if not any(np.linalg.norm(p_flat - ep) < 2 for ep in existing_pts):
                annotation_trackers.append(p)

def mouse_callback(event, x, y, flags, param):
    global annotation_point, drawing, current_stroke, annotations, prev_gray

    if prev_gray is None: return

    if event == cv2.EVENT_LBUTTONDOWN:
        if annotation_point is None:
            set_anchor(x, y, prev_gray)
        
        drawing = True
        current_stroke = []
        print("Started drawing")

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Save relative position to anchor
        if annotation_point:
            dx = x - annotation_point[0]
            dy = y - annotation_point[1]
            current_stroke.append((dx, dy))

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        if current_stroke:
            annotations.append(current_stroke)
            print(f"Stroke saved.")

def main():
    global prev_gray, annotation_point, template_patch, frame_count
    global paused, annotation_trackers, is_lost, last_known_pos

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    default_delay = int(1000 / fps) if fps > 0 else 33

    cv2.namedWindow("Tracking")
    cv2.setMouseCallback("Tracking", mouse_callback)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret: break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_count += 1
            
            # --- LOGIC BRANCH: TRACKING vs. SEARCHING ---
            
            if annotation_point is not None and template_patch is not None:
                
                # 1. IF LOST: Search only around the last known position
                if is_lost:
                    # Define search area around LAST KNOWN position (not current)
                    lx, ly = last_known_pos
                    h, w = template_patch.shape
                    
                    search_r = SEARCH_WINDOW_SIZE # Larger radius to catch re-entry
                    y1 = max(0, int(ly - search_r))
                    y2 = min(gray.shape[0], int(ly + search_r))
                    x1 = max(0, int(lx - search_r))
                    x2 = min(gray.shape[1], int(lx + search_r))
                    
                    search_area = gray[y1:y2, x1:x2]
                    
                    # Only search if area is big enough
                    if search_area.shape[0] > h and search_area.shape[1] > w:
                        res = cv2.matchTemplate(search_area, template_patch, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        
                        if max_val > CONFIDENCE_THRESHOLD:
                            # FOUND IT! Snap back.
                            is_lost = False
                            # Convert local search coordinates back to global
                            new_x = x1 + max_loc[0] + w // 2
                            new_y = y1 + max_loc[1] + h // 2
                            annotation_point = (new_x, new_y)
                            
                            # Immediately regenerate trackers at the new spot
                            annotation_trackers = []
                            replenish_trackers(gray, annotation_point)
                            print(f"Re-acquired at {annotation_point} (Conf: {max_val:.2f})")

                # 2. IF TRACKING: Normal Optical Flow + Drift Correct
                else:
                    dx_list, dy_list = [], []
                    
                    # A. Optical Flow
                    if len(annotation_trackers) > 0 and prev_gray is not None:
                        pts_in = np.array(annotation_trackers, dtype=np.float32)
                        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts_in, None)
                        
                        good_new = new_pts[status == 1]
                        good_old = pts_in[status == 1]
                        
                        if len(good_new) > 0:
                            movement = good_new - good_old
                            dx_list = movement[:, 0]
                            dy_list = movement[:, 1]
                            annotation_trackers = [p.reshape(1, 2) for p in good_new]

                    # B. Update Anchor
                    if len(dx_list) > 0:
                        dx = np.clip(np.median(dx_list), -20, 20)
                        dy = np.clip(np.median(dy_list), -20, 20)
                        annotation_point = (annotation_point[0] + dx, annotation_point[1] + dy)

                    # C. Check if we are lost (No trackers left OR out of bounds)
                    h, w = gray.shape
                    x, y = annotation_point
                    margin = 10
                    out_of_bounds = not (margin < x < w - margin and margin < y < h - margin)
                    
                    if len(annotation_trackers) < 3 or out_of_bounds:
                        is_lost = True
                        # Don't update last_known_pos here if it's out of bounds, 
                        # keep the last VALID one inside the frame roughly.
                        if not out_of_bounds:
                            last_known_pos = (x, y)
                        print("Target Lost. Waiting for re-entry...")

                    # D. Drift Correction & Auto-Replenish
                    elif not is_lost:
                        # Periodically verify position with template
                        if frame_count % 5 == 0:
                            res = cv2.matchTemplate(gray, template_patch, cv2.TM_CCOEFF_NORMED)
                            # Look in a small window around current estimate
                            # (Simplified logic: OpenCV creates a result map, we just mask it or check max)
                            # Ideally, crop search region like before for speed:
                            sx, sy = int(x), int(y)
                            r = 40
                            y1, y2 = max(0, sy-r), min(h, sy+r)
                            x1, x2 = max(0, sx-r), min(w, sx+r)
                            patch_search = gray[y1:y2, x1:x2]
                            
                            if patch_search.shape[0] > template_patch.shape[0]:
                                res = cv2.matchTemplate(patch_search, template_patch, cv2.TM_CCOEFF_NORMED)
                                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                                
                                if max_val > CONFIDENCE_THRESHOLD:
                                    # Adjust Anchor
                                    new_x = x1 + max_loc[0] + template_patch.shape[1]//2
                                    new_y = y1 + max_loc[1] + template_patch.shape[0]//2
                                    annotation_point = (float(new_x), float(new_y))
                                    last_known_pos = annotation_point # Update valid pos

                        # Auto-Replenish Trackers if count gets low
                        if len(annotation_trackers) < REPLENISH_THRESHOLD:
                            replenish_trackers(gray, annotation_point)

            prev_gray = gray.copy()
        
        # --- RENDERING ---
        display_frame = frame.copy()
        
        # Only draw strokes if we are NOT lost
        if annotation_point and not is_lost:
            # Draw anchor center
            cv2.circle(display_frame, (int(annotation_point[0]), int(annotation_point[1])), 5, (0, 255, 0), -1)
            
            for stroke in annotations + ([current_stroke] if drawing else []):
                if len(stroke) < 2: continue
                # Convert relative stroke to absolute screen points
                pts = np.array(stroke) + np.array(annotation_point)
                cv2.polylines(display_frame, [pts.astype(np.int32)], False, (0, 0, 255), 2)
        
        elif is_lost and last_known_pos:
            # Visualize the search area
            lx, ly = int(last_known_pos[0]), int(last_known_pos[1])
            cv2.rectangle(display_frame, 
                         (lx - SEARCH_WINDOW_SIZE, ly - SEARCH_WINDOW_SIZE), 
                         (lx + SEARCH_WINDOW_SIZE, ly + SEARCH_WINDOW_SIZE), 
                         (0, 255, 255), 2)
            cv2.putText(display_frame, "SEARCHING...", (lx - 40, ly - 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # UI Text
        status_text = "LOST" if is_lost else f"TRACKING ({len(annotation_trackers)} pts)"
        color = (0, 0, 255) if is_lost else (0, 255, 0)
        cv2.putText(display_frame, f"FPS: {fps:.1f} | Status: {status_text}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow("Tracking", display_frame)

        key = cv2.waitKey(default_delay) & 0xFF
        if key == 27: break
        elif key == 32: paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
