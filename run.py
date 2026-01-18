import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os

# =========================
# GLOBAL VARS (For the Frontend)
# =========================
SELECTED_VIDEO_PATH = ""

# =========================
# ORIGINAL TRACKING LOGIC
# =========================
# (Logic preserved exactly as requested, with UI hooks added)

VIDEO_PATH = "Microscopy1.mp4" 

# General tracking parameters
SEARCH_WINDOW_SIZE = 80
CONFIDENCE_THRESHOLD = 0.65
CENTER_RADIUS = 25
MID_RADIUS = 45
OUTER_RADIUS = 70
EXTRA_RADIUS = 120
MAX_FEATURES_TOTAL = 20000
REPLENISH_THRESHOLD = 40
OUTLIER_THRESHOLD = 18
MASK_RADIUS_DRAW = 45
MASK_RADIUS_TRACK = 35
MIN_TRACK_POINTS = 25

lk_params = dict(
    winSize=(20, 20),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.003)
)

annotations = []
current_stroke = []
drawing = False
annotation_point = None
template_patch = None
annotation_trackers = []
track_points = None
annotation_contours = []
prev_gray = None
paused = False
frame_count = 0
is_lost = False
last_known_pos = None

def replenish_trackers(gray, pos):
    global annotation_trackers
    x, y = int(pos[0]), int(pos[1])
    def sample(radius, count):
        mask = np.zeros_like(gray)
        cv2.circle(mask, (x, y), radius, 255, -1)
        return cv2.goodFeaturesToTrack(gray, maxCorners=count, qualityLevel=0.001, minDistance=2, mask=mask)
    pts = []
    for r, c in [(CENTER_RADIUS, 60), (MID_RADIUS, 80), (OUTER_RADIUS, 100), (EXTRA_RADIUS, 50)]:
        p = sample(r, c)
        if p is not None:
            pts.extend([pt.reshape(1, 2) for pt in p])
    annotation_trackers = pts[:MAX_FEATURES_TOTAL]

def set_anchor(x, y, gray):
    global annotation_point, template_patch, is_lost, last_known_pos, track_points
    annotation_point = (float(x), float(y))
    last_known_pos = annotation_point
    is_lost = False
    patch = 25
    y1, y2 = max(0, y - patch), min(gray.shape[0], y + patch)
    x1, x2 = max(0, x - patch), min(gray.shape[1], x + patch)
    template_patch = gray[y1:y2, x1:x2].copy()
    replenish_trackers(gray, annotation_point)
    mask = np.zeros_like(gray)
    cv2.circle(mask, (x, y), MASK_RADIUS_DRAW, 255, -1)
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=MAX_FEATURES_TOTAL, qualityLevel=0.001, minDistance=2, mask=mask)
    if pts is not None:
        track_points = pts.reshape(-1, 2)
    print("Anchor set")

def mouse_callback(event, x, y, flags, param):
    global drawing, current_stroke, annotations, annotation_point, prev_gray, track_points, annotation_contours
    if prev_gray is None: return
    if event == cv2.EVENT_LBUTTONDOWN:
        if annotation_point is None:
            set_anchor(x, y, prev_gray)
        drawing = True
        current_stroke = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_stroke.append((x, y))
        mask = np.zeros_like(prev_gray)
        cv2.circle(mask, (x, y), MASK_RADIUS_DRAW, 255, -1)
        new_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=40, qualityLevel=0.005, minDistance=2, blockSize=7, mask=mask)
        if new_pts is not None:
            if track_points is None: track_points = new_pts.reshape(-1, 2)
            else: track_points = np.vstack([track_points, new_pts.reshape(-1, 2)])
            annotation_trackers.extend([p.reshape(1, 2) for p in new_pts])
    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        if len(current_stroke) > 5:
            annotation_contours.append(np.array(current_stroke, dtype=np.float32))
            print("Contour saved")

def main():
    global prev_gray, frame_count, paused, VIDEO_PATH, annotation_point, annotation_trackers, track_points, is_lost, last_known_pos, template_patch
    
    # Use selected path if available
    if SELECTED_VIDEO_PATH:
        VIDEO_PATH = SELECTED_VIDEO_PATH
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video file.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 33
    
    cv2.namedWindow("Hybrid Annotation Tracker")
    cv2.setMouseCallback("Hybrid Annotation Tracker", mouse_callback)
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_count += 1
            
            # --- TRACKING LOGIC START ---
            if annotation_point is not None and template_patch is not None:
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
                    if annotation_trackers:
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
                            dx = np.clip(np.median(movement[:, 0]), -8, 8)
                            dy = np.clip(np.median(movement[:, 1]), -8, 8)
                            annotation_point = (annotation_point[0] + dx, annotation_point[1] + dy)
                            x, y = annotation_point
                            h, w = gray.shape
                            if len(annotation_trackers) < 3 or not (10 < x < w-10 and 10 < y < h-10):
                                is_lost = True
                                print("Target lost")
                            else:
                                last_known_pos = annotation_point
                        if len(annotation_trackers) < REPLENISH_THRESHOLD:
                            replenish_trackers(gray, annotation_point)
            
            if track_points is not None and len(track_points) >= MIN_TRACK_POINTS:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, track_points.astype(np.float32), None, **lk_params)
                good_new = next_pts[status.flatten() == 1]
                good_old = track_points[status.flatten() == 1]
                if len(good_new) >= MIN_TRACK_POINTS:
                    M, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    if M is not None:
                        for i in range(len(annotation_contours)):
                            pts = annotation_contours[i]
                            pts = cv2.transform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)
                            annotation_contours[i] = pts
                        track_points = good_new
                else: track_points = None
            
            if track_points is not None and len(track_points) < REPLENISH_THRESHOLD:
                mask = np.zeros_like(gray)
                for p in track_points.astype(int): cv2.circle(mask, tuple(p), MASK_RADIUS_TRACK, 255, -1)
                new_pts = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.005, minDistance=3, blockSize=7, mask=mask)
                if new_pts is not None: track_points = np.vstack([track_points, new_pts.reshape(-1, 2)])
            
            prev_gray = gray.copy()
            # --- TRACKING LOGIC END ---
        
        # =========================
        # RENDERING & OVERLAY
        # =========================
        display = frame.copy()
        
        # Draw Strokes
        for contour in annotation_contours:
            cv2.polylines(display, [contour.astype(np.int32)], False, (0, 100, 255), 2)
        if drawing and len(current_stroke) > 1:
            cv2.polylines(display, [np.array(current_stroke, dtype=np.int32)], False, (0, 255, 255), 2)
        
        # Draw Status Overlay
        status_color = (0, 0, 255) if is_lost else (0, 255, 0)
        status_text = "LOST" if is_lost else "ACTIVE"
        
        # Simple dashboard at top left
        cv2.rectangle(display, (0, 0), (220, 85), (30, 30, 30), -1)
        cv2.putText(display, f"STATUS: {status_text}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(display, f"Confidence Score: {len(annotation_trackers)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, f"Frame: {frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Instructions Overlay at bottom
        h, w = display.shape[:2]
        cv2.rectangle(display, (0, h-30), (w, h), (30, 30, 30), -1)
        cv2.putText(display, "[SPACE] Pause   [ESC] Quit   [L-CLICK] Anchor   [DRAG] Draw Region", (15, h-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if is_lost and last_known_pos:
            lx, ly = int(last_known_pos[0]), int(last_known_pos[1])
            cv2.putText(display, "SEARCHING...", (lx-40, ly-90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Hybrid Annotation Tracker", display)
        
        key = cv2.waitKey(delay) & 0xFF
        if key == 27: break
        elif key == 32: paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()


# =========================
# PROFESSIONAL FRONTEND (GUI)
# =========================
def create_professional_frontend():
    # --- Color Palette (Nord Theme Inspired) ---
    BG_COLOR = "#2E3440"       # Dark Grey
    FG_COLOR = "#D8DEE9"       # Off-White
    ACCENT_COLOR = "#88C0D0"   # Cyan/Blue
    BTN_BG = "#4C566A"         # Lighter Grey
    BTN_ACTIVE = "#5E81AC"     # Blueish
    
    root = tk.Tk()
    root.title("VisionTrack Pro")
    root.geometry("600x450")
    root.configure(bg=BG_COLOR)
    
    # --- Styling ---
    style = ttk.Style()
    style.theme_use('clam') # Use 'clam' as base for better customizability
    
    # Configure generic styles
    style.configure("TFrame", background=BG_COLOR)
    style.configure("TLabel", background=BG_COLOR, foreground=FG_COLOR, font=("Segoe UI", 10))
    style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"), foreground=ACCENT_COLOR)
    style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground=FG_COLOR)
    
    # Button Styling
    style.configure("TButton", 
                    font=("Segoe UI", 10, "bold"), 
                    background=BTN_BG, 
                    foreground="white", 
                    borderwidth=0, 
                    focuscolor=BTN_ACTIVE)
    style.map("TButton", background=[("active", BTN_ACTIVE)])
    
    # Main Container
    main_frame = ttk.Frame(root, padding=30)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # 1. Header Section
    header_frame = ttk.Frame(main_frame)
    header_frame.pack(fill=tk.X, pady=(0, 20))
    ttk.Label(header_frame, text="VISIONTRACK PRO", style="Title.TLabel").pack(side=tk.LEFT)
    ttk.Label(header_frame, text="v1.0", style="TLabel").pack(side=tk.RIGHT, pady=10)
    
    ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=10)

    # 2. File Selection Section
    file_frame = ttk.Frame(main_frame)
    file_frame.pack(fill=tk.X, pady=10)
    
    ttk.Label(file_frame, text="SOURCE VIDEO", style="Header.TLabel").pack(anchor=tk.W)
    
    path_var = tk.StringVar(value="Default: 'Microscopy1.mp4'")
    path_label = tk.Label(file_frame, textvariable=path_var, bg="#3B4252", fg="#E5E9F0", 
                          height=2, anchor="w", padx=10, font=("Consolas", 9))
    path_label.pack(fill=tk.X, pady=5)
    
    def browse_file():
        global SELECTED_VIDEO_PATH
        filename = filedialog.askopenfilename(title="Select Video Source")
        if filename:
            SELECTED_VIDEO_PATH = filename
            path_var.set(f"Selected: {os.path.basename(filename)}")
            
    ttk.Button(file_frame, text="BROWSE FILES...", command=browse_file).pack(anchor=tk.E, pady=5)

    # 3. Controls / Instructions Section
    info_frame = ttk.Frame(main_frame)
    info_frame.pack(fill=tk.BOTH, expand=True, pady=15)
    
    ttk.Label(info_frame, text="CONTROL GUIDE", style="Header.TLabel").pack(anchor=tk.W, pady=(0,5))
    
    # Instruction Grid
    grid_frame = ttk.Frame(info_frame)
    grid_frame.pack(fill=tk.X, padx=10)
    
    def add_instruction(row, key, desc):
        k_lbl = tk.Label(grid_frame, text=key, bg="#3B4252", fg=ACCENT_COLOR, 
                         width=12, font=("Segoe UI", 9, "bold"))
        k_lbl.grid(row=row, column=0, pady=2, padx=(0, 10), sticky="w")
        
        d_lbl = tk.Label(grid_frame, text=desc, bg=BG_COLOR, fg=FG_COLOR, font=("Segoe UI", 9))
        d_lbl.grid(row=row, column=1, pady=2, sticky="w")

    add_instruction(0, "L-CLICK", "Set tracking anchor (Blue Dot)")
    add_instruction(1, "DRAG MOUSE", "Draw contour regions to track")
    add_instruction(2, "SPACEBAR", "Pause / Resume playback")
    add_instruction(3, "ESC", "Terminate session")
    
    # 4. Action Section
    def start_app():
        if not SELECTED_VIDEO_PATH and not os.path.exists(VIDEO_PATH):
             messagebox.showwarning("Missing File", "Please select a video file to begin.")
             return
        root.destroy()
        main()

    btn_launch = tk.Button(main_frame, text="INITIALIZE TRACKING ENGINE", command=start_app,
                           bg=ACCENT_COLOR, fg="#2E3440", font=("Segoe UI", 11, "bold"),
                           relief="flat", pady=10, cursor="hand2")
    btn_launch.pack(fill=tk.X, side=tk.BOTTOM, pady=10)

    # Hover effect for launch button manually since it's a tk.Button (for better color control than ttk)
    def on_enter(e): btn_launch['bg'] = "#81A1C1"
    def on_leave(e): btn_launch['bg'] = ACCENT_COLOR
    btn_launch.bind("<Enter>", on_enter)
    btn_launch.bind("<Leave>", on_leave)

    root.mainloop()

if __name__ == "__main__":
    create_professional_frontend()