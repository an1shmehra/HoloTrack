import cv2
import numpy as np
import os

annotations = []          # list of strokes
current_stroke = []
drawing = False

VIDEO_PATH = os.path.join("data", "Lapchole1.mp4")

prev_gray = None
tracking_points = None
annotation_point = None
template_patch = None
frame_count = 0
paused = False
speed_delay = 33  # default ~30 FPS

def mouse_callback(event, x, y, flags, param):
    global annotation_point, tracking_points, prev_gray
    global template_patch, drawing, current_stroke, annotations

    # LEFT CLICK (no shift) → set anchor
    if event == cv2.EVENT_LBUTTONDOWN and not (flags & cv2.EVENT_FLAG_SHIFTKEY):
        if prev_gray is None:
            return

        annotation_point = (x, y)

        mask = np.zeros_like(prev_gray)
        cv2.circle(mask, (x, y), 30, 255, -1)

        tracking_points = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=50,
            qualityLevel=0.01,
            minDistance=3,
            mask=mask
        )

        patch_size = 20
        y1, y2 = max(0, y - patch_size), min(prev_gray.shape[0], y + patch_size)
        x1, x2 = max(0, x - patch_size), min(prev_gray.shape[1], x + patch_size)
        template_patch = prev_gray[y1:y2, x1:x2]

        print("Anchor set")

    # SHIFT + LEFT → start drawing
    elif event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY):
        if annotation_point is None:
            return
        drawing = True
        current_stroke = []

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        dx = x - annotation_point[0]
        dy = y - annotation_point[1]
        current_stroke.append((dx, dy))

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        if current_stroke:
            annotations.append(current_stroke)
            print("Stroke saved")

def main():
    global prev_gray, tracking_points, annotation_point, template_patch, frame_count, paused, speed_delay

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error opening video")
        return

    # Default FPS from video
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    default_delay = int(1000 / video_fps)

    cv2.namedWindow("Tracking")
    cv2.setMouseCallback("Tracking", mouse_callback)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_count += 1

            # Optical flow tracking
            if tracking_points is not None and prev_gray is not None:
                new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, tracking_points, None
                )
                good_new = new_points[status == 1]
                good_old = tracking_points[status == 1]

                if len(good_new) > 0:
                    dx = (good_new[:, 0] - good_old[:, 0]).mean()
                    dy = (good_new[:, 1] - good_old[:, 1]).mean()

                    dx = np.clip(dx, -5, 5)
                    dy = np.clip(dy, -5, 5)

                    annotation_point = (
                        int(annotation_point[0] + dx),
                        int(annotation_point[1] + dy)
                    )
                    tracking_points = good_new.reshape(-1, 1, 2)

            # Drift correction using template matching every 5 frames
            if template_patch is not None and annotation_point is not None and frame_count % 5 == 0:
                patch_h, patch_w = template_patch.shape
                res = cv2.matchTemplate(gray, template_patch, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(res)
                annotation_point = (max_loc[0] + patch_w // 2, max_loc[1] + patch_h // 2)

            prev_gray = gray.copy()
        else:
            # When paused, keep showing the same frame
            frame = frame.copy()

        # Overlay speed options on the frame
        cv2.putText(frame, "Press 1=Slow, 2=Medium, 3=Fast", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Press SPACE=Pause/Play, ESC=Exit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw tracking points
        if tracking_points is not None:
            for p in tracking_points:
                x, y = p.ravel()
                cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)

        # Draw annotations
        if annotation_point is not None:
            for stroke in annotations:
                for i in range(1, len(stroke)):
                    p1 = (
                        annotation_point[0] + stroke[i - 1][0],
                        annotation_point[1] + stroke[i - 1][1]
                    )
                    p2 = (
                        annotation_point[0] + stroke[i][0],
                        annotation_point[1] + stroke[i][1]
                    )
                    cv2.line(frame, p1, p2, (0, 0, 255), 2)
                    
            # Draw current stroke live
        if drawing and annotation_point is not None and len(current_stroke) > 1:
            for i in range(1, len(current_stroke)):
                p1 = (
                    annotation_point[0] + current_stroke[i - 1][0],
                    annotation_point[1] + current_stroke[i - 1][1]
                )
                p2 = (
                    annotation_point[0] + current_stroke[i][0],
                    annotation_point[1] + current_stroke[i][1]
                )
                cv2.line(frame, p1, p2, (0, 0, 255), 2)

        cv2.imshow("Tracking", frame)

        key = cv2.waitKey(speed_delay) & 0xFF
        if key == 27:  # ESC to quit
            break
        elif key == 32:  # SPACE to pause/play
            paused = not paused
            print("Video paused." if paused else "Video resumed.")
        elif key == ord('1'):
            speed_delay = default_delay * 3  # slow
            print("Speed set to SLOW")
        elif key == ord('2'):
            speed_delay = default_delay  # medium
            print("Speed set to MEDIUM")
        elif key == ord('3'):
            speed_delay = max(1, default_delay // 2)  # fast
            print("Speed set to FAST")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
