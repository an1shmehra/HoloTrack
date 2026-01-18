import cv2
import numpy as np
import os

VIDEO_PATH = os.path.join("data", "Lapchole1.mp4")

prev_gray = None
tracking_points = None
annotation_point = None
template_patch = None
frame_count = 0
paused = False
speed_delay = 33  # default ~30 FPS

def mouse_callback(event, x, y, flags, param):
    global annotation_point, tracking_points, prev_gray, template_patch

    if event == cv2.EVENT_LBUTTONDOWN and prev_gray is not None:
        annotation_point = (x, y)

        # Create a slightly larger region mask around the click
        mask = np.zeros_like(prev_gray)
        cv2.circle(mask, (x, y), 30, 255, -1)
        
        # Detect good feature points in the selected area
        tracking_points = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=50,
            qualityLevel=0.01,
            minDistance=3,
            mask=mask
        )

        # Save a template patch for drift correction
        patch_size = 20
        y1 = max(0, y - patch_size)
        y2 = min(prev_gray.shape[0], y + patch_size)
        x1 = max(0, x - patch_size)
        x2 = min(prev_gray.shape[1], x + patch_size)
        template_patch = prev_gray[y1:y2, x1:x2]

        print(f"Tracking initialized at: {annotation_point}")

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

        # Draw annotation
        if annotation_point is not None:
            cv2.circle(frame, annotation_point, 6, (0, 255, 0), -1)

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
