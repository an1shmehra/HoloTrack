#!/usr/bin/env python3
"""
HoloRay Motion-Tracked Annotation System
CLI Demo - Test tracking without web UI

Usage:
    python demo_cli.py path/to/video.mp4
    python demo_cli.py path/to/video.mp4 --output tracked_output.mp4

Controls:
    - Click to add point annotations
    - Press 'r' then drag to add region annotations
    - Press 'c' to clear all annotations
    - Press 'q' to quit
    - Press 'p' to pause/play
    - Press 's' to save current frame
"""

import cv2
import numpy as np
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.tracker import HybridMotionTracker, TrackingState


class DemoApp:
    def __init__(self, video_path: str, output_path: str = None):
        self.video_path = video_path
        self.output_path = output_path

        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Tracker
        self.tracker = HybridMotionTracker(
            use_kalman=True,
            drift_correction_interval=30,
            max_lost_frames=15
        )

        # State
        self.current_frame = None
        self.frame_num = 0
        self.paused = True  # Start paused for annotation placement
        self.annotation_count = 0

        # Region drawing
        self.drawing_region = False
        self.region_start = None
        self.region_mode = False

        # Output
        self.writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        # Performance
        self.process_times = []
        
        # Display scaling for small videos
        self.display_scale = 1.0
        if self.width < 400 or self.height < 400:
            self.display_scale = max(640 / self.width, 640 / self.height)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for annotation placement."""
        # Convert mouse coordinates from scaled display back to original video coordinates
        if self.display_scale > 1.0:
            x = int(x / self.display_scale)
            y = int(y / self.display_scale)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.region_mode:
                self.drawing_region = True
                self.region_start = (x, y)
            else:
                # Add point annotation
                self.annotation_count += 1
                point_id = f"Point {self.annotation_count}"
                self.tracker.add_point(point_id, x, y, self.current_frame)
                print(f"Added point annotation '{point_id}' at ({x}, {y})")

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing_region and self.region_start:
                # Draw preview (handled in main loop)
                pass

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing_region and self.region_start:
                # Finalize region
                x1 = min(self.region_start[0], x)
                y1 = min(self.region_start[1], y)
                w = abs(x - self.region_start[0])
                h = abs(y - self.region_start[1])

                if w > 10 and h > 10:
                    self.annotation_count += 1
                    region_id = f"Region {self.annotation_count}"
                    self.tracker.add_region(region_id, (x1, y1, w, h), self.current_frame, "CSRT")
                    print(f"Added region annotation '{region_id}' at ({x1}, {y1}) size {w}x{h}")

                self.drawing_region = False
                self.region_start = None
                self.region_mode = False

    def draw_annotations(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """Draw annotations on frame."""
        frame = frame.copy()

        # Draw tracked points
        for pid, data in results.get("points", {}).items():
            x, y = int(data["x"]), int(data["y"])
            state = data["state"]
            confidence = data.get("confidence", 1.0)

            # Color based on state
            if state == "tracking":
                color = (0, 255, 0)  # Green
            elif state == "occluded":
                color = (0, 165, 255)  # Orange
            elif state == "reinitializing":
                color = (255, 255, 0)  # Cyan
            else:
                color = (0, 0, 255)  # Red

            # Draw marker
            cv2.circle(frame, (x, y), 12, color, -1)
            cv2.circle(frame, (x, y), 14, (255, 255, 255), 2)

            # Draw confidence ring
            if confidence < 1.0:
                cv2.circle(frame, (x, y), 18, color, 1)

            # Draw label
            cv2.putText(frame, pid, (x + 16, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, pid, (x + 16, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw state indicator
            cv2.putText(frame, state[:4], (x + 16, y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # Draw tracked regions
        for rid, data in results.get("regions", {}).items():
            bbox = data["bbox"]
            state = data["state"]
            x, y, w, h = bbox

            color = (0, 255, 0) if state == "tracking" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw label
            cv2.putText(frame, rid, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def draw_hud(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """Draw heads-up display with stats."""
        h, w = frame.shape[:2]

        # Semi-transparent background for HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (250, 100), (0, 0, 0), -1)
        cv2.rectangle(overlay, (w - 200, 0), (w, 60), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Left panel - info
        y_offset = 20
        cv2.putText(frame, f"HoloRay Tracker Demo", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 212, 170), 1)

        y_offset += 20
        cv2.putText(frame, f"Frame: {self.frame_num}/{self.total_frames}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        y_offset += 18
        points_tracking = sum(1 for p in results.get("points", {}).values() if p["state"] == "tracking")
        total_points = len(results.get("points", {}))
        cv2.putText(frame, f"Points: {points_tracking}/{total_points} tracking", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        y_offset += 18
        status = "PAUSED" if self.paused else "PLAYING"
        status_color = (0, 165, 255) if self.paused else (0, 255, 0)
        cv2.putText(frame, f"Status: {status}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

        # Right panel - performance
        fps = results.get("fps", 0)
        latency = results.get("process_time_ms", 0)

        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 180, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Latency: {latency:.1f}ms", (w - 180, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Bottom instructions
        if self.paused:
            instructions = "Click: add point | R+drag: add region | P: play | C: clear | Q: quit"
            cv2.putText(frame, instructions, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Mode indicator
        if self.region_mode:
            cv2.putText(frame, "REGION MODE - Drag to draw", (10, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        return frame

    def run(self):
        """Main demo loop."""
        window_name = "HoloRay Motion Tracker"
        # Create resizable window and set initial size
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Calculate display size - scale up small videos for better visibility
        # Minimum display size: 640x640, scale up if video is smaller
        scale_factor = max(640 / self.width, 640 / self.height, 1.0)
        display_width = int(self.width * scale_factor)
        display_height = int(self.height * scale_factor)
        cv2.resizeWindow(window_name, display_width, display_height)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n" + "=" * 50)
        print("  HoloRay Motion-Tracked Annotation Demo")
        print("=" * 50)
        print(f"\nVideo: {self.video_path}")
        print(f"Resolution: {self.width}x{self.height} @ {self.fps:.1f} FPS")
        print(f"Total frames: {self.total_frames}")
        print("\nControls:")
        print("  Click        - Add point annotation")
        print("  R + drag     - Add region annotation")
        print("  P            - Play/Pause")
        print("  C            - Clear all annotations")
        print("  S            - Save current frame")
        print("  Q            - Quit")
        print("\nStart by adding annotations while paused, then press P to play.")
        print("=" * 50 + "\n")

        # Read first frame
        ret, self.current_frame = self.cap.read()
        if not ret:
            print("Error: Cannot read video")
            return

        results = {"points": {}, "regions": {}, "fps": 0, "process_time_ms": 0}

        while True:
            if not self.paused:
                ret, frame = self.cap.read()

                if not ret:
                    # Loop video
                    print("\nVideo ended, looping...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_num = 0

                    # Re-initialize tracker with current annotations
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    # Reset tracker but keep annotation definitions
                    self.tracker.prev_gray = None
                    self.tracker.frame_count = 0

                self.current_frame = frame.copy()
                self.frame_num += 1

                # Update tracking
                results = self.tracker.update(frame)
            else:
                frame = self.current_frame.copy()

            # Draw annotations
            display_frame = self.draw_annotations(frame, results)

            # Draw region preview if drawing
            if self.drawing_region and self.region_start:
                # Get current mouse position (approximated from last event)
                pass

            # Draw HUD
            display_frame = self.draw_hud(display_frame, results)

            # Scale up frame for display if video is small
            # This makes small videos (like 112x112 echo) more visible
            if self.display_scale > 1.0:
                display_width = int(self.width * self.display_scale)
                display_height = int(self.height * self.display_scale)
                display_frame = cv2.resize(display_frame, (display_width, display_height), interpolation=cv2.INTER_NEAREST)

            # Write to output if enabled (use original size, not scaled)
            if self.writer and not self.paused:
                # Write original frame, not scaled version
                original_frame = self.draw_annotations(frame, results)
                original_frame = self.draw_hud(original_frame, results)
                self.writer.write(original_frame)

            # Display scaled frame
            cv2.imshow(window_name, display_frame)

            # Handle keyboard
            key = cv2.waitKey(1 if not self.paused else 30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('p'):
                self.paused = not self.paused
                status = "Paused" if self.paused else "Playing"
                print(f"Video {status}")
            elif key == ord('c'):
                self.tracker.clear_all()
                self.annotation_count = 0
                results = {"points": {}, "regions": {}, "fps": 0, "process_time_ms": 0}
                print("Cleared all annotations")
            elif key == ord('r'):
                self.region_mode = True
                print("Region mode enabled - drag to draw a region")
            elif key == ord('s'):
                filename = f"holoray_frame_{self.frame_num}.png"
                cv2.imwrite(filename, display_frame)
                print(f"Saved frame to {filename}")

        # Cleanup
        self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

        # Print summary
        stats = self.tracker.get_stats()
        print("\n" + "=" * 50)
        print("  Session Summary")
        print("=" * 50)
        print(f"  Frames processed: {stats['frame_count']}")
        print(f"  Average FPS: {stats['avg_fps']:.1f}")
        print(f"  Total annotations tracked: {stats['tracked_points'] + stats['tracked_regions']}")
        print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='HoloRay Motion Tracker CLI Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('--output', '-o', help='Path to save output video')

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    try:
        app = DemoApp(args.video, args.output)
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
