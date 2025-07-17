#!/usr/bin/env python3
"""
Debug version of head tracker with fixed servo control for tracking
Unfilled head bounding box visualization with enhanced OpenCV display
Uses cv2.VideoCapture for IMX519 on CAM0
"""

import cv2
from jetson_inference import poseNet
from jetson_utils import videoOutput, cudaFromNumpy
import argparse
import sys
import time
import math
import numpy as np

try:
    from adafruit_servokit import ServoKit
    ADAFRUIT_AVAILABLE = True
    print("✓ Adafruit libraries imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    ADAFRUIT_AVAILABLE = False
    sys.exit(1)

class DebugServoController:
    def __init__(self, pan_channel=0, tilt_channel=1):
        self.pan_channel = pan_channel
        self.tilt_channel = tilt_channel
        self.last_pan = 90
        self.last_tilt = 90
        
        print("Initializing ServoKit...")
        try:
            self.kit = ServoKit(channels=16)
            print("✓ ServoKit initialized")
            self.kit.servo[self.pan_channel].angle = 90
            self.kit.servo[self.tilt_channel].angle = 90
            time.sleep(0.5)
            print("Moving servos to test positions...")
            self.kit.servo[self.pan_channel].angle = 60
            time.sleep(0.5)
            self.kit.servo[self.pan_channel].angle = 120
            time.sleep(0.5)
            self.kit.servo[self.pan_channel].angle = 90
            print("✓ Initial servo test complete")
        except Exception as e:
            print(f"✗ ServoKit initialization failed: {e}")
            sys.exit(1)
    
    def move_to_raw(self, pan_servo_angle, tilt_servo_angle):
        try:
            print(f"DEBUG: Moving servos to Pan={pan_servo_angle:.1f}°, Tilt={tilt_servo_angle:.1f}°")
            pan_servo_angle = max(0, min(180, pan_servo_angle))
            tilt_servo_angle = max(0, min(180, tilt_servo_angle))
            self.kit.servo[self.pan_channel].angle = pan_servo_angle
            self.kit.servo[self.tilt_channel].angle = tilt_servo_angle
            time.sleep(0.2)
            print(f"✓ Servos moved to Pan={pan_servo_angle:.1f}°, Tilt={tilt_servo_angle:.1f}°")
        except Exception as e:
            print(f"✗ Servo movement error: {e}")
            import traceback
            traceback.print_exc()
    
    def move_to_tracking(self, pan_angle, tilt_angle):
        if pan_angle is None or tilt_angle is None:
            print("DEBUG: Received None angles, skipping movement")
            return
        print(f"DEBUG: Tracking angles - Pan={pan_angle:.1f}°, Tilt={tilt_angle:.1f}°")
        servo_pan = pan_angle + 90
        servo_tilt = tilt_angle + 90
        max_angle_change = 5
        servo_pan = self.last_pan + max(-max_angle_change, min(max_angle_change, servo_pan - self.last_pan))
        servo_tilt = self.last_tilt + max(-max_angle_change, min(max_angle_change, servo_tilt - self.last_tilt))
        print(f"DEBUG: Converted to servo angles - Pan={servo_pan:.1f}°, Tilt={servo_tilt:.1f}°")
        self.last_pan = servo_pan
        self.last_tilt = servo_tilt
        self.move_to_raw(servo_pan, servo_tilt)
    
    def test_servo_range(self):
        print("Testing full servo range...")
        angles = [0, 45, 90, 135, 180]
        for angle in angles:
            print(f"Moving to Pan={angle}°, Tilt={angle}°")
            self.move_to_raw(angle, angle)
            time.sleep(1)
        print("Returning to center...")
        self.move_to_raw(90, 90)
    
    def cleanup(self):
        print("Returning servos to center...")
        self.move_to_raw(90, 90)
        cv2.destroyAllWindows()

def calculate_head_center_and_bbox(pose):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    head_points = []
    keypoints = pose.Keypoints
    keypoint_names = {0: "Nose", 1: "Left Eye", 2: "Right Eye", 3: "Left Ear", 4: "Right Ear"}
    for idx in [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]:
        if idx < len(keypoints):
            keypoint = keypoints[idx]
            if keypoint.x > 0 and keypoint.y > 0:
                head_points.append((keypoint.x, keypoint.y))
                print(f"DEBUG: Detected {keypoint_names[idx]} at ({keypoint.x:.1f}, {keypoint.y:.1f})")
            else:
                print(f"DEBUG: {keypoint_names[idx]} not detected")
        else:
            print(f"DEBUG: {keypoint_names[idx]} keypoint not available")
    if len(head_points) < 1:
        print("DEBUG: No valid head keypoints detected")
        return None, None
    avg_x = sum(point[0] for point in head_points) / len(head_points)
    avg_y = sum(point[1] for point in head_points) / len(head_points)
    center = (avg_x, avg_y)
    min_x = min(point[0] for point in head_points)
    max_x = max(point[0] for point in head_points)
    min_y = min(point[1] for point in head_points)
    max_y = max(point[1] for point in head_points)
    width = max_x - min_x
    height = max_y - min_y
    padding_x = width * 0.5
    padding_y = height * 0.5
    bbox = {
        'x1': max(0, min_x - padding_x),
        'y1': max(0, min_y - padding_y),
        'x2': max_x + padding_x,
        'y2': max_y + padding_y,
        'width': width + 2 * padding_x,
        'height': height + 2 * padding_y
    }
    return center, bbox

def draw_bounding_box(cuda_img, bbox, center):
    if bbox is None or center is None:
        return
    try:
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        jetson_utils.cudaDrawLine(cuda_img, (x1, y1), (x2, y1), (0, 255, 0, 255))
        jetson_utils.cudaDrawLine(cuda_img, (x2, y1), (x2, y2), (0, 255, 0, 255))
        jetson_utils.cudaDrawLine(cuda_img, (x2, y2), (x1, y2), (0, 255, 0, 255))
        jetson_utils.cudaDrawLine(cuda_img, (x1, y2), (x1, y1), (0, 255, 0, 255))
        jetson_utils.cudaDrawCircle(cuda_img, 
                                  (int(center[0]), int(center[1])), 
                                  5, (255, 0, 0, 255))
        jetson_utils.cudaDrawLine(cuda_img,
                                 (int(center[0] - 10), int(center[1])),
                                 (int(center[0] + 10), int(center[1])),
                                 (255, 0, 0, 255))
        jetson_utils.cudaDrawLine(cuda_img,
                                 (int(center[0]), int(center[1] - 10)),
                                 (int(center[0]), int(center[1] + 10)),
                                 (255, 0, 0, 255))
        print(f"✓ Drew unfilled bounding box at center ({int(center[0])}, {int(center[1])})")
        for person_id, tracked_center, _ in tracked_people:
            if tracked_center == center:
                jetson_utils.cudaDrawText(cuda_img, f"Person {person_id}", 
                                         (x1, max(0, y1 - 10)), (255, 255, 0, 255), 20)
                print(f"DEBUG: Drew label 'Person {person_id}' at ({x1}, {max(0, y1 - 10)})")
                break
    except Exception as e:
        print(f"⚠ Could not draw bounding box: {e}")

def calculate_servo_angles(head_pos, frame_width, frame_height, fov_horizontal=66.7, fov_vertical=37.2):
    if head_pos is None:
        return None, None
    x, y = head_pos
    print(f"DEBUG: Head position - x={x:.1f}, y={y:.1f}")
    print(f"DEBUG: Frame size - {frame_width}x{frame_height}")
    rel_x = (x - frame_width/2) / (frame_width/2)
    rel_y = (y - frame_height/2) / (frame_height/2)
    print(f"DEBUG: Relative position - x={rel_x:.3f}, y={rel_y:.3f}")
    pan_angle = max(-fov_horizontal/2, min(fov_horizontal/2, rel_x * (fov_horizontal / 2)))
    tilt_angle = max(-fov_vertical/2, min(fov_vertical/2, -rel_y * (fov_vertical / 2)))
    print(f"DEBUG: Calculated angles - Pan={pan_angle:.1f}°, Tilt={tilt_angle:.1f}°")
    return pan_angle, tilt_angle

tracked_people = []
next_person_id = 1
TRACKING_THRESHOLD = 100
last_tracked_id = None

def track_people(current_centers):
    global tracked_people, next_person_id, frame_count
    new_tracked = []
    used_centers = set()
    for person_id, last_center, last_frame in tracked_people:
        if not current_centers:
            continue
        distances = [(i, math.hypot(center[0] - last_center[0], center[1] - last_center[1])) 
                     for i, center in enumerate(current_centers)]
        closest_idx, min_distance = min(distances, key=lambda x: x[1])
        if min_distance < TRACKING_THRESHOLD:
            new_tracked.append((person_id, current_centers[closest_idx], frame_count))
            used_centers.add(closest_idx)
        elif frame_count - last_frame < 30:
            new_tracked.append((person_id, last_center, last_frame))
    for i, center in enumerate(current_centers):
        if i not in used_centers:
            new_tracked.append((next_person_id, center, frame_count))
            print(f"DEBUG: New person detected, assigned Person {next_person_id}")
            next_person_id += 1
    tracked_people = new_tracked

parser = argparse.ArgumentParser()
parser.add_argument("--network", type=str, default="resnet18-body")
parser.add_argument("--overlay", type=str, default="links,keypoints")
parser.add_argument("--threshold", type=float, default=0.15)
parser.add_argument("--pan-channel", type=int, default=0)
parser.add_argument("--tilt-channel", type=int, default=1)
parser.add_argument("--sensor-id", type=int, default=0, help="Camera sensor ID (0 for CAM0, 1 for CAM1)")
parser.add_argument("--input", type=str, default="nvarguscamerasrc sensor-id={sensor_id} ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGR ! appsink")
parser.add_argument("output", type=str, nargs="?", default="display://0")
args = parser.parse_args()

# Format input pipeline with sensor-id
args.input = args.input.format(sensor_id=args.sensor_id)

print("Loading pose detection network...")
net = poseNet(args.network, sys.argv, args.threshold)

print("Initializing video source with cv2.VideoCapture...")
try:
    cap = cv2.VideoCapture(args.input, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("Failed to capture initial frame")
    frame_height, frame_width = frame.shape[:2]
    print(f"✓ Camera initialized: {frame_width}x{frame_height}")
except Exception as e:
    print(f"✗ Failed to initialize cv2.VideoCapture: {e}")
    print("Try running with --sensor-id=1 or check 'dmesg | grep -i imx519'")
    print("Ensure nvargus-daemon is running: 'sudo systemctl status nvargus-daemon'")
    print("Verify IMX519 connection to CAM0 port")
    print("Test pipeline: gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1' ! nvvidconv ! video/x-raw, format=BGR ! videoconvert ! autovideosink")
    sys.exit(1)

print("Creating output stream...")
output = videoOutput(args.output, argv=sys.argv)

print("Initializing servo controller...")
servo_controller = DebugServoController(
    pan_channel=args.pan_channel,
    tilt_channel=args.tilt_channel
)
servo_controller.test_servo_range()

frame_count = 0
last_detection_frame = 0

print("\n=== DEBUG HEAD TRACKING WITH UNFILLED BOUNDING BOX ===")
print("Watch the console for detailed servo movement logs")
print("Green unfilled box = head bounding box, Red circle/crosshair = head center")
print("Yellow line = tracking from center to head (OpenCV display)")
print("Press Ctrl+C or Q to stop")

try:
    while True:
        # Capture frame
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            print("Failed to capture frame")
            continue
        frame_count += 1
        frame_height, frame_width = frame_bgr.shape[:2]
        # Convert to CUDA for poseNet and CUDA drawing
        cuda_img = cudaFromNumpy(frame_bgr)
        poses = net.Process(cuda_img, overlay=args.overlay)
        print(f"DEBUG: Processing frame {frame_count}, {len(poses)} poses detected")
        current_centers = []
        person_data = []
        for pose in poses:
            head_center, head_bbox = calculate_head_center_and_bbox(pose)
            if head_center and head_bbox:
                current_centers.append(head_center)
                person_data.append((head_center, head_bbox))
        track_people(current_centers)
        head_center = None
        head_bbox = None
        if len(poses) > 0:
            print(f"\n--- FRAME {frame_count} ---")
            print(f"Detected {len(poses)} people")
            for person_id, tracked_center, _ in tracked_people:
                if person_id == last_tracked_id and tracked_center in current_centers:
                    head_center = tracked_center
                    for center, bbox in person_data:
                        if center == head_center:
                            head_bbox = bbox
                            break
                    break
            if head_center is None and person_data:
                head_center = person_data[0][0]
                head_bbox = person_data[0][1]
                for person_id, tracked_center, _ in tracked_people:
                    if tracked_center == head_center:
                        last_tracked_id = person_id
                        break
            if head_center and head_bbox:
                last_detection_frame = frame_count
                print(f"Head detected at: {head_center}")
                print(f"Head bbox: {head_bbox['width']:.1f}x{head_bbox['height']:.1f}")
                for center, bbox in person_data:
                    draw_bounding_box(cuda_img, bbox, center)
                pan_angle, tilt_angle = calculate_servo_angles(
                    head_center, frame_width, frame_height
                )
                servo_controller.move_to_tracking(pan_angle, tilt_angle)
                time.sleep(0.2)
            else:
                print("Head center could not be calculated")
        else:
            if frame_count - last_detection_frame < 30:
                print(f"Frame {frame_count}: No people detected, maintaining last position")
            else:
                print(f"Frame {frame_count}: No people detected, returning to center")
                servo_controller.move_to_raw(90, 90)
        output.Render(cuda_img)
        status_text = f"Debug Mode | Frame {frame_count} | Last detection: {last_detection_frame}"
        output.SetStatus(status_text)
        # OpenCV display
        cv2.line(frame_bgr, (frame_width//2-25, frame_height//2), 
                (frame_width//2+25, frame_height//2), (255, 0, 0), 2)
        cv2.line(frame_bgr, (frame_width//2, frame_height//2-25), 
                (frame_width//2, frame_height//2+25), (255, 0, 0), 2)
        status_color = (0, 255, 0) if len(poses) > 0 else (0, 0, 255)
        status_text = "TRACKING" if len(poses) > 0 else "NO DETECTION"
        cv2.putText(frame_bgr, f'Status: {status_text}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame_bgr, f'Pan: {servo_controller.last_pan:5.1f}°  Tilt: {servo_controller.last_tilt:5.1f}°', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if head_center:
            cv2.line(frame_bgr, (frame_width//2, frame_height//2), 
                    (int(head_center[0]), int(head_center[1])), (0, 255, 255), 2)
        cv2.imshow('Head Tracker - Press Q to quit', frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        if not output.IsStreaming():
            break
except KeyboardInterrupt:
    print("\n\nStopping...")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("Cleaning up...")
    servo_controller.cleanup()
    cap.release()
    output.Close()
    print("Done!")
