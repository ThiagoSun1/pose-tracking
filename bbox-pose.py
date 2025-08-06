#!/usr/bin/env python3
"""
Debug version of head tracker optimized for MG995 servos
Unfilled head bounding box visualization with prominent FPS display ON VIDEO
"""

import cv2
from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput
import jetson_utils
import argparse
import sys
import time
import math
import numpy as np

# Import Adafruit libraries for servo control
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
        self.last_pan = 90  # Initialize to center
        self.last_tilt = 90
        
        # Initialize ServoKit
        print("Initializing ServoKit...")
        try:
            self.kit = ServoKit(channels=16)
            print("✓ ServoKit initialized")
            
            # Test immediate movement to confirm it works
            print("Testing immediate servo movement...")
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
        """Move servos using raw servo angles (0-180)"""
        try:
            print(f"DEBUG: Moving servos to Pan={pan_servo_angle:.1f}°, Tilt={tilt_servo_angle:.1f}°")
            
            # Clamp to valid servo range
            pan_servo_angle = max(0, min(180, pan_servo_angle))
            tilt_servo_angle = max(0, min(180, tilt_servo_angle))
            
            # Move servos
            self.kit.servo[self.pan_channel].angle = pan_servo_angle
            self.kit.servo[self.tilt_channel].angle = tilt_servo_angle
            
            # Minimal delay for small movements to prevent jitter
            if abs(pan_servo_angle - self.last_pan) > 2 or abs(tilt_servo_angle - self.last_tilt) > 2:
                time.sleep(0.01)  # Minimal delay for significant movements
            else:
                time.sleep(0.005)  # Ultra-minimal delay for tiny adjustments
            
            print(f"✓ Servos moved to Pan={pan_servo_angle:.1f}°, Tilt={tilt_servo_angle:.1f}°")
            
            # Update last angles
            self.last_pan = pan_servo_angle
            self.last_tilt = tilt_servo_angle
            
        except Exception as e:
            print(f"✗ Servo movement error: {e}")
            import traceback
            traceback.print_exc()
    
    def move_to_tracking(self, pan_angle, tilt_angle):
        """Convert tracking angles to servo angles and move"""
        if pan_angle is None or tilt_angle is None:
            print("DEBUG: Received None angles, skipping movement")
            return
        
        print(f"DEBUG: Tracking angles - Pan={pan_angle:.1f}°, Tilt={tilt_angle:.1f}°")
        
        # Convert from tracking coordinates to servo coordinates
        servo_pan = pan_angle + 90
        servo_tilt = tilt_angle + 90
        
        # Smooth movement by limiting angle change
        max_angle_change = 30  # Increased from 15 to 30 for faster MG995 response
        servo_pan = self.last_pan + max(-max_angle_change, min(max_angle_change, servo_pan - self.last_pan))
        servo_tilt = self.last_tilt + max(-max_angle_change, min(max_angle_change, servo_tilt - self.last_tilt))
        
        print(f"DEBUG: Converted to servo angles - Pan={servo_pan:.1f}°, Tilt={servo_tilt:.1f}°")
        
        # Move servos
        self.move_to_raw(servo_pan, servo_tilt)
    
    def test_servo_range(self):
        """Test full servo range"""
        print("Testing full servo range...")
        angles = [0, 45, 90, 135, 180]
        for angle in angles:
            print(f"Moving to Pan={angle}°, Tilt={angle}°")
            self.move_to_raw(angle, angle)
            time.sleep(0.5)  # Reduced delay for faster testing
        print("Returning to center...")
        self.move_to_raw(90, 90)
    
    def cleanup(self):
        """Return to center"""
        print("Returning servos to center...")
        self.move_to_raw(90, 90)

def calculate_head_center_and_bbox(pose):
    """Calculate head center and bounding box from pose keypoints"""
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

    # Calculate center
    avg_x = sum(point[0] for point in head_points) / len(head_points)
    avg_y = sum(point[1] for point in head_points) / len(head_points)
    center = (avg_x, avg_y)

    # Calculate bounding box
    min_x = min(point[0] for point in head_points)
    max_x = max(point[0] for point in head_points)
    min_y = min(point[1] for point in head_points)
    max_y = max(point[1] for point in head_points)
    
    # Add padding around the head (expand by 50%)
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
    """Draw unfilled bounding box and center point using native Jetson CUDA functions"""
    if bbox is None or center is None:
        return
    
    try:
        # Draw green bounding box outline (unfilled)
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        
        # Draw four lines to create unfilled rectangle
        jetson_utils.cudaDrawLine(cuda_img, (x1, y1), (x2, y1), (0, 255, 0, 255))  # Top line
        jetson_utils.cudaDrawLine(cuda_img, (x2, y1), (x2, y2), (0, 255, 0, 255))  # Right line
        jetson_utils.cudaDrawLine(cuda_img, (x2, y2), (x1, y2), (0, 255, 0, 255))  # Bottom line
        jetson_utils.cudaDrawLine(cuda_img, (x1, y2), (x1, y1), (0, 255, 0, 255))  # Left line
        
        # Draw red center point circle
        jetson_utils.cudaDrawCircle(cuda_img, 
                                   (int(center[0]), int(center[1])), 
                                   5, (255, 0, 0, 255))  # Red circle
        
        # Draw crosshair lines at center
        jetson_utils.cudaDrawLine(cuda_img,
                                 (int(center[0] - 10), int(center[1])),
                                 (int(center[0] + 10), int(center[1])),
                                 (255, 0, 0, 255))  # Red horizontal line
        jetson_utils.cudaDrawLine(cuda_img,
                                 (int(center[0]), int(center[1] - 10)),
                                 (int(center[0]), int(center[1] + 10)),
                                 (255, 0, 0, 255))  # Red vertical line
        
        print(f"✓ Drew unfilled bounding box at center ({int(center[0])}, {int(center[1])})")
        
    except Exception as e:
        print(f"⚠ Could not draw bounding box: {e}")

def draw_fps_overlay(cuda_img, fps_controller, frame_count, detection_count):
    """Draw FPS and status information on the video feed"""
    try:
        # Get image dimensions
        width = cuda_img.width
        height = cuda_img.height
        
        # Create background rectangle for FPS display (top-left corner)
        jetson_utils.cudaDrawRect(cuda_img, (10, 10, 350, 140), (0, 0, 0, 200))  # Semi-transparent black
        
        # Target FPS text (large, bright green)
        target_fps_text = f"Target: {fps_controller.target_fps} FPS"
        jetson_utils.cudaDrawText(cuda_img, target_fps_text, (20, 30), (0, 255, 0, 255), 
                                 jetson_utils.cudaFont.Make(size=20))
        
        # Actual FPS text (large, cyan)
        actual_fps_text = f"Actual: {fps_controller.actual_fps:.1f} FPS"
        jetson_utils.cudaDrawText(cuda_img, actual_fps_text, (20, 55), (0, 255, 255, 255), 
                                 jetson_utils.cudaFont.Make(size=20))
        
        # Frame count (smaller, white)
        frame_text = f"Frame: {frame_count}"
        jetson_utils.cudaDrawText(cuda_img, frame_text, (20, 80), (255, 255, 255, 255), 
                                 jetson_utils.cudaFont.Make(size=16))
        
        # Detection status (smaller, yellow)
        detection_text = f"Detections: {detection_count}"
        jetson_utils.cudaDrawText(cuda_img, detection_text, (20, 100), (255, 255, 0, 255), 
                                 jetson_utils.cudaFont.Make(size=16))
        
        # Control instructions (top-right corner)
        control_text = "Press: +/- to change FPS"
        jetson_utils.cudaDrawText(cuda_img, control_text, (width - 250, 30), 
                                 (255, 255, 255, 255), jetson_utils.cudaFont.Make(size=14))
        
        # Status indicator (bottom-right corner)
        status_text = f"HEAD TRACKING @ {fps_controller.target_fps}FPS"
        jetson_utils.cudaDrawText(cuda_img, status_text, (width - 280, height - 30), 
                                 (0, 255, 255, 255), jetson_utils.cudaFont.Make(size=16))
        
        print(f"✓ Drew FPS overlay: Target {fps_controller.target_fps} | Actual {fps_controller.actual_fps:.1f}")
        
    except Exception as e:
        print(f"⚠ Could not draw FPS overlay: {e}")

def calculate_servo_angles(head_pos, frame_width, frame_height, fov_horizontal=54, fov_vertical=30):
    """Calculate pan/tilt angles based on head position in frame"""
    if head_pos is None:
        return None, None

    x, y = head_pos
    
    print(f"DEBUG: Head position - x={x:.1f}, y={y:.1f}")
    print(f"DEBUG: Frame size - {frame_width}x{frame_height}")

    # Calculate relative position from center (-1 to 1)
    rel_x = (x - frame_width/2) / (frame_width/2)
    rel_y = (y - frame_height/2) / (frame_height/2)
    
    print(f"DEBUG: Relative position - x={rel_x:.3f}, y={rel_y:.3f}")

    # Convert to servo angles based on field of view and clamp to FoV/2
    pan_angle = max(-fov_horizontal/2, min(fov_horizontal/2, rel_x * (fov_horizontal / 2)))
    tilt_angle = max(-fov_vertical/2, min(fov_vertical/2, -rel_y * (fov_vertical / 2)))

    print(f"DEBUG: Calculated angles - Pan={pan_angle:.1f}°, Tilt={tilt_angle:.1f}°")
    
    return pan_angle, tilt_angle

# FPS Control Class
class FPSController:
    def __init__(self, target_fps=5):
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps
        self.last_frame_time = time.time()
        self.actual_fps = 0
        print(f"✓ FPS Controller initialized - Target: {target_fps} FPS")
    
    def set_target_fps(self, fps):
        """Set new target FPS"""
        self.target_fps = max(1, min(fps, 60))  # Clamp between 1-60
        self.target_frame_time = 1.0 / self.target_fps
        print(f"Target FPS changed to: {self.target_fps}")
    
    def wait_for_next_frame(self):
        """Wait to maintain target FPS"""
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        
        if elapsed < self.target_frame_time:
            sleep_time = self.target_frame_time - elapsed
            time.sleep(sleep_time)
            current_time = time.time()
        
        # Calculate actual FPS
        actual_elapsed = current_time - self.last_frame_time
        self.actual_fps = 1.0 / actual_elapsed if actual_elapsed > 0 else 0
        self.last_frame_time = current_time
        
        return self.actual_fps
    
    def update_fps(self, current_fps):
        """Update the actual FPS value"""
        self.actual_fps = current_fps
    
    def get_status(self):
        """Get FPS status string"""
        return f"Target: {self.target_fps} | Actual: {self.actual_fps:.1f}"

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--network", type=str, default="resnet18-body")
parser.add_argument("--overlay", type=str, default="links,keypoints")
parser.add_argument("--threshold", type=float, default=0.15)
parser.add_argument("--pan-channel", type=int, default=0)
parser.add_argument("--tilt-channel", type=int, default=1)
parser.add_argument("--input", type=str, default="/dev/video0")
parser.add_argument("--target-fps", type=float, default=5, help="Target FPS (default: 5)")
parser.add_argument("output", type=str, nargs="?", default="display://0")
args = parser.parse_args()

# Initialize everything
print("Loading pose detection network...")
net = poseNet(args.network, sys.argv, args.threshold)

print("Initializing video source...")
print(f"Attempting to connect to: {args.input}")

# Try different video source configurations
video_sources_to_try = [
    args.input,  # User specified or default
    "/dev/video0",  # Standard USB camera
    "/dev/video1",  # Alternative USB camera
    "csi://0",  # CSI camera
    "v4l2:///dev/video0",  # V4L2 explicit
    "0",  # Simple camera index
]

input_source = None
for source in video_sources_to_try:
    try:
        print(f"Trying video source: {source}")
        input_source = videoSource(source, argv=sys.argv)
        print(f"✓ Successfully connected to: {source}")
        break
    except Exception as e:
        print(f"✗ Failed to connect to {source}: {e}")
        continue

if input_source is None:
    print("\n" + "="*50)
    print("❌ CAMERA CONNECTION FAILED")
    print("="*50)
    print("Troubleshooting steps:")
    print("1. Check if camera is connected:")
    print("   ls /dev/video*")
    print("2. Check camera permissions:")
    print("   sudo chmod 666 /dev/video0")
    print("3. Check if camera is in use:")
    print("   sudo lsof /dev/video0")
    print("4. Try with different input:")
    print("   python3 script.py --input /dev/video1")
    print("   python3 script.py --input csi://0")
    print("5. Test camera with:")
    print("   v4l2-ctl --list-devices")
    print("   nvgstcapture-1.0")
    print("="*50)
    sys.exit(1)

print(f"DEBUG: Camera resolution - {input_source.GetWidth()}x{input_source.GetHeight()}")

print("Creating output stream...")
output = videoOutput(args.output, argv=sys.argv)

print("Initializing servo controller...")
servo_controller = DebugServoController(
    pan_channel=args.pan_channel,
    tilt_channel=args.tilt_channel
)
servo_controller.test_servo_range()  # Test servo range on startup

# Initialize FPS controller
fps_controller = FPSController(target_fps=args.target_fps)

# Tracking variables
frame_count = 0
last_detection_frame = 0
detection_count = 0

# FPS calculation variables
fps_history = []
fps_window_size = 10  # Calculate FPS over last 10 frames for smoothing

print("\n=== DEBUG HEAD TRACKING WITH UNFILLED BOUNDING BOX ===")
print("Watch the console for detailed servo movement logs")
print("Green unfilled box = head bounding box, Red circle/crosshair = head center")
print("FPS display will appear on the video feed")
print("Press Ctrl+C to stop")

try:
    while True:
        frame_start_time = time.time()
        
        # Capture image
        cuda_img = input_source.Capture()
        if cuda_img is None:
            continue

        frame_count += 1
        
        # Get dimensions
        frame_width = cuda_img.width
        frame_height = cuda_img.height

        # Run pose estimation
        poses = net.Process(cuda_img, overlay=args.overlay)

        # Process detections
        print(f"DEBUG: Processing frame {frame_count}, {len(poses)} poses detected")
        if len(poses) > 0:
            detection_count += 1
            print(f"\n--- FRAME {frame_count} ---")
            print(f"Detected {len(poses)} people")
            
            # Use first detected person
            pose = poses[0]
            head_center, head_bbox = calculate_head_center_and_bbox(pose)
            
            if head_center and head_bbox:
                last_detection_frame = frame_count
                print(f"Head detected at: {head_center}")
                print(f"Head bbox: {head_bbox['width']:.1f}x{head_bbox['height']:.1f}")
                
                # Draw unfilled bounding box and center
                draw_bounding_box(cuda_img, head_bbox, head_center)
                
                # Calculate servo angles
                pan_angle, tilt_angle = calculate_servo_angles(
                    head_center, frame_width, frame_height
                )
                
                # Move servos
                servo_controller.move_to_tracking(pan_angle, tilt_angle)
                
            else:
                print("Head center could not be calculated")
        else:
            if frame_count - last_detection_frame < 30:
                print(f"Frame {frame_count}: No people detected, maintaining last position")
            else:
                print(f"Frame {frame_count}: No people detected, returning to center")
                servo_controller.move_to_raw(90, 90)

        # Calculate FPS
        frame_end_time = time.time()
        frame_processing_time = frame_end_time - frame_start_time
        current_fps = 1 / frame_processing_time if frame_processing_time > 0 else 0
        
        # Add to FPS history and calculate smooth average
        fps_history.append(current_fps)
        if len(fps_history) > fps_window_size:
            fps_history.pop(0)
        
        smooth_fps = sum(fps_history) / len(fps_history)
        
        # Update FPS controller with current FPS
        fps_controller.update_fps(smooth_fps)
        
        # Draw FPS overlay on video - now passing the fps_controller object
        draw_fps_overlay(cuda_img, fps_controller, frame_count, detection_count)
        
        # Console FPS display every 30 frames
        if frame_count % 30 == 0:
            print(f"\n{'='*50}")
            print(f"🎯 FPS STATUS - Frame {frame_count}")
            print(f"   Current FPS: {current_fps:.1f}")
            print(f"   Average FPS: {smooth_fps:.1f}")
            print(f"   Total Detections: {detection_count}")
            print(f"   Processing Time: {frame_processing_time:.3f}s")
            print(f"{'='*50}\n")
            sys.stdout.flush()

        # Render output
        output.Render(cuda_img)
        status_text = f"Debug Mode | Frame {frame_count} | FPS: {smooth_fps:.1f} | Detections: {detection_count}"
        output.SetStatus(status_text)

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
    input_source.Close()
    output.Close()
    print("Done!")
