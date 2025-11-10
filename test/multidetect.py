"""
live_track_seg_debug.py - FIXED VERSION
Complete tracking solution with console-based target selection
Fixes: Detection format, tracking activation, path window display
"""

import time
import cv2
import numpy as np
import torch
import threading
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import queue

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "yolov8n-seg.pt"
DEVICE = "cpu"  # or "cuda" if GPU available
DETECT_WIDTH = 640
DETECT_EVERY_N = 1  # REDUCED from 3 - detect every frame for better tracking
CONF_THRESH = 0.45
MAX_COSINE_DISTANCE = 0.4  # Increased from 0.2 for more lenient matching
NMS_IOU_THRESH = 0.5
USE_MASKS = False
ONLY_PERSONS = True

# Path tracking parameters
MAX_PATH_LENGTH = 300
PATH_WINDOW_SIZE = (900, 700)

# ----------------------------
# Initialize model & tracker
# ----------------------------
print("Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)
try:
    model.fuse()
    print("Model fused successfully")
except Exception as e:
    print(f"Model fusion failed: {e}")

print("Initializing DeepSort tracker...")
# Using default DeepSort parameters for reliability
tracker = DeepSort(
    max_age=50,           # Increased from 30
    n_init=3,             # Number of frames before track is confirmed
    max_cosine_distance=MAX_COSINE_DISTANCE,
    nn_budget=100
)

# ----------------------------
# Global state management
# ----------------------------
track_paths = defaultdict(list)
selected_track_id = None
all_active_tracks = []
running = True
command_queue = queue.Queue()
path_window_lock = threading.Lock()

# ----------------------------
# Helper Functions
# ----------------------------
def resize_frame_keep_aspect(frame, width):
    """Resize frame maintaining aspect ratio"""
    h, w = frame.shape[:2]
    if w == width:
        return frame, 1.0
    scale = width / float(w)
    new_h = int(h * scale)
    frame_resized = cv2.resize(frame, (width, new_h))
    return frame_resized, scale

def extract_detections_yolov8(results, conf_thresh=0.45):
    """
    Extract detections from YOLOv8 results in proper format for DeepSort.
    Returns list of: (x1, y1, x2, y2, confidence, class_id)
    """
    detections = []
    
    if results is None or len(results) == 0:
        return detections
    
    result = results[0] if isinstance(results, (list, tuple)) else results
    
    try:
        # Extract boxes, confidence, and class information
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            return detections
        
        xyxy = boxes.xyxy.cpu().numpy()       # [N, 4] - x1, y1, x2, y2
        confs = boxes.conf.cpu().numpy()      # [N] - confidence scores
        clss = boxes.cls.cpu().numpy()        # [N] - class IDs
        
        for i in range(len(xyxy)):
            conf = float(confs[i])
            cls_id = int(clss[i])
            
            # Filter by confidence
            if conf < conf_thresh:
                continue
            
            # Filter by class (0 = person in COCO)
            if ONLY_PERSONS and cls_id != 0:
                continue
            
            x1, y1, x2, y2 = xyxy[i]
            detections.append({
                'x1': float(x1),
                'y1': float(y1),
                'x2': float(x2),
                'y2': float(y2),
                'conf': conf,
                'cls': cls_id
            })
    
    except Exception as e:
        print(f"Error extracting detections: {e}")
    
    return detections

def convert_to_deepsort_format(detections, frame_shape):
    """
    Convert detections to DeepSort format: list of (tlwh, confidence, class_id)
    tlwh = (top, left, width, height)
    """
    deepsort_detections = []
    
    for det in detections:
        x1 = int(det['x1'])
        y1 = int(det['y1'])
        x2 = int(det['x2'])
        y2 = int(det['y2'])
        conf = det['conf']
        cls_id = det['cls']
        
        # Ensure coordinates are within frame
        x1 = max(0, min(x1, frame_shape[1]))
        y1 = max(0, min(y1, frame_shape[0]))
        x2 = max(0, min(x2, frame_shape[1]))
        y2 = max(0, min(y2, frame_shape[0]))
        
        w = x2 - x1
        h = y2 - y1
        
        if w <= 0 or h <= 0:
            continue
        
        # TLWH format for DeepSort
        tlwh = [x1, y1, w, h]
        deepsort_detections.append((tlwh, conf, cls_id))
    
    return deepsort_detections

def get_color_by_target_count(num_targets, track_id=None, selected_id=None, is_confirmed=True):
    """
    Dynamic color based on target count and selection status.
    """
    # Selected target gets special color
    if track_id is not None and track_id == selected_id:
        return (255, 0, 255)  # Magenta
    
    # Unconfirmed tracks get cyan
    if not is_confirmed:
        return (255, 255, 0)  # Cyan
    
    # Color based on number of confirmed targets
    if num_targets <= 1:
        return (0, 255, 0)  # Green
    elif num_targets == 2:
        return (0, 255, 255)  # Yellow
    elif num_targets <= 4:
        return (0, 165, 255)  # Orange
    elif num_targets <= 6:
        return (0, 100, 255)  # Dark orange
    else:
        return (0, 0, 255)  # Red

def draw_path_visualization(track_id, path_history):
    """
    Create path visualization window for selected target.
    """
    canvas = np.zeros((PATH_WINDOW_SIZE[1], PATH_WINDOW_SIZE[0], 3), dtype=np.uint8)
    
    # Draw grid
    grid_spacing = 50
    for i in range(0, PATH_WINDOW_SIZE[0], grid_spacing):
        cv2.line(canvas, (i, 0), (i, PATH_WINDOW_SIZE[1]), (40, 40, 40), 1)
    for i in range(0, PATH_WINDOW_SIZE[1], grid_spacing):
        cv2.line(canvas, (0, i), (PATH_WINDOW_SIZE[0], i), (40, 40, 40), 1)
    
    # Draw center axes
    cv2.line(canvas, (PATH_WINDOW_SIZE[0]//2, 0), 
             (PATH_WINDOW_SIZE[0]//2, PATH_WINDOW_SIZE[1]), (80, 80, 80), 2)
    cv2.line(canvas, (0, PATH_WINDOW_SIZE[1]//2), 
             (PATH_WINDOW_SIZE[0], PATH_WINDOW_SIZE[1]//2), (80, 80, 80), 2)
    
    # Draw header
    cv2.rectangle(canvas, (0, 0), (PATH_WINDOW_SIZE[0], 60), (20, 20, 20), -1)
    cv2.putText(canvas, f"Track ID: {track_id} - Path Visualization", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    if len(path_history) < 2:
        cv2.putText(canvas, "Collecting path data...", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        return canvas
    
    # Normalize path to canvas
    path_array = np.array(path_history)
    min_x, min_y = path_array.min(axis=0)
    max_x, max_y = path_array.max(axis=0)
    
    range_x = max(max_x - min_x, 1)
    range_y = max(max_y - min_y, 1)
    
    margin = 80
    max_width = PATH_WINDOW_SIZE[0] - 2 * margin
    max_height = PATH_WINDOW_SIZE[1] - 2 * margin
    
    scale_x = max_width / range_x
    scale_y = max_height / range_y
    scale = min(scale_x, scale_y)
    
    # Convert to canvas coordinates
    normalized_path = []
    for px, py in path_history:
        nx = int((px - min_x) * scale + margin)
        ny = int((py - min_y) * scale + margin)
        normalized_path.append((nx, ny))
    
    # Draw path with color gradient
    for i in range(1, len(normalized_path)):
        alpha = i / len(normalized_path)
        color_val = int(100 + 155 * alpha)
        color = (color_val, 0, color_val)  # Purple gradient
        thickness = max(1, int(1 + 2 * alpha))
        cv2.line(canvas, normalized_path[i-1], normalized_path[i], color, thickness)
    
    # Draw start and end points
    if normalized_path:
        cv2.circle(canvas, normalized_path[0], 8, (0, 255, 0), -1)    # Green start
        cv2.circle(canvas, normalized_path[-1], 8, (0, 0, 255), -1)    # Red end
        
        # Draw direction arrow
        if len(normalized_path) > 1:
            p1 = normalized_path[-2]
            p2 = normalized_path[-1]
            cv2.arrowedLine(canvas, p1, p2, (255, 0, 0), 2, tipLength=0.3)
    
    # Draw stats
    y_offset = PATH_WINDOW_SIZE[1] - 60
    cv2.rectangle(canvas, (0, y_offset), (PATH_WINDOW_SIZE[0], PATH_WINDOW_SIZE[1]), (20, 20, 20), -1)
    cv2.putText(canvas, f"Total Points: {len(path_history)} | Distance Coverage: {range_x:.0f} x {range_y:.0f}",
                (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(canvas, "Green=Start | Red=Current | Blue=Direction",
                (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return canvas

def console_input_handler():
    """
    Handle user console input in separate thread.
    """
    global selected_track_id, running
    
    print("\n" + "="*80)
    print("TRACKING CONSOLE - COMMANDS:")
    print("="*80)
    print("  [ID]    - Enter target ID to track (e.g., 1, 2, 3)")
    print("  [L]     - List all active targets")
    print("  [C]     - Clear selection")
    print("  [Q]     - Quit application")
    print("="*80 + "\n")
    
    while running:
        try:
            cmd = input(">>> ").strip().upper()
            
            if cmd == 'Q':
                print("[SYSTEM] Quit command received. Closing...")
                running = False
                break
            elif cmd == 'C':
                selected_track_id = None
                print("[INFO] Selection cleared")
            elif cmd == 'L':
                if all_active_tracks:
                    print(f"[INFO] Active targets: {sorted(all_active_tracks)}")
                else:
                    print("[INFO] No active targets")
            else:
                try:
                    target_id = int(cmd)
                    if target_id in all_active_tracks:
                        selected_track_id = target_id
                        print(f"[SUCCESS] Tracking target ID: {target_id}")
                    else:
                        active_str = str(sorted(all_active_tracks)) if all_active_tracks else "None"
                        print(f"[ERROR] Target {target_id} not found. Active: {active_str}")
                except ValueError:
                    print(f"[ERROR] Invalid input '{cmd}'. Use L/C/Q or enter a target ID")
        
        except EOFError:
            break
        except KeyboardInterrupt:
            running = False
            break
        except Exception as e:
            print(f"[ERROR] {e}")

# ----------------------------
# Main Tracking Loop
# ----------------------------
def run_tracking(source=0):
    global selected_track_id, track_paths, all_active_tracks, running
    
    print("\n[INIT] Opening video source...")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return
    
    print("[INIT] Video source opened successfully")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Video: {frame_width}x{frame_height} @ {fps} FPS")
    
    # Start console input thread
    console_thread = threading.Thread(target=console_input_handler, daemon=True)
    console_thread.start()
    
    # Create main window
    cv2.namedWindow("YOLOv8 + DeepSort Tracking", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    fps_smooth = 0.0
    last_time = time.time()
    
    print("[INIT] Starting tracking loop...\n")
    
    while running:
        ret, frame = cap.read()
        if not ret:
            print("[SYSTEM] End of video or stream lost")
            break
        
        frame_count += 1
        current_time = time.time()
        
        # Resize for detection
        frame_resized, scale_factor = resize_frame_keep_aspect(frame, DETECT_WIDTH)
        
        # Detection phase
        detections_list = []
        
        if frame_count % DETECT_EVERY_N == 0:
            try:
                with torch.no_grad():
                    results = model.predict(
                        source=frame_resized,
                        device=DEVICE,
                        imgsz=DETECT_WIDTH,
                        conf=CONF_THRESH,
                        iou=NMS_IOU_THRESH,
                        verbose=False
                    )
                
                # Extract detections
                detections_raw = extract_detections_yolov8(results, CONF_THRESH)
                
                # Convert to original frame scale
                for det in detections_raw:
                    det['x1'] *= (1.0 / scale_factor)
                    det['y1'] *= (1.0 / scale_factor)
                    det['x2'] *= (1.0 / scale_factor)
                    det['y2'] *= (1.0 / scale_factor)
                
                # Convert to DeepSort format
                detections_list = convert_to_deepsort_format(detections_raw, frame.shape)
                
                if frame_count % 30 == 0:  # Log every 30 frames
                    print(f"[DETECT] Frame {frame_count}: {len(detections_list)} detections")
            
            except Exception as e:
                print(f"[ERROR] Detection failed: {e}")
                detections_list = []
        
        # Tracking phase - CRITICAL FIX
        try:
            # Update DeepSort with detections
            # Pass frame as uint8 array
            frame_uint8 = frame.astype(np.uint8)
            tracks = tracker.update_tracks(detections_list, frame=frame_uint8)
            
            # Filter confirmed tracks only
            confirmed_tracks = [t for t in tracks if t.is_confirmed()]
            all_active_tracks = [t.track_id for t in confirmed_tracks]
            
            if frame_count % 30 == 0:
                print(f"[TRACK] Frame {frame_count}: {len(confirmed_tracks)} confirmed tracks, "
                      f"IDs: {sorted(all_active_tracks)}")
        
        except Exception as e:
            print(f"[ERROR] Tracking failed: {e}")
            confirmed_tracks = []
            all_active_tracks = []
        
        # Visualization
        display_frame = frame.copy()
        num_targets = len(confirmed_tracks)
        
        # Draw detections (light boxes)
        for det in detections_raw:
            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (200, 150, 100), 1)
        
        # Draw tracks
        for track in confirmed_tracks:
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            
            # Update path
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            track_paths[track_id].append((cx, cy))
            if len(track_paths[track_id]) > MAX_PATH_LENGTH:
                track_paths[track_id].pop(0)
            
            # Get color
            color = get_color_by_target_count(num_targets, track_id, selected_track_id, True)
            
            # Draw bbox
            thickness = 3 if track_id == selected_track_id else 2
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw path trail
            if len(track_paths[track_id]) > 1:
                pts = np.array(track_paths[track_id][-20:], np.int32)
                for i in range(1, len(pts)):
                    cv2.line(display_frame, tuple(pts[i-1]), tuple(pts[i]), color, 1)
            
            # Draw label
            label = f"ID:{track_id}"
            if hasattr(track, 'last_detection_confidence') and track.last_detection_confidence:
                label += f" {track.last_detection_confidence:.2f}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(display_frame, (x1, y1-label_size[1]-8),
                         (x1+label_size[0]+5, y1), color, -1)
            cv2.putText(display_frame, label, (x1+2, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw selection indicator
            if track_id == selected_track_id:
                cv2.putText(display_frame, "★ SELECTED ★", (x1, y2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # FPS calculation
        dt = current_time - last_time
        last_time = current_time
        fps_current = 1.0 / (dt + 1e-6)
        fps_smooth = 0.85 * fps_smooth + 0.15 * fps_current
        
        # Info panel
        panel_color = (25, 25, 25)
        cv2.rectangle(display_frame, (5, 5), (420, 140), panel_color, -1)
        cv2.rectangle(display_frame, (5, 5), (420, 140), (150, 150, 150), 2)
        
        cv2.putText(display_frame, f"FPS: {fps_smooth:.1f}", (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Frame: {frame_count}", (15, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
        cv2.putText(display_frame, f"Active Targets: {num_targets}", (15, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(display_frame, f"Target IDs: {sorted(all_active_tracks) if all_active_tracks else 'None'}",
                   (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        cv2.putText(display_frame, f"Selected: {selected_track_id if selected_track_id else 'None'}",
                   (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
        
        # Display main window
        cv2.imshow("YOLOv8 + DeepSort Tracking", display_frame)
        
        # Display path window if target selected
        if selected_track_id is not None and selected_track_id in track_paths:
            path_canvas = draw_path_visualization(selected_track_id, track_paths[selected_track_id])
            cv2.imshow("Target Path Visualization", path_canvas)
        else:
            # Ensure window is closed if no selection
            cv2.destroyWindow("Target Path Visualization") if "Target Path Visualization" in cv2.getWindowProperty.__doc__ else None
        
        # Keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("[SYSTEM] ESC pressed. Shutting down...\n")
            running = False
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM] Tracking closed successfully\n")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("YOLOv8 DETECTION + DeepSORT TRACKING WITH PATH VISUALIZATION")
    print("="*80)
    print("Features:")
    print("  ✓ Real-time person detection with YOLOv8")
    print("  ✓ Multi-target tracking with DeepSort")
    print("  ✓ Console-based target selection")
    print("  ✓ Path history visualization")
    print("  ✓ Dynamic bounding box colors")
    print("="*80 + "\n")
    
    run_tracking(0)
 