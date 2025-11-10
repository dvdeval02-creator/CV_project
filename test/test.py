"""
live_track_seg_debug.py
Modified version with:
1. Dynamic bbox colors based on target count
2. Unique ID display (already implemented by DeepSort)
3. Console-based target selection with path mapping in separate window
"""

import time
import cv2
import numpy as np
import torch
import threading
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

# ----------------------------
# Config (tweak for speed)
# ----------------------------
MODEL_PATH = "yolov8n-seg.pt"
DEVICE = "cpu"                 # set "cuda" if you have GPU + torch
DETECT_WIDTH = 640
DETECT_EVERY_N = 3
CONF_THRESH = 0.35
MAX_COSINE_DISTANCE = 0.2
NMS_IOU_THRESH = 0.5
USE_MASKS = False
ONLY_PERSONS = True

# Path tracking parameters
MAX_PATH_LENGTH = 200  # Maximum number of points to store in path
PATH_WINDOW_SIZE = (800, 600)  # Size of path visualization window

# ----------------------------
# Initialize model & tracker
# ----------------------------
print("Loading model...")
model = YOLO(MODEL_PATH)
try:
    model.fuse()
except Exception:
    pass

print("Initializing tracker...")
tracker = DeepSort(max_age=30,
                   n_init=1,
                   max_cosine_distance=MAX_COSINE_DISTANCE)

# ----------------------------
# Global variables for path tracking
# ----------------------------
track_paths = defaultdict(list)  # Store path history for each track ID
selected_track_id = None  # Currently selected track for path visualization
all_active_tracks = []  # List of currently active track IDs
input_lock = threading.Lock()  # Thread safety for console input

# ----------------------------
# Helpers
# ----------------------------
def resize_frame_keep_aspect(frame, width):
    h, w = frame.shape[:2]
    if w == width:
        return frame, 1.0
    scale = width / float(w)
    new_h = int(h * scale)
    frame_resized = cv2.resize(frame, (width, new_h))
    return frame_resized, scale

def get_detections_from_result(res, conf_thresh=0.35, only_persons=True):
    dets = []
    try:
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
    except Exception:
        try:
            arr = res.xyxy
            xyxy = arr[0][:, :4].cpu().numpy()
            confs = arr[0][:, 4].cpu().numpy()
            clss = arr[0][:, 5].cpu().numpy().astype(int)
        except Exception:
            return dets

    masks = None
    if USE_MASKS:
        try:
            masks_obj = res.masks
            masks = masks_obj.data.cpu().numpy()
        except Exception:
            masks = None

    for i, box in enumerate(xyxy):
        conf = float(confs[i])
        cls_id = int(clss[i])
        if conf < conf_thresh:
            continue
        if only_persons and cls_id != 0:
            continue
        x1, y1, x2, y2 = box.tolist()
        mask = None
        if masks is not None:
            try:
                mask = masks[i]
            except Exception:
                mask = None
        dets.append({'xyxy':[x1,y1,x2,y2], 'conf':conf, 'cls':cls_id, 'mask':mask})
    return dets

def get_color_for_target_count(num_targets, track_id=None, selected_id=None):
    """
    Returns BGR color based on number of targets.
    Different color schemes based on target count.
    Highlights selected target.
    """
    # If this is the selected target, use a distinct color
    if track_id is not None and track_id == selected_id:
        return (255, 0, 255)  # Magenta for selected target
    
    # Color scheme changes based on total number of targets
    if num_targets == 0:
        return (0, 200, 0)  # Green
    elif num_targets == 1:
        return (0, 255, 0)  # Bright green
    elif num_targets == 2:
        return (0, 255, 255)  # Yellow
    elif num_targets <= 4:
        return (0, 165, 255)  # Orange
    elif num_targets <= 6:
        return (0, 100, 255)  # Dark orange
    else:
        return (0, 0, 255)  # Red (many targets)

def draw_path_window(track_id, path_history, frame_shape):
    """
    Draw the path of selected target in a separate window.
    """
    h, w = PATH_WINDOW_SIZE[1], PATH_WINDOW_SIZE[0]
    path_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Draw grid
    grid_spacing = 50
    for i in range(0, w, grid_spacing):
        cv2.line(path_canvas, (i, 0), (i, h), (40, 40, 40), 1)
    for i in range(0, h, grid_spacing):
        cv2.line(path_canvas, (0, i), (w, i), (40, 40, 40), 1)
    
    # Draw axes
    cv2.line(path_canvas, (w//2, 0), (w//2, h), (80, 80, 80), 1)
    cv2.line(path_canvas, (0, h//2), (w, h//2), (80, 80, 80), 1)
    
    if len(path_history) < 2:
        cv2.putText(path_canvas, f"Tracking ID: {track_id}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(path_canvas, "Waiting for movement...", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        return path_canvas
    
    # Normalize path to fit window
    path_array = np.array(path_history)
    min_x, min_y = path_array.min(axis=0)
    max_x, max_y = path_array.max(axis=0)
    
    range_x = max_x - min_x if max_x > min_x else 1
    range_y = max_y - min_y if max_y > min_y else 1
    
    margin = 50
    scale_x = (w - 2 * margin) / range_x
    scale_y = (h - 2 * margin) / range_y
    scale = min(scale_x, scale_y)
    
    # Convert path to window coordinates
    normalized_path = []
    for px, py in path_history:
        norm_x = int((px - min_x) * scale + margin)
        norm_y = int((py - min_y) * scale + margin)
        normalized_path.append((norm_x, norm_y))
    
    # Draw path with gradient (older = darker, newer = brighter)
    num_points = len(normalized_path)
    for i in range(1, num_points):
        alpha = i / num_points
        color_intensity = int(100 + 155 * alpha)
        color = (color_intensity, 0, color_intensity)  # Purple gradient
        thickness = 2 if i > num_points * 0.8 else 1
        cv2.line(path_canvas, normalized_path[i-1], normalized_path[i], color, thickness)
    
    # Draw start and end markers
    if normalized_path:
        cv2.circle(path_canvas, normalized_path[0], 5, (0, 255, 0), -1)  # Green start
        cv2.circle(path_canvas, normalized_path[-1], 7, (0, 0, 255), -1)  # Red current position
    
    # Draw info
    cv2.putText(path_canvas, f"Tracking ID: {track_id}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(path_canvas, f"Path points: {len(path_history)}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(path_canvas, "Green=Start, Red=Current", (10, h-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    return path_canvas

def console_input_thread():
    """
    Thread function to handle console input for target selection.
    Runs in background and allows user to enter target IDs.
    """
    global selected_track_id
    
    print("\n" + "="*70)
    print("CONSOLE INPUT THREAD ACTIVE")
    print("="*70)
    print("Commands:")
    print("  - Enter a number: Select target by ID (e.g., 1, 2, 3)")
    print("  - Press 'l' or 'L': List all active targets")
    print("  - Press 'c' or 'C': Clear selection")
    print("  - Press 'q' or 'Q': Quit application")
    print("="*70 + "\n")
    
    while True:
        try:
            user_input = input(">>> Enter target ID or command: ").strip().lower()
            
            with input_lock:
                if user_input == 'q':
                    print("Quit command received. Press ESC in video window to exit.")
                    return
                elif user_input == 'c':
                    selected_track_id = None
                    print("✓ Selection cleared")
                elif user_input == 'l':
                    if all_active_tracks:
                        print(f"✓ Active targets: {sorted(all_active_tracks)}")
                    else:
                        print("✗ No active targets")
                else:
                    try:
                        target_id = int(user_input)
                        if target_id in all_active_tracks:
                            selected_track_id = target_id
                            print(f"✓ Selected target ID: {target_id}")
                        else:
                            print(f"✗ Target ID {target_id} not found. Available: {sorted(all_active_tracks)}")
                    except ValueError:
                        print("✗ Invalid input. Enter a number, 'l' (list), 'c' (clear), or 'q' (quit)")
        except EOFError:
            # Handle pipe closure gracefully
            break
        except Exception as e:
            print(f"✗ Error: {e}")

# ----------------------------
# Main loop
# ----------------------------
def run_live(source=0):
    global selected_track_id, track_paths, all_active_tracks
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("ERROR: cannot open video source:", source)
        return

    # Start console input thread
    input_thread = threading.Thread(target=console_input_thread, daemon=True)
    input_thread.start()
    
    frame_idx = 0
    last_time = time.time()
    fps_smooth = 0.0
    scale = 1.0
    
    # Create windows
    main_window = "Live Track Debug (ESC to quit)"
    cv2.namedWindow(main_window)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        proc_frame, scale = resize_frame_keep_aspect(frame, DETECT_WIDTH)
        detections_for_tracker = []

        if frame_idx % DETECT_EVERY_N == 0:
            with torch.no_grad():
                res_list = model.predict(source=proc_frame, device=DEVICE, imgsz=DETECT_WIDTH, 
                                        conf=CONF_THRESH, iou=NMS_IOU_THRESH, verbose=False)
            res = res_list[0] if isinstance(res_list, (list, tuple)) else res_list
            dets = get_detections_from_result(res, conf_thresh=CONF_THRESH, only_persons=ONLY_PERSONS)

            # Convert to DeepSort expected format: tlwh (left, top, width, height)
            for d in dets:
                x1, y1, x2, y2 = d['xyxy']
                inv_scale = 1.0 / scale
                left = int(round(x1 * inv_scale))
                top = int(round(y1 * inv_scale))
                right = int(round(x2 * inv_scale))
                bottom = int(round(y2 * inv_scale))
                w = right - left
                h = bottom - top
                if w <= 0 or h <= 0:
                    continue
                conf = d['conf']
                cls = d['cls']
                tlwh = [left, top, w, h]
                detections_for_tracker.append((tlwh, conf, cls))

            print(f"[Frame {frame_idx}] Detections -> {len(detections_for_tracker)} sent to tracker")
        else:
            detections_for_tracker = []

        # Update tracker
        tracks = tracker.update_tracks(detections_for_tracker, frame=frame)
        tracks_list = [tr for tr in tracks if tr.is_confirmed()]
        
        # Update list of active track IDs
        all_active_tracks = [tr.track_id for tr in tracks_list]
        
        # Validate selected track is still active
        with input_lock:
            if selected_track_id is not None and selected_track_id not in all_active_tracks:
                print(f"[System] Selected target ID {selected_track_id} lost. Selection cleared.")
                selected_track_id = None
        
        # Count confirmed tracks for color scheme
        num_confirmed = len(tracks_list)

        # Visualization
        display = frame.copy()

        # Draw raw detections (for debugging)
        if detections_for_tracker:
            for (tlwh, conf, cls) in detections_for_tracker:
                l, t, w, h = [int(v) for v in tlwh]
                cv2.rectangle(display, (l, t), (l+w, t+h), (200, 120, 0), 1)
                cv2.putText(display, f"{conf:.2f}", (l, t-6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 120, 0), 1)

        # Update paths and draw tracks
        for tr in tracks:
            if not tr.is_confirmed():
                continue
                
            tid = tr.track_id
            ltrb = tr.to_ltrb()  # left, top, right, bottom
            left, top, right, bottom = [int(v) for v in ltrb]
            
            # Calculate center point for path
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            
            # Update path history
            track_paths[tid].append((center_x, center_y))
            if len(track_paths[tid]) > MAX_PATH_LENGTH:
                track_paths[tid].pop(0)
            
            # Get dynamic color based on target count
            with input_lock:
                current_selected = selected_track_id
            color = get_color_for_target_count(num_confirmed, tid, current_selected)
            
            # Draw bounding box with thickness based on selection
            thickness = 3 if tid == current_selected else 2
            cv2.rectangle(display, (left, top), (right, bottom), color, thickness)
            
            # Draw path trail on main window (last 30 points)
            if len(track_paths[tid]) > 1:
                path_points = track_paths[tid][-30:]
                for i in range(1, len(path_points)):
                    cv2.line(display, path_points[i-1], path_points[i], color, 1)
            
            # Draw label with unique ID
            label = f"ID:{tid}"
            if hasattr(tr, 'last_detection_confidence'):
                label += f" {tr.last_detection_confidence:.2f}"
            
            # Draw label background for better visibility
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(display, (left, top-label_size[1]-8), 
                         (left+label_size[0], top), color, -1)
            cv2.putText(display, label, (left, top-6), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw selection indicator
            if tid == current_selected:
                cv2.putText(display, "SELECTED", (left, bottom+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # FPS and info overlay
        now = time.time()
        dt = now - last_time
        last_time = now
        fps = 1.0 / (dt + 1e-6)
        fps_smooth = 0.8 * fps_smooth + 0.2 * fps
        
        # Info panel background
        cv2.rectangle(display, (5, 5), (350, 120), (0, 0, 0), -1)
        cv2.rectangle(display, (5, 5), (350, 120), (100, 100, 100), 2)
        
        # Info text
        cv2.putText(display, f"FPS: {fps_smooth:.1f}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display, f"Targets: {num_confirmed}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        with input_lock:
            current_selected = selected_track_id
        cv2.putText(display, f"Selected: {current_selected if current_selected else 'None'}", 
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(display, f"Active IDs: {sorted(all_active_tracks) if all_active_tracks else 'None'}", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)

        # Show main window
        cv2.imshow(main_window, display)
        
        # Show path window if a target is selected
        with input_lock:
            current_selected = selected_track_id
        
        if current_selected is not None and current_selected in track_paths:
            path_canvas = draw_path_window(current_selected, 
                                          track_paths[current_selected], 
                                          frame.shape)
            cv2.imshow("Path Visualization", path_canvas)
        else:
            # Close path window if no target selected
            try:
                cv2.destroyWindow("Path Visualization")
            except:
                pass

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            print("\n[System] ESC pressed. Shutting down...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[System] Application closed.")

if __name__ == "__main__":
    run_live(0)
