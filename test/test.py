import time
import cv2
import numpy as np
import torch
import threading
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import queue

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_PATH = "yolov8n.pt"  # Use yolov8n.pt for speed, yolov8n-seg.pt for segmentation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DETECT_WIDTH = 640
DETECT_EVERY_N = 1  # Detect every N frames (1 = every frame)
CONF_THRESH = 0.45
NMS_IOU_THRESH = 0.5
ONLY_PERSONS = True

# Tracker parameters
MAX_COSINE_DISTANCE = 0.4
TRACKER_MAX_AGE = 50
TRACKER_N_INIT = 3

# Path tracking
MAX_PATH_LENGTH = 300
PATH_WINDOW_SIZE = (900, 700)

# Video recording
OUTPUT_VIDEO_CODEC = 'mp4v'  # or 'XVID', 'MJPG'
OUTPUT_VIDEO_FPS = 20.0

# =============================================================================
# GLOBAL STATE
# =============================================================================
track_paths = defaultdict(list)
selected_track_id = None
uid_map = {}  # track_id -> user-assigned UID
uid_map_lock = threading.Lock()
all_active_tracks = []
running = True

# Video recording
is_recording = False
video_writer = None
video_writer_lock = threading.Lock()
recording_start_time = None

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
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
    """Extract detections from YOLOv8 results"""
    detections = []
    
    if results is None or len(results) == 0:
        return detections
    
    result = results[0] if isinstance(results, (list, tuple)) else results
    
    try:
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            return detections
        
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()
        
        for i in range(len(xyxy)):
            conf = float(confs[i])
            cls_id = int(clss[i])
            
            if conf < conf_thresh:
                continue
            
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
        print(f"[ERROR] Detection extraction: {e}")
    
    return detections

def convert_to_deepsort_format(detections, frame_shape):
    """Convert detections to DeepSort format: (tlwh, confidence, class_id)"""
    deepsort_detections = []
    
    for det in detections:
        x1 = int(det['x1'])
        y1 = int(det['y1'])
        x2 = int(det['x2'])
        y2 = int(det['y2'])
        conf = det['conf']
        cls_id = det['cls']
        
        # Clamp to frame boundaries
        x1 = max(0, min(x1, frame_shape[1]))
        y1 = max(0, min(y1, frame_shape[0]))
        x2 = max(0, min(x2, frame_shape[1]))
        y2 = max(0, min(y2, frame_shape[0]))
        
        w = x2 - x1
        h = y2 - y1
        
        if w <= 0 or h <= 0:
            continue
        
        tlwh = [x1, y1, w, h]
        deepsort_detections.append((tlwh, conf, cls_id))
    
    return deepsort_detections

def get_color_for_track(track_id, num_targets, selected_id, has_uid=False):

    # UID assigned = Magenta
    if has_uid:
        return (255, 0, 255)
    
    # Selected target = Cyan
    if track_id == selected_id:
        return (255, 255, 0)
    
    # Color based on target density
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

def draw_path_visualization(track_id, path_history, uid=None):
   
    canvas = np.zeros((PATH_WINDOW_SIZE[1], PATH_WINDOW_SIZE[0], 3), dtype=np.uint8)
    
    # Grid
    grid_spacing = 50
    for i in range(0, PATH_WINDOW_SIZE[0], grid_spacing):
        cv2.line(canvas, (i, 0), (i, PATH_WINDOW_SIZE[1]), (40, 40, 40), 1)
    for i in range(0, PATH_WINDOW_SIZE[1], grid_spacing):
        cv2.line(canvas, (0, i), (PATH_WINDOW_SIZE[0], i), (40, 40, 40), 1)
    
    # Center axes
    cv2.line(canvas, (PATH_WINDOW_SIZE[0]//2, 0), 
             (PATH_WINDOW_SIZE[0]//2, PATH_WINDOW_SIZE[1]), (80, 80, 80), 2)
    cv2.line(canvas, (0, PATH_WINDOW_SIZE[1]//2), 
             (PATH_WINDOW_SIZE[0], PATH_WINDOW_SIZE[1]//2), (80, 80, 80), 2)
    
    # Header
    cv2.rectangle(canvas, (0, 0), (PATH_WINDOW_SIZE[0], 70), (20, 20, 20), -1)
    title = f"Track ID: {track_id}"
    if uid:
        title += f" | UID: {uid}"
    title += " - Path Visualization"
    cv2.putText(canvas, title, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    if len(path_history) < 2:
        cv2.putText(canvas, "Collecting path data...", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        return canvas
    
    # Normalize path
    path_array = np.array(path_history)
    min_x, min_y = path_array.min(axis=0)
    max_x, max_y = path_array.max(axis=0)
    
    range_x = max(max_x - min_x, 1)
    range_y = max(max_y - min_y, 1)
    
    margin = 90
    max_width = PATH_WINDOW_SIZE[0] - 2 * margin
    max_height = PATH_WINDOW_SIZE[1] - 2 * margin - 70
    
    scale_x = max_width / range_x
    scale_y = max_height / range_y
    scale = min(scale_x, scale_y)
    
    # Convert to canvas coordinates
    normalized_path = []
    for px, py in path_history:
        nx = int((px - min_x) * scale + margin)
        ny = int((py - min_y) * scale + margin + 70)
        normalized_path.append((nx, ny))
    
    # Draw path with gradient
    for i in range(1, len(normalized_path)):
        alpha = i / len(normalized_path)
        color_val = int(100 + 155 * alpha)
        color = (color_val, 0, color_val)
        thickness = max(1, int(1 + 2 * alpha))
        cv2.line(canvas, normalized_path[i-1], normalized_path[i], color, thickness)
    
    # Start and end markers
    if normalized_path:
        cv2.circle(canvas, normalized_path[0], 8, (0, 255, 0), -1)
        cv2.circle(canvas, normalized_path[-1], 8, (0, 0, 255), -1)
        
        if len(normalized_path) > 1:
            p1 = normalized_path[-2]
            p2 = normalized_path[-1]
            cv2.arrowedLine(canvas, p1, p2, (255, 0, 0), 2, tipLength=0.3)
    
    # Stats footer
    y_offset = PATH_WINDOW_SIZE[1] - 60
    cv2.rectangle(canvas, (0, y_offset), (PATH_WINDOW_SIZE[0], PATH_WINDOW_SIZE[1]), (20, 20, 20), -1)
    cv2.putText(canvas, f"Points: {len(path_history)} | Coverage: {range_x:.0f}x{range_y:.0f}",
                (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(canvas, "Green=Start | Red=Current | Blue=Direction",
                (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return canvas

def start_recording(frame_width, frame_height, fps):
    """Start video recording"""
    global video_writer, is_recording, recording_start_time
    
    if is_recording:
        print("[RECORD] Already recording!")
        return False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tracking_output_{timestamp}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC)
    
    with video_writer_lock:
        video_writer = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
        
        if not video_writer.isOpened():
            print(f"[ERROR] Failed to create video writer: {filename}")
            video_writer = None
            return False
        
        is_recording = True
        recording_start_time = time.time()
    
    print(f"[RECORD] Started recording to: {filename}")
    return True

def stop_recording():
    """Stop video recording"""
    global video_writer, is_recording, recording_start_time
    
    if not is_recording:
        print("[RECORD] Not currently recording!")
        return
    
    with video_writer_lock:
        if video_writer is not None:
            video_writer.release()
            video_writer = None
        
        is_recording = False
        duration = time.time() - recording_start_time
        recording_start_time = None
    
    print(f"[RECORD] Stopped recording (duration: {duration:.1f}s)")

def write_frame_to_video(frame):
    """Write frame to video file if recording"""
    if not is_recording:
        return
    
    with video_writer_lock:
        if video_writer is not None:
            try:
                video_writer.write(frame)
            except Exception as e:
                print(f"[ERROR] Failed to write frame: {e}")

# =============================================================================
# CONSOLE INPUT HANDLER
# =============================================================================
def console_input_handler():
    """Handle console commands in separate thread"""
    global selected_track_id, running, uid_map
    
    print("\n" + "="*90)
    print("UNIFIED TRACKING CONSOLE - COMMANDS")
    print("="*90)
    print("  [ID]              - Select target by ID (e.g., 1, 2, 3)")
    print("  assign [ID] [UID] - Assign custom UID to track ID (e.g., assign 2 Person_A)")
    print("  unassign [ID]     - Remove UID assignment")
    print("  list              - Show all active tracks and UID mappings")
    print("  clear             - Clear target selection")
    print("  record            - Start/stop video recording")
    print("  quit / q          - Exit application")
    print("="*90 + "\n")
    
    while running:
        try:
            cmd = input(">>> ").strip()
            
            if not cmd:
                continue
            
            parts = cmd.split()
            cmd_lower = parts[0].lower()
            
            # Quit
            if cmd_lower in ('quit', 'q', 'exit'):
                print("[SYSTEM] Quit command received. Closing...")
                running = False
                break
            
            # Clear selection
            elif cmd_lower == 'clear' or cmd_lower == 'c':
                selected_track_id = None
                print("[INFO] Selection cleared")
            
            # List tracks
            elif cmd_lower == 'list' or cmd_lower == 'l':
                if all_active_tracks:
                    print(f"[INFO] Active tracks: {sorted(all_active_tracks)}")
                    with uid_map_lock:
                        if uid_map:
                            print(f"[INFO] UID mappings: {dict(uid_map)}")
                        else:
                            print("[INFO] No UID mappings")
                else:
                    print("[INFO] No active tracks")
            
            # Assign UID
            elif cmd_lower == 'assign' and len(parts) >= 3:
                try:
                    track_id = int(parts[1])
                    uid = " ".join(parts[2:])
                    
                    if track_id in all_active_tracks:
                        with uid_map_lock:
                            uid_map[track_id] = uid
                        print(f"[SUCCESS] Assigned UID '{uid}' to track {track_id}")
                    else:
                        print(f"[ERROR] Track {track_id} not active. Active: {sorted(all_active_tracks)}")
                except ValueError:
                    print("[ERROR] Invalid track ID. Usage: assign [ID] [UID]")
            
            # Unassign UID
            elif cmd_lower == 'unassign' and len(parts) == 2:
                try:
                    track_id = int(parts[1])
                    with uid_map_lock:
                        if track_id in uid_map:
                            removed_uid = uid_map.pop(track_id)
                            print(f"[SUCCESS] Removed UID '{removed_uid}' from track {track_id}")
                        else:
                            print(f"[INFO] Track {track_id} has no UID assignment")
                except ValueError:
                    print("[ERROR] Invalid track ID")
            
            # Recording
            elif cmd_lower == 'record' or cmd_lower == 'rec':
                if is_recording:
                    stop_recording()
                else:
                    print("[INFO] Recording will start with next frame...")
            
            # Select track by ID
            else:
                try:
                    target_id = int(cmd)
                    if target_id in all_active_tracks:
                        selected_track_id = target_id
                        print(f"[SUCCESS] Selected track ID: {target_id}")
                    else:
                        active_str = str(sorted(all_active_tracks)) if all_active_tracks else "None"
                        print(f"[ERROR] Track {target_id} not found. Active: {active_str}")
                except ValueError:
                    print(f"[ERROR] Unknown command: '{cmd}'")
        
        except EOFError:
            break
        except KeyboardInterrupt:
            running = False
            break
        except Exception as e:
            print(f"[ERROR] Console error: {e}")

# =============================================================================
# MAIN TRACKING FUNCTION
# =============================================================================
def run_tracking(source=0):
    """Main tracking loop"""
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
    if fps == 0:
        fps = 30.0
    
    print(f"[INFO] Video: {frame_width}x{frame_height} @ {fps:.1f} FPS")
    print(f"[INFO] Device: {DEVICE}")
    
    # Initialize model and tracker
    print("[INIT] Loading YOLOv8 model...")
    model = YOLO(MODEL_PATH)
    try:
        model.to(DEVICE)
        model.fuse()
        print("[INIT] Model loaded and fused")
    except Exception as e:
        print(f"[WARN] Model setup: {e}")
    
    print("[INIT] Initializing DeepSort tracker...")
    tracker = DeepSort(
        max_age=TRACKER_MAX_AGE,
        n_init=TRACKER_N_INIT,
        max_cosine_distance=MAX_COSINE_DISTANCE,
        nn_budget=100
    )
    
    # Start console thread
    console_thread = threading.Thread(target=console_input_handler, daemon=True)
    console_thread.start()
    
    # Create windows
    cv2.namedWindow("Unified Tracker", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    fps_smooth = 0.0
    last_time = time.time()
    
    print("[INIT] Starting tracking loop...\n")
    
    try:
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
            detections_raw = []
            
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
                    
                    detections_raw = extract_detections_yolov8(results, CONF_THRESH)
                    
                    # Scale back to original frame
                    for det in detections_raw:
                        det['x1'] *= (1.0 / scale_factor)
                        det['y1'] *= (1.0 / scale_factor)
                        det['x2'] *= (1.0 / scale_factor)
                        det['y2'] *= (1.0 / scale_factor)
                    
                    detections_list = convert_to_deepsort_format(detections_raw, frame.shape)
                
                except Exception as e:
                    print(f"[ERROR] Detection: {e}")
                    detections_list = []
            
            # Tracking phase
            try:
                frame_uint8 = frame.astype(np.uint8)
                tracks = tracker.update_tracks(detections_list, frame=frame_uint8)
                confirmed_tracks = [t for t in tracks if t.is_confirmed()]
                all_active_tracks = [t.track_id for t in confirmed_tracks]
            
            except Exception as e:
                print(f"[ERROR] Tracking: {e}")
                confirmed_tracks = []
                all_active_tracks = []
            
            # Visualization
            display_frame = frame.copy()
            num_targets = len(confirmed_tracks)
            
            # Draw all tracks
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
                
                # Get UID if assigned
                with uid_map_lock:
                    assigned_uid = uid_map.get(track_id, None)
                
                # Determine color
                color = get_color_for_track(track_id, num_targets, selected_track_id, 
                                           has_uid=(assigned_uid is not None))
                
                # Draw bounding box
                thickness = 3 if track_id == selected_track_id else 2
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw path trail
                if len(track_paths[track_id]) > 1:
                    pts = np.array(track_paths[track_id][-30:], np.int32)
                    for i in range(1, len(pts)):
                        alpha = i / len(pts)
                        trail_thickness = max(1, int(1 + alpha))
                        cv2.line(display_frame, tuple(pts[i-1]), tuple(pts[i]), 
                                color, trail_thickness)
                
                # Label construction
                label = f"ID:{track_id}"
                if assigned_uid:
                    label += f" | {assigned_uid}"
                
                # Draw label background and text
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                label_y = max(label_size[1] + 10, y1 - 5)
                cv2.rectangle(display_frame, 
                             (x1, label_y - label_size[1] - 8),
                             (x1 + label_size[0] + 10, label_y + 2), 
                             color, -1)
                cv2.putText(display_frame, label, (x1 + 5, label_y - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Selection indicator
                if track_id == selected_track_id:
                    cv2.putText(display_frame, ">>> SELECTED <<<", (x1, y2 + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # FPS calculation
            dt = current_time - last_time
            last_time = current_time
            fps_current = 1.0 / (dt + 1e-6)
            fps_smooth = 0.85 * fps_smooth + 0.15 * fps_current
            
            # Info panel
            panel_h = 200
            cv2.rectangle(display_frame, (5, 5), (450, panel_h), (25, 25, 25), -1)
            cv2.rectangle(display_frame, (5, 5), (450, panel_h), (100, 100, 100), 2)
            
            y_pos = 35
            cv2.putText(display_frame, f"FPS: {fps_smooth:.1f}", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_pos += 30
            cv2.putText(display_frame, f"Frame: {frame_count}", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_pos += 25
            cv2.putText(display_frame, f"Active Targets: {num_targets}", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_pos += 25
            
            # Show active IDs
            ids_str = str(sorted(all_active_tracks)) if all_active_tracks else "None"
            if len(ids_str) > 40:
                ids_str = ids_str[:40] + "..."
            cv2.putText(display_frame, f"IDs: {ids_str}", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
            y_pos += 25
            
            # Show selected
            sel_str = str(selected_track_id) if selected_track_id else "None"
            cv2.putText(display_frame, f"Selected: {sel_str}", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            y_pos += 25
            
            # Show UID mappings
            with uid_map_lock:
                uid_count = len(uid_map)
            cv2.putText(display_frame, f"UID Mappings: {uid_count}", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            y_pos += 25
            
            # Recording status
            if is_recording:
                elapsed = time.time() - recording_start_time
                cv2.putText(display_frame, f"REC {elapsed:.1f}s", (15, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # Recording indicator
                cv2.circle(display_frame, (frame_width - 30, 30), 10, (0, 0, 255), -1)
            
            # Display main window
            cv2.imshow("Unified Tracker", display_frame)
            
            # Path visualization window
            if selected_track_id is not None and selected_track_id in track_paths:
                with uid_map_lock:
                    sel_uid = uid_map.get(selected_track_id, None)
                path_canvas = draw_path_visualization(selected_track_id, 
                                                     track_paths[selected_track_id],
                                                     sel_uid)
                cv2.imshow("Target Path Visualization", path_canvas)
            else:
                try:
                    cv2.destroyWindow("Target Path Visualization")
                except:
                    pass
            
            # Video recording
            if not is_recording and frame_count == 1:
                # Auto-start recording on first frame if needed
                pass
            
            # Write frame if recording
            if is_recording:
                write_frame_to_video(display_frame)
            
            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("[SYSTEM] ESC pressed. Shutting down...")
                running = False
                break
            elif key == ord('r') or key == ord('R'):
                if is_recording:
                    stop_recording()
                else:
                    start_recording(frame_width, frame_height, OUTPUT_VIDEO_FPS)
    
    except KeyboardInterrupt:
        print("\n[SYSTEM] Keyboard interrupt received")
        running = False
    
    finally:
        # Cleanup
        if is_recording:
            stop_recording()
        
        cap.release()
        cv2.destroyAllWindows()
        print("[SYSTEM] Tracking closed successfully\n")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*90)
    print("UNIFIED MULTI-TARGET TRACKING SYSTEM")
    print("="*90)
    print("Features:")
    print("  ✓ Real-time person detection with YOLOv8")
    print("  ✓ Multi-target tracking with DeepSort")
    print("  ✓ Color-coded visualization based on target density")
    print("  ✓ User-assigned UID mapping system")
    print("  ✓ Console-based target selection and control")
    print("  ✓ Path history visualization")
    print("  ✓ Video recording capability")
    print("  ✓ Compatible with Visual Studio 2022")
    print("="*90)
    print("\nControls:")
    print("  ESC or 'q'    - Quit")
    print("  'r' or 'R'    - Start/stop recording")
    print("  Console       - See console for commands")
    print("="*90 + "\n")
    
    # Run tracking (0 = default webcam, or specify video file path)
    run_tracking(0)
