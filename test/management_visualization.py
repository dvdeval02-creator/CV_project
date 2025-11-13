"""
MANAGEMENT & VISUALIZATION MODULE
Stages 4-7: Track Management, Visualization, Multi-threading, Recording

Course Topics:
- Stage 4: Lectures 34-35 (Track Management), 18 (Motion), 30 (Sequential Data)
- Stage 5: Lectures 3-5 (Image Processing, Visualization)
- Stage 6: Multi-threading concepts
- Stage 7: Video I/O
"""

import cv2
import numpy as np
import threading
import time
from datetime import datetime
from collections import defaultdict, deque
import config


# =============================================================================
# STAGE 4: TRACK MANAGEMENT
# =============================================================================
class TrackManager:
    """
    Track lifecycle, UID assignment, path storage
    Course:
    - Lecture 34-35 (Track Management, Identity)
    - Lecture 18 (Trajectories)
    - Lecture 30 (Sequential Data for RNN)
    """
    
    def __init__(self):
        print("[STAGE 4] Initializing Track Manager")
        
        # Path storage (Lecture 18: Trajectories, Lecture 30: Sequential data)
        self.track_paths = defaultdict(lambda: deque(maxlen=config.MAX_PATH_LENGTH))
        
        # UID mapping (Lecture 34-35: Identity management)
        self.uid_map = {}
        self.uid_map_lock = threading.Lock()
        
        self.all_active_tracks = []
    
    def update_paths(self, track_info):
        """
        Store path history
        Course: Lecture 18 (Trajectories), Lecture 30 (Temporal sequences)
        """
        track_id = track_info['track_id']
        x1, y1, x2, y2 = track_info['x1'], track_info['y1'], track_info['x2'], track_info['y2']
        
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        self.track_paths[track_id].append((cx, cy))
    
    def get_path(self, track_id):
        """Get path history"""
        return list(self.track_paths[track_id])
    
    def assign_uid(self, track_id, uid):
        """
        Assign UID to track
        Course: Lecture 34-35 (Identity Management)
        """
        if track_id not in self.all_active_tracks:
            return False
        
        with self.uid_map_lock:
            self.uid_map[track_id] = uid
        
        print(f"[STAGE 4] Assigned UID '{uid}' to track {track_id}")
        return True
    
    def unassign_uid(self, track_id):
        """Remove UID assignment"""
        with self.uid_map_lock:
            if track_id in self.uid_map:
                removed = self.uid_map.pop(track_id)
                print(f"[STAGE 4] Removed UID '{removed}' from track {track_id}")
                return removed
        return None
    
    def get_uid(self, track_id):
        """Get UID (thread-safe)"""
        with self.uid_map_lock:
            return self.uid_map.get(track_id, None)
    
    def get_all_uids(self):
        """Get all UIDs"""
        with self.uid_map_lock:
            return dict(self.uid_map)
    
    def update_active_tracks(self, track_ids):
        """Update active track list"""
        self.all_active_tracks = track_ids
    
    def cleanup_deleted_tracks(self, current_ids):
        """
        Clean up deleted tracks
        Course: Lecture 34-35 (Track Termination, Memory Management)
        """
        all_stored = set(self.track_paths.keys())
        current = set(current_ids)
        deleted = all_stored - current
        
        for tid in deleted:
            if tid in self.track_paths:
                del self.track_paths[tid]
            
            with self.uid_map_lock:
                if tid in self.uid_map:
                    removed = self.uid_map.pop(tid)
                    print(f"[STAGE 4] Auto-removed UID '{removed}' from deleted track {tid}")


# =============================================================================
# STAGE 5: VISUALIZATION
# =============================================================================
class Visualizer:
    """
    Drawing, colors, path visualization
    Course: Lectures 3-5 (Spatial Domain Operations, Color Spaces)
    """
    
    @staticmethod
    def get_color_for_track(track_id, num_targets, selected_id, has_uid=False):
        """
        Determine track color
        Priority: UID > Selected > Density-based
        """
        if has_uid:
            return config.COLOR_UID_ASSIGNED
        if track_id == selected_id:
            return config.COLOR_SELECTED
        
        if num_targets <= 1:
            return config.COLOR_LOW_DENSITY
        elif num_targets == 2:
            return config.COLOR_MEDIUM_DENSITY
        elif num_targets <= 4:
            return config.COLOR_HIGH_DENSITY
        else:
            return config.COLOR_VERY_HIGH
    
    @staticmethod
    def draw_track(frame, track_info, color, is_selected, uid=None):
        """
        Draw bounding box and label
        Course: Lecture 3-5 (Drawing operations, color spaces)
        """
        x1, y1, x2, y2 = track_info['x1'], track_info['y1'], track_info['x2'], track_info['y2']
        track_id = track_info['track_id']
        
        thickness = 3 if is_selected else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Label
        label = f"ID:{track_id}"
        if uid:
            label += f" | {uid}"
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_y = max(label_size[1] + 10, y1 - 5)
        
        cv2.rectangle(frame, 
                     (x1, label_y - label_size[1] - 8),
                     (x1 + label_size[0] + 10, label_y + 2),
                     color, -1)
        cv2.putText(frame, label, (x1 + 5, label_y - 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if is_selected:
            cv2.putText(frame, ">>> SELECTED <<<", (x1, y2 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_SELECTED, 2)
    
    @staticmethod
    def draw_path_trail(frame, path, color):
        """
        Draw path trail on frame
        Course: Lecture 3-5 (Line drawing, gradient effects)
        """
        if len(path) < 2:
            return
        
        pts = np.array(path[-config.TRAIL_LENGTH:], np.int32)
        
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            thickness = max(1, int(1 + alpha))
            cv2.line(frame, tuple(pts[i-1]), tuple(pts[i]), color, thickness)
    
    @staticmethod
    def draw_path_visualization(track_id, path_history, uid=None):
        """
        Create dedicated path visualization window
        Course: Lecture 3-5 (Coordinate transformations, visualization)
        """
        canvas = np.zeros((config.PATH_WINDOW_SIZE[1], config.PATH_WINDOW_SIZE[0], 3), dtype=np.uint8)
        
        # Grid
        for i in range(0, config.PATH_WINDOW_SIZE[0], 50):
            cv2.line(canvas, (i, 0), (i, config.PATH_WINDOW_SIZE[1]), (40, 40, 40), 1)
        for i in range(0, config.PATH_WINDOW_SIZE[1], 50):
            cv2.line(canvas, (0, i), (config.PATH_WINDOW_SIZE[0], i), (40, 40, 40), 1)
        
        # Header
        cv2.rectangle(canvas, (0, 0), (config.PATH_WINDOW_SIZE[0], 70), (20, 20, 20), -1)
        title = f"Track ID: {track_id}"
        if uid:
            title += f" | UID: {uid}"
        title += " - Path Visualization"
        cv2.putText(canvas, title, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
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
        max_w = config.PATH_WINDOW_SIZE[0] - 2 * margin
        max_h = config.PATH_WINDOW_SIZE[1] - 2 * margin - 70
        
        scale = min(max_w / range_x, max_h / range_y)
        
        normalized = []
        for px, py in path_history:
            nx = int((px - min_x) * scale + margin)
            ny = int((py - min_y) * scale + margin + 70)
            normalized.append((nx, ny))
        
        # Draw path
        for i in range(1, len(normalized)):
            alpha = i / len(normalized)
            color_val = int(100 + 155 * alpha)
            color = (color_val, 0, color_val)
            thickness = max(1, int(1 + 2 * alpha))
            cv2.line(canvas, normalized[i-1], normalized[i], color, thickness)
        
        # Markers
        if normalized:
            cv2.circle(canvas, normalized[0], 8, (0, 255, 0), -1)    # Start
            cv2.circle(canvas, normalized[-1], 8, (0, 0, 255), -1)  # Current
        
        # Stats
        y_offset = config.PATH_WINDOW_SIZE[1] - 60
        cv2.rectangle(canvas, (0, y_offset), (config.PATH_WINDOW_SIZE[0], config.PATH_WINDOW_SIZE[1]), (20, 20, 20), -1)
        cv2.putText(canvas, f"Points: {len(path_history)} | Coverage: {range_x:.0f}x{range_y:.0f}",
                   (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        return canvas
    
    @staticmethod
    def draw_info_panel(frame, fps, frame_count, num_targets, active_ids, selected_id, uid_count, is_recording, rec_time):
        """Draw information panel"""
        panel_h = 180
        panel_w = 190
        cv2.rectangle(frame, (5, 5), (panel_w, panel_h), (25, 25, 25), -1)
        cv2.rectangle(frame, (5, 5), (panel_w, panel_h), (100, 100, 100), 2)
        
        y = 35
        cv2.putText(frame, f"FPS: {fps:.1f}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y += 30
        cv2.putText(frame, f"Frame: {frame_count}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 25
        cv2.putText(frame, f"Active Targets: {num_targets}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        y += 25
        
        ids_str = str(sorted(active_ids)) if active_ids else "None"
        if len(ids_str) > 40:
            ids_str = ids_str[:40] + "..."
        cv2.putText(frame, f"IDs: {ids_str}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
        y += 25
        
        sel_str = str(selected_id) if selected_id else "None"
        cv2.putText(frame, f"Selected: {sel_str}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        y += 25
        
        cv2.putText(frame, f"UID Mappings: {uid_count}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        y += 25
        
        if is_recording:
            cv2.putText(frame, f"REC {rec_time:.1f}s", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)


# =============================================================================
# STAGE 6: CONSOLE INPUT HANDLER (Multi-threading)
# =============================================================================
class ConsoleHandler:
    """
    Console command handler running in separate thread
    Course: Multi-threading concepts for non-blocking I/O
    """
    
    def __init__(self, track_manager):
        self.track_manager = track_manager
        self.selected_track_id = None
        self.running = True
        self.is_recording = False
        self.thread = None
    
    def start(self):
        """Start console thread"""
        print("\n" + "="*80)
        print("CONSOLE COMMANDS:")
        print("  [ID]              - Select track")
        print("  assign [ID] [UID] - Assign UID")
        print("  unassign [ID]     - Remove UID")
        print("  list              - Show all tracks")
        print("  clear             - Clear selection")
        print("  record            - Toggle recording")
        print("  quit / q          - Exit")
        print("="*80 + "\n")
        
        self.thread = threading.Thread(target=self._console_loop, daemon=True)
        self.thread.start()
    
    def _console_loop(self):
        """Console input loop"""
        while self.running:
            try:
                cmd = input(">>> ").strip()
                if not cmd:
                    continue
                
                parts = cmd.split()
                cmd_lower = parts[0].lower()
                
                if cmd_lower in ('quit', 'q', 'exit'):
                    self.running = False
                    break
                
                elif cmd_lower in ('clear', 'c'):
                    self.selected_track_id = None
                    print("[INFO] Selection cleared")
                
                elif cmd_lower in ('list', 'l'):
                    if self.track_manager.all_active_tracks:
                        print(f"[INFO] Active: {sorted(self.track_manager.all_active_tracks)}")
                        uids = self.track_manager.get_all_uids()
                        if uids:
                            print(f"[INFO] UIDs: {uids}")
                    else:
                        print("[INFO] No active tracks")
                
                elif cmd_lower == 'assign' and len(parts) >= 3:
                    tid = int(parts[1])
                    uid = " ".join(parts[2:])
                    if self.track_manager.assign_uid(tid, uid):
                        print(f"[SUCCESS] Assigned '{uid}' to {tid}")
                    else:
                        print(f"[ERROR] Track {tid} not active")
                
                elif cmd_lower == 'unassign' and len(parts) == 2:
                    tid = int(parts[1])
                    self.track_manager.unassign_uid(tid)
                
                elif cmd_lower in ('record', 'rec'):
                    self.is_recording = not self.is_recording
                    print(f"[RECORDING] {'Started' if self.is_recording else 'Stopped'}")
                
                else:
                    # Try as track ID
                    try:
                        tid = int(cmd)
                        if tid in self.track_manager.all_active_tracks:
                            self.selected_track_id = tid
                            print(f"[SUCCESS] Selected track {tid}")
                        else:
                            print(f"[ERROR] Track {tid} not active")
                    except ValueError:
                        print(f"[ERROR] Unknown command: '{cmd}'")
            
            except (EOFError, KeyboardInterrupt):
                self.running = False
                break
            except Exception as e:
                print(f"[ERROR] Console: {e}")


# =============================================================================
# STAGE 7: VIDEO RECORDING
# =============================================================================
class VideoRecorder:
    """
    Video recording system
    Course: Video I/O, codec handling
    """
    
    def __init__(self):
        self.writer = None
        self.is_recording = False
        self.start_time = None
        self.lock = threading.Lock()
    
    def start(self, width, height, fps):
        """Start recording"""
        if self.is_recording:
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tracking_output_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_VIDEO_CODEC)
        
        with self.lock:
            self.writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            
            if not self.writer.isOpened():
                self.writer = None
                return False
            
            self.is_recording = True
            self.start_time = time.time()
        
        print(f"[STAGE 7] Recording to: {filename}")
        return True
    
    def stop(self):
        """Stop recording"""
        if not self.is_recording:
            return
        
        with self.lock:
            if self.writer:
                self.writer.release()
                self.writer = None
            
            duration = time.time() - self.start_time
            self.is_recording = False
            self.start_time = None
        
        print(f"[STAGE 7] Stopped (duration: {duration:.1f}s)")
    
    def write_frame(self, frame):
        """Write frame to video"""
        if not self.is_recording:
            return
        
        with self.lock:
            if self.writer:
                try:
                    self.writer.write(frame)
                except Exception as e:
                    print(f"[STAGE 7 ERROR] {e}")
    
    def get_elapsed_time(self):
        """Get recording elapsed time"""
        if not self.is_recording or not self.start_time:
            return 0.0
        return time.time() - self.start_time
