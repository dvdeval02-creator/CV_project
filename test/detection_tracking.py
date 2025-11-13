"""
DETECTION & TRACKING MODULE
Stages 1-3: Input Processing, YOLOv8 Detection, DeepSort Tracking

Course Topics:
- Stage 1: Lectures 2-5 (Image Processing, Geometric Transformations)
- Stage 2: Lectures 27-28 (CNNs for Detection)
- Stage 3: Lectures 18, 27-28, 34-35 (Motion, Appearance, Association)
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import config


# =============================================================================
# STAGE 1: INPUT PROCESSING
# =============================================================================
class InputProcessor:
    """
    Video capture and frame preprocessing
    Course: Lectures 2-5 (Geometric Transformations, Spatial Domain Operations)
    """
    
    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0.0
    
    def open(self):
        """Open video source"""
        print(f"[STAGE 1] Opening video source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        print(f"[STAGE 1] Video: {self.frame_width}x{self.frame_height} @ {self.fps:.1f} FPS")
        return True
    
    def read_frame(self):
        """Read next frame"""
        return self.cap.read() if self.cap else (False, None)
    
    @staticmethod
    def resize_frame_keep_aspect(frame, target_width):
        """
        Resize maintaining aspect ratio
        Course: Lecture 2-5 (Geometric Transformations - Bilinear Interpolation)
        """
        h, w = frame.shape[:2]
        if w == target_width:
            return frame, 1.0
        
        scale = target_width / float(w)
        new_h = int(h * scale)
        frame_resized = cv2.resize(frame, (target_width, new_h), 
                                   interpolation=cv2.INTER_LINEAR)
        return frame_resized, scale
    
    def release(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
            print("[STAGE 1] Video capture released")
    
    def get_properties(self):
        """Get video properties"""
        return {
            'width': self.frame_width,
            'height': self.frame_height,
            'fps': self.fps
        }


# =============================================================================
# STAGE 2: YOLOV8 DETECTION
# =============================================================================
class YOLOv8Detector:
    """
    YOLOv8 object detection
    Course: Lectures 27-28 (CNNs - Backbone, Neck, Head, NMS)
    """
    
    def __init__(self):
        print(f"[STAGE 2] Loading YOLOv8: {config.MODEL_PATH}")
        print(f"[STAGE 2] Device: {config.DEVICE}")
        
        self.model = YOLO(config.MODEL_PATH)
        self.device = config.DEVICE
        
        try:
            self.model.to(self.device)
            self.model.fuse()
            print("[STAGE 2] Model loaded successfully")
        except Exception as e:
            print(f"[STAGE 2 WARN] {e}")
    
    def detect(self, frame):
        """
        Detect objects in frame
        Course: Lecture 27-28 (CNN Detection Pipeline)
        - Backbone: Feature extraction (CSPDarknet)
        - Neck: Multi-scale fusion (PANet)
        - Head: Bounding box prediction
        - Post-processing: Confidence filter + NMS
        """
        detections = []
        
        try:
            with torch.no_grad():
                results = self.model.predict(
                    source=frame,
                    device=self.device,
                    imgsz=config.DETECT_WIDTH,
                    conf=config.CONF_THRESH,
                    iou=config.NMS_IOU_THRESH,
                    verbose=False
                )
            
            detections = self._extract_detections(results)
        
        except Exception as e:
            print(f"[STAGE 2 ERROR] {e}")
        
        return detections
    
    def _extract_detections(self, results):
        """Extract bounding boxes from results"""
        detections = []
        
        if not results or len(results) == 0:
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
                
                if conf < config.CONF_THRESH:
                    continue
                
                if config.ONLY_PERSONS and cls_id != 0:
                    continue
                
                x1, y1, x2, y2 = xyxy[i]
                detections.append({
                    'x1': float(x1), 'y1': float(y1),
                    'x2': float(x2), 'y2': float(y2),
                    'conf': conf, 'cls': cls_id
                })
        
        except Exception as e:
            print(f"[STAGE 2 ERROR] Extraction: {e}")
        
        return detections
    
    @staticmethod
    def scale_detections(detections, scale_factor):
        """Scale coordinates back to original size"""
        for det in detections:
            det['x1'] /= scale_factor
            det['y1'] /= scale_factor
            det['x2'] /= scale_factor
            det['y2'] /= scale_factor
        return detections
    
    @staticmethod
    def convert_to_deepsort_format(detections, frame_shape):
        """Convert XYXY → TLWH format for DeepSort"""
        deepsort_detections = []
        
        for det in detections:
            x1 = max(0, min(int(det['x1']), frame_shape[1]))
            y1 = max(0, min(int(det['y1']), frame_shape[0]))
            x2 = max(0, min(int(det['x2']), frame_shape[1]))
            y2 = max(0, min(int(det['y2']), frame_shape[0]))
            
            w, h = x2 - x1, y2 - y1
            
            if w > 0 and h > 0:
                tlwh = [x1, y1, w, h]
                deepsort_detections.append((tlwh, det['conf'], det['cls']))
        
        return deepsort_detections


# =============================================================================
# STAGE 3: DEEPSORT TRACKING
# =============================================================================
class DeepSortTracker:
    """
    DeepSort tracking with association
    Course: 
    - Lecture 18 (Kalman Filter, Motion Prediction)
    - Lecture 27-28 (CNN Appearance Features)
    - Lecture 34-35 (Matching Cascade, Hungarian Algorithm)
    """
    
    def __init__(self):
        print("[STAGE 3] Initializing DeepSort")
        print(f"[STAGE 3] max_age={config.TRACKER_MAX_AGE}, "
              f"n_init={config.TRACKER_N_INIT}, "
              f"max_cosine_dist={config.MAX_COSINE_DISTANCE}")
        
        self.tracker = DeepSort(
            max_age=config.TRACKER_MAX_AGE,
            n_init=config.TRACKER_N_INIT,
            max_cosine_distance=config.MAX_COSINE_DISTANCE,
            nn_budget=config.NN_BUDGET
        )
    
    def update(self, detections, frame):
        """
        Update tracks with detections
        
        ASSOCIATION PROCESS (inside DeepSort):
        1. Kalman Prediction (Lecture 18):
           - Predict track positions using motion model
           
        2. Appearance Features (Lectures 27-28):
           - Extract 128D CNN features from detections
           
        3. Cost Matrix (Lectures 18, 27-28):
           - Motion cost: Mahalanobis distance
           - Appearance cost: Cosine distance
           - Combined: λ*motion + (1-λ)*appearance
           
        4. Matching Cascade (Lecture 34-35):
           - Hungarian algorithm for optimal assignment
           - Prioritize recent tracks
           
        5. Track Update (Lecture 18):
           - Kalman update for matched tracks
           - Aging for unmatched tracks
        """
        try:
            tracks = self.tracker.update_tracks(detections, frame=frame)
            return tracks
        except Exception as e:
            print(f"[STAGE 3 ERROR] {e}")
            return []
    
    @staticmethod
    def get_confirmed_tracks(tracks):
        """
        Filter confirmed tracks only
        Course: Lecture 34-35 (Track Confirmation)
        """
        return [t for t in tracks if t.is_confirmed()]
    
    @staticmethod
    def extract_track_info(track):
        """Extract track information"""
        ltrb = track.to_ltrb()
        return {
            'track_id': track.track_id,
            'ltrb': ltrb,
            'x1': int(ltrb[0]), 'y1': int(ltrb[1]),
            'x2': int(ltrb[2]), 'y2': int(ltrb[3]),
            'state': track.state,
            'age': track.age,
            'hits': track.hits
        }
