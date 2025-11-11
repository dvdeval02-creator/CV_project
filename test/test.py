# live_fast_with_uid.py
"""
Fast live tracking with user-assigned UIDs.
- Detection runs in background thread (non-blocking)
- Tracker updates every frame
- Console thread accepts commands to assign/unassign UIDs to tracker IDs
Commands:
  list
  assign <track_id> <uid>
  unassign <track_id>
  clear
  quit / q
"""

import time
import threading
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ----------------------------
# CONFIG (tune for speed)
# ----------------------------
MODEL_PATH = "yolov8n.pt"   # tiny detection model (no segmentation)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DETECT_WIDTH = 416          # small detection size -> faster
DETECT_INTERVAL = 0.25      # seconds between detection runs
CONF_THRESH = 0.35
NMS_IOU_THRESH = 0.5
ONLY_PERSONS = True
DEEPSORT_MAX_AGE = 30
DEEPSORT_N_INIT = 1         # confirm faster (1 = immediate)

# ----------------------------
# Shared state between threads
# ----------------------------
latest_frame = None
latest_frame_lock = threading.Lock()
latest_detection = []       # list of tuples (x1,y1,x2,y2,conf,cls)
latest_detection_lock = threading.Lock()
stop_event = threading.Event()

# UID mapping: track_id -> user-supplied UID (string)
uid_map = {}
uid_map_lock = threading.Lock()

# ----------------------------
# Initialize model and tracker
# ----------------------------
print(f"[INIT] Device: {DEVICE}  Model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
try:
    model.to(DEVICE)
except Exception:
    pass

try:
    model.fuse()
except Exception:
    pass

tracker = DeepSort(max_age=DEEPSORT_MAX_AGE, n_init=DEEPSORT_N_INIT)

# ----------------------------
# Detection thread
# ----------------------------
def detection_worker():
    global latest_detection
    print("[DETECT] Thread started (interval {:.3f}s)".format(DETECT_INTERVAL))
    while not stop_event.is_set():
        t0 = time.time()
        frame = None
        with latest_frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
        if frame is None:
            # no frame yet; sleep briefly
            if stop_event.wait(0.01):
                break
            continue

        h, w = frame.shape[:2]
        # resize small for detection
        if w != DETECT_WIDTH:
            scale = DETECT_WIDTH / float(w)
            new_h = int(h * scale)
            frame_resized = cv2.resize(frame, (DETECT_WIDTH, new_h), interpolation=cv2.INTER_LINEAR)
            inv_scale = w / float(frame_resized.shape[1]) if frame_resized.shape[1] != 0 else 1.0
        else:
            frame_resized = frame
            inv_scale = 1.0

        # run model
        dets_local = []
        try:
            with torch.no_grad():
                res_list = model.predict(source=frame_resized, device=DEVICE, imgsz=DETECT_WIDTH,
                                         conf=CONF_THRESH, iou=NMS_IOU_THRESH, verbose=False)
            res = res_list[0] if isinstance(res_list, (list, tuple)) else res_list

            # parse
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
                    xyxy = np.array([]); confs = np.array([]); clss = np.array([])

            for i, box in enumerate(xyxy):
                conf = float(confs[i])
                cls_id = int(clss[i])
                if conf < CONF_THRESH:
                    continue
                if ONLY_PERSONS and cls_id != 0:
                    continue
                x1, y1, x2, y2 = box.tolist()
                # map back to original frame scale
                x1 = int(x1 * inv_scale); y1 = int(y1 * inv_scale)
                x2 = int(x2 * inv_scale); y2 = int(y2 * inv_scale)
                dets_local.append((x1, y1, x2, y2, conf, cls_id))
        except Exception as e:
            # print minimal debug to avoid console spam
            print("[DETECT] inference error:", e)
            dets_local = []

        # atomic replacement
        with latest_detection_lock:
            latest_detection = dets_local

        # wait remaining time
        elapsed = time.time() - t0
        wait_for = max(0.0, DETECT_INTERVAL - elapsed)
        if stop_event.wait(wait_for):
            break
    print("[DETECT] Thread exiting")

# ----------------------------
# Console command thread
# ----------------------------
def console_worker():
    """
    Simple console input loop running in background.
    Commands:
      list
      assign <track_id> <uid>
      unassign <track_id>
      clear
      quit / q
    """
    print("\nConsole commands: list | assign <track_id> <uid> | unassign <track_id> | clear | quit\n")
    while not stop_event.is_set():
        try:
            cmd = input().strip()
        except Exception:
            break
        if not cmd:
            continue
        parts = cmd.split()
        if parts[0].lower() in ('quit', 'q', 'exit'):
            print("[CMD] quitting...")
            stop_event.set()
            break
        if parts[0].lower() == 'list':
            with latest_detection_lock:
                dets = list(latest_detection)
            with uid_map_lock:
                mapped = dict(uid_map)
            print("[CMD] Active detections (latest):", len(dets))
            print("[CMD] UID map:", mapped)
            continue
        if parts[0].lower() == 'assign' and len(parts) >= 3:
            try:
                tid = int(parts[1])
                uid = " ".join(parts[2:])
                with uid_map_lock:
                    uid_map[tid] = uid
                print(f"[CMD] Assigned UID '{uid}' to track {tid}")
            except ValueError:
                print("[CMD] assign: invalid track id")
            continue
        if parts[0].lower() == 'unassign' and len(parts) == 2:
            try:
                tid = int(parts[1])
                with uid_map_lock:
                    if tid in uid_map:
                        del uid_map[tid]
                        print(f"[CMD] Unassigned track {tid}")
                    else:
                        print(f"[CMD] Track {tid} not assigned")
            except ValueError:
                print("[CMD] unassign: invalid track id")
            continue
        if parts[0].lower() == 'clear':
            with uid_map_lock:
                uid_map.clear()
            print("[CMD] Cleared all UID mappings")
            continue
        print("[CMD] Unknown/invalid command")

# ----------------------------
# Main loop
# ----------------------------
def run_live(source=0):
    global latest_frame
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[MAIN] cannot open source", source)
        return

    # try set resolution (may be ignored by some cams)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # start threads
    det_thread = threading.Thread(target=detection_worker, daemon=True)
    det_thread.start()
    console_thread = threading.Thread(target=console_worker, daemon=True)
    console_thread.start()

    frame_idx = 0
    last_time = time.time()
    fps_smoothed = 0.0

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[MAIN] frame read failed")
                break
            frame_idx += 1

            # publish latest frame for detection
            with latest_frame_lock:
                latest_frame = frame

            # read latest detections
            with latest_detection_lock:
                dets = list(latest_detection)

            # convert to deepsort format (tlwh, conf, cls)
            detections_for_tracker = []
            for (x1,y1,x2,y2,conf,cls) in dets:
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue
                tlwh = [int(x1), int(y1), int(w), int(h)]
                detections_for_tracker.append((tlwh, float(conf), int(cls)))

            # update tracker
            try:
                tracks = tracker.update_tracks(detections_for_tracker, frame=frame)
            except Exception as e:
                print("[MAIN] tracker.update error:", e)
                tracks = []

            # draw
            display = frame
            drawn = 0
            active_ids = []
            for tr in tracks:
                try:
                    tid = tr.track_id
                    active_ids.append(tid)
                    l, t, r, b = [int(v) for v in tr.to_ltrb()]
                    # check confirmation
                    confirmed = tr.is_confirmed()
                    # color: magenta if assigned UID else green/cyan
                    with uid_map_lock:
                        assigned_uid = uid_map.get(tid, None)
                    if assigned_uid is not None:
                        color = (255, 0, 255)  # magenta for assigned
                    else:
                        color = (0, 200, 0) if confirmed else (0, 200, 200)
                    cv2.rectangle(display, (l,t), (r,b), color, 2)
                    # label shows track id and optional UID
                    label = f"ID:{tid}"
                    if assigned_uid:
                        label += f" | UID:{assigned_uid}"
                    elif hasattr(tr, 'last_detection_confidence') and tr.last_detection_confidence:
                        label += f" {tr.last_detection_confidence:.2f}"
                    cv2.putText(display, label, (l, max(15,t-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
                except Exception:
                    continue
                drawn += 1
                if drawn >= 40:
                    break

            # small overlay info
            now = time.time()
            dt = now - last_time
            last_time = now
            fps = 1.0 / (dt + 1e-9)
            fps_smoothed = 0.9*fps_smoothed + 0.1*fps
            cv2.putText(display, f"FPS:{fps_smoothed:.1f}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.putText(display, f"Active IDs: {active_ids}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,255,180), 1)
            with uid_map_lock:
                mapped = dict(uid_map)
            cv2.putText(display, f"UID Map: {mapped}", (10,75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

            cv2.imshow("FAST Live Track w/UID", display)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                stop_event.set()
                break

    except KeyboardInterrupt:
        stop_event.set()

    # cleanup
    det_thread.join(timeout=1.0)
    console_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()
    print("[MAIN] Exited cleanly")

if __name__ == "__main__":
    run_live(0)
