"""
Main entry point for Joint Tracking and Segmentation Project
Provides simple CLI interface
"""

# import sys
# import os

# # Add src to path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# from pipeline import main

# if __name__ == "__main__":
#     main()

"""
MAIN TRACKING APPLICATION
Integrates all stages: Input → Detection → Tracking → Management → Visualization → Recording

Run: python main.py
"""

import cv2
import time
import config
from detection_tracking import InputProcessor, YOLOv8Detector, DeepSortTracker
from management_visualization import TrackManager, Visualizer, ConsoleHandler, VideoRecorder


def main(source=0):
    """
    Main tracking loop
    
    Args:
        source: Video source (0 for webcam, or path to video file)
    """
    print("\n" + "="*80)
    print("UNIFIED MULTI-TARGET TRACKING SYSTEM")
    print("="*80)
    print("Stages:")
    print("  1. Input Processing (Lectures 2-5)")
    print("  2. YOLOv8 Detection (Lectures 27-28)")
    print("  3. DeepSort Tracking (Lectures 18, 27-28, 34-35)")
    print("  4. Track Management (Lectures 34-35, 18, 30)")
    print("  5. Visualization (Lectures 3-5)")
    print("  6. Multi-threading (Console input)")
    print("  7. Video Recording")
    print("="*80 + "\n")
    
    # Initialize all components
    print("[INIT] Initializing components...")
    
    # Stage 1: Input
    input_processor = InputProcessor(source)
    input_processor.open()
    props = input_processor.get_properties()
    
    # Stage 2: Detection
    detector = YOLOv8Detector()
    
    # Stage 3: Tracking
    tracker = DeepSortTracker()
    
    # Stage 4: Management
    track_manager = TrackManager()
    
    # Stage 5: Visualization
    visualizer = Visualizer()
    
    # Stage 6: Console
    console = ConsoleHandler(track_manager)
    console.start()
    
    # Stage 7: Recording
    recorder = VideoRecorder()
    
    # Create window
    cv2.namedWindow("Unified Tracker", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    fps_smooth = 0.0
    last_time = time.time()
    
    print("[INIT] Starting tracking loop...\n")
    
    try:
        while console.running:
            # Stage 1: Read frame
            ret, frame = input_processor.read_frame()
            if not ret:
                print("[SYSTEM] End of video")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Stage 1: Resize for detection
            frame_resized, scale = InputProcessor.resize_frame_keep_aspect(frame, config.DETECT_WIDTH)
            
            # Stage 2: Detect objects
            detections = []
            if frame_count % config.DETECT_EVERY_N == 0:
                detections = detector.detect(frame_resized)
                detections = detector.scale_detections(detections, scale)
            
            detections_ds = detector.convert_to_deepsort_format(detections, frame.shape)
            
            # Stage 3: Track objects
            tracks = tracker.update(detections_ds, frame)
            confirmed = tracker.get_confirmed_tracks(tracks)
            
            # Stage 4: Manage tracks
            track_infos = []
            for track in confirmed:
                info = tracker.extract_track_info(track)
                track_infos.append(info)
                track_manager.update_paths(info)
            
            active_ids = [t['track_id'] for t in track_infos]
            track_manager.update_active_tracks(active_ids)
            track_manager.cleanup_deleted_tracks(active_ids)
            
            # Stage 5: Visualize
            display_frame = frame.copy()
            num_targets = len(track_infos)
            
            for info in track_infos:
                tid = info['track_id']
                uid = track_manager.get_uid(tid)
                is_selected = (tid == console.selected_track_id)
                
                color = visualizer.get_color_for_track(tid, num_targets, console.selected_track_id, uid is not None)
                
                visualizer.draw_track(display_frame, info, color, is_selected, uid)
                
                path = track_manager.get_path(tid)
                visualizer.draw_path_trail(display_frame, path, color)
            
            # FPS
            dt = current_time - last_time
            last_time = current_time
            fps_current = 1.0 / (dt + 1e-6)
            fps_smooth = 0.85 * fps_smooth + 0.15 * fps_current
            
            # Info panel
            uid_count = len(track_manager.get_all_uids())
            rec_time = recorder.get_elapsed_time() if recorder.is_recording else 0.0
            visualizer.draw_info_panel(display_frame, fps_smooth, frame_count, num_targets, 
                                      active_ids, console.selected_track_id, uid_count, 
                                      recorder.is_recording, rec_time)
            
            # Main window
            cv2.imshow("Unified Tracker", display_frame)
            
            # Path visualization
            if console.selected_track_id and console.selected_track_id in active_ids:
                path = track_manager.get_path(console.selected_track_id)
                uid = track_manager.get_uid(console.selected_track_id)
                path_canvas = visualizer.draw_path_visualization(console.selected_track_id, path, uid)
                cv2.imshow("Target Path Visualization", path_canvas)
            else:
                try:
                    cv2.destroyWindow("Target Path Visualization")
                except:
                    pass
            
            # Stage 7: Recording
            if console.is_recording and not recorder.is_recording:
                recorder.start(props['width'], props['height'], config.OUTPUT_VIDEO_FPS)
            elif not console.is_recording and recorder.is_recording:
                recorder.stop()
            
            if recorder.is_recording:
                recorder.write_frame(display_frame)
            
            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                console.running = False
                break
            elif key == ord('r'):
                console.is_recording = not console.is_recording
    
    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted")
    
    finally:
        # Cleanup
        if recorder.is_recording:
            recorder.stop()
        
        input_processor.release()
        cv2.destroyAllWindows()
        print("[SYSTEM] Closed successfully\n")


if __name__ == "__main__":
    # Run tracking (0 = webcam, or specify video path)
    main(0)


