"""
Traffic Tracking and Counting System using YOLOv8
This script uses a trained YOLOv8 model to detect, track, and count vehicles in video streams.
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import argparse
from pathlib import Path


class TrafficCounter:
    """Class to handle traffic detection, tracking, and counting"""
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the traffic counter
        
        Args:
            model_path: Path to the trained YOLOv8 model weights (.pt file)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Class names
        self.class_names = {0: 'car', 1: 'bus', 2: 'van', 3: 'others'}
        self.class_colors = {
            0: (0, 255, 0),    # car - green
            1: (255, 0, 0),    # bus - blue
            2: (0, 0, 255),    # van - red
            3: (255, 255, 0)   # others - cyan
        }
        
        # Tracking and counting variables
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.counted_ids = set()  # Track IDs that have been counted
        self.counts = defaultdict(int)  # Count per class
        self.total_count = 0
        
        # Counting line (horizontal line across the frame)
        self.counting_line_position = None
        self.counting_line_set = False
        
    def set_counting_line(self, y_position):
        """
        Set the counting line position
        
        Args:
            y_position: Y coordinate of the counting line (horizontal line)
        """
        self.counting_line_position = y_position
        self.counting_line_set = True
        
    def draw_counting_line(self, frame):
        """Draw the counting line on the frame"""
        if self.counting_line_set and self.counting_line_position is not None:
            cv2.line(frame, 
                    (0, self.counting_line_position), 
                    (frame.shape[1], self.counting_line_position), 
                    (0, 255, 255), 2)
            cv2.putText(frame, "Counting Line", 
                       (10, self.counting_line_position - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def has_crossed_line(self, track_id, current_y, prev_y):
        """
        Check if a vehicle has crossed the counting line
        
        Args:
            track_id: ID of the tracked object
            current_y: Current Y coordinate (center of bounding box)
            prev_y: Previous Y coordinate
            
        Returns:
            True if the line was crossed, False otherwise
        """
        if not self.counting_line_set or self.counting_line_position is None:
            return False
            
        if track_id in self.counted_ids:
            return False
            
        # Check if crossing from above to below (or vice versa)
        if prev_y is not None:
            if (prev_y < self.counting_line_position <= current_y) or \
               (prev_y > self.counting_line_position >= current_y):
                return True
        return False
    
    def process_frame(self, frame):
        """
        Process a single frame: detect, track, and count vehicles
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            Annotated frame with detections, tracks, and counts
        """
        # Run YOLOv8 tracking
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            persist=True,
            tracker="bytetrack.yaml",  # Use ByteTrack tracker
            verbose=False
        )
        
        annotated_frame = frame.copy()
        
        # Draw counting line
        self.draw_counting_line(annotated_frame)
        
        # Process detections
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            if track_ids is not None:
                track_ids = track_ids.cpu().numpy().astype(int)
                
                for i, (box, track_id, cls, conf) in enumerate(zip(boxes, track_ids, classes, confidences)):
                    cls = int(cls)
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Calculate center point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Get previous position
                    prev_y = None
                    if len(self.track_history[track_id]) > 0:
                        prev_y = self.track_history[track_id][-1][1]
                    
                    # Check if line was crossed
                    if self.has_crossed_line(track_id, center_y, prev_y):
                        self.counted_ids.add(track_id)
                        self.counts[self.class_names[cls]] += 1
                        self.total_count += 1
                    
                    # Update track history
                    self.track_history[track_id].append((center_x, center_y))
                    
                    # Draw bounding box
                    color = self.class_colors.get(cls, (255, 255, 255))
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw track ID and class label
                    label = f"ID:{track_id} {self.class_names[cls]} {conf:.2f}"
                    if track_id in self.counted_ids:
                        label += " [COUNTED]"
                    
                    # Label background for better visibility
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_frame, 
                                 (x1, y1 - text_height - 10), 
                                 (x1 + text_width, y1), 
                                 color, -1)
                    cv2.putText(annotated_frame, label, 
                               (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Draw tracking trail
                    if len(self.track_history[track_id]) > 1:
                        points = np.array(self.track_history[track_id], dtype=np.int32)
                        cv2.polylines(annotated_frame, [points], False, color, 2)
                    
                    # Draw center point
                    cv2.circle(annotated_frame, (center_x, center_y), 5, color, -1)
        
        # Draw count information
        self.draw_count_info(annotated_frame)
        
        return annotated_frame
    
    def draw_count_info(self, frame):
        """Draw counting statistics on the frame"""
        y_offset = 30
        x_offset = 10
        
        # Background rectangle for text
        text_lines = [
            f"Total Count: {self.total_count}",
            f"Car: {self.counts['car']}",
            f"Bus: {self.counts['bus']}",
            f"Van: {self.counts['van']}",
            f"Others: {self.counts['others']}"
        ]
        
        max_width = max([cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] 
                        for line in text_lines])
        
        cv2.rectangle(frame, 
                     (x_offset - 5, y_offset - 25), 
                     (x_offset + max_width + 10, y_offset + len(text_lines) * 25 + 5), 
                     (0, 0, 0), -1)
        
        for i, line in enumerate(text_lines):
            cv2.putText(frame, line, 
                       (x_offset, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def reset_counts(self):
        """Reset all counting statistics"""
        self.counted_ids.clear()
        self.counts.clear()
        self.total_count = 0
        self.track_history.clear()


def main():
    parser = argparse.ArgumentParser(description='Traffic Tracking and Counting with YOLOv8')
    parser.add_argument('--model', type=str, 
                       default='detect/traffic_model6/weights/best.pt',
                       help='Path to trained YOLOv8 model weights')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--line-y', type=int, default=None,
                       help='Y position of counting line (if not set, will use middle of frame)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (optional)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found at {args.model}")
        print("Please provide the correct path to your trained model weights.")
        return
    
    # Initialize traffic counter
    print(f"Loading model from {args.model}...")
    counter = TrafficCounter(args.model, args.conf, args.iou)
    
    # Set counting line
    if args.source.isdigit():
        # Webcam - set line to middle of frame (will be adjusted after first frame)
        pass
    else:
        # Video file - read first frame to get dimensions
        cap = cv2.VideoCapture(args.source)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if args.line_y is None:
                    args.line_y = frame.shape[0] // 2
                counter.set_counting_line(args.line_y)
            cap.release()
    
    # Open video source
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set counting line if not set and using webcam
    if not counter.counting_line_set:
        if args.line_y is None:
            args.line_y = height // 2
        counter.set_counting_line(args.line_y)
    
    # Setup video writer if output is specified
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print("Starting traffic tracking and counting...")
    print("Press 'q' to quit, 'r' to reset counts, 's' to save screenshot")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame = counter.process_frame(frame)
            
            # Display frame
            cv2.imshow('Traffic Tracking and Counting', annotated_frame)
            
            # Save frame if output is specified
            if out:
                out.write(annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                counter.reset_counts()
                print("Counts reset!")
            elif key == ord('s'):
                screenshot_path = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"Screenshot saved: {screenshot_path}")
            
            frame_count += 1
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames. Total count: {counter.total_count}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n=== Final Statistics ===")
        print(f"Total vehicles counted: {counter.total_count}")
        print(f"  - Cars: {counter.counts['car']}")
        print(f"  - Buses: {counter.counts['bus']}")
        print(f"  - Vans: {counter.counts['van']}")
        print(f"  - Others: {counter.counts['others']}")
        print(f"Total frames processed: {frame_count}")


if __name__ == '__main__':
    main()

