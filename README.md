# Traffic Counting System

A comprehensive traffic tracking and counting system using YOLOv8 for vehicle detection, tracking, and counting.

## Features

- **Real-time Vehicle Detection**: Uses trained YOLOv8 model to detect vehicles (cars, buses, vans, others)
- **Object Tracking**: Implements ByteTrack algorithm for robust multi-object tracking
- **Traffic Counting**: Counts vehicles crossing a counting line with direction detection
- **Visualization**: Real-time visualization with bounding boxes, track IDs, trails, and statistics
- **Class-based Counting**: Separate counts for different vehicle types

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main tracking and counting script:

```bash
python track_and_count.py --model detect/traffic_model6/weights/best.pt --source 0
```

### Command Line Arguments

- `--model`: Path to trained YOLOv8 model weights (default: `detect/traffic_model6/weights/best.pt`)
- `--source`: Video source - `0` for webcam, or path to video file (default: `0`)
- `--conf`: Confidence threshold for detections (default: `0.25`)
- `--iou`: IoU threshold for NMS (default: `0.45`)
- `--line-y`: Y position of counting line in pixels (default: middle of frame)
- `--output`: Optional output video path to save results

### Examples

**Using webcam:**
```bash
python track_and_count.py --model detect/traffic_model6/weights/best.pt --source 0
```

**Using video file:**
```bash
python track_and_count.py --model detect/traffic_model6/weights/best.pt --source path/to/video.mp4
```

**With custom counting line position:**
```bash
python track_and_count.py --model detect/traffic_model6/weights/best.pt --source video.mp4 --line-y 400
```

**Save output video:**
```bash
python track_and_count.py --model detect/traffic_model6/weights/best.pt --source video.mp4 --output output.mp4
```

### Simple Example Script

For quick testing, use the simplified example:

```bash
python example_usage.py
```

Edit `example_usage.py` to change the video source or model path.

## Controls

While the application is running:

- **'q'**: Quit the application
- **'r'**: Reset all counts
- **'s'**: Save a screenshot of the current frame

## How It Works

1. **Detection**: YOLOv8 model detects vehicles in each frame
2. **Tracking**: ByteTrack algorithm assigns unique IDs to each vehicle and tracks them across frames
3. **Counting**: When a vehicle crosses the counting line, it's counted once (prevents double counting)
4. **Visualization**: 
   - Bounding boxes with class labels and confidence scores
   - Track IDs for each vehicle
   - Tracking trails showing vehicle paths
   - Real-time count statistics

## Model Information

The system uses a trained YOLOv8 model with the following classes:
- **Car** (class 0)
- **Bus** (class 1)
- **Van** (class 2)
- **Others** (class 3)

The trained model weights are located at: `detect/traffic_model6/weights/best.pt`

## Output

The system displays:
- Total vehicle count
- Count per vehicle class (car, bus, van, others)
- Visual tracking with bounding boxes and trails
- Final statistics printed to console when exiting

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for real-time performance)
- Trained YOLOv8 model weights

## Notes

- The counting line should be positioned where you want to count vehicles crossing
- Each vehicle is counted only once when it crosses the line
- The system tracks vehicles even when temporarily occluded
- Adjust confidence and IoU thresholds based on your use case for better accuracy
