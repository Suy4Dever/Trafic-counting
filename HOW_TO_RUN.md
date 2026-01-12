# How to Run Traffic Tracking on Video

## Quick Start

### Basic Command (Your Video)
```bash
python track_and_count.py --model detect/traffic_model6/weights/best.pt --source video_test/video1.mp4
```

## All Command Options

### 1. Run on Your Video File
```bash
python track_and_count.py --model detect/traffic_model6/weights/best.pt --source video_test/video1.mp4
```

### 2. Run with Custom Counting Line Position
Set the counting line at Y=400 pixels from top:
```bash
python track_and_count.py --model detect/traffic_model6/weights/best.pt --source video_test/video1.mp4 --line-y 400
```

### 3. Save Output Video
Save the tracked video with annotations:
```bash
python track_and_count.py --model detect/traffic_model6/weights/best.pt --source video_test/video1.mp4 --output output_tracked.mp4
```

### 4. Adjust Detection Sensitivity
Lower confidence threshold (detect more objects, may include false positives):
```bash
python track_and_count.py --model detect/traffic_model6/weights/best.pt --source video_test/video1.mp4 --conf 0.15
```

Higher confidence threshold (more strict, fewer false positives):
```bash
python track_and_count.py --model detect/traffic_model6/weights/best.pt --source video_test/video1.mp4 --conf 0.4
```

### 5. Complete Example with All Options
```bash
python track_and_count.py \
  --model detect/traffic_model6/weights/best.pt \
  --source video_test/video1.mp4 \
  --conf 0.25 \
  --iou 0.45 \
  --line-y 400 \
  --output tracked_output.mp4
```

## Controls While Running

- **'q'** - Quit the application
- **'r'** - Reset all counts
- **'s'** - Save a screenshot of current frame

## What You'll See

- **Bounding boxes** around detected vehicles (color-coded by type)
- **Track IDs** for each vehicle
- **Tracking trails** showing vehicle paths
- **Counting line** (yellow horizontal line)
- **Real-time statistics** showing:
  - Total vehicle count
  - Count per vehicle type (car, bus, van, others)

## Tips

1. **Counting Line Position**: Adjust `--line-y` to position the counting line where vehicles cross
2. **Confidence Threshold**: Start with 0.25, adjust based on detection quality
3. **Output Video**: Use `--output` to save results for later analysis
4. **Frame Rate**: Processing speed depends on your GPU. On CPU it will be slower.

## Troubleshooting

**Video won't open?**
- Check the video path is correct
- Ensure video format is supported (mp4, avi, etc.)

**No detections?**
- Lower the `--conf` threshold (try 0.15)
- Check that your model weights file exists

**Too many false positives?**
- Increase the `--conf` threshold (try 0.4 or 0.5)

