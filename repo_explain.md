# 🚀 Traffic Flow Analytics System

## 🧠 0. System Overview

```
User upload video + draw ROI (frontend)
        ↓
Backend receives request
        ↓
Call inference pipeline
        ↓
Detect → Track → ROI → Count
        ↓
Render video + stats
        ↓
Return result to frontend
```

---

# 🔵 1. Frontend (`frontend/`)

### 🎯 Purpose

User interaction layer:

* Upload video
* Draw ROI
* View results

## 📁 `src/components/`

### `VideoUploader.jsx`

* Upload video to backend (`/upload`)
* Returns `video_path` or `video_id`

### `ROISelector.jsx`

* Core component
* Allows drawing polygon ROI
* Output format:

```json
[[x1,y1], [x2,y2], ...]
```

* Sent to `/roi`

### `VideoPlayer.jsx`

* Displays processed video
* Loads from `storage/outputs`

### `StatsPanel.jsx`

* Displays:

  * Total vehicle count
  * Per-class stats
  * ROI-based stats

## 📁 `src/pages/`

* Page-level components (Home, Upload, Result)

## 📁 `src/services/api.js`

* API calls abstraction:

```js
uploadVideo()
setROI()
processVideo()
```

## `package.json`

* React dependencies & config

---

# 🟢 2. Backend (`backend/`)

### 🎯 Purpose

* API layer
* Orchestrates pipeline
* Does NOT perform heavy CV

## 📁 `app/main.py`

* FastAPI entry point

## 📁 `app/routes/`

### `upload.py`

* Receives video
* Saves to `storage/uploads/`

### `roi.py`

* Stores ROI per video

### `inference.py`

* Triggers pipeline execution
* Returns output path

## 📁 `app/schemas/`

* Request/response models (Pydantic)

## 📁 `app/services/`

### `video_service.py`

* File handling
* Path management

### `pipeline_service.py`

* Bridge to inference layer

## 📁 `app/config.py`

* System config

## `requirements.txt`

* Backend dependencies

---

# 🔴 3. Inference (`inference/`)

### 🎯 Purpose

Core AI processing

## 📁 `pipeline/`

### `detector.py`

* Loads YOLO model
* Output: bounding boxes

### `tracker.py`

* Assigns track IDs

### `roi.py`

* Filters objects inside ROI

### `counter.py`

* Counting logic
* Avoids double counting

### `pipeline.py`

* Core flow:

```
detect → track → roi → count
```

## 📁 `utils/`

### `visualization.py`

* Draws:

  * Bounding boxes
  * IDs
  * ROI
  * Count

### `video_io.py`

* Video read/write

## 📁 `weights/`

* Stores trained model

## `run_inference.py`

* Standalone pipeline execution

---

# 🟡 4. Edge (`edge/`)

### 🎯 Purpose

Optimize for Jetson Orin

## 📁 `trt/`

### `export_onnx.py`

* Convert PyTorch → ONNX

### `build_engine.py`

* ONNX → TensorRT

### `infer_trt.py`

* TensorRT inference

## 📁 `configs/`

* Runtime configs (FP16, batch size)

---

# 🟣 5. Storage (`storage/`)

### `uploads/`

* Raw videos

### `outputs/`

* Processed videos

### `temp/`

* Temporary files

---

# ⚫ 6. Docker (`docker/`)

### `backend.Dockerfile`

* Build API service

### `inference.Dockerfile`

* Build CV environment

### `docker-compose.yml`

* Run full system

---

# ⚙️ 7. Scripts (`scripts/`)

### `start.sh`

* Start system

### `test_pipeline.py`

* Quick pipeline testing

---

# 📄 8. README.md

Should include:

* Architecture diagram
* Setup instructions
* Demo flow

---

# 🔥 Key Design Principles

## Separation of Concerns

* Frontend → UI
* Backend → orchestration
* Inference → AI
* Edge → optimization

## Scalability

* Modular components
* Easy to swap models/tracker

## Production-ready

* Clean structure
* Deployable with Docker

---

# 💀 Final Insight

> Good models are not enough.
> Clean system design is what makes your project stand out.
