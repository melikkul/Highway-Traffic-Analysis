# Traffic Flow Analysis

Highway traffic analysis pipeline written in C++. Detects and classifies vehicles using YOLOv8 (via ONNXRuntime), tracks them with a centroid tracker to avoid double-counting, estimates per-direction speeds, and reconstructs a clean background image from the video.

## Architecture

Three threads, connected by lock-free queues:

1. **Reader** — decodes frames and resizes to the configured resolution
2. **Inference** — runs YOLOv8 through ONNXRuntime; produces bounding boxes with class labels per frame
3. **Tracker** (main thread) — matches detections across frames, counts vehicles crossing the zone, estimates speed

C++ was chosen over Python primarily because of the GIL — Python threads can't do real parallel CPU work with shared memory, whereas this pipeline naturally maps onto C++ threads sharing the same frame buffers. ONNXRuntime is used instead of OpenCV's built-in DNN module because OpenCV 4.6 (Ubuntu 22.04) mishandles YOLOv8's dynamic output tensor shape.

## Building

### Docker (recommended)

```bash
docker build -t traffic-analyzer .
docker run --gpus all -v "$(pwd)/input:/input" -v "$(pwd)/output:/output" traffic-analyzer
```

The `--gpus all` flag is passed through but inference runs on CPU via ONNXRuntime. GPU acceleration can be enabled by swapping in the CUDA execution provider if the host has a CUDA-capable card.

### Local

Requires OpenCV and ONNXRuntime installed system-wide.

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
cd ..
./build/main
```

## Usage

Drop input videos (`.mp4`, `.avi`, `.ts`) into `input/`, then:

```bash
./build/main
```

To open a live preview window with bounding boxes, track IDs, and the counting zone:

```bash
./build/main --gui
```

Press **ESC** to stop. GUI mode processes every frame through YOLO so it's slower than real-time on CPU.

## Direction mapping

The application assumes the camera is positioned at the roadside, roughly perpendicular to the traffic lanes, so vehicles travel horizontally across the frame. Direction is determined from lateral centroid movement at the moment of zone crossing:

- **Direction A** — centroid moving left (decreasing X)
- **Direction B** — centroid moving right (increasing X)

In the provided highway footage this cleanly separates the two dominant traffic flows. See `analysis.md` for a fuller discussion of the geometry and limitations.

## Configuration

Edit `config.json` to tune the pipeline. Key parameters:

- `model.confidence_threshold` — minimum detection score (0–1)
- `model.nms_threshold` — IoU threshold for duplicate box suppression
- `zones.y_min` / `zones.y_max` — pixel Y range of the counting line
- `speed.pixels_per_meter` — camera calibration constant; needs to be set per scene (see `analysis.md`)
- `speed.ema_alpha` — EMA smoothing factor; lower values are smoother but slower to react
- `tracker.max_disappeared` — frames before a lost track is dropped
- `tracker.max_distance` — maximum centroid movement (px) between frames for track association

## Output

For each input video, two files are written to `output/`:

- `<name>.csv` — single-row results: total count, frame time, per-direction breakdown by type, average speeds
- `<name>_background.jpg` — estimated background (temporal mean of sampled frames, vehicles averaged out)
