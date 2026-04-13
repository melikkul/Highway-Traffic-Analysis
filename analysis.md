# Design Notes

## Why C++ and not Python

The multi-threading requirement was the deciding factor. Python threads share the GIL, so you can't actually run CPU-bound work in parallel without reaching for `multiprocessing`. That's fine for embarrassingly parallel tasks but awkward when you have a pipeline where threads need to share large frame buffers — the IPC overhead adds up. C++ threads share the same address space, which maps directly onto this problem. It's more boilerplate to write correctly, but the runtime model is simpler.

Secondary benefit: no garbage collector. The tracker thread has soft latency requirements (needs to stay roughly in sync with the inference thread), and GC pauses are an unpredictable source of jitter.

## Why YOLOv8 and not background subtraction

First approach was `cv::createBackgroundSubtractorMOG2()`. It's fast — under 5ms per frame on CPU — and requires no model download. But it fell apart on three things:

1. It gives you blobs, not class labels. You'd need a second classification stage anyway to split cars from trucks from motorcycles, and that stage would be as expensive as just running YOLO.
2. Shadows. MOG2 frequently treated vehicle shadows as foreground, inflating counts.
3. Occlusion. When two vehicles were close together, MOG2 merged them into one blob. YOLO outputs separate bounding boxes with class labels regardless.

YOLOv8 costs ~30–60ms per frame on CPU. For an offline processing task that's acceptable. If this were a real-time stream the calculus would be different.

## ONNXRuntime vs. OpenCV DNN

I tried loading the model with `cv::dnn::readNetFromONNX()` first. The model loads without complaint but crashes during inference — shape mismatch on the output tensor. After some digging this appears to be a known limitation in OpenCV 4.6's DNN module: it doesn't handle YOLOv8's dynamic output axes correctly. ONNXRuntime 1.20 handles it out of the box, so I switched.

## Direction mapping

The application defines two directions based on lateral (horizontal) centroid movement within the frame:

- **Direction A** — vehicle's centroid moves left (decreasing X) as it crosses the counting zone
- **Direction B** — vehicle's centroid moves right (increasing X) as it crosses the counting zone

This assumes the camera is positioned at the roadside, oriented roughly perpendicular to the traffic lanes, so vehicles appear to travel horizontally across the frame. In the provided test footage the two dominant traffic flows travel in opposite directions along the highway, and left vs. right horizontal movement cleanly separates them. If the camera is angled differently, the direction labels can be swapped by reinterpreting the output without changing the code.

Direction is determined from the centroid's lateral movement between the previous frame and the counting-zone crossing frame, not from screen-side position. This handles edge cases where a vehicle enters the counting zone from an off-centre position.

## Tracker

The centroid tracker uses greedy nearest-neighbour matching on a Euclidean distance matrix. For each new frame it:

1. Builds the full distance matrix between existing track centroids and new detection centroids
2. Sorts existing tracks by their closest unmatched detection
3. Assigns greedily, rejecting any pair further apart than `max_distance`

The Hungarian algorithm would give globally optimal assignments and handle dense, overlapping traffic better. For highway traffic where vehicles are mostly well-separated, greedy matching is reliable enough and meaningfully simpler — both to implement and to reason about when something goes wrong.

The `max_disappeared` parameter (default 30 frames) controls how long a track survives without a matching detection. Setting it too low causes ID flicker during brief occlusions; too high and a vehicle that leaves the frame might get "reactivated" by a new vehicle entering in a similar position.

## Speed estimation

Speed is estimated from the frame-to-frame horizontal displacement of each tracked centroid. The core formula is:

```
speed_kph = (dx_pixels / pixels_per_meter) / (1 / fps) * 3.6
```

**Camera and geometry assumptions:**

- The camera is mounted at the roadside looking across the lanes, so vehicle forward motion maps to horizontal pixel displacement. This is an assumption about the camera orientation — a camera mounted overhead would require a different axis.
- `pixels_per_meter` represents how many pixels correspond to one metre at the depth of the counting zone. The config default of `0.7` was calibrated empirically against the provided footage by observing centroid displacements for vehicles at known typical highway speeds (80–130 km/h) and tuning until the EMA estimates converged to a plausible range. There is no ground-truth reference, so this value carries inherent uncertainty. A proper calibration would measure a known physical distance (e.g., lane width, painted road markings) in the background image and derive the constant from that.
- Errors in `pixels_per_meter` are proportional: a ±30% calibration error produces ±30% speed error. The spec acknowledges this by stating perfect accuracy is not expected.

**Noise handling:**

Raw frame-to-frame displacement is extremely noisy: YOLO bounding boxes jitter 3–5 pixels per frame even for stationary objects. Three filters are applied:

- Sub-pixel movements (`dx < 2.0px`) are discarded as jitter
- Readings outside [25, 160] km/h are discarded as unrealistic for highway traffic
- After collecting a baseline (5+ samples), sudden jumps greater than ±15 km/h are clamped — this prevents a single bad detection from corrupting the running average
- An EMA (alpha = 0.1) provides the final smoothing

**Known limitation:** The `pixels_per_meter` constant is only accurate at one depth. Vehicles close to the camera appear to move faster in pixels than vehicles far away for the same physical speed. A perspective-correct implementation would require a homography calibrated to real-world reference points. Without calibration data this isn't possible, and the spec acknowledges this by stating that perfect accuracy isn't expected.

## Background reconstruction

The background image is generated by averaging ~30 frames sampled at even intervals across the video. Moving vehicles appear in different positions across samples; averaging washes them out, leaving the static road. This works reasonably well when vehicle density is moderate — if the road is always congested, the same vehicles appear in multiple samples and leave ghost artifacts.

Per-pixel median across the sample set would be more robust (outliers from vehicles don't affect the median the way they affect the mean), but sorting pixel values across 30 frames for every pixel in an 800×600×3 image is slow in serial C++ without vectorisation. The mean approach produces good enough results for this use case.
