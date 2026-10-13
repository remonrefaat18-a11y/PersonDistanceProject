# Person and Car Distance Estimation

A computer-vision project that detects people and cars in a live camera feed with
Ultralytics YOLO11n, estimates their distance from the camera using apparent
object height, and includes a Flutter mobile application that runs inference
on-device with ONNX Runtime.

The repository contains two Python camera demos and a Flutter app. They share
the same general detection-and-distance idea, but use different model formats,
object-height assumptions, and user interfaces.

> **Important:** Distance is an approximate monocular-camera estimate, not a
> safety measurement. Do not use it for driving, collision avoidance, or other
> safety-critical decisions.

## Features

- Live camera detection using the YOLO11n model.
- Person and car detection; the Flutter app lets you choose one mode at a time.
- Approximate distance in centimeters and meters using object bounding-box
  height.
- Flutter camera overlay, confidence labels, and distance display.
- Optional focal-length calibration in the Flutter app and a basic calibration
  key in the Python distance demo.
- On-device inference in the Flutter app using an ONNX model.

## How distance is estimated

The project uses the pinhole-camera relationship:

```text
distance = (real object height × focal length in pixels) / object height in pixels
```

Where:

- `real object height` is an assumed physical height for the detected class.
- `focal length in pixels` is a camera-dependent value.
- `object height in pixels` is the height of the detected bounding box in the
  camera image.

The app's calibration uses a known object height and a measured distance to
estimate the focal length:

```text
focal length in pixels = (object height in pixels × measured distance) / real object height
```

This method assumes the object is upright, its full height is visible, and the
selected real-world height is representative. A partially cropped object,
unusual object dimensions, camera changes, perspective, or a poor detection can
make the result inaccurate. Calibration adjusts the focal value for the current
camera setup; it does not add a depth sensor or make the estimate universally
accurate.

### Distance settings by implementation

| Implementation | Classes measured | Assumed real heights | Initial focal value |
| --- | --- | --- | --- |
| `distance.py` | Person, car, bottle, cell phone | Person: 170 cm; car: 140 cm; bottle: 23 cm; cell phone: 16 cm | Fixed at 487.5 px |
| Flutter app | Person or car, selected on the start screen | Person: 170 cm; car: 150 cm | `0.72 × rotated camera-frame height` until calibrated |

The Python and Flutter values are defined separately in their respective source
files. Results from the two implementations may therefore differ.

## Repository layout

```text
.
├── main.py                          # Basic live YOLO detection window
├── distance.py                      # Python detection and distance demo
├── yolo11n.pt                        # PyTorch model used by the Python demos
├── yolo11n.onnx                      # ONNX model copy at the repository root
├── requirements.txt                  # Current dependency notes (see setup)
├── Remon.jpg                         # Sample image
├── runs/detect/predict/Remon.jpg      # Example Ultralytics output
└── mobile_app/
    ├── lib/main.dart                 # Flutter UI, camera, inference, and distance logic
    ├── pubspec.yaml                  # Flutter dependencies and model asset declaration
    ├── assets/models/yolo11n.onnx    # ONNX model bundled with the Flutter app
    └── android/                      # Android app configuration
```

The Flutter app loads `mobile_app/assets/models/yolo11n.onnx`; it does not load
the root-level `.pt` file. Keep the bundled ONNX asset in place or update the
asset declaration in `mobile_app/pubspec.yaml` and the asset path in
`mobile_app/lib/main.dart` if you replace it.

## Requirements

### Python demos

- Python 3.9 or newer is recommended.
- A camera accessible to OpenCV (the scripts open camera index `0`).
- The `yolo11n.pt` model file in the repository root.

Install the Python packages used by the demos:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python
```

If PowerShell does not allow activating the virtual environment, run the Python
commands with the environment's executable directly, or activate it using a
shell appropriate for your system.

**About `requirements.txt`:** The current file contains `pip install` and
`flutter pub add` command notes rather than standard pinned Python requirement
entries. For that reason, `pip install -r requirements.txt` is not the setup
command for this repository. Use the commands above for the Python demos and
`flutter pub get` for the Flutter app.

### Flutter application

- Flutter SDK with a Dart SDK satisfying the constraint in
  `mobile_app/pubspec.yaml` (`^3.13.5`).
- Android SDK/toolchain for Android builds.
- An Android device or emulator with a camera. A physical device is recommended
  for live camera inference.

The Android manifest requests camera permission, and the Android module sets
`minSdk` to 24. Flutter platform folders for other targets are present, but the
live camera/inference workflow is primarily documented here for Android.

## Run the Python demos

From the repository root, with the virtual environment activated:

### Detection only

```powershell
python main.py
```

This opens the default camera and draws YOLO detections in an OpenCV window.
Press **Q** in the window to stop.

### Detection with distance estimates

```powershell
python distance.py
```

This draws boxes and displays an estimated distance for detected people, cars,
bottles, and cell phones. Press **Q** in the OpenCV window to stop.

The Python distance demo uses its fixed `FOCAL = 487.5` value by default. Its
calibration key sets the assumed calibration distance to 50 cm; before using
that key, make sure the detected supported object is actually at that distance.
For a different known distance, update `CALIB_DIST` in `distance.py` before
running the script. The recalculated focal value only remains in memory for
that run.

## Run the Flutter application

From the repository root:

```powershell
cd mobile_app
flutter pub get
flutter devices
flutter run
```

Choose an available Android device from `flutter devices` if Flutter asks.
Accept the camera permission request when prompted.

To build an Android APK:

```powershell
flutter build apk
```

The generated APK is normally written under
`mobile_app/build/app/outputs/flutter-apk/`.

### Using the app

1. Start the app and choose **Person** or **Car (Live)**.
2. Point the camera at a single, clearly visible target.
3. In **Person** mode, the app collects five detections, uses the median
   bounding-box height, and shows a captured crop with the estimated distance.
   Use the screen's measure-again action to start another measurement.
4. In **Car (Live)** mode, the app continuously updates the estimate using a
   rolling median of up to five detected frame heights. The displayed result
   clears after several consecutive frames without a car.
5. Use **Calibrate** while a target is detected. Enter its measured distance
   and real-world height to update the focal-length estimate.

The app's focal override is held in memory only. It is not saved between app
launches. To use a fixed focal value when launching the app, set
`focalPxOverride` in `mobile_app/lib/main.dart` after calibration.

## Flutter inference pipeline

The camera screen performs the following work locally on the device:

1. Reads camera frames and accounts for camera image format and sensor rotation.
2. Resizes each frame into a 640 × 640 letterboxed RGB input and normalizes the
   values for the model.
3. Runs the bundled `assets/models/yolo11n.onnx` model with
   `flutter_onnxruntime`.
4. Decodes the model output as `[1, 84, 8400]`, applies a 0.5 confidence
   threshold and non-maximum suppression with an IoU threshold of 0.45.
5. Selects the requested COCO class (person class ID 0 or car class ID 2), draws
   its bounding box, and estimates distance from its pixel height.

Inference is rate-limited to approximately one attempt every 150 ms. The app
does not upload camera frames to a server. Detections are not persistent object
tracks: each update uses the highest-confidence applicable detection.

## Model files

- `yolo11n.pt` is used by the Python scripts through the Ultralytics Python
  package.
- `mobile_app/assets/models/yolo11n.onnx` is packaged into the Flutter app and
  loaded through ONNX Runtime.
- `yolo11n.onnx` in the repository root is another ONNX model copy; the Flutter
  asset path is the copy under `mobile_app/assets/models/`.

The Flutter decoder expects the ONNX model's current 640 × 640 input and
`[1, 84, 8400]` output layout. If you replace or re-export the model, confirm
that its preprocessing, class ordering, output tensor layout, and input/output
names still match the app implementation.

## Troubleshooting

- **Camera cannot open in a Python demo:** Check that the camera is connected,
  not being used exclusively by another application, and available at index
  `0`. Change `cv2.VideoCapture(0)` if the camera uses another index.
- **`yolo11n.pt` cannot be loaded:** Run the script from the repository root
  and verify that the model file is present there.
- **No camera permission or preview in Flutter:** Grant camera access in Android
  settings, reconnect the device, and verify that `flutter devices` lists it.
- **Model fails to load in Flutter:** Confirm that
  `mobile_app/assets/models/yolo11n.onnx` exists and is listed in
  `mobile_app/pubspec.yaml`, then run `flutter pub get` again.
- **Distance seems implausible:** Ensure the target is fully in frame, check
  the real-height assumptions, keep the same camera setup, and calibrate against
  a measured distance. Treat the output as an estimate.

## Limitations

- A single RGB camera cannot directly measure depth. This project estimates
  distance using assumed object dimensions.
- People and cars vary in physical size, posture, orientation, and visibility;
  a bounding box may not correspond to the assumed real-world height.
- Image blur, lighting, camera zoom, lens distortion, perspective, occlusion,
  and partial cropping can affect both detection and distance estimates.
- Flutter calibration is session-only. The Python demo's calibration is also
  not persisted and assumes 50 cm unless its source is changed.
- The app processes the highest-confidence detection for the selected class;
  it does not identify a particular object over time.


  ## Download link for android:

https://www.mediafire.com/file/rubd9tt9wzm2c1f/app-arm64-v8a-release.apk/file

## License and model provenance

This repository does not define a project-wide license. Check the licenses and
terms for the code, model weights, and dependencies before redistributing or
using them commercially.
