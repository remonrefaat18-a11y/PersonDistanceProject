import 'dart:io' show Platform;
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;

// ============================================================
// Constants
// ============================================================

const int modelSize = 640;
const int numClasses = 80;
const int numDetections = 8400;

const int personClass = 0;
const int carClass = 2;

const double confidenceThreshold = 0.5;
const double nmsIouThreshold = 0.45;

/// Person: عدد الفريمات اللي بناخد منها median قبل ما نثبّت النتيجة
const int samplesNeeded = 5;

/// Car (live): عدد الفريمات في الـ rolling median (بيثبّت الرقم من غير ما يوقف)
const int liveWindow = 5;

/// Car (live): كام فريم متتالي من غير كشف قبل ما نقول "مفيش عربية"
const int liveMissedLimit = 3;

const int inferenceIntervalMs = 150;

/// الأطوال الحقيقية المتوقعة (سم)
const double personHeightCm = 170.0;
const double carHeightCm = 150.0;

/// بعد ما تعمل Calibrate هيظهر الرقم في رسالة وفي الـ console.
/// حطه هنا عشان يتثبت للأبد، مثال: double? focalPxOverride = 512.3;
double? focalPxOverride;

/// تقدير مبدئي للـ focal (بالبكسل) لو لسه ماعملتش Calibrate.
double focalPx(int imageHeight) => focalPxOverride ?? imageHeight * 0.72;

double distanceFor(int classId, double pixelHeight, int imageHeight) {
  if (pixelHeight <= 0) return 0;
  final known = classId == personClass ? personHeightCm : carHeightCm;
  return known * focalPx(imageHeight) / pixelHeight;
}

// ============================================================
// main
// ============================================================

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);

  final cameras = await availableCameras();
  final camera = cameras.firstWhere(
    (c) => c.lensDirection == CameraLensDirection.back,
    orElse: () => cameras.first,
  );

  runApp(MyApp(camera: camera));
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;

  const MyApp({super.key, required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: DetectionChoiceScreen(camera: camera),
    );
  }
}

// ============================================================
// Choice screen
// ============================================================

class DetectionChoiceScreen extends StatelessWidget {
  final CameraDescription camera;

  const DetectionChoiceScreen({super.key, required this.camera});

  void _open(BuildContext context, int cls) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => CameraScreen(camera: camera, selectedClass: cls),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Distance')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text(
              'What do you want to detect?',
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 40),
            ElevatedButton(
              onPressed: () => _open(context, personClass),
              child: const Text('Person'),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () => _open(context, carClass),
              child: const Text('Car (Live)'),
            ),
          ],
        ),
      ),
    );
  }
}

// ============================================================
// Data classes
// ============================================================

class Detection {
  final double x1;
  final double y1;
  final double x2;
  final double y2;
  final double confidence;
  final int classId;
  final double distance;

  /// الصندوق ملامس لحافة الصورة من فوق أو تحت (الطول غير كامل)
  final bool truncated;

  Detection({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    required this.confidence,
    required this.classId,
    required this.distance,
    this.truncated = false,
  });

  Detection copyWith({double? distance}) => Detection(
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2,
        confidence: confidence,
        classId: classId,
        distance: distance ?? this.distance,
        truncated: truncated,
      );
}

class LetterboxInfo {
  final double scale;
  final int newW;
  final int newH;
  final int padX;
  final int padY;

  LetterboxInfo._(this.scale, this.newW, this.newH, this.padX, this.padY);

  factory LetterboxInfo(int srcW, int srcH, int target) {
    final scale = math.min(target / srcW, target / srcH);
    final newW = (srcW * scale).round();
    final newH = (srcH * scale).round();
    return LetterboxInfo._(
      scale,
      newW,
      newH,
      (target - newW) ~/ 2,
      (target - newH) ~/ 2,
    );
  }
}

// ============================================================
// Frame sampler: YUV420 / BGRA + rotation + letterbox في خطوة واحدة
// ============================================================

class FrameSampler {
  final CameraImage image;
  final int rotation; // 0 / 90 / 180 / 270

  late final int rotW;
  late final int rotH;

  late final bool _isBgra;
  late final Uint8List _y, _u, _v;
  late final int _yStride, _uStride, _vStride, _uPix, _vPix;

  FrameSampler(this.image, this.rotation) {
    if (rotation == 90 || rotation == 270) {
      rotW = image.height;
      rotH = image.width;
    } else {
      rotW = image.width;
      rotH = image.height;
    }

    _isBgra = image.planes.length == 1;

    if (_isBgra) {
      _y = image.planes[0].bytes;
      _u = _y;
      _v = _y;
      _yStride = image.planes[0].bytesPerRow;
      _uStride = 0;
      _vStride = 0;
      _uPix = 0;
      _vPix = 0;
    } else {
      _y = image.planes[0].bytes;
      _u = image.planes[1].bytes;
      _v = image.planes[2].bytes;
      _yStride = image.planes[0].bytesPerRow;
      _uStride = image.planes[1].bytesPerRow;
      _vStride = image.planes[2].bytesPerRow;
      _uPix = image.planes[1].bytesPerPixel ?? 1;
      _vPix = image.planes[2].bytesPerPixel ?? 1;
    }
  }

  static int _c(int v) => v < 0 ? 0 : (v > 255 ? 255 : v);

  int _original(int ox, int oy) {
    if (_isBgra) {
      final i = oy * _yStride + ox * 4;
      final b = _y[i];
      final g = _y[i + 1];
      final r = _y[i + 2];
      return (r << 16) | (g << 8) | b;
    }

    final yp = _y[oy * _yStride + ox];
    final up = _u[(oy >> 1) * _uStride + (ox >> 1) * _uPix] - 128;
    final vp = _v[(oy >> 1) * _vStride + (ox >> 1) * _vPix] - 128;

    final r = _c((yp + 1.402 * vp).round());
    final g = _c((yp - 0.344136 * up - 0.714136 * vp).round());
    final b = _c((yp + 1.772 * up).round());

    return (r << 16) | (g << 8) | b;
  }

  int pixel(int rx, int ry) {
    int ox, oy;
    switch (rotation) {
      case 90:
        ox = ry;
        oy = image.height - 1 - rx;
        break;
      case 180:
        ox = image.width - 1 - rx;
        oy = image.height - 1 - ry;
        break;
      case 270:
        ox = image.width - 1 - ry;
        oy = rx;
        break;
      default:
        ox = rx;
        oy = ry;
    }
    return _original(ox, oy);
  }

  Float32List toTensor(LetterboxInfo lb) {
    const size = modelSize;
    const plane = size * size;

    final data = Float32List(3 * plane);
    data.fillRange(0, 3 * plane, 114 / 255.0);

    for (int ty = lb.padY; ty < lb.padY + lb.newH; ty++) {
      final ry = math.min(rotH - 1, ((ty - lb.padY + 0.5) / lb.scale).floor());

      for (int tx = lb.padX; tx < lb.padX + lb.newW; tx++) {
        final rx =
            math.min(rotW - 1, ((tx - lb.padX + 0.5) / lb.scale).floor());

        final p = pixel(rx, ry);
        final idx = ty * size + tx;

        data[idx] = ((p >> 16) & 0xFF) / 255.0;
        data[plane + idx] = ((p >> 8) & 0xFF) / 255.0;
        data[2 * plane + idx] = (p & 0xFF) / 255.0;
      }
    }

    return data;
  }

  img.Image crop(int x1, int y1, int x2, int y2) {
    final w = x2 - x1;
    final h = y2 - y1;
    final out = img.Image(width: w, height: h);

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final p = pixel(x1 + x, y1 + y);
        out.setPixelRgb(x, y, (p >> 16) & 0xFF, (p >> 8) & 0xFF, p & 0xFF);
      }
    }
    return out;
  }
}

// ============================================================
// Painter
// ============================================================

class DetectionPainter extends CustomPainter {
  final List<Detection> detections;
  final int imageWidth;
  final int imageHeight;

  DetectionPainter({
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final scaleX = size.width / imageWidth;
    final scaleY = size.height / imageHeight;

    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = Colors.greenAccent;

    final textPainter = TextPainter(textDirection: TextDirection.ltr);

    for (final d in detections) {
      final rect = Rect.fromLTRB(
        d.x1 * scaleX,
        d.y1 * scaleY,
        d.x2 * scaleX,
        d.y2 * scaleY,
      );

      canvas.drawRect(rect, boxPaint);

      final label = d.classId == personClass ? 'PERSON' : 'CAR';

      // لو المسافة محسوبة (live للعربية) نكتبها على الصندوق
      final distText =
          d.distance > 0 ? '\n${(d.distance / 100).toStringAsFixed(2)} m' : '';

      textPainter.text = TextSpan(
        text: '$label  ${(d.confidence * 100).toStringAsFixed(0)}%$distText',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 16,
          fontWeight: FontWeight.bold,
          backgroundColor: Colors.black54,
        ),
      );
      textPainter.layout();

      final dy = rect.top - textPainter.height - 4;
      textPainter.paint(
        canvas,
        Offset(rect.left, dy < 0 ? rect.top + 4 : dy),
      );
    }
  }

  @override
  bool shouldRepaint(covariant DetectionPainter oldDelegate) => true;
}

// ============================================================
// Camera screen
// ============================================================

class CameraScreen extends StatefulWidget {
  final CameraDescription camera;
  final int selectedClass;

  const CameraScreen({
    super.key,
    required this.camera,
    required this.selectedClass,
  });

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController controller;

  OnnxRuntime? ort;
  OrtSession? session;

  bool cameraReady = false;
  bool modelLoaded = false;
  bool processingFrame = false;
  bool measurementDone = false;

  /// Car = live mode (بيقيس باستمرار)، Person = قياس واحد
  bool get isLive => widget.selectedClass == carClass;

  DateTime lastInferenceTime = DateTime.fromMillisecondsSinceEpoch(0);

  // shared live state
  List<Detection> currentDetections = [];
  int frameW = 480;
  int frameH = 720;

  // Person: multi-frame measurement
  final List<double> samples = [];
  int truncatedCount = 0;

  // Person: final result
  Detection? finalDetection;
  Uint8List? capturedJpg;
  double medianPixelHeight = 0;

  // Car: live state
  final List<double> liveHeights = [];
  int missedFrames = 0;
  double livePixelHeight = 0;

  @override
  void initState() {
    super.initState();
    initializeCamera();
  }

  Future<void> initializeCamera() async {
    try {
      controller = CameraController(
        widget.camera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup:
            Platform.isIOS ? ImageFormatGroup.bgra8888 : ImageFormatGroup.yuv420,
      );

      await controller.initialize();

      if (!mounted) return;
      setState(() => cameraReady = true);

      await loadYOLO();

      await controller.startImageStream(processCameraFrame);
    } catch (e) {
      debugPrint('CAMERA INIT ERROR: $e');
    }
  }

  Future<void> loadYOLO() async {
    try {
      ort = OnnxRuntime();

      session = await ort!.createSessionFromAsset(
        'assets/models/yolo11n.onnx',
      );

      if (mounted) setState(() => modelLoaded = true);

      debugPrint('YOLO LOADED');
      debugPrint('Input: ${session!.inputNames}');
      debugPrint('Output: ${session!.outputNames}');
    } catch (e) {
      debugPrint('YOLO ERROR: $e');
    }
  }

  // ==========================================================
  // Frame processing
  // ==========================================================

  Future<void> processCameraFrame(CameraImage cameraImage) async {
    if (processingFrame || !modelLoaded || measurementDone) return;

    final now = DateTime.now();
    if (now.difference(lastInferenceTime).inMilliseconds <
        inferenceIntervalMs) {
      return;
    }
    lastInferenceTime = now;
    processingFrame = true;

    OrtValue? input;
    Map<String, OrtValue>? outputs;

    try {
      final sampler =
          FrameSampler(cameraImage, widget.camera.sensorOrientation);
      final lb = LetterboxInfo(sampler.rotW, sampler.rotH, modelSize);

      final tensor = sampler.toTensor(lb);

      input = await OrtValue.fromList(
        tensor,
        [1, 3, modelSize, modelSize],
      );

      outputs = await session!.run({session!.inputNames.first: input});

      final output = outputs[session!.outputNames.first];
      if (output == null) return;

      final raw = await output.asFlattenedList();

      final detections = applyNMS(
        decode(raw.cast<num>(), lb, sampler.rotW, sampler.rotH),
      );

      if (!mounted) return;

      // ======================================================
      // CAR: live distance (مفيش capture ومفيش توقف)
      // ======================================================
      if (isLive) {
        if (detections.isEmpty) {
          missedFrames++;
          // نستنى كام فريم قبل ما نمسح، عشان الرقم ما يرمشش
          if (missedFrames >= liveMissedLimit) {
            liveHeights.clear();
            livePixelHeight = 0;
            setState(() => currentDetections = []);
          }
          return;
        }

        missedFrames = 0;

        final best = detections.first;

        liveHeights.add(best.y2 - best.y1);
        if (liveHeights.length > liveWindow) {
          liveHeights.removeAt(0);
        }

        // rolling median
        final sorted = [...liveHeights]..sort();
        final medianH = sorted[sorted.length ~/ 2];
        final dist = distanceFor(best.classId, medianH, sampler.rotH);

        setState(() {
          frameW = sampler.rotW;
          frameH = sampler.rotH;
          livePixelHeight = medianH;
          currentDetections = [best.copyWith(distance: dist)];
        });
        return;
      }

      // ======================================================
      // PERSON: قياس واحد (زي ما كان)
      // ======================================================
      if (detections.isEmpty) {
        samples.clear();
        truncatedCount = 0;
        setState(() => currentDetections = []);
        return;
      }

      final best = detections.first;

      samples.add(best.y2 - best.y1);
      if (best.truncated) truncatedCount++;

      setState(() {
        frameW = sampler.rotW;
        frameH = sampler.rotH;
        currentDetections = [best];
      });

      if (samples.length >= samplesNeeded) {
        final sorted = [...samples]..sort();
        final medianH = sorted[sorted.length ~/ 2];

        final dist = distanceFor(best.classId, medianH, sampler.rotH);

        final left = best.x1.floor().clamp(0, sampler.rotW - 1);
        final top = best.y1.floor().clamp(0, sampler.rotH - 1);
        final right = best.x2.ceil().clamp(left + 1, sampler.rotW);
        final bottom = best.y2.ceil().clamp(top + 1, sampler.rotH);

        final crop = sampler.crop(left, top, right, bottom);
        final jpg = Uint8List.fromList(img.encodeJpg(crop));

        final result = Detection(
          x1: best.x1,
          y1: best.y1,
          x2: best.x2,
          y2: best.y2,
          confidence: best.confidence,
          classId: best.classId,
          distance: dist,
          truncated: truncatedCount > samples.length / 2,
        );

        if (mounted) {
          setState(() {
            finalDetection = result;
            medianPixelHeight = medianH;
            capturedJpg = jpg;
            currentDetections = [result];
            measurementDone = true;
          });
        }
      }
    } catch (e) {
      debugPrint('INFERENCE ERROR: $e');
    } finally {
      await input?.dispose();
      if (outputs != null) {
        for (final t in outputs.values) {
          await t.dispose();
        }
      }
      processingFrame = false;
    }
  }

  // ==========================================================
  // Decode YOLO output [1, 84, 8400]
  // ==========================================================

  List<Detection> decode(
    List<num> out,
    LetterboxInfo lb,
    int rotW,
    int rotH,
  ) {
    final result = <Detection>[];
    final clsOffset = (4 + widget.selectedClass) * numDetections;

    for (int i = 0; i < numDetections; i++) {
      final score = out[clsOffset + i].toDouble();
      if (score < confidenceThreshold) continue;

      final cx = out[i].toDouble();
      final cy = out[numDetections + i].toDouble();
      final w = out[2 * numDetections + i].toDouble();
      final h = out[3 * numDetections + i].toDouble();

      double x1 = (cx - w / 2 - lb.padX) / lb.scale;
      double y1 = (cy - h / 2 - lb.padY) / lb.scale;
      double x2 = (cx + w / 2 - lb.padX) / lb.scale;
      double y2 = (cy + h / 2 - lb.padY) / lb.scale;

      x1 = x1.clamp(0.0, rotW.toDouble()).toDouble();
      x2 = x2.clamp(0.0, rotW.toDouble()).toDouble();
      y1 = y1.clamp(0.0, rotH.toDouble()).toDouble();
      y2 = y2.clamp(0.0, rotH.toDouble()).toDouble();

      if (y2 - y1 < 2 || x2 - x1 < 2) continue;

      final truncated = y1 <= 3 || y2 >= rotH - 3;

      result.add(Detection(
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2,
        confidence: score,
        classId: widget.selectedClass,
        distance: 0,
        truncated: truncated,
      ));
    }

    return result;
  }

  double calculateIoU(Detection a, Detection b) {
    final ix1 = math.max(a.x1, b.x1);
    final iy1 = math.max(a.y1, b.y1);
    final ix2 = math.min(a.x2, b.x2);
    final iy2 = math.min(a.y2, b.y2);

    final iw = math.max(0.0, ix2 - ix1);
    final ih = math.max(0.0, iy2 - iy1);
    final inter = iw * ih;

    final areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    final areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    final union = areaA + areaB - inter;

    return union <= 0 ? 0 : inter / union;
  }

  List<Detection> applyNMS(List<Detection> input) {
    final list = [...input]
      ..sort((a, b) => b.confidence.compareTo(a.confidence));

    final selected = <Detection>[];

    while (list.isNotEmpty) {
      final best = list.removeAt(0);
      selected.add(best);
      list.removeWhere((d) => calculateIoU(best, d) > nmsIouThreshold);
    }

    return selected;
  }

  // ==========================================================
  // Calibration (شغال في الوضعين)
  // ==========================================================

  Future<void> showCalibrationDialog() async {
    // Person: النتيجة النهائية / Car: القياس الحي الحالي
    final Detection? det = isLive
        ? (currentDetections.isEmpty ? null : currentDetections.first)
        : finalDetection;
    final double pixelH = isLive ? livePixelHeight : medianPixelHeight;

    if (det == null || pixelH <= 0) return;

    final result = await showDialog<List<double>>(
      context: context,
      builder: (_) => CalibrationDialog(
        defaultHeight:
            det.classId == personClass ? personHeightCm : carHeightCm,
      ),
    );

    if (result == null || !mounted) return;

    final realDist = result[0];
    final realHeight = result[1];

    // focal = pixelHeight * realDistance / realHeight
    final newFocal = pixelH * realDist / realHeight;
    focalPxOverride = newFocal;

    debugPrint('CALIBRATED focalPx = $newFocal');

    setState(() {
      if (!isLive) {
        finalDetection = det.copyWith(
          distance: distanceFor(det.classId, medianPixelHeight, frameH),
        );
      }
      // في الـ live الفريم الجاي هيحسب بالـ focal الجديد لوحده
    });

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('focalPx = ${newFocal.toStringAsFixed(1)}'),
      ),
    );
  }

  void measureAgain() {
    setState(() {
      measurementDone = false;
      currentDetections = [];
      finalDetection = null;
      capturedJpg = null;
      samples.clear();
      truncatedCount = 0;
    });
  }

  @override
  void dispose() {
    if (cameraReady && controller.value.isStreamingImages) {
      controller.stopImageStream();
    }
    if (cameraReady) controller.dispose();
    session?.close();
    super.dispose();
  }

  // ==========================================================
  // UI
  // ==========================================================

  String get statusText {
    if (!modelLoaded) return 'Loading model...';
    if (isLive) {
      return currentDetections.isEmpty
          ? 'Point the camera at a car'
          : 'Live distance';
    }
    return samples.isEmpty
        ? 'Point the camera at the object'
        : 'Measuring ${samples.length}/$samplesNeeded';
  }

  Widget buildLivePanel() {
    final hasCar = currentDetections.isNotEmpty;
    final det = hasCar ? currentDetections.first : null;

    return Card(
      elevation: 8,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text('CAR', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const Text('Distance', style: TextStyle(fontSize: 14)),
            Text(
              hasCar
                  ? '${(det!.distance / 100).toStringAsFixed(2)} m'
                  : '--',
              style: const TextStyle(fontSize: 32, fontWeight: FontWeight.bold),
            ),
            if (hasCar)
              Text(
                '${det!.distance.toStringAsFixed(0)} cm',
                style: const TextStyle(fontSize: 14, color: Colors.grey),
              ),
            if (hasCar && det!.truncated)
              const Padding(
                padding: EdgeInsets.only(top: 6),
                child: Text(
                  'العربية مش ظاهرة كاملة، ابعد شوية عشان النتيجة تبقى أدق',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.orange, fontSize: 12),
                ),
              ),
            const SizedBox(height: 10),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton(
                onPressed: hasCar ? showCalibrationDialog : null,
                child: const Text('Calibrate'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (!cameraReady || !controller.value.isInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    final previewAspect = 1 / controller.value.aspectRatio;

    return Scaffold(
      resizeToAvoidBottomInset: false,
      appBar: AppBar(title: const Text('Distance')),
      body: Stack(
        fit: StackFit.expand,
        children: [
          Center(
            child: AspectRatio(
              aspectRatio: previewAspect,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  CameraPreview(controller),
                  CustomPaint(
                    painter: DetectionPainter(
                      detections: currentDetections,
                      imageWidth: frameW,
                      imageHeight: frameH,
                    ),
                  ),
                ],
              ),
            ),
          ),

          if (!measurementDone)
            Positioned(
              top: 12,
              left: 0,
              right: 0,
              child: Center(
                child: Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    statusText,
                    style: const TextStyle(color: Colors.white),
                  ),
                ),
              ),
            ),

          // Car: لوحة live دايماً ظاهرة
          if (isLive && modelLoaded)
            Positioned(
              left: 20,
              right: 20,
              bottom: 25,
              child: buildLivePanel(),
            ),

          // Person: كارت النتيجة بعد القياس
          if (!isLive &&
              measurementDone &&
              capturedJpg != null &&
              finalDetection != null)
            Positioned(
              left: 20,
              right: 20,
              bottom: 25,
              child: ResultCard(
                jpg: capturedJpg!,
                detection: finalDetection!,
                onMeasureAgain: measureAgain,
                onCalibrate: showCalibrationDialog,
              ),
            ),
        ],
      ),
    );
  }
}

// ============================================================
// Result card (Person)
// ============================================================

class ResultCard extends StatelessWidget {
  final Uint8List jpg;
  final Detection detection;
  final VoidCallback onMeasureAgain;
  final VoidCallback onCalibrate;

  const ResultCard({
    super.key,
    required this.jpg,
    required this.detection,
    required this.onMeasureAgain,
    required this.onCalibrate,
  });

  @override
  Widget build(BuildContext context) {
    final label = detection.classId == personClass ? 'PERSON' : 'CAR';
    final cm = detection.distance;
    final meters = (cm / 100).toStringAsFixed(2);

    return Card(
      elevation: 8,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(15),
              child: Image.memory(
                jpg,
                height: 150,
                width: double.infinity,
                fit: BoxFit.contain,
              ),
            ),
            const SizedBox(height: 10),
            Text(
              label,
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const Text('Distance', style: TextStyle(fontSize: 14)),
            Text(
              '$meters m',
              style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
            ),
            Text(
              '${cm.toStringAsFixed(0)} cm',
              style: const TextStyle(fontSize: 14, color: Colors.grey),
            ),
            if (detection.truncated)
              const Padding(
                padding: EdgeInsets.only(top: 6),
                child: Text(
                  'الجسم مش ظاهر كامل في الصورة، ابعد شوية عشان النتيجة تبقى أدق',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.orange, fontSize: 12),
                ),
              ),
            const SizedBox(height: 10),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: onMeasureAgain,
                    child: const Text('Measure Again'),
                  ),
                ),
                const SizedBox(width: 10),
                OutlinedButton(
                  onPressed: onCalibrate,
                  child: const Text('Calibrate'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

// ============================================================
// Calibration dialog
// ============================================================

class CalibrationDialog extends StatefulWidget {
  final double defaultHeight;

  const CalibrationDialog({super.key, required this.defaultHeight});

  @override
  State<CalibrationDialog> createState() => _CalibrationDialogState();
}

class _CalibrationDialogState extends State<CalibrationDialog> {
  late final TextEditingController distCtrl = TextEditingController();
  late final TextEditingController heightCtrl = TextEditingController(
    text: widget.defaultHeight.toStringAsFixed(0),
  );

  @override
  void dispose() {
    distCtrl.dispose();
    heightCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Calibrate'),
      content: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              'قِس المسافة الحقيقية بشريط قياس واكتب الطول الحقيقي للجسم.',
            ),
            TextField(
              controller: distCtrl,
              keyboardType:
                  const TextInputType.numberWithOptions(decimal: true),
              decoration:
                  const InputDecoration(labelText: 'Real distance (cm)'),
            ),
            TextField(
              controller: heightCtrl,
              keyboardType:
                  const TextInputType.numberWithOptions(decimal: true),
              decoration:
                  const InputDecoration(labelText: 'Real object height (cm)'),
            ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancel'),
        ),
        TextButton(
          onPressed: () {
            final d = double.tryParse(distCtrl.text);
            final h = double.tryParse(heightCtrl.text);
            if (d == null || h == null || d <= 0 || h <= 0) return;
            Navigator.pop(context, [d, h]);
          },
          child: const Text('Save'),
        ),
      ],
    );
  }
}