import 'dart:io';
import 'dart:ui' as ui;

/// Result of a local pre-upload image quality check.
class ImageQualityReport {
  const ImageQualityReport({
    required this.width,
    required this.height,
    required this.fileBytes,
    required this.warnings,
  });

  final int width;
  final int height;
  final int fileBytes;
  final List<String> warnings;

  bool get isAcceptable => warnings.isEmpty;

  static const ImageQualityReport unknown = ImageQualityReport(
    width: 0,
    height: 0,
    fileBytes: 0,
    warnings: [],
  );
}

/// Validates an image before uploading it for prediction.
///
/// The CNN was trained on 150×150 crops, so anything below ~200 px on the
/// short side loses disease texture detail after server-side resizing.
/// These are soft warnings — the user can still submit the photo.
Future<ImageQualityReport> checkImageQuality(String path) async {
  final file = File(path);
  final bytes = await file.readAsBytes();
  final warnings = <String>[];

  int width = 0;
  int height = 0;
  try {
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    width = frame.image.width;
    height = frame.image.height;
    frame.image.dispose();
    codec.dispose();
  } catch (_) {
    warnings.add('The image could not be decoded — it may be corrupt.');
    return ImageQualityReport(
      width: 0,
      height: 0,
      fileBytes: bytes.length,
      warnings: warnings,
    );
  }

  final shortSide = width < height ? width : height;
  if (shortSide < 200) {
    warnings.add(
      'Low resolution (${width}x$height). Disease details may be lost — '
      'try a closer or higher-quality photo.',
    );
  }

  final longSide = width > height ? width : height;
  if (shortSide > 0 && longSide / shortSide > 3) {
    warnings.add(
      'Unusual aspect ratio — the fish may be cropped or distorted '
      'during analysis.',
    );
  }

  if (bytes.length < 15 * 1024) {
    warnings.add(
      'Very small file size — the image may be over-compressed or blurry.',
    );
  }

  return ImageQualityReport(
    width: width,
    height: height,
    fileBytes: bytes.length,
    warnings: warnings,
  );
}
