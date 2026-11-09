class ShrimpBoundingBox {
  const ShrimpBoundingBox({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
  });

  final double x1;
  final double y1;
  final double x2;
  final double y2;

  factory ShrimpBoundingBox.fromJson(Map<String, dynamic> json) {
    return ShrimpBoundingBox(
      x1: (json['x1'] as num).toDouble(),
      y1: (json['y1'] as num).toDouble(),
      x2: (json['x2'] as num).toDouble(),
      y2: (json['y2'] as num).toDouble(),
    );
  }
}

class ShrimpDetectionModel {
  const ShrimpDetectionModel({
    required this.id,
    required this.className,
    required this.confidence,
    required this.lengthPx,
    required this.lengthMm,
    required this.boundingBox,
    required this.maskUrl,
    required this.measurementNote,
  });

  final int id;
  final String className;
  final double confidence;
  final double? lengthPx;
  final double? lengthMm;
  final ShrimpBoundingBox boundingBox;
  final String maskUrl;
  final String? measurementNote;

  factory ShrimpDetectionModel.fromJson(Map<String, dynamic> json) {
    return ShrimpDetectionModel(
      id: json['id'] as int,
      className: json['class_name'] as String? ?? 'shrimp',
      confidence: (json['confidence'] as num).toDouble(),
      lengthPx: (json['length_px'] as num?)?.toDouble(),
      lengthMm: (json['length_mm'] as num?)?.toDouble(),
      boundingBox: ShrimpBoundingBox.fromJson(
        json['bounding_box'] as Map<String, dynamic>,
      ),
      maskUrl: json['mask_url'] as String,
      measurementNote: json['measurement_note'] as String?,
    );
  }
}

class ShrimpResultImages {
  const ShrimpResultImages({
    required this.originalUrl,
    required this.waterMaskUrl,
    required this.waterGatedUrl,
    required this.annotatedUrl,
  });

  final String originalUrl;
  final String waterMaskUrl;
  final String waterGatedUrl;
  final String annotatedUrl;

  factory ShrimpResultImages.fromJson(Map<String, dynamic> json) {
    return ShrimpResultImages(
      originalUrl: json['original_url'] as String,
      waterMaskUrl: json['water_mask_url'] as String,
      waterGatedUrl: json['water_gated_url'] as String,
      annotatedUrl: json['annotated_url'] as String,
    );
  }
}

class ShrimpAnalysisModel {
  const ShrimpAnalysisModel({
    required this.analysisId,
    required this.totalShrimp,
    required this.averageLengthPx,
    required this.minimumLengthPx,
    required this.maximumLengthPx,
    required this.averageConfidence,
    required this.processingTime,
    required this.calibrationAvailable,
    required this.scaleMmPerPx,
    required this.samWaterScores,
    required this.detections,
    required this.images,
    required this.warnings,
    required this.methodologyNote,
  });

  final String analysisId;
  final int totalShrimp;
  final double? averageLengthPx;
  final double? minimumLengthPx;
  final double? maximumLengthPx;
  final double? averageConfidence;
  final double processingTime;
  final bool calibrationAvailable;
  final double? scaleMmPerPx;
  final List<double> samWaterScores;
  final List<ShrimpDetectionModel> detections;
  final ShrimpResultImages images;
  final List<String> warnings;
  final String methodologyNote;

  factory ShrimpAnalysisModel.fromJson(Map<String, dynamic> json) {
    return ShrimpAnalysisModel(
      analysisId: json['analysis_id'] as String,
      totalShrimp: json['total_shrimp'] as int,
      averageLengthPx: (json['average_length_px'] as num?)?.toDouble(),
      minimumLengthPx: (json['minimum_length_px'] as num?)?.toDouble(),
      maximumLengthPx: (json['maximum_length_px'] as num?)?.toDouble(),
      averageConfidence: (json['average_confidence'] as num?)?.toDouble(),
      processingTime: (json['processing_time'] as num).toDouble(),
      calibrationAvailable: json['calibration_available'] as bool? ?? false,
      scaleMmPerPx: (json['scale_mm_per_px'] as num?)?.toDouble(),
      samWaterScores: (json['sam_water_scores'] as List<dynamic>? ?? const [])
          .map((value) => (value as num).toDouble())
          .toList(growable: false),
      detections: (json['detections'] as List<dynamic>? ?? const [])
          .map(
            (value) => ShrimpDetectionModel.fromJson(
              value as Map<String, dynamic>,
            ),
          )
          .toList(growable: false),
      images: ShrimpResultImages.fromJson(
        json['images'] as Map<String, dynamic>,
      ),
      warnings: (json['warnings'] as List<dynamic>? ?? const [])
          .map((value) => value.toString())
          .toList(growable: false),
      methodologyNote: json['methodology_note'] as String? ??
          'Ellipse length is an approximate geometric proxy.',
    );
  }
}
