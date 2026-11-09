import 'package:aquaculture/models/shrimp_analysis_model.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('parses real shrimp analysis values without defaults or fabrication',
      () {
    final result = ShrimpAnalysisModel.fromJson({
      'analysis_id': 'abc123',
      'total_shrimp': 1,
      'average_length_px': 42.5,
      'minimum_length_px': 42.5,
      'maximum_length_px': 42.5,
      'average_confidence': 0.962,
      'processing_time': 2.4,
      'calibration_available': false,
      'scale_mm_per_px': null,
      'sam_water_scores': [0.91],
      'detections': [
        {
          'id': 1,
          'class_name': 'shrimp',
          'confidence': 0.962,
          'length_px': 42.5,
          'length_mm': null,
          'bounding_box': {'x1': 10, 'y1': 20, 'x2': 40, 'y2': 50},
          'mask_url': 'http://localhost/mask.png',
          'measurement_note': null,
        },
      ],
      'images': {
        'original_url': 'http://localhost/original.png',
        'water_mask_url': 'http://localhost/water.png',
        'water_gated_url': 'http://localhost/gated.png',
        'annotated_url': 'http://localhost/annotated.png',
      },
      'warnings': <String>[],
      'methodology_note': 'Approximate proxy.',
    });

    expect(result.totalShrimp, 1);
    expect(result.detections, hasLength(1));
    expect(result.detections.single.confidence, closeTo(0.962, 0.0001));
    expect(result.detections.single.lengthPx, 42.5);
    expect(result.detections.single.lengthMm, isNull);
    expect(result.calibrationAvailable, isFalse);
  });
}
