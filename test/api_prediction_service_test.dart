import 'dart:io';

import 'package:aquaculture/services/api_prediction_service.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final File sampleImage = File(
    'Freshwater_Fish_Disease_Aquaculture_in_south_asia/Train/'
    'Bacterial Red disease/Bacterial Red disease (106).jpg',
  );
  final bool runLiveApiTest = Platform.environment['RUN_LIVE_API_TEST'] == '1';

  test(
    'ApiPredictionService uploads an image to the real backend',
    () async {
      expect(sampleImage.existsSync(), isTrue);

      final ApiPredictionService service = ApiPredictionService(
        baseUrl: 'http://127.0.0.1:8000',
      );

      final prediction = await service.predictDiseaseFromImage(sampleImage.path);

      expect(prediction.disease.id, greaterThan(0));
      expect(prediction.disease.name, isNotEmpty);
      expect(prediction.confidence, greaterThan(0));
      expect(prediction.source, contains('onnxruntime'));
    },
    timeout: const Timeout(Duration(seconds: 30)),
    skip: runLiveApiTest && sampleImage.existsSync()
        ? false
        : 'Set RUN_LIVE_API_TEST=1 and ensure the sample image exists locally.',
  );
}
