import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import '../models/prediction_result_model.dart';

class ApiPredictionService {
  ApiPredictionService({String? baseUrl}) : baseUrl = baseUrl ?? _defaultBaseUrl();

  final String baseUrl;

  static String _defaultBaseUrl() {
    if (Platform.isAndroid) {
      return 'http://10.0.2.2:8000';
    }
    return 'http://127.0.0.1:8000';
  }

  Future<PredictionResultModel> predictDiseaseFromImage(String imagePath) async {
    final Uri uri = Uri.parse('$baseUrl/predict');
    final http.MultipartRequest request = http.MultipartRequest('POST', uri);

    request.files.add(await http.MultipartFile.fromPath('file', imagePath));

    final http.StreamedResponse streamedResponse = await request.send().timeout(
      const Duration(seconds: 12),
    );
    final String responseBody = await streamedResponse.stream.bytesToString();

    if (streamedResponse.statusCode != 200) {
      throw Exception(
        'Backend prediction failed (${streamedResponse.statusCode}): '
        '$responseBody',
      );
    }

    final Map<String, dynamic> parsed = json.decode(responseBody) as Map<String, dynamic>;
    return PredictionResultModel.fromJson(parsed);
  }
}
