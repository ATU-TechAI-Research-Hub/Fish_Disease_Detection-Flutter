import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

import '../models/prediction_result_model.dart';

class ApiPredictionService {
  ApiPredictionService({String? baseUrl})
      : baseUrl = baseUrl ?? _defaultBaseUrl();

  final String baseUrl;

  // Replace with your PC's Wi-Fi IP for physical device testing.
  // Run `ipconfig` and use the IPv4 address.
  static const String _lanIp = 'http://10.201.1.55:8001';

  static String _defaultBaseUrl() {
    if (Platform.isAndroid) {
      if (_lanIp.isNotEmpty) return _lanIp;
      return 'http://10.0.2.2:8001';
    }
    return 'http://127.0.0.1:8001';
  }

  Future<PredictionResultModel> predictDiseaseFromImage(
      String imagePath) async {
    final file = File(imagePath);
    if (!await file.exists()) {
      throw Exception('Image file not found at: $imagePath');
    }

    final Uri uri = Uri.parse('$baseUrl/predict');
    final http.MultipartRequest request = http.MultipartRequest('POST', uri);

    final ext = imagePath.split('.').last.toLowerCase();
    final mimeType = switch (ext) {
      'jpg' || 'jpeg' => 'image/jpeg',
      'png' => 'image/png',
      'webp' => 'image/webp',
      'gif' => 'image/gif',
      'bmp' => 'image/bmp',
      _ => 'image/jpeg',
    };
    request.files.add(await http.MultipartFile.fromPath(
      'file',
      imagePath,
      contentType: MediaType.parse(mimeType),
    ));

    final http.StreamedResponse streamedResponse;
    try {
      streamedResponse = await request.send().timeout(
            const Duration(seconds: 30),
          );
    } on SocketException {
      throw Exception(
        'Cannot connect to the server. '
        'Make sure the backend is running at $baseUrl.',
      );
    } on TimeoutException {
      throw Exception(
        'Connection timed out. The server may be busy or unreachable.',
      );
    } on HttpException catch (e) {
      throw Exception('HTTP error: $e');
    }

    final String responseBody =
        await streamedResponse.stream.bytesToString();

    if (streamedResponse.statusCode != 200) {
      throw Exception(
        'Backend prediction failed (${streamedResponse.statusCode}): '
        '$responseBody',
      );
    }

    final Map<String, dynamic> parsed =
        json.decode(responseBody) as Map<String, dynamic>;
    return PredictionResultModel.fromJson(parsed);
  }
}
