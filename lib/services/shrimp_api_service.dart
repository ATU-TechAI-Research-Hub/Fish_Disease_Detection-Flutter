import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

import '../models/shrimp_analysis_model.dart';
import 'api_prediction_service.dart';

class ShrimpApiException implements Exception {
  const ShrimpApiException(this.message, {this.code, this.statusCode});

  final String message;
  final String? code;
  final int? statusCode;

  @override
  String toString() => message;
}

class ShrimpServiceStatus {
  const ShrimpServiceStatus({
    required this.ready,
    required this.samState,
    required this.yoloState,
    required this.message,
  });

  final bool ready;
  final String samState;
  final String yoloState;
  final String? message;

  factory ShrimpServiceStatus.fromJson(Map<String, dynamic> json) {
    final sam = json['sam3'] as Map<String, dynamic>? ?? const {};
    final yolo = json['yolo11'] as Map<String, dynamic>? ?? const {};
    return ShrimpServiceStatus(
      ready: json['ready'] as bool? ?? false,
      samState: sam['state']?.toString() ?? 'unknown',
      yoloState: yolo['state']?.toString() ?? 'unknown',
      message: json['message']?.toString(),
    );
  }
}

/// Multipart client for the independent `/api/shrimp/analyze` endpoint.
class ShrimpApiService {
  ShrimpApiService({String? baseUrl})
      : baseUrl = baseUrl ?? ApiPredictionService.defaultBaseUrl();

  final String baseUrl;

  String resolveArtifactUrl(String path) {
    final uri = Uri.tryParse(path);
    if (uri != null && uri.hasScheme) return path;
    return Uri.parse(baseUrl).resolve(path).toString();
  }

  Future<ShrimpServiceStatus> status() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/api/shrimp/status'))
          .timeout(const Duration(seconds: 5));
      final payload = json.decode(response.body) as Map<String, dynamic>;
      if (response.statusCode != 200) {
        throw ShrimpApiException(
          'Could not check shrimp model status.',
          statusCode: response.statusCode,
        );
      }
      return ShrimpServiceStatus.fromJson(payload);
    } on ShrimpApiException {
      rethrow;
    } on SocketException {
      throw const ShrimpApiException(
        'Cannot reach the AquaScan backend.',
        code: 'server_unavailable',
      );
    } on TimeoutException {
      throw const ShrimpApiException(
        'Timed out while checking shrimp model status.',
        code: 'request_timeout',
      );
    } catch (_) {
      throw const ShrimpApiException(
        'The backend returned an invalid shrimp model status.',
        code: 'invalid_response',
      );
    }
  }

  Future<ShrimpAnalysisModel> analyze(
    String imagePath, {
    double? calibrationReferenceMm,
    double? calibrationReferencePx,
  }) async {
    final file = File(imagePath);
    if (!await file.exists()) {
      throw const ShrimpApiException('The selected image could not be found.');
    }

    final request = http.MultipartRequest(
      'POST',
      Uri.parse('$baseUrl/api/shrimp/analyze'),
    );
    if (calibrationReferenceMm != null) {
      request.fields['calibration_reference_mm'] =
          calibrationReferenceMm.toString();
    }
    if (calibrationReferencePx != null) {
      request.fields['calibration_reference_px'] =
          calibrationReferencePx.toString();
    }

    final extension = imagePath.split('.').last.toLowerCase();
    final mediaType = switch (extension) {
      'jpg' || 'jpeg' => 'image/jpeg',
      'png' => 'image/png',
      'webp' => 'image/webp',
      'gif' => 'image/gif',
      'bmp' => 'image/bmp',
      _ => 'application/octet-stream',
    };
    request.files.add(
      await http.MultipartFile.fromPath(
        'file',
        imagePath,
        contentType: MediaType.parse(mediaType),
      ),
    );

    final http.StreamedResponse streamed;
    try {
      streamed = await request.send().timeout(const Duration(minutes: 4));
    } on SocketException {
      throw ShrimpApiException(
        'Cannot reach the AquaScan backend at $baseUrl. '
        'Start the server and confirm this device is on the same network.',
        code: 'server_unavailable',
      );
    } on TimeoutException {
      throw const ShrimpApiException(
        'Shrimp analysis timed out. SAM 3 can be slow without a compatible GPU.',
        code: 'request_timeout',
      );
    }

    final body = await streamed.stream.bytesToString();
    Map<String, dynamic>? payload;
    try {
      payload = json.decode(body) as Map<String, dynamic>;
    } catch (_) {
      payload = null;
    }

    if (streamed.statusCode != 200) {
      final detail = payload?['detail'];
      String message = body.isEmpty ? 'Shrimp analysis failed.' : body;
      String? code;
      if (detail is Map<String, dynamic>) {
        message = detail['message']?.toString() ?? message;
        code = detail['code']?.toString();
      } else if (detail != null) {
        message = detail.toString();
      }
      throw ShrimpApiException(
        message,
        code: code,
        statusCode: streamed.statusCode,
      );
    }
    if (payload == null) {
      throw const ShrimpApiException(
        'The backend returned an unreadable shrimp analysis response.',
        code: 'invalid_response',
      );
    }

    final images = Map<String, dynamic>.from(
      payload['images'] as Map<String, dynamic>,
    );
    for (final key in images.keys.toList()) {
      images[key] = resolveArtifactUrl(images[key].toString());
    }
    payload['images'] = images;

    final detections =
        (payload['detections'] as List<dynamic>? ?? const []).map((item) {
      final detection = Map<String, dynamic>.from(
        item as Map<String, dynamic>,
      );
      detection['mask_url'] =
          resolveArtifactUrl(detection['mask_url'].toString());
      return detection;
    }).toList(growable: false);
    payload['detections'] = detections;

    return ShrimpAnalysisModel.fromJson(payload);
  }
}
