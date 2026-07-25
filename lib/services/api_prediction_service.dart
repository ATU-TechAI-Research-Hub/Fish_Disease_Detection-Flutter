import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

import '../models/prediction_result_model.dart';

/// HTTP client for the AquaScan FastAPI backend.
///
/// Default base URL is resolved automatically:
///   - Android emulator → `http://10.0.2.2:8000`
///   - iOS simulator / desktop / web → `http://127.0.0.1:8000`
///   - Physical device → set [lanIp] below to your computer's LAN IP.
///
/// For physical Android/iOS devices, set [lanIp] to your PC/Mac IPv4 address
/// (run `ipconfig` on Windows or `ipconfig getifaddr en0` on macOS).
class ApiPredictionService {
  ApiPredictionService({String? baseUrl})
      : baseUrl = baseUrl ?? defaultBaseUrl();

  final String baseUrl;

  /// Optional LAN IP override. Leave blank for emulators / desktop / web.
  /// Example: `'http://192.168.1.74:8000'`
  ///
  /// Set to this PC's Wi-Fi IPv4 so a physical phone on the same network can
  /// reach the backend. Phone and PC must share the same Wi-Fi.
  static const String lanIp = 'http://192.168.1.74:8000';

  /// Launch-time override, for example:
  /// `--dart-define=AQUASCAN_BACKEND_URL=http://127.0.0.1:8000`
  static const String configuredBaseUrl = String.fromEnvironment(
    'AQUASCAN_BACKEND_URL',
    defaultValue: '',
  );

  /// Default port used by the FastAPI backend (see `run_backend.bat`).
  static const int defaultPort = 8000;

  static String defaultBaseUrl() {
    if (configuredBaseUrl.isNotEmpty) return configuredBaseUrl;
    if (lanIp.isNotEmpty) return lanIp;
    if (kIsWeb) return 'http://127.0.0.1:$defaultPort';
    if (Platform.isAndroid) return 'http://10.0.2.2:$defaultPort';
    return 'http://127.0.0.1:$defaultPort';
  }

  /// Send an image to `/predict` and parse the response.
  ///
  /// Throws an [Exception] with a friendly message on connection / parsing /
  /// HTTP errors so the UI can render it directly.
  Future<PredictionResultModel> predictDiseaseFromImage(
    String imagePath, {
    String? assistantSessionId,
  }) async {
    final file = File(imagePath);
    if (!await file.exists()) {
      throw Exception('Image file not found at: $imagePath');
    }

    final Uri uri = Uri.parse('$baseUrl/predict');
    final request = http.MultipartRequest('POST', uri);
    if (assistantSessionId != null && assistantSessionId.isNotEmpty) {
      request.fields['assistant_session_id'] = assistantSessionId;
    }

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
      streamedResponse =
          await request.send().timeout(const Duration(seconds: 30));
    } on SocketException {
      throw Exception(
        'Cannot reach the AquaScan backend at $baseUrl. '
        'Make sure the server is running (run_backend.bat).',
      );
    } on TimeoutException {
      throw Exception(
        'The request timed out. The server may be busy or unreachable.',
      );
    } on HttpException catch (e) {
      throw Exception('HTTP error: $e');
    }

    final body = await streamedResponse.stream.bytesToString();

    if (streamedResponse.statusCode != 200) {
      // Try to surface the structured `detail` field from FastAPI.
      String detail = body;
      try {
        final parsed = json.decode(body);
        if (parsed is Map<String, dynamic> && parsed['detail'] != null) {
          detail = parsed['detail'].toString();
        }
      } catch (_) {
        // Body wasn't JSON; fall back to the raw text.
      }
      throw Exception(
        'Backend prediction failed (${streamedResponse.statusCode}): $detail',
      );
    }

    final Map<String, dynamic> parsed =
        json.decode(body) as Map<String, dynamic>;
    return PredictionResultModel.fromJson(parsed);
  }
}
