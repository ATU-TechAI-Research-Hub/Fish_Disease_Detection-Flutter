import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import '../models/assistant_message.dart';
import '../models/prediction_result_model.dart';
import 'api_prediction_service.dart';

class AssistantApiService {
  AssistantApiService({
    String? baseUrl,
    http.Client? client,
  })  : baseUrl = baseUrl ?? ApiPredictionService.defaultBaseUrl(),
        _client = client ?? http.Client(),
        _ownsClient = client == null;

  final String baseUrl;
  http.Client _client;
  final bool _ownsClient;

  Future<Map<String, dynamic>> health() async {
    final response = await _client
        .get(Uri.parse('$baseUrl/assistant/health'))
        .timeout(const Duration(seconds: 5));
    return _decodeResponse(response);
  }

  Future<List<AssistantModelOption>> models() async {
    final response = await _client
        .get(Uri.parse('$baseUrl/assistant/models'))
        .timeout(const Duration(seconds: 8));
    final body = _decodeResponse(response);
    final raw = body['models'] as List<dynamic>? ?? const [];
    return raw
        .whereType<Map<String, dynamic>>()
        .map(AssistantModelOption.fromJson)
        .toList(growable: false);
  }

  Future<List<AssistantMessage>> history(String sessionId) async {
    final response = await _client
        .get(Uri.parse('$baseUrl/assistant/history/$sessionId'))
        .timeout(const Duration(seconds: 8));
    final body = _decodeResponse(response);
    final raw = body['messages'] as List<dynamic>? ?? const [];
    return raw
        .whereType<Map<String, dynamic>>()
        .map(AssistantMessage.fromJson)
        .toList(growable: false);
  }

  Future<void> clearHistory(String sessionId) async {
    final response = await _client
        .delete(Uri.parse('$baseUrl/assistant/history/$sessionId'))
        .timeout(const Duration(seconds: 8));
    _decodeResponse(response);
  }

  Future<void> deleteSession(String sessionId) async {
    final response = await _client
        .delete(Uri.parse('$baseUrl/assistant/session/$sessionId'))
        .timeout(const Duration(seconds: 8));
    _decodeResponse(response);
  }

  Future<void> publishPrediction(
    String sessionId,
    PredictionResultModel prediction,
  ) async {
    final response = await _client
        .post(
          Uri.parse('$baseUrl/assistant/prediction-context'),
          headers: const {'Content-Type': 'application/json'},
          body: jsonEncode({
            'session_id': sessionId,
            'prediction': prediction.toJson(),
          }),
        )
        .timeout(const Duration(seconds: 8));
    _decodeResponse(response);
  }

  Stream<Map<String, dynamic>> streamChat({
    required String sessionId,
    required String question,
    required String model,
    bool regenerate = false,
  }) async* {
    final request = http.Request(
      'POST',
      Uri.parse('$baseUrl/assistant/chat/stream'),
    )
      ..headers['Content-Type'] = 'application/json'
      ..body = jsonEncode({
        'session_id': sessionId,
        'question': question,
        'model': model,
        'regenerate': regenerate,
      });

    final http.StreamedResponse response;
    try {
      response =
          await _client.send(request).timeout(const Duration(seconds: 30));
    } on SocketException {
      throw Exception(
        'Cannot reach the local AquaScan assistant at $baseUrl. '
        'Start the backend and try again.',
      );
    } on TimeoutException {
      throw Exception(
        'The local assistant did not start responding in time. '
        'A GGUF model can take a while to load on first use.',
      );
    } on http.ClientException {
      throw Exception(
        'The connection to the local AquaScan assistant was interrupted.',
      );
    }

    if (response.statusCode != 200) {
      final body = await response.stream.bytesToString();
      throw Exception(_extractDetail(body, response.statusCode));
    }

    await for (final line in response.stream
        .transform(utf8.decoder)
        .transform(const LineSplitter())) {
      if (line.trim().isEmpty) continue;
      final decoded = jsonDecode(line);
      if (decoded is Map<String, dynamic>) {
        yield decoded;
      }
    }
  }

  Map<String, dynamic> _decodeResponse(http.Response response) {
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw Exception(_extractDetail(response.body, response.statusCode));
    }
    final decoded = jsonDecode(response.body);
    if (decoded is! Map<String, dynamic>) {
      throw const FormatException('Unexpected backend response.');
    }
    return decoded;
  }

  String _extractDetail(String body, int statusCode) {
    try {
      final decoded = jsonDecode(body);
      if (decoded is Map<String, dynamic> && decoded['detail'] != null) {
        return decoded['detail'].toString();
      }
    } catch (_) {
      // Use the body below when FastAPI did not return structured JSON.
    }
    return body.trim().isEmpty
        ? 'Assistant request failed ($statusCode).'
        : body.trim();
  }

  void cancelActiveRequests() {
    if (!_ownsClient) return;
    _client.close();
    _client = http.Client();
  }

  void close() => _client.close();
}
