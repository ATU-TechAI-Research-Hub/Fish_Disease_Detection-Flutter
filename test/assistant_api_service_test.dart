import 'dart:convert';

import 'package:aquaculture/services/assistant_api_service.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

void main() {
  test('assistant service parses NDJSON streaming events', () async {
    late http.Request captured;
    final client = MockClient((request) async {
      captured = request;
      return http.Response(
        '${jsonEncode({
              'type': 'start',
              'model': 'qwen',
              'sources': [],
            })}\n'
        '${jsonEncode({
              'type': 'token',
              'text': 'Local response',
            })}\n'
        '${jsonEncode({
              'type': 'done',
              'model': 'qwen',
              'message': null,
            })}\n',
        200,
        headers: {'content-type': 'application/x-ndjson'},
      );
    });
    final service = AssistantApiService(
      baseUrl: 'http://local.test',
      client: client,
    );

    final events = await service
        .streamChat(
          sessionId: 'session_123456',
          question: 'What is fin rot?',
          model: 'qwen',
        )
        .toList();

    expect(events.map((event) => event['type']), ['start', 'token', 'done']);
    expect(events[1]['text'], 'Local response');
    expect(captured.url.path, '/assistant/chat/stream');
    final requestBody = jsonDecode(captured.body) as Map<String, dynamic>;
    expect(requestBody['session_id'], 'session_123456');
    expect(requestBody['question'], 'What is fin rot?');
  });
}
