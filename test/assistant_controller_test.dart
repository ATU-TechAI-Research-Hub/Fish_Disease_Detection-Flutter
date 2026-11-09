import 'dart:async';

import 'package:aquaculture/models/assistant_message.dart';
import 'package:aquaculture/models/prediction_result_model.dart';
import 'package:aquaculture/services/assistant_api_service.dart';
import 'package:aquaculture/services/assistant_controller.dart';
import 'package:aquaculture/widgets/assistant_panel.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

class _FakeAssistantApiService extends AssistantApiService {
  _FakeAssistantApiService() : super(baseUrl: 'http://local.test');

  bool cleared = false;
  bool predictionPublished = false;
  bool regenerated = false;

  @override
  Future<List<AssistantMessage>> history(String sessionId) async => [];

  @override
  Future<List<AssistantModelOption>> models() async => const [
        AssistantModelOption(
          key: 'qwen',
          displayName: 'Qwen3 8B',
          available: true,
          active: false,
        ),
      ];

  @override
  Stream<Map<String, dynamic>> streamChat({
    required String sessionId,
    required String question,
    required String model,
    bool regenerate = false,
  }) async* {
    regenerated = regenerate;
    yield {
      'type': 'start',
      'model': model,
      'sources': [
        {
          'title': 'Water quality',
          'source_name': 'water_quality.md',
          'score': 0.9,
        }
      ],
    };
    yield {'type': 'token', 'text': 'Increase aeration '};
    yield {'type': 'token', 'text': 'and verify oxygen readings. [1]'};
    yield {
      'type': 'done',
      'model': model,
      'message': {
        'id': 'saved-assistant-message',
        'session_id': sessionId,
        'role': 'assistant',
        'content': 'Increase aeration and verify oxygen readings. [1]',
        'created_at': '2026-01-01T00:00:00Z',
        'model': model,
        'sources': [
          {
            'title': 'Water quality',
            'source_name': 'water_quality.md',
            'score': 0.9,
          }
        ],
      },
    };
  }

  @override
  Future<void> clearHistory(String sessionId) async {
    cleared = true;
  }

  @override
  Future<void> deleteSession(String sessionId) async {
    cleared = true;
  }

  @override
  Future<void> publishPrediction(
    String sessionId,
    PredictionResultModel prediction,
  ) async {
    predictionPublished = true;
  }
}

class _SlowAssistantApiService extends _FakeAssistantApiService {
  final started = Completer<void>();
  final release = Completer<void>();

  @override
  Stream<Map<String, dynamic>> streamChat({
    required String sessionId,
    required String question,
    required String model,
    bool regenerate = false,
  }) async* {
    yield {'type': 'start', 'model': model, 'sources': <dynamic>[]};
    started.complete();
    await release.future;
    yield {'type': 'token', 'text': 'stale response'};
    yield {
      'type': 'done',
      'model': model,
      'message': {
        'id': 'stale',
        'session_id': sessionId,
        'role': 'assistant',
        'content': 'stale response',
        'created_at': '2026-01-01T00:00:00Z',
        'model': model,
        'sources': <dynamic>[],
      },
    };
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('controller streams, regenerates, and clears local conversation',
      () async {
    SharedPreferences.setMockInitialValues({});
    final api = _FakeAssistantApiService();
    final controller = AssistantController.forTest(api);
    await controller.init();
    await controller.refresh();

    await controller.sendMessage('How do I improve oxygen?');
    expect(controller.messages, hasLength(2));
    expect(controller.messages.first.role, AssistantRole.user);
    expect(controller.messages.last.content, contains('Increase aeration'));
    expect(controller.messages.last.sources.single.title, 'Water quality');
    expect(controller.isStreaming, isFalse);

    await controller.regenerateLast();
    expect(api.regenerated, isTrue);
    expect(controller.messages, hasLength(2));

    await controller.clearConversation();
    expect(api.cleared, isTrue);
    expect(controller.messages, isEmpty);

    final previousSession = controller.sessionId;
    await controller.resetSession();
    expect(controller.sessionId, isNot(previousSession));
  });

  testWidgets('assistant panel sends and renders a streamed Markdown response',
      (tester) async {
    tester.view.physicalSize = const Size(400, 900);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.reset);
    SharedPreferences.setMockInitialValues({});
    final controller = AssistantController.forTest(_FakeAssistantApiService());
    await controller.init();
    await controller.refresh();

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SizedBox(
            width: 400,
            height: 700,
            child: ListenableBuilder(
              listenable: controller,
              builder: (context, _) => AssistantPanel(controller: controller),
            ),
          ),
        ),
      ),
    );
    expect(find.text('Aquaculture Assistant'), findsOneWidget);
    expect(find.text('Ask about fish health'), findsOneWidget);

    await tester.enterText(
      find.byType(TextField),
      'How do I improve dissolved oxygen?',
    );
    await tester.tap(find.byTooltip('Send message'));
    await tester.pumpAndSettle();

    expect(find.textContaining('Increase aeration'), findsOneWidget);
    expect(find.textContaining('[1] Water quality'), findsOneWidget);
  });

  test('reset discards an in-flight response from the previous session',
      () async {
    SharedPreferences.setMockInitialValues({});
    final api = _SlowAssistantApiService();
    final controller = AssistantController.forTest(api);
    await controller.init();
    final oldSession = controller.sessionId;

    final sending = controller.sendMessage('Explain the current scan.');
    await api.started.future;
    final resetting = controller.resetSession();
    api.release.complete();
    await Future.wait([sending, resetting]);

    expect(controller.sessionId, isNot(oldSession));
    expect(controller.messages, isEmpty);
    expect(controller.isStreaming, isFalse);
  });
}
