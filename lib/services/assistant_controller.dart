import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../models/assistant_message.dart';
import '../models/prediction_result_model.dart';
import 'assistant_api_service.dart';

class AssistantController extends ChangeNotifier {
  AssistantController._({AssistantApiService? api})
      : _api = api ?? AssistantApiService();

  static final AssistantController instance = AssistantController._();

  @visibleForTesting
  factory AssistantController.forTest(AssistantApiService api) =>
      AssistantController._(api: api);

  static const _sessionKey = 'aquascan_assistant_session_id';
  static const _modelKey = 'aquascan_assistant_model';

  final AssistantApiService _api;
  final List<AssistantMessage> _messages = [];
  List<AssistantModelOption> _models = const [
    AssistantModelOption(
      key: 'qwen',
      displayName: 'Qwen3 8B',
      available: true,
      active: false,
    ),
    AssistantModelOption(
      key: 'mistral',
      displayName: 'Mistral 7B',
      available: true,
      active: false,
    ),
    AssistantModelOption(
      key: 'llama',
      displayName: 'Llama 3.1 8B',
      available: true,
      active: false,
    ),
  ];

  String _sessionId = '';
  String _selectedModel = 'qwen';
  bool _isOpen = false;
  bool _isStreaming = false;
  bool _isLoadingHistory = false;
  bool _backendAvailable = false;
  String? _lastError;
  PredictionResultModel? _currentPrediction;
  int _temporaryId = 0;
  int _sessionGeneration = 0;
  Completer<void>? _activeStreamDone;

  List<AssistantMessage> get messages => List.unmodifiable(_messages);
  List<AssistantModelOption> get models => List.unmodifiable(_models);
  String get sessionId => _sessionId;
  String get selectedModel => _selectedModel;
  bool get isOpen => _isOpen;
  bool get isStreaming => _isStreaming;
  bool get isLoadingHistory => _isLoadingHistory;
  bool get backendAvailable => _backendAvailable;
  String? get lastError => _lastError;
  PredictionResultModel? get currentPrediction => _currentPrediction;

  Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    _sessionId = prefs.getString(_sessionKey) ?? _newSessionId();
    _selectedModel = prefs.getString(_modelKey) ?? 'qwen';
    await prefs.setString(_sessionKey, _sessionId);
  }

  String _newSessionId() {
    final random = Random.secure().nextInt(0x7fffffff).toRadixString(16);
    return 'aquascan_${DateTime.now().microsecondsSinceEpoch}_$random';
  }

  void toggle() => _isOpen ? close() : open();

  void open() {
    _isOpen = true;
    notifyListeners();
    unawaited(refresh());
  }

  void close() {
    _isOpen = false;
    notifyListeners();
  }

  Future<void> refresh() async {
    if (_sessionId.isEmpty || _isStreaming) return;
    final generation = _sessionGeneration;
    _isLoadingHistory = true;
    _lastError = null;
    notifyListeners();
    try {
      final results = await Future.wait([
        _api.history(_sessionId),
        _api.models(),
      ]);
      if (generation != _sessionGeneration) return;
      _messages
        ..clear()
        ..addAll(results[0] as List<AssistantMessage>);
      _models = results[1] as List<AssistantModelOption>;
      _backendAvailable = true;
      if (!_models.any(
        (option) => option.key == _selectedModel && option.available,
      )) {
        final available = _models.where((option) => option.available);
        if (available.isNotEmpty) {
          await setModel(available.first.key);
        }
      }
    } catch (error) {
      if (generation != _sessionGeneration) return;
      _backendAvailable = false;
      _lastError = _friendlyError(error);
    } finally {
      if (generation == _sessionGeneration) {
        _isLoadingHistory = false;
        notifyListeners();
      }
    }
  }

  Future<void> setModel(String model) async {
    if (_isStreaming || model == _selectedModel) return;
    _selectedModel = model;
    notifyListeners();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_modelKey, model);
  }

  Future<void> sendMessage(String question) async {
    final clean = question.trim();
    if (clean.isEmpty ||
        _isStreaming ||
        _isLoadingHistory ||
        _sessionId.isEmpty) {
      return;
    }
    _messages.add(
      AssistantMessage(
        id: _nextTemporaryId('user'),
        role: AssistantRole.user,
        content: clean,
        createdAt: DateTime.now(),
      ),
    );
    await _streamAnswer(question: clean, regenerate: false);
  }

  Future<void> regenerateLast() async {
    if (_isStreaming) return;
    final user = _messages.lastWhere(
      (message) => message.role == AssistantRole.user,
      orElse: () => AssistantMessage(
        id: '',
        role: AssistantRole.user,
        content: '',
        createdAt: DateTime.now(),
      ),
    );
    if (user.content.isEmpty) return;
    final assistantIndex = _messages.lastIndexWhere(
      (message) => message.role == AssistantRole.assistant,
    );
    if (assistantIndex >= 0) {
      _messages.removeAt(assistantIndex);
    }
    await _streamAnswer(question: user.content, regenerate: true);
  }

  Future<void> _streamAnswer({
    required String question,
    required bool regenerate,
  }) async {
    final generation = _sessionGeneration;
    final streamDone = Completer<void>();
    _activeStreamDone = streamDone;
    final placeholderId = _nextTemporaryId('assistant');
    _messages.add(
      AssistantMessage(
        id: placeholderId,
        role: AssistantRole.assistant,
        content: '',
        createdAt: DateTime.now(),
        model: _selectedModel,
        isStreaming: true,
      ),
    );
    _isStreaming = true;
    _lastError = null;
    notifyListeners();

    var receivedDone = false;
    try {
      await for (final event in _api.streamChat(
        sessionId: _sessionId,
        question: question,
        model: _selectedModel,
        regenerate: regenerate,
      )) {
        if (generation != _sessionGeneration) break;
        final type = event['type']?.toString();
        if (type == 'start') {
          final sources = (event['sources'] as List<dynamic>? ?? const [])
              .whereType<Map<String, dynamic>>()
              .map(AssistantSource.fromJson)
              .toList(growable: false);
          _updateMessage(
            placeholderId,
            (message) => message.copyWith(
              model: event['model']?.toString(),
              sources: sources,
            ),
          );
        } else if (type == 'token') {
          final token = event['text']?.toString() ?? '';
          _updateMessage(
            placeholderId,
            (message) => message.copyWith(content: message.content + token),
          );
        } else if (type == 'error') {
          throw Exception(
            event['message']?.toString() ?? 'Local model generation failed.',
          );
        } else if (type == 'done') {
          receivedDone = true;
          final rawMessage = event['message'];
          if (rawMessage is Map<String, dynamic>) {
            final saved = AssistantMessage.fromJson(rawMessage);
            _updateMessage(
              placeholderId,
              (_) => saved.copyWith(isStreaming: false),
            );
          }
        }
        notifyListeners();
      }
      if (generation != _sessionGeneration) return;
      if (!receivedDone) {
        throw Exception('The local assistant stream ended unexpectedly.');
      }
      _backendAvailable = true;
      _updateMessage(
        placeholderId,
        (message) => message.copyWith(isStreaming: false),
      );
    } catch (error) {
      if (generation != _sessionGeneration) return;
      _backendAvailable = false;
      _lastError = _friendlyError(error);
      _updateMessage(
        placeholderId,
        (message) => message.copyWith(
          content: message.content.isEmpty ? _lastError! : message.content,
          isStreaming: false,
          isError: true,
        ),
      );
    } finally {
      if (!streamDone.isCompleted) streamDone.complete();
      if (identical(_activeStreamDone, streamDone)) {
        _activeStreamDone = null;
      }
      if (generation == _sessionGeneration) {
        _isStreaming = false;
        notifyListeners();
      }
    }
  }

  void _updateMessage(
    String id,
    AssistantMessage Function(AssistantMessage message) update,
  ) {
    final index = _messages.indexWhere((message) => message.id == id);
    if (index >= 0) _messages[index] = update(_messages[index]);
  }

  Future<void> clearConversation() async {
    if (_isStreaming || _sessionId.isEmpty) return;
    try {
      await _api.clearHistory(_sessionId);
      _backendAvailable = true;
      _lastError = null;
    } catch (error) {
      _backendAvailable = false;
      _lastError = _friendlyError(error);
    }
    _messages.clear();
    notifyListeners();
  }

  Future<void> resetSession() async {
    final previousSession = _sessionId;
    final activeStream = _activeStreamDone?.future;
    _sessionGeneration++;
    _api.cancelActiveRequests();
    _isOpen = false;
    _isStreaming = false;
    _isLoadingHistory = false;
    _messages.clear();
    _currentPrediction = null;
    _lastError = null;
    _sessionId = _newSessionId();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_sessionKey, _sessionId);
    notifyListeners();
    if (activeStream != null) {
      try {
        await activeStream.timeout(const Duration(seconds: 10));
      } catch (_) {
        // The old local model stream is detached from the new account session.
      }
    }
    if (previousSession.isEmpty) return;
    try {
      await _api.deleteSession(previousSession);
    } catch (_) {
      // Local account isolation is complete even when the backend is offline.
    }
  }

  Future<void> attachPrediction(
    PredictionResultModel prediction, {
    bool publishToBackend = false,
  }) async {
    _currentPrediction = prediction;
    notifyListeners();
    if (!publishToBackend || _sessionId.isEmpty) return;
    try {
      await _api.publishPrediction(_sessionId, prediction);
      _backendAvailable = true;
      _lastError = null;
    } catch (error) {
      _backendAvailable = false;
      _lastError = _friendlyError(error);
    }
    notifyListeners();
  }

  String _nextTemporaryId(String role) =>
      'local_${role}_${DateTime.now().microsecondsSinceEpoch}_${_temporaryId++}';

  String _friendlyError(Object error) {
    return error.toString().replaceFirst(
          RegExp(
            r'^(Exception|SocketException|ClientException|FormatException):\s*',
          ),
          '',
        );
  }
}
