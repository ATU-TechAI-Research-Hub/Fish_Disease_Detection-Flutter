import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import 'api_prediction_service.dart';

/// Connectivity / model-status states surfaced in the UI.
enum BackendStatus {
  /// Initial / waiting state before the first probe completes.
  unknown,

  /// `/health` returned 200 and `model_ready: true`.
  online,

  /// Server reachable but model is not ready (e.g. missing `model.h5`).
  degraded,

  /// Cannot reach the backend at all.
  offline,
}

/// Polls the FastAPI `/health` endpoint and exposes a `ChangeNotifier`.
///
/// The UI uses this both as the live offline-mode banner and to short-circuit
/// the call-to-action buttons when the backend is unreachable.
class BackendStatusService extends ChangeNotifier {
  BackendStatusService._();
  static final BackendStatusService instance = BackendStatusService._();

  final ApiPredictionService _api = ApiPredictionService();

  BackendStatus _status = BackendStatus.unknown;
  String _backendName = 'unknown';
  String _deviceName = 'n/a';
  DateTime? _lastCheck;
  Timer? _timer;

  BackendStatus get status => _status;
  String get backendName => _backendName;
  String get deviceName => _deviceName;
  DateTime? get lastCheck => _lastCheck;
  bool get isOnline => _status == BackendStatus.online;
  bool get isReachable =>
      _status == BackendStatus.online || _status == BackendStatus.degraded;

  /// Start probing `/health` every [interval] seconds. Safe to call multiple
  /// times — repeated calls are no-ops after the first.
  void start({Duration interval = const Duration(seconds: 15)}) {
    _timer ??= Timer.periodic(interval, (_) => probe());
    probe();
  }

  Future<void> probe() async {
    final uri = Uri.parse('${_api.baseUrl}/health');
    BackendStatus newStatus;
    String backendName = _backendName;
    String deviceName = _deviceName;

    try {
      final response = await http.get(uri).timeout(const Duration(seconds: 4));
      if (response.statusCode == 200) {
        final body = json.decode(response.body) as Map<String, dynamic>;
        final modelReady = body['model_ready'] == true;
        backendName = body['backend']?.toString() ?? backendName;
        deviceName = body['device']?.toString() ?? deviceName;
        newStatus = modelReady ? BackendStatus.online : BackendStatus.degraded;
      } else {
        newStatus = BackendStatus.degraded;
      }
    } on SocketException {
      newStatus = BackendStatus.offline;
    } on TimeoutException {
      newStatus = BackendStatus.offline;
    } catch (_) {
      newStatus = BackendStatus.offline;
    }

    final changed = newStatus != _status ||
        backendName != _backendName ||
        deviceName != _deviceName;
    _status = newStatus;
    _backendName = backendName;
    _deviceName = deviceName;
    _lastCheck = DateTime.now();

    if (changed) notifyListeners();
  }

  @override
  void dispose() {
    _timer?.cancel();
    _timer = null;
    super.dispose();
  }
}
