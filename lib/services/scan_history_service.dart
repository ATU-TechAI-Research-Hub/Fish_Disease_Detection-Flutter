import 'package:flutter/foundation.dart';

import '../models/prediction_result_model.dart';

class ScanEntry {
  ScanEntry({
    required this.result,
    required this.imagePath,
    required this.timestamp,
  });

  final PredictionResultModel result;
  final String imagePath;
  final DateTime timestamp;
}

class ScanHistoryService extends ChangeNotifier {
  ScanHistoryService._();
  static final ScanHistoryService instance = ScanHistoryService._();

  final List<ScanEntry> _entries = [];

  List<ScanEntry> get entries => List.unmodifiable(_entries);
  int get count => _entries.length;

  void add(PredictionResultModel result, String imagePath) {
    _entries.insert(
      0,
      ScanEntry(
        result: result,
        imagePath: imagePath,
        timestamp: DateTime.now(),
      ),
    );
    notifyListeners();
  }

  void clear() {
    _entries.clear();
    notifyListeners();
  }
}
