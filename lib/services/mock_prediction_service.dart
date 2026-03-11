import 'dart:convert';
import 'dart:math';

import 'package:flutter/services.dart' show rootBundle;

import '../models/disease_model.dart';

class MockPredictionService {
  static const String _assetPath = 'assets/diseases.json';
  final Random _random = Random();

  // This simulates a backend + ML call by waiting 2 seconds and returning
  // a random disease entry from local JSON data.
  Future<DiseaseModel> predictDisease() async {
    await Future.delayed(const Duration(seconds: 2));

    final String jsonString = await rootBundle.loadString(_assetPath);
    final List<dynamic> jsonList = json.decode(jsonString) as List<dynamic>;

    if (jsonList.isEmpty) {
      throw StateError('No diseases found in local JSON data.');
    }

    final List<DiseaseModel> diseases = jsonList
        .map((dynamic item) =>
            DiseaseModel.fromJson(item as Map<String, dynamic>))
        .toList();

    return diseases[_random.nextInt(diseases.length)];
  }
}
