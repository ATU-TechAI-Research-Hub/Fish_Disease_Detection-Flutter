import 'disease_model.dart';

class ClassProbability {
  const ClassProbability({
    required this.diseaseId,
    required this.diseaseName,
    required this.confidence,
  });

  final int diseaseId;
  final String diseaseName;
  final double confidence;

  factory ClassProbability.fromJson(Map<String, dynamic> json) {
    final dynamic rawConf = json['confidence'];
    return ClassProbability(
      diseaseId: json['disease_id'] as int? ?? 0,
      diseaseName: json['disease_name']?.toString() ?? 'Unknown',
      confidence: rawConf is num
          ? rawConf.toDouble()
          : double.tryParse(rawConf?.toString() ?? '') ?? 0,
    );
  }
}

class PredictionResultModel {
  const PredictionResultModel({
    required this.disease,
    required this.confidence,
    required this.source,
    required this.filename,
    this.inferenceMs = 0,
    this.topPredictions = const [],
  });

  final DiseaseModel disease;
  final double confidence;
  final String source;
  final String filename;
  final double inferenceMs;
  final List<ClassProbability> topPredictions;

  factory PredictionResultModel.fromJson(Map<String, dynamic> json) {
    final dynamic rawConfidence = json['confidence'];
    final dynamic rawInferenceMs = json['inference_ms'];
    final List<dynamic> topList =
        json['top_predictions'] as List<dynamic>? ?? [];

    return PredictionResultModel(
      disease: DiseaseModel.fromJson(
        json['prediction'] as Map<String, dynamic>? ?? <String, dynamic>{},
      ),
      confidence: rawConfidence is num
          ? rawConfidence.toDouble()
          : double.tryParse(rawConfidence?.toString() ?? '') ?? 0,
      source: json['source']?.toString() ?? 'unknown',
      filename: json['filename']?.toString() ?? '',
      inferenceMs: rawInferenceMs is num
          ? rawInferenceMs.toDouble()
          : double.tryParse(rawInferenceMs?.toString() ?? '') ?? 0,
      topPredictions: topList
          .map((e) =>
              ClassProbability.fromJson(e as Map<String, dynamic>))
          .toList(),
    );
  }
}
