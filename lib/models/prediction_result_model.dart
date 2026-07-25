import 'disease_model.dart';

enum ConfidenceTier {
  high,
  medium,
  low;

  static ConfidenceTier fromString(String? raw) {
    switch ((raw ?? '').toLowerCase()) {
      case 'high':
        return ConfidenceTier.high;
      case 'medium':
        return ConfidenceTier.medium;
      case 'low':
      default:
        return ConfidenceTier.low;
    }
  }

  String get label {
    switch (this) {
      case ConfidenceTier.high:
        return 'High confidence';
      case ConfidenceTier.medium:
        return 'Medium confidence';
      case ConfidenceTier.low:
        return 'Low confidence';
    }
  }
}

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
      diseaseId: (json['disease_id'] as num?)?.toInt() ?? 0,
      diseaseName: json['disease_name']?.toString() ?? 'Unknown',
      confidence: rawConf is num
          ? rawConf.toDouble()
          : double.tryParse(rawConf?.toString() ?? '') ?? 0,
    );
  }

  Map<String, dynamic> toJson() => {
        'disease_id': diseaseId,
        'disease_name': diseaseName,
        'confidence': confidence,
      };
}

class PredictionResultModel {
  const PredictionResultModel({
    required this.disease,
    required this.confidence,
    required this.confidenceTier,
    required this.source,
    required this.filename,
    this.inferenceMs = 0,
    this.topPredictions = const [],
    this.warning,
    this.recommendation,
  });

  final DiseaseModel disease;
  final double confidence;
  final ConfidenceTier confidenceTier;
  final String source;
  final String filename;
  final double inferenceMs;
  final List<ClassProbability> topPredictions;
  final String? warning;
  final String? recommendation;

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
      confidenceTier:
          ConfidenceTier.fromString(json['confidence_tier']?.toString()),
      source: json['source']?.toString() ?? 'unknown',
      filename: json['filename']?.toString() ?? '',
      inferenceMs: rawInferenceMs is num
          ? rawInferenceMs.toDouble()
          : double.tryParse(rawInferenceMs?.toString() ?? '') ?? 0,
      topPredictions: topList
          .map((e) => ClassProbability.fromJson(e as Map<String, dynamic>))
          .toList(),
      warning: (json['warning'] as String?)?.trim().isEmpty == true
          ? null
          : json['warning']?.toString(),
      recommendation:
          (json['recommendation'] as String?)?.trim().isEmpty == true
              ? null
              : json['recommendation']?.toString(),
    );
  }

  Map<String, dynamic> toJson() => {
        'prediction': disease.toJson(),
        'confidence': confidence,
        'confidence_tier': confidenceTier.name,
        'source': source,
        'filename': filename,
        'inference_ms': inferenceMs,
        'top_predictions': topPredictions.map((item) => item.toJson()).toList(),
        'warning': warning,
        'recommendation': recommendation,
      };
}
