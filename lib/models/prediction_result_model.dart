import 'disease_model.dart';

class PredictionResultModel {
  const PredictionResultModel({
    required this.disease,
    required this.confidence,
    required this.source,
    required this.filename,
  });

  final DiseaseModel disease;
  final double confidence;
  final String source;
  final String filename;

  factory PredictionResultModel.fromJson(Map<String, dynamic> json) {
    final dynamic rawConfidence = json['confidence'];
    return PredictionResultModel(
      disease: DiseaseModel.fromJson(
        json['prediction'] as Map<String, dynamic>? ?? <String, dynamic>{},
      ),
      confidence: rawConfidence is num
          ? rawConfidence.toDouble()
          : double.tryParse(rawConfidence?.toString() ?? '') ?? 0,
      source: json['source']?.toString() ?? 'unknown',
      filename: json['filename']?.toString() ?? '',
    );
  }
}
