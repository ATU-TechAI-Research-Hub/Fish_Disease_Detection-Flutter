class DiseaseModel {
  const DiseaseModel({
    required this.id,
    required this.name,
    required this.type,
    required this.cause,
    required this.symptoms,
    required this.treatment,
    required this.prevention,
  });

  final int id;
  final String name;
  final String type;
  final String cause;
  final String symptoms;
  final String treatment;
  final String prevention;

  bool get isHealthy => name.toLowerCase() == 'healthy fish';
  bool get isUnknown => type.toLowerCase() == 'unknown' || name.toLowerCase().contains('no fish');

  factory DiseaseModel.fromJson(Map<String, dynamic> json) {
    final dynamic rawId = json['id'];
    return DiseaseModel(
      id: rawId is int ? rawId : int.tryParse(rawId.toString()) ?? 0,
      name: json['name']?.toString() ?? 'Unknown Disease',
      type: json['type']?.toString() ?? 'Unknown Type',
      cause: json['cause']?.toString() ?? 'No cause available.',
      symptoms: json['symptoms']?.toString() ?? 'No symptoms available.',
      treatment: json['treatment']?.toString() ?? 'No treatment available.',
      prevention: json['prevention']?.toString() ?? 'No prevention available.',
    );
  }
}
