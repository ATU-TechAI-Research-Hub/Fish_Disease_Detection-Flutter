import 'dart:io';

import 'package:flutter/material.dart';

import '../models/prediction_result_model.dart';
import '../services/api_prediction_service.dart';

class ResultScreen extends StatefulWidget {
  const ResultScreen({required this.imagePath, super.key});

  final String imagePath;

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  late Future<PredictionResultModel> _predictionFuture;
  final ApiPredictionService _apiPredictionService = ApiPredictionService();

  @override
  void initState() {
    super.initState();
    _predictionFuture = _apiPredictionService.predictDiseaseFromImage(widget.imagePath);
  }

  Future<void> _retryPrediction() async {
    setState(() {
      _predictionFuture = _apiPredictionService.predictDiseaseFromImage(
        widget.imagePath,
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Detection Result'),
      ),
      body: FutureBuilder<PredictionResultModel>(
        future: _predictionFuture,
        builder: (
          BuildContext context,
          AsyncSnapshot<PredictionResultModel> snapshot,
        ) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 12),
                  Text('Analyzing image...'),
                ],
              ),
            );
          }

          if (snapshot.hasError) {
            final String errorMessage = snapshot.error.toString();
            return Center(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.error_outline, size: 56),
                    const SizedBox(height: 12),
                    const Text(
                      'Failed to get a real backend prediction.',
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      errorMessage,
                      textAlign: TextAlign.center,
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                    const SizedBox(height: 14),
                    FilledButton.icon(
                      onPressed: _retryPrediction,
                      icon: const Icon(Icons.refresh_rounded),
                      label: const Text('Try Again'),
                    ),
                  ],
                ),
              ),
            );
          }

          if (!snapshot.hasData) {
            return const Center(child: Text('No result available.'));
          }

          final PredictionResultModel predictionResult = snapshot.data!;
          final disease = predictionResult.disease;
          final ColorScheme colorScheme = Theme.of(context).colorScheme;
          final String confidenceLabel =
              '${(predictionResult.confidence * 100).toStringAsFixed(1)}%';

          return SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(18),
                  child: Image.file(
                    File(widget.imagePath),
                    width: double.infinity,
                    height: 240,
                    fit: BoxFit.cover,
                  ),
                ),
                const SizedBox(height: 16),
                Text(
                  disease.name,
                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                        fontWeight: FontWeight.w800,
                      ),
                ),
                const SizedBox(height: 10),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: [
                    Chip(
                      avatar: const Icon(Icons.science_rounded, size: 18),
                      label: Text('Type: ${disease.type}'),
                    ),
                    Chip(
                      avatar: const Icon(Icons.analytics_rounded, size: 18),
                      label: Text('Confidence: $confidenceLabel'),
                    ),
                    if (disease.isHealthy)
                      Chip(
                        backgroundColor: Colors.green.shade100,
                        side: BorderSide(color: Colors.green.shade300),
                        avatar: const Icon(
                          Icons.verified_rounded,
                          size: 18,
                          color: Colors.green,
                        ),
                        label: const Text(
                          'Healthy Fish',
                          style: TextStyle(
                            color: Colors.green,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                    Chip(
                      avatar: const Icon(Icons.memory_rounded, size: 18),
                      label: Text(predictionResult.source),
                    ),
                  ],
                ),
                const SizedBox(height: 14),
                _DetailCard(
                  icon: Icons.coronavirus_rounded,
                  title: 'Cause',
                  content: disease.cause,
                  color: colorScheme.primary,
                ),
                _DetailCard(
                  icon: Icons.warning_amber_rounded,
                  title: 'Symptoms',
                  content: disease.symptoms,
                  color: colorScheme.tertiary,
                ),
                _DetailCard(
                  icon: Icons.medical_services_rounded,
                  title: 'Treatment',
                  content: disease.treatment,
                  color: colorScheme.secondary,
                ),
                _DetailCard(
                  icon: Icons.shield_rounded,
                  title: 'Prevention',
                  content: disease.prevention,
                  color: Colors.green.shade700,
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}

class _DetailCard extends StatelessWidget {
  const _DetailCard({
    required this.icon,
    required this.title,
    required this.content,
    required this.color,
  });

  final IconData icon;
  final String title;
  final String content;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(
          color: Theme.of(context).colorScheme.outlineVariant,
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 36,
              height: 36,
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(icon, color: color, size: 20),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.w700,
                        ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    content,
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
