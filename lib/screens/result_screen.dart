import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';

import '../models/prediction_result_model.dart';
import '../services/api_prediction_service.dart';
import '../services/assistant_controller.dart';
import '../services/scan_history_service.dart';
import '../theme/app_theme.dart';
import '../widgets/bubble_background.dart';
import '../widgets/confidence_ring.dart';
import '../widgets/wave_clipper.dart';

class ResultScreen extends StatefulWidget {
  const ResultScreen({
    required this.imagePath,
    this.existingResult,
    super.key,
  });

  final String imagePath;
  final PredictionResultModel? existingResult;

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen>
    with SingleTickerProviderStateMixin {
  late Future<PredictionResultModel> _predictionFuture;
  final ApiPredictionService _apiService = ApiPredictionService();
  late AnimationController _fadeController;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();
    _fadeController = AnimationController(
      duration: const Duration(milliseconds: 600),
      vsync: this,
    );
    _fadeAnimation = CurvedAnimation(
      parent: _fadeController,
      curve: Curves.easeOut,
    );

    if (widget.existingResult != null) {
      _predictionFuture = Future.value(widget.existingResult!);
      unawaited(
        AssistantController.instance.attachPrediction(
          widget.existingResult!,
          publishToBackend: true,
        ),
      );
      _fadeController.forward();
    } else {
      _predictionFuture = _predict();
    }
  }

  Future<PredictionResultModel> _predict() async {
    final result = await _apiService.predictDiseaseFromImage(
      widget.imagePath,
      assistantSessionId: AssistantController.instance.sessionId,
    );
    await AssistantController.instance.attachPrediction(result);
    ScanHistoryService.instance.add(result, widget.imagePath);
    _fadeController.forward();
    return result;
  }

  void _retryPrediction() {
    _fadeController.reset();
    setState(() {
      _predictionFuture = _predict();
    });
  }

  @override
  void dispose() {
    _fadeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: FutureBuilder<PredictionResultModel>(
        future: _predictionFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return _LoadingView();
          }
          if (snapshot.hasError) {
            return _ErrorView(
              error: snapshot.error.toString(),
              onRetry: _retryPrediction,
            );
          }
          if (!snapshot.hasData) {
            return const Center(child: Text('No result available.'));
          }
          return FadeTransition(
            opacity: _fadeAnimation,
            child: _ResultBody(
              result: snapshot.data!,
              imagePath: widget.imagePath,
            ),
          );
        },
      ),
    );
  }
}

class _LoadingView extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return BubbleBackground(
      child: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [AppColors.deepOcean, AppColors.seaBlue],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                child: Row(
                  children: [
                    IconButton(
                      icon: const Icon(Icons.arrow_back_rounded,
                          color: Colors.white),
                      onPressed: () => Navigator.of(context).pop(),
                    ),
                  ],
                ),
              ),
              const Spacer(),
              Container(
                width: 110,
                height: 110,
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.08),
                  borderRadius: BorderRadius.circular(32),
                ),
                child: const Center(
                  child: SizedBox(
                    width: 50,
                    height: 50,
                    child: CircularProgressIndicator(
                      strokeWidth: 3,
                      color: AppColors.seafoam,
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 28),
              const Text(
                'Analyzing Fish...',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.w800,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 10),
              Text(
                'Our AI model is identifying\nthe fish condition',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 15,
                  color: Colors.white.withValues(alpha: 0.65),
                  height: 1.5,
                ),
              ),
              const Spacer(flex: 2),
            ],
          ),
        ),
      ),
    );
  }
}

class _ErrorView extends StatelessWidget {
  const _ErrorView({required this.error, required this.onRetry});

  final String error;
  final VoidCallback onRetry;

  bool get _isConnectionError {
    final lower = error.toLowerCase();
    return lower.contains('socket') ||
        lower.contains('connect') ||
        lower.contains('timeout') ||
        lower.contains('connection');
  }

  @override
  Widget build(BuildContext context) {
    final isConn = _isConnectionError;
    final icon = isConn ? Icons.wifi_off_rounded : Icons.error_outline_rounded;
    final title = isConn ? 'Connection Failed' : 'Analysis Failed';
    final subtitle = isConn
        ? 'Could not reach the backend server.\nMake sure it is running and try again.'
        : 'Something went wrong during analysis.\nPlease try again with a different image.';

    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          children: [
            Align(
              alignment: Alignment.centerLeft,
              child: IconButton(
                icon: const Icon(Icons.arrow_back_rounded),
                onPressed: () => Navigator.of(context).pop(),
              ),
            ),
            const Spacer(),
            Container(
              width: 88,
              height: 88,
              decoration: BoxDecoration(
                color: AppColors.coral.withValues(alpha: 0.08),
                borderRadius: BorderRadius.circular(26),
              ),
              child: Icon(icon, size: 42, color: AppColors.coral),
            ),
            const SizedBox(height: 24),
            Text(title, style: Theme.of(context).textTheme.headlineSmall),
            const SizedBox(height: 8),
            Text(
              subtitle,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            const SizedBox(height: 6),
            Text(
              error,
              textAlign: TextAlign.center,
              maxLines: 3,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(fontSize: 11, color: Color(0xFFB0BEC5)),
            ),
            const SizedBox(height: 28),
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                onPressed: onRetry,
                icon: const Icon(Icons.refresh_rounded),
                label: const Text('Try Again'),
              ),
            ),
            const Spacer(),
          ],
        ),
      ),
    );
  }
}

class _ResultBody extends StatelessWidget {
  const _ResultBody({required this.result, required this.imagePath});

  final PredictionResultModel result;
  final String imagePath;

  Color get _badgeColor {
    final d = result.disease;
    if (d.isUnknown) return const Color(0xFF6B7280);
    if (d.isHealthy) return AppColors.emerald;
    return AppColors.coral;
  }

  String get _badgeLabel {
    final d = result.disease;
    if (d.isUnknown) return 'NOT RECOGNIZED';
    if (d.isHealthy) return 'HEALTHY';
    return d.type.toUpperCase();
  }

  Color _tierColor(ConfidenceTier tier) {
    switch (tier) {
      case ConfidenceTier.high:
        return AppColors.emerald;
      case ConfidenceTier.medium:
        return AppColors.amber;
      case ConfidenceTier.low:
        return AppColors.coral;
    }
  }

  IconData _tierIcon(ConfidenceTier tier) {
    switch (tier) {
      case ConfidenceTier.high:
        return Icons.verified_rounded;
      case ConfidenceTier.medium:
        return Icons.balance_rounded;
      case ConfidenceTier.low:
        return Icons.help_outline_rounded;
    }
  }

  @override
  Widget build(BuildContext context) {
    final disease = result.disease;
    final isUnknown = disease.isUnknown;

    return CustomScrollView(
      slivers: [
        SliverToBoxAdapter(
          child: Stack(
            children: [
              ClipPath(
                clipper: WaveClipper(waveHeight: 16),
                child: SizedBox(
                  height: 300,
                  width: double.infinity,
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      Image.file(
                        File(imagePath),
                        fit: BoxFit.cover,
                        errorBuilder: (_, __, ___) => Container(
                          color: AppColors.deepOcean,
                          child: const Center(
                            child: Icon(Icons.image_not_supported_rounded,
                                color: Colors.white54, size: 60),
                          ),
                        ),
                      ),
                      Container(
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            begin: Alignment.topCenter,
                            end: Alignment.bottomCenter,
                            colors: [
                              Colors.black.withValues(alpha: 0.05),
                              Colors.black.withValues(alpha: 0.55),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              Positioned(
                bottom: 36,
                left: 20,
                right: 20,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: _badgeColor,
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        _badgeLabel,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 11,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 0.8,
                        ),
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      disease.name,
                      style: const TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.w800,
                        color: Colors.white,
                        height: 1.2,
                      ),
                    ),
                  ],
                ),
              ),
              SafeArea(
                child: Padding(
                  padding: const EdgeInsets.all(8),
                  child: Container(
                    decoration: BoxDecoration(
                      color: Colors.black.withValues(alpha: 0.25),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: IconButton(
                      icon: const Icon(Icons.arrow_back_rounded,
                          color: Colors.white),
                      onPressed: () => Navigator.of(context).pop(),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 4, 20, 32),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildAskAssistant(context),
                const SizedBox(height: 14),
                if (isUnknown) ...[
                  _buildNoFishCard(context),
                  if (result.warning != null) ...[
                    const SizedBox(height: 12),
                    _buildWarningCard(context, result.warning!),
                  ],
                  if (result.recommendation != null) ...[
                    const SizedBox(height: 12),
                    _buildRecommendationCard(context, result.recommendation!),
                  ],
                  const SizedBox(height: 12),
                  _buildMetaRow(context),
                  if (result.topPredictions.length > 1) ...[
                    const SizedBox(height: 16),
                    _buildTopPredictions(context),
                  ],
                  const SizedBox(height: 20),
                  _buildDetail(
                    context,
                    icon: Icons.info_outline_rounded,
                    title: 'What Happened',
                    text: disease.cause,
                    color: const Color(0xFF6B7280),
                  ),
                  const SizedBox(height: 12),
                  _buildDetail(
                    context,
                    icon: Icons.camera_alt_rounded,
                    title: 'How to Get Better Results',
                    text: disease.treatment,
                    color: AppColors.seaBlue,
                  ),
                  const SizedBox(height: 12),
                  _buildDetail(
                    context,
                    icon: Icons.tips_and_updates_rounded,
                    title: 'Photography Tips',
                    text: disease.prevention,
                    color: AppColors.amber,
                  ),
                ] else ...[
                  _buildConfidenceCard(context),
                  const SizedBox(height: 12),
                  _buildTierBadge(context),
                  if (result.warning != null) ...[
                    const SizedBox(height: 12),
                    _buildWarningCard(context, result.warning!),
                  ],
                  if (result.recommendation != null) ...[
                    const SizedBox(height: 12),
                    _buildRecommendationCard(context, result.recommendation!),
                  ],
                  const SizedBox(height: 12),
                  _buildMetaRow(context),
                  if (result.topPredictions.length > 1) ...[
                    const SizedBox(height: 16),
                    _buildTopPredictions(context),
                  ],
                  const SizedBox(height: 20),
                  _buildDetail(
                    context,
                    icon: Icons.coronavirus_rounded,
                    title: 'Cause',
                    text: disease.cause,
                    color: AppColors.seaBlue,
                  ),
                  const SizedBox(height: 12),
                  _buildDetail(
                    context,
                    icon: Icons.warning_amber_rounded,
                    title: 'Symptoms',
                    text: disease.symptoms,
                    color: AppColors.amber,
                  ),
                  const SizedBox(height: 12),
                  _buildDetail(
                    context,
                    icon: Icons.medical_services_rounded,
                    title: 'Treatment',
                    text: disease.treatment,
                    color: AppColors.purple,
                  ),
                  const SizedBox(height: 12),
                  _buildDetail(
                    context,
                    icon: Icons.shield_rounded,
                    title: 'Prevention',
                    text: disease.prevention,
                    color: AppColors.emerald,
                  ),
                ],
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildAskAssistant(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      child: FilledButton.icon(
        onPressed: () {
          unawaited(
            AssistantController.instance.attachPrediction(
              result,
              publishToBackend: true,
            ),
          );
          AssistantController.instance.open();
        },
        icon: const Icon(Icons.auto_awesome_rounded),
        label: const Text('Ask AI about this result'),
        style: FilledButton.styleFrom(
          backgroundColor:
              context.isDarkMode ? AppColors.seaBlue : AppColors.ocean,
        ),
      ),
    );
  }

  Widget _buildNoFishCard(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: context.surfaceCard,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: context.subtleBorder),
        boxShadow: [
          BoxShadow(
            color: context.cardShadow,
            blurRadius: 20,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        children: [
          Container(
            width: 64,
            height: 64,
            decoration: BoxDecoration(
              color: context.subtleBorder,
              borderRadius: BorderRadius.circular(20),
            ),
            child: const Icon(
              Icons.search_off_rounded,
              size: 32,
              color: Color(0xFF6B7280),
            ),
          ),
          const SizedBox(height: 14),
          Text(
            'No Fish Detected',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w800,
              color: context.textPrimary,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            'The AI model could not confidently identify a fish disease '
            'in this image. This could mean the image does not contain a '
            'recognizable fish, or the photo quality is too low.',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 14,
              color: AppColors.textSecondary,
              height: 1.5,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildConfidenceCard(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: context.surfaceCard,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: context.cardShadow,
            blurRadius: 20,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        children: [
          ConfidenceRing(
            confidence: result.confidence,
            tier: result.confidenceTier,
            size: 88,
            strokeWidth: 7,
          ),
          const SizedBox(width: 20),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Detection Result',
                    style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 4),
                Text(
                  result.disease.isHealthy
                      ? 'No disease detected. The fish appears healthy.'
                      : 'A ${result.disease.type.toLowerCase()} condition was identified.',
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTierBadge(BuildContext context) {
    final tier = result.confidenceTier;
    final color = _tierColor(tier);
    final icon = _tierIcon(tier);
    final pct = (result.confidence * 100).toStringAsFixed(1);
    return Semantics(
      label: '${tier.label}, $pct percent. ${result.disease.name}.',
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: color.withValues(alpha: 0.3)),
        ),
        child: Row(
          children: [
            Icon(icon, color: color, size: 20),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                '${tier.label} • $pct%',
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w800,
                  color: color,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildWarningCard(BuildContext context, String warning) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.amber.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.amber.withValues(alpha: 0.3)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.warning_amber_rounded,
              color: AppColors.amber, size: 22),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              warning,
              style: TextStyle(
                fontSize: 13,
                color: context.textPrimary,
                height: 1.45,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRecommendationCard(BuildContext context, String recommendation) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.seaBlue.withValues(alpha: 0.06),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.seaBlue.withValues(alpha: 0.2)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.lightbulb_outline_rounded,
              color: AppColors.seaBlue, size: 22),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Recommended next step',
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w800,
                    color: AppColors.ocean,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  recommendation,
                  style: TextStyle(
                    fontSize: 13,
                    color: context.textPrimary,
                    height: 1.45,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMetaRow(BuildContext context) {
    final chips = <Widget>[];
    if (result.inferenceMs > 0) {
      chips.add(_MetaChip(
        icon: Icons.speed_rounded,
        label: '${result.inferenceMs.toStringAsFixed(0)} ms',
      ));
    }
    chips.add(_MetaChip(
      icon: Icons.memory_rounded,
      label: _prettySource(result.source),
    ));
    return Wrap(spacing: 8, runSpacing: 8, children: chips);
  }

  String _prettySource(String source) {
    final lower = source.toLowerCase();
    if (lower.contains('keras')) return 'KERAS H5';
    if (lower.contains('onnx')) return 'ONNX';
    return source.toUpperCase();
  }

  Widget _buildTopPredictions(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: context.tintedSurface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.aqua.withValues(alpha: 0.15)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.leaderboard_rounded,
                  size: 18, color: AppColors.seaBlue),
              const SizedBox(width: 8),
              Text(
                'Top Predictions',
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                  color: context.textPrimary,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          for (int i = 0; i < result.topPredictions.length; i++)
            _buildPredictionBar(context, result.topPredictions[i], i),
        ],
      ),
    );
  }

  Widget _buildPredictionBar(
      BuildContext context, ClassProbability pred, int rank) {
    final pct = (pred.confidence * 100).toStringAsFixed(1);
    final isTop = rank == 0;
    final barColor = isTop ? AppColors.seaBlue : AppColors.aqua;

    return Padding(
      padding: EdgeInsets.only(bottom: rank < 2 ? 10 : 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 22,
                height: 22,
                decoration: BoxDecoration(
                  color: barColor.withValues(alpha: isTop ? 0.15 : 0.08),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Center(
                  child: Text(
                    '${rank + 1}',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w800,
                      color: barColor,
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  pred.diseaseName,
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: isTop ? FontWeight.w700 : FontWeight.w500,
                    color: context.textPrimary,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              Text(
                '$pct%',
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  color: barColor,
                ),
              ),
            ],
          ),
          const SizedBox(height: 5),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: pred.confidence,
              backgroundColor: barColor.withValues(alpha: 0.08),
              color: barColor,
              minHeight: 5,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDetail(
    BuildContext context, {
    required IconData icon,
    required String title,
    required String text,
    required Color color,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: context.surfaceCard,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: context.cardShadow,
            blurRadius: 10,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(13),
            ),
            child: Icon(icon, color: color, size: 22),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                    color: context.textPrimary,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  text,
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _MetaChip extends StatelessWidget {
  const _MetaChip({required this.icon, required this.label});

  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: context.tintedSurface,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: AppColors.aqua.withValues(alpha: 0.15)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: AppColors.ocean),
          const SizedBox(width: 5),
          Flexible(
            child: Text(
              label,
              overflow: TextOverflow.ellipsis,
              maxLines: 1,
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: AppColors.ocean,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
