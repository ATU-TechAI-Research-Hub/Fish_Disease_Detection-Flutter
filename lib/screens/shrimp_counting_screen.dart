import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';

import '../models/shrimp_analysis_model.dart';
import '../services/image_quality.dart';
import '../services/scan_flow.dart';
import '../services/shrimp_api_service.dart';
import '../theme/app_theme.dart';
import '../widgets/section_header.dart';

enum _ResultImageView { annotated, original, waterMask, waterGated }

class ShrimpCountingScreen extends StatefulWidget {
  const ShrimpCountingScreen({super.key});

  @override
  State<ShrimpCountingScreen> createState() => _ShrimpCountingScreenState();
}

class _ShrimpCountingScreenState extends State<ShrimpCountingScreen> {
  final ShrimpApiService _api = ShrimpApiService();
  final TextEditingController _referenceMm = TextEditingController();
  final TextEditingController _referencePx = TextEditingController();

  String? _imagePath;
  ImageQualityReport? _quality;
  ShrimpAnalysisModel? _result;
  ShrimpServiceStatus? _serviceStatus;
  String? _error;
  String? _errorCode;
  bool _analyzing = false;
  int _progressStage = 0;
  Timer? _progressTimer;
  _ResultImageView _imageView = _ResultImageView.annotated;

  @override
  void initState() {
    super.initState();
    unawaited(_loadServiceStatus());
  }

  @override
  void dispose() {
    _progressTimer?.cancel();
    _referenceMm.dispose();
    _referencePx.dispose();
    super.dispose();
  }

  Future<void> _loadServiceStatus() async {
    try {
      final status = await _api.status();
      if (mounted) setState(() => _serviceStatus = status);
    } on ShrimpApiException {
      // General backend connectivity is already surfaced by AquaScan's
      // persistent status banner and by the analysis request itself.
    }
  }

  Future<void> _pickImage({required bool camera}) async {
    final Future<String?> Function(BuildContext) picker =
        camera ? ScanFlow.captureImage : ScanFlow.selectGalleryImage;
    final path = await picker(context);
    if (!mounted || path == null) return;
    setState(() {
      _imagePath = path;
      _quality = null;
      _result = null;
      _error = null;
      _errorCode = null;
    });
    final quality = await checkImageQuality(path);
    if (mounted && _imagePath == path) {
      setState(() => _quality = quality);
    }
  }

  void _removeImage() {
    setState(() {
      _imagePath = null;
      _quality = null;
      _result = null;
      _error = null;
      _errorCode = null;
      _imageView = _ResultImageView.annotated;
    });
  }

  double? _optionalNumber(TextEditingController controller) {
    final value = controller.text.trim();
    return value.isEmpty ? null : double.tryParse(value);
  }

  Future<void> _analyze() async {
    if (_serviceStatus case final status? when !status.ready) {
      setState(() {
        _error = status.message ??
            'The trained shrimp segmentation checkpoint is not configured.';
        _errorCode = 'model_unavailable';
      });
      return;
    }

    final imagePath = _imagePath;
    if (imagePath == null) {
      ScanFlow.showInfo(context, 'Take a photo or upload an image first.');
      return;
    }

    final mmText = _referenceMm.text.trim();
    final pxText = _referencePx.text.trim();
    final mm = _optionalNumber(_referenceMm);
    final px = _optionalNumber(_referencePx);
    if ((mmText.isEmpty != pxText.isEmpty) ||
        (mmText.isNotEmpty && (mm == null || mm <= 0)) ||
        (pxText.isNotEmpty && (px == null || px <= 0))) {
      setState(() {
        _error =
            'Calibration needs both a positive reference length in millimeters '
            'and its measured pixel length.';
        _errorCode = 'invalid_calibration';
      });
      return;
    }

    setState(() {
      _analyzing = true;
      _progressStage = 0;
      _result = null;
      _error = null;
      _errorCode = null;
    });
    _progressTimer?.cancel();
    _progressTimer = Timer.periodic(const Duration(seconds: 3), (_) {
      if (mounted && _progressStage < 4) {
        setState(() => _progressStage++);
      }
    });

    try {
      final result = await _api.analyze(
        imagePath,
        calibrationReferenceMm: mm,
        calibrationReferencePx: px,
      );
      if (mounted) {
        setState(() {
          _result = result;
          _imageView = _ResultImageView.annotated;
        });
      }
    } on ShrimpApiException catch (error) {
      if (mounted) {
        setState(() {
          _error = error.message;
          _errorCode = error.code;
        });
      }
    } catch (error) {
      if (mounted) {
        setState(() {
          _error = 'Shrimp analysis failed: $error';
          _errorCode = 'processing_failed';
        });
      }
    } finally {
      _progressTimer?.cancel();
      if (mounted) setState(() => _analyzing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Shrimp Counting & Analysis')),
      body: CustomScrollView(
        slivers: [
          SliverPadding(
            padding: const EdgeInsets.fromLTRB(20, 8, 20, 40),
            sliver: SliverList(
              delegate: SliverChildListDelegate([
                _buildProtocolCard(context),
                if (_serviceStatus case final status? when !status.ready) ...[
                  const SizedBox(height: 12),
                  _InlineNotice(
                    icon: Icons.memory_rounded,
                    color: AppColors.coral,
                    text: status.message ??
                        'Shrimp model setup is required before analysis.',
                  ),
                ],
                const SizedBox(height: 24),
                const SectionHeader(
                  title: 'Capture image',
                  subtitle:
                      'Use a top-down photo with the entire bowl visible.',
                ),
                const SizedBox(height: 12),
                _buildCaptureCard(context),
                if (_imagePath != null) ...[
                  const SizedBox(height: 14),
                  _buildCalibrationCard(context),
                  const SizedBox(height: 14),
                  FilledButton.icon(
                    key: const Key('analyze_shrimp_action'),
                    onPressed: _analyzing || _serviceStatus?.ready == false
                        ? null
                        : _analyze,
                    icon: const Icon(Icons.auto_awesome_rounded),
                    label: Text(
                      _analyzing
                          ? 'Analyzing Shrimp…'
                          : _serviceStatus?.ready == false
                              ? 'Model Setup Required'
                              : 'Analyze Shrimp',
                    ),
                    style: FilledButton.styleFrom(
                      minimumSize: const Size.fromHeight(54),
                      backgroundColor: AppColors.teal,
                    ),
                  ),
                ],
                if (_analyzing) ...[
                  const SizedBox(height: 18),
                  _AnalysisProgress(stage: _progressStage),
                ],
                if (_error != null) ...[
                  const SizedBox(height: 18),
                  _buildErrorCard(context),
                ],
                if (_result != null) ...[
                  const SizedBox(height: 28),
                  _buildResults(context, _result!),
                ],
              ]),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildProtocolCard(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            AppColors.ocean,
            AppColors.teal.withValues(alpha: 0.88),
          ],
        ),
        borderRadius: BorderRadius.circular(20),
      ),
      child: const Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.center_focus_strong_rounded,
              color: Colors.white, size: 28),
          SizedBox(width: 14),
          Expanded(
            child: Text(
              'Place shrimp in a white bowl containing water. Capture a clear '
              'top-down photograph with the entire bowl visible. Keep camera '
              'distance and orientation consistent and avoid glare.',
              style: TextStyle(color: Colors.white, height: 1.45, fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCaptureCard(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: context.surfaceCard,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: context.subtleBorder),
      ),
      child: Column(
        children: [
          if (_imagePath == null)
            Container(
              height: 190,
              decoration: BoxDecoration(
                color: context.tintedSurface,
                borderRadius: BorderRadius.circular(16),
              ),
              child: const Center(
                child: Icon(
                  Icons.water_drop_outlined,
                  size: 64,
                  color: AppColors.aqua,
                ),
              ),
            )
          else
            ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: SizedBox(
                height: 260,
                width: double.infinity,
                child: InteractiveViewer(
                  minScale: 1,
                  maxScale: 4,
                  child: Image.file(
                    File(_imagePath!),
                    width: double.infinity,
                    fit: BoxFit.contain,
                    errorBuilder: (_, __, ___) => const Center(
                      child: Icon(Icons.broken_image_rounded, size: 48),
                    ),
                  ),
                ),
              ),
            ),
          if (_quality != null && !_quality!.isAcceptable) ...[
            const SizedBox(height: 12),
            _InlineNotice(
              icon: Icons.warning_amber_rounded,
              color: AppColors.amber,
              text: _quality!.warnings.join('\n'),
            ),
          ],
          const SizedBox(height: 14),
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  key: const Key('shrimp_camera_action'),
                  onPressed: ScanFlow.cameraSupported
                      ? () => _pickImage(camera: true)
                      : null,
                  icon: const Icon(Icons.camera_alt_rounded),
                  label: Text(_imagePath == null ? 'Take Photo' : 'Retake'),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: OutlinedButton.icon(
                  key: const Key('shrimp_gallery_action'),
                  onPressed: () => _pickImage(camera: false),
                  icon: const Icon(Icons.photo_library_rounded),
                  label: const Text('Upload Image'),
                ),
              ),
            ],
          ),
          if (_imagePath != null) ...[
            const SizedBox(height: 6),
            TextButton.icon(
              onPressed: _analyzing ? null : _removeImage,
              icon: const Icon(Icons.delete_outline_rounded),
              label: const Text('Remove image'),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildCalibrationCard(BuildContext context) {
    return Card(
      child: ExpansionTile(
        leading: const Icon(Icons.straighten_rounded, color: AppColors.purple),
        title: const Text('Optional physical calibration'),
        subtitle: const Text(
          'Only enter values when a measured fiducial is visible.',
        ),
        childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
        children: [
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _referenceMm,
                  keyboardType:
                      const TextInputType.numberWithOptions(decimal: true),
                  decoration: const InputDecoration(
                    labelText: 'Reference length (mm)',
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: TextField(
                  controller: _referencePx,
                  keyboardType:
                      const TextInputType.numberWithOptions(decimal: true),
                  decoration: const InputDecoration(
                    labelText: 'Reference length (px)',
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildErrorCard(BuildContext context) {
    final waterFailure = _errorCode == 'water_isolation_failed';
    final modelFailure = _errorCode == 'model_unavailable';
    return _InlineNotice(
      icon: waterFailure
          ? Icons.water_drop_outlined
          : modelFailure
              ? Icons.memory_rounded
              : Icons.error_outline_rounded,
      color: AppColors.coral,
      text: _error!,
    );
  }

  Widget _buildResults(BuildContext context, ShrimpAnalysisModel result) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SectionHeader(
          title: 'Analysis results',
          subtitle: 'Review masks before relying on the count or measurements.',
        ),
        const SizedBox(height: 14),
        _buildMetrics(context, result),
        if (result.warnings.isNotEmpty) ...[
          const SizedBox(height: 14),
          ...result.warnings.map(
            (warning) => Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: _InlineNotice(
                icon: Icons.info_outline_rounded,
                color:
                    result.totalShrimp == 0 ? AppColors.coral : AppColors.amber,
                text: warning,
              ),
            ),
          ),
        ],
        const SizedBox(height: 24),
        const SectionHeaderAccent(title: 'Visual inspection'),
        const SizedBox(height: 10),
        _buildImageInspector(context, result),
        const SizedBox(height: 24),
        const SectionHeader(
          title: 'Individual measurements',
          subtitle: 'Major-axis ellipse estimates in original-image pixels.',
        ),
        const SizedBox(height: 10),
        if (result.detections.isEmpty)
          _InlineNotice(
            icon: Icons.search_off_rounded,
            color: AppColors.coral,
            text: 'No shrimp were detected. Please check the image quality, '
                'ensure the shrimp are visible inside the water, and try again.',
          )
        else
          ...result.detections.map(
            (detection) => Padding(
              padding: const EdgeInsets.only(bottom: 10),
              child: _DetectionCard(detection: detection),
            ),
          ),
        const SizedBox(height: 14),
        _InlineNotice(
          icon: Icons.science_outlined,
          color: AppColors.seaBlue,
          text: '${result.methodologyNote} Touching or overlapping shrimp can '
              'merge, pale shrimp can be missed, and glare can corrupt the '
              'water mask.',
        ),
        const SizedBox(height: 18),
        OutlinedButton.icon(
          onPressed: _removeImage,
          icon: const Icon(Icons.refresh_rounded),
          label: const Text('Analyze Another Image'),
          style: OutlinedButton.styleFrom(
            minimumSize: const Size.fromHeight(50),
          ),
        ),
      ],
    );
  }

  Widget _buildMetrics(BuildContext context, ShrimpAnalysisModel result) {
    String px(double? value) =>
        value == null ? '—' : '${value.toStringAsFixed(1)} px';
    final confidence = result.averageConfidence == null
        ? '—'
        : '${(result.averageConfidence! * 100).toStringAsFixed(1)}%';
    final metrics = [
      ('Total shrimp', '${result.totalShrimp}', Icons.set_meal_rounded),
      ('Average length', px(result.averageLengthPx), Icons.straighten_rounded),
      ('Minimum length', px(result.minimumLengthPx), Icons.south_east_rounded),
      ('Maximum length', px(result.maximumLengthPx), Icons.north_east_rounded),
      ('Avg confidence', confidence, Icons.verified_outlined),
      (
        'Processing time',
        '${result.processingTime.toStringAsFixed(1)} s',
        Icons.timer_outlined,
      ),
    ];
    return LayoutBuilder(
      builder: (context, constraints) {
        final width = (constraints.maxWidth - 10) / 2;
        return Wrap(
          spacing: 10,
          runSpacing: 10,
          children: metrics
              .map(
                (metric) => SizedBox(
                  width: width,
                  child: _MetricCard(
                    label: metric.$1,
                    value: metric.$2,
                    icon: metric.$3,
                  ),
                ),
              )
              .toList(growable: false),
        );
      },
    );
  }

  Widget _buildImageInspector(
    BuildContext context,
    ShrimpAnalysisModel result,
  ) {
    final options = <_ResultImageView, (String, String)>{
      _ResultImageView.annotated: ('Annotated', result.images.annotatedUrl),
      _ResultImageView.original: ('Original', result.images.originalUrl),
      _ResultImageView.waterMask: ('Water mask', result.images.waterMaskUrl),
      _ResultImageView.waterGated: (
        'Water-gated',
        result.images.waterGatedUrl,
      ),
    };
    final selected = options[_imageView]!;
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: context.surfaceCard,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: context.subtleBorder),
      ),
      child: Column(
        children: [
          Wrap(
            spacing: 7,
            runSpacing: 7,
            children: options.entries
                .map(
                  (entry) => ChoiceChip(
                    label: Text(entry.value.$1),
                    selected: _imageView == entry.key,
                    onSelected: (_) => setState(() => _imageView = entry.key),
                  ),
                )
                .toList(growable: false),
          ),
          const SizedBox(height: 12),
          ClipRRect(
            borderRadius: BorderRadius.circular(14),
            child: SizedBox(
              height: 310,
              width: double.infinity,
              child: InteractiveViewer(
                minScale: 1,
                maxScale: 6,
                child: Image.network(
                  selected.$2,
                  key: ValueKey(selected.$2),
                  fit: BoxFit.contain,
                  loadingBuilder: (context, child, progress) => progress == null
                      ? child
                      : const Center(child: CircularProgressIndicator()),
                  errorBuilder: (_, __, ___) => const Center(
                    child: Text('Result image expired or could not be loaded.'),
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Pinch to zoom • ${selected.$1}',
            style: TextStyle(fontSize: 12, color: context.textSecondary),
          ),
        ],
      ),
    );
  }
}

class _AnalysisProgress extends StatelessWidget {
  const _AnalysisProgress({required this.stage});

  final int stage;

  static const stages = [
    'Processing image',
    'Isolating water using SAM 3',
    'Detecting shrimp using YOLO11',
    'Calculating shrimp measurements',
    'Preparing results',
  ];

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: context.surfaceCard,
        borderRadius: BorderRadius.circular(18),
      ),
      child: Row(
        children: [
          const CircularProgressIndicator(color: AppColors.teal),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  stages[stage.clamp(0, stages.length - 1)],
                  style: const TextStyle(fontWeight: FontWeight.w800),
                ),
                const SizedBox(height: 4),
                Text(
                  'The first run may take longer while model weights load.',
                  style: TextStyle(fontSize: 12, color: context.textSecondary),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _MetricCard extends StatelessWidget {
  const _MetricCard({
    required this.label,
    required this.value,
    required this.icon,
  });

  final String label;
  final String value;
  final IconData icon;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: context.surfaceCard,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: context.subtleBorder),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 20, color: AppColors.teal),
          const SizedBox(height: 8),
          Text(
            value,
            style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w800),
          ),
          Text(
            label,
            style: TextStyle(fontSize: 12, color: context.textSecondary),
          ),
        ],
      ),
    );
  }
}

class _DetectionCard extends StatelessWidget {
  const _DetectionCard({required this.detection});

  final ShrimpDetectionModel detection;

  @override
  Widget build(BuildContext context) {
    final px = detection.lengthPx == null
        ? 'Not measurable'
        : '${detection.lengthPx!.toStringAsFixed(1)} px';
    final mm = detection.lengthMm == null
        ? null
        : '${detection.lengthMm!.toStringAsFixed(2)} mm';
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: context.surfaceCard,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: context.subtleBorder),
      ),
      child: Row(
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(10),
            child: Container(
              width: 58,
              height: 58,
              color: Colors.black,
              child: Image.network(
                detection.maskUrl,
                fit: BoxFit.contain,
                errorBuilder: (_, __, ___) => const Icon(
                    Icons.hide_image_outlined,
                    color: Colors.white54),
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Shrimp ${detection.id}',
                  style: const TextStyle(fontWeight: FontWeight.w800),
                ),
                const SizedBox(height: 3),
                Text(
                  '${(detection.confidence * 100).toStringAsFixed(1)}% confidence',
                  style: TextStyle(fontSize: 12, color: context.textSecondary),
                ),
                Text(
                  mm == null ? px : '$px • $mm',
                  style: const TextStyle(
                    fontSize: 13,
                    color: AppColors.seaBlue,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                if (detection.measurementNote != null)
                  Text(
                    detection.measurementNote!,
                    style:
                        const TextStyle(fontSize: 11, color: AppColors.amber),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _InlineNotice extends StatelessWidget {
  const _InlineNotice({
    required this.icon,
    required this.color,
    required this.text,
  });

  final IconData icon;
  final Color color;
  final String text;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(13),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color.withValues(alpha: 0.28)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color, size: 21),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              text,
              style: TextStyle(
                color: context.textPrimary,
                fontSize: 13,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
