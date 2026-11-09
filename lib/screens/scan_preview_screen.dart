import 'dart:io';

import 'package:flutter/material.dart';

import '../services/image_quality.dart';
import '../theme/app_theme.dart';
import 'result_screen.dart';

/// Review the captured image before sending it to the AI backend.
/// Runs a local quality check (resolution, aspect ratio, file size) and
/// surfaces soft warnings so users can retake poor photos before uploading.
class ScanPreviewScreen extends StatefulWidget {
  const ScanPreviewScreen({required this.imagePath, super.key});

  final String imagePath;

  @override
  State<ScanPreviewScreen> createState() => _ScanPreviewScreenState();
}

class _ScanPreviewScreenState extends State<ScanPreviewScreen> {
  late final Future<ImageQualityReport> _quality;

  String get imagePath => widget.imagePath;

  @override
  void initState() {
    super.initState();
    _quality = checkImageQuality(imagePath);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.deepOcean,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        foregroundColor: Colors.white,
        elevation: 0,
        title: const Text(
          'Preview',
          style: TextStyle(
            fontWeight: FontWeight.w800,
            color: Colors.white,
          ),
        ),
      ),
      body: Column(
        children: [
          Expanded(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 16),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(24),
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    Image.file(
                      File(imagePath),
                      fit: BoxFit.cover,
                      errorBuilder: (_, __, ___) => Container(
                        color: AppColors.ocean,
                        child: const Icon(
                          Icons.broken_image_rounded,
                          color: Colors.white54,
                          size: 48,
                        ),
                      ),
                    ),
                    Positioned(
                      left: 0,
                      right: 0,
                      bottom: 0,
                      child: Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            begin: Alignment.topCenter,
                            end: Alignment.bottomCenter,
                            colors: [
                              Colors.transparent,
                              Colors.black.withValues(alpha: 0.65),
                            ],
                          ),
                        ),
                        child: Row(
                          children: [
                            Icon(
                              Icons.lightbulb_outline_rounded,
                              color: AppColors.seafoam,
                              size: 20,
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: Text(
                                'Use a clear, well-lit side view of the fish. '
                                'Avoid blur and heavy shadows.',
                                style: TextStyle(
                                  fontSize: 13,
                                  color: Colors.white.withValues(alpha: 0.92),
                                  height: 1.35,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          Container(
            width: double.infinity,
            padding: EdgeInsets.fromLTRB(
              20,
              20,
              20,
              20 + MediaQuery.paddingOf(context).bottom,
            ),
            decoration: BoxDecoration(
              color: context.surfaceCard,
              borderRadius:
                  const BorderRadius.vertical(top: Radius.circular(28)),
              boxShadow: [
                BoxShadow(
                  color: AppColors.deepOcean.withValues(alpha: 0.12),
                  blurRadius: 24,
                  offset: const Offset(0, -6),
                ),
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(
                  'Ready to analyze?',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                        fontWeight: FontWeight.w800,
                      ),
                ),
                const SizedBox(height: 6),
                Text(
                  'The image will be sent to your local AquaScan server '
                  'for CNN classification.',
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
                FutureBuilder<ImageQualityReport>(
                  future: _quality,
                  builder: (context, snapshot) {
                    final report = snapshot.data;
                    if (report == null || report.isAcceptable) {
                      return const SizedBox.shrink();
                    }
                    return Padding(
                      padding: const EdgeInsets.only(top: 12),
                      child: Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: AppColors.amber.withValues(alpha: 0.1),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                            color: AppColors.amber.withValues(alpha: 0.35),
                          ),
                        ),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Icon(Icons.warning_amber_rounded,
                                color: AppColors.amber, size: 20),
                            const SizedBox(width: 10),
                            Expanded(
                              child: Text(
                                report.warnings.join('\n'),
                                style: TextStyle(
                                  fontSize: 12.5,
                                  height: 1.4,
                                  color: context.textPrimary,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    );
                  },
                ),
                const SizedBox(height: 20),
                FilledButton.icon(
                  onPressed: () {
                    Navigator.of(context).pushReplacement(
                      MaterialPageRoute<void>(
                        builder: (_) => ResultScreen(imagePath: imagePath),
                      ),
                    );
                  },
                  icon: const Icon(Icons.auto_awesome_rounded),
                  label: const Text('Analyze Fish'),
                  style: FilledButton.styleFrom(
                    backgroundColor: AppColors.seaBlue,
                    minimumSize: const Size.fromHeight(54),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                  ),
                ),
                const SizedBox(height: 10),
                OutlinedButton.icon(
                  onPressed: () => Navigator.of(context).pop(),
                  icon: const Icon(Icons.refresh_rounded),
                  label: const Text('Choose Another'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: context.textSecondary,
                    minimumSize: const Size.fromHeight(50),
                    side: BorderSide(color: context.subtleBorder),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
