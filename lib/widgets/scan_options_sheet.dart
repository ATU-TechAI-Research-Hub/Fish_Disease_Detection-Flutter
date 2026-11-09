import 'dart:io';

import 'package:flutter/material.dart';

import '../theme/app_theme.dart';
import 'primary_action_card.dart';

/// Bottom sheet for choosing camera vs gallery — used from the center FAB.
class ScanOptionsSheet extends StatelessWidget {
  const ScanOptionsSheet({
    required this.onCamera,
    required this.onGallery,
    this.cameraEnabled = true,
    super.key,
  });

  final VoidCallback? onCamera;
  final VoidCallback? onGallery;
  final bool cameraEnabled;

  static Future<void> show(
    BuildContext context, {
    required VoidCallback? onCamera,
    required VoidCallback? onGallery,
    bool cameraEnabled = true,
  }) {
    return showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) => ScanOptionsSheet(
        onCamera: onCamera,
        onGallery: onGallery,
        cameraEnabled: cameraEnabled,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final bottom = MediaQuery.paddingOf(context).bottom;
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 0, 16, 16),
      padding: EdgeInsets.fromLTRB(20, 12, 20, 20 + bottom),
      decoration: BoxDecoration(
        color: context.surfaceCard,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: AppColors.deepOcean.withValues(alpha: 0.15),
            blurRadius: 24,
            offset: const Offset(0, -4),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 40,
            height: 4,
            decoration: BoxDecoration(
              color: context.subtleBorder,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(height: 20),
          Text(
            'Scan a fish',
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
                  fontWeight: FontWeight.w800,
                ),
          ),
          const SizedBox(height: 6),
          Text(
            'Choose how you want to add an image for AI analysis.',
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.bodyMedium,
          ),
          const SizedBox(height: 20),
          Row(
            children: [
              Expanded(
                child: SizedBox(
                  height: 140,
                  child: PrimaryActionCard(
                    icon: Icons.camera_alt_rounded,
                    title: 'Camera',
                    subtitle:
                        cameraEnabled && (Platform.isAndroid || Platform.isIOS)
                            ? 'Take a live photo'
                            : 'Mobile only',
                    colors: const [AppColors.seaBlue, AppColors.aqua],
                    onTap: cameraEnabled ? onCamera : null,
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: SizedBox(
                  height: 140,
                  child: PrimaryActionCard(
                    icon: Icons.photo_library_rounded,
                    title: 'Gallery',
                    subtitle: 'Pick saved image',
                    colors: const [AppColors.teal, AppColors.emerald],
                    onTap: onGallery,
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
