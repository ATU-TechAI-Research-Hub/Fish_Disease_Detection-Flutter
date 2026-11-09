import 'package:flutter/material.dart';

import '../services/backend_status_service.dart';
import '../theme/app_theme.dart';

/// Compact banner that reflects [BackendStatusService] state.
///
/// Reuses one place to render the offline/degraded indicator so the UI stays
/// consistent across screens (home, result, history, library).
class BackendStatusBanner extends StatefulWidget {
  const BackendStatusBanner({super.key, this.compact = false});

  /// Compact mode renders a small pill suitable for app headers.
  final bool compact;

  @override
  State<BackendStatusBanner> createState() => _BackendStatusBannerState();
}

class _BackendStatusBannerState extends State<BackendStatusBanner> {
  @override
  void initState() {
    super.initState();
    BackendStatusService.instance.addListener(_onChanged);
  }

  @override
  void dispose() {
    BackendStatusService.instance.removeListener(_onChanged);
    super.dispose();
  }

  void _onChanged() {
    if (mounted) setState(() {});
  }

  ({IconData icon, String label, Color color, String tooltip}) _statusFor(
      BackendStatus s) {
    switch (s) {
      case BackendStatus.online:
        return (
          icon: Icons.cloud_done_rounded,
          label: 'AI online',
          color: AppColors.emerald,
          tooltip:
              'Backend reachable and the disease detection model is loaded.',
        );
      case BackendStatus.degraded:
        return (
          icon: Icons.warning_amber_rounded,
          label: 'Model not loaded',
          color: AppColors.amber,
          tooltip: 'Backend is running but no model.h5/ONNX file was found. '
              'Place model.h5 in the /model folder or run training.',
        );
      case BackendStatus.offline:
        return (
          icon: Icons.cloud_off_rounded,
          label: 'Offline mode',
          color: AppColors.coral,
          tooltip:
              'Cannot reach the AquaScan backend. Make sure run_backend.bat '
              'is running on your computer.',
        );
      case BackendStatus.unknown:
        return (
          icon: Icons.cloud_sync_rounded,
          label: 'Checking…',
          color: AppColors.seaBlue,
          tooltip: 'Probing the backend connection…',
        );
    }
  }

  @override
  Widget build(BuildContext context) {
    final service = BackendStatusService.instance;
    final info = _statusFor(service.status);

    if (widget.compact) {
      return Semantics(
        label: '${info.label}. ${info.tooltip}',
        child: Tooltip(
          message: info.tooltip,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.14),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(info.icon, color: info.color, size: 14),
                const SizedBox(width: 6),
                Text(
                  info.label,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
          ),
        ),
      );
    }

    return Semantics(
      label: '${info.label}. ${info.tooltip}',
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: info.color.withValues(alpha: 0.08),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: info.color.withValues(alpha: 0.25)),
        ),
        child: Row(
          children: [
            Icon(info.icon, color: info.color, size: 22),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    info.label,
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w800,
                      color: info.color,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    info.tooltip,
                    style: const TextStyle(
                      fontSize: 12,
                      color: AppColors.textSecondary,
                      height: 1.4,
                    ),
                  ),
                ],
              ),
            ),
            IconButton(
              tooltip: 'Re-check connection',
              onPressed: () => BackendStatusService.instance.probe(),
              icon: Icon(Icons.refresh_rounded,
                  color: info.color.withValues(alpha: 0.9)),
            ),
          ],
        ),
      ),
    );
  }
}
