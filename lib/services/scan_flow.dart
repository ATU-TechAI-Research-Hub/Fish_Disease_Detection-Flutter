import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';

import '../screens/scan_preview_screen.dart';
import '../theme/app_theme.dart';
import 'backend_status_service.dart';

/// Picks images and routes through preview before analysis.
class ScanFlow {
  ScanFlow._();

  static final ImagePicker _picker = ImagePicker();

  static bool get cameraSupported => Platform.isAndroid || Platform.isIOS;

  static bool get backendReachable => BackendStatusService.instance.isReachable;

  static void showBackendError(BuildContext context) {
    _snack(
      context,
      'The AquaScan backend is not reachable. '
      'Start the server (run_backend.bat) and try again.',
      AppColors.coral,
    );
  }

  static void showInfo(BuildContext context, String msg) {
    _snack(context, msg, AppColors.seaBlue);
  }

  static void _snack(BuildContext context, String msg, Color bg) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        backgroundColor: bg,
        content: Text(msg, style: const TextStyle(color: Colors.white)),
      ),
    );
  }

  static Future<bool> _ensureCameraPermission(BuildContext context) async {
    if (!cameraSupported) return true;

    var status = await Permission.camera.status;
    if (status.isGranted) return true;

    status = await Permission.camera.request();
    if (status.isGranted) return true;

    if (!context.mounted) return false;

    if (status.isPermanentlyDenied) {
      await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          title: const Text('Camera Permission Required'),
          content: const Text(
            'Camera access was denied. Please enable it in settings.',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () {
                Navigator.pop(ctx);
                openAppSettings();
              },
              child: const Text('Open Settings'),
            ),
          ],
        ),
      );
      return false;
    }
    return false;
  }

  static Future<void> scanWithCamera(BuildContext context) async {
    if (!backendReachable) {
      showBackendError(context);
      return;
    }

    final path = await captureImage(context);
    if (!context.mounted || path == null) return;
    await _openPreview(context, path);
  }

  /// Capture an image without choosing what analysis flow receives it.
  ///
  /// Fish diagnosis and shrimp analysis both use this shared permission and
  /// picker implementation, then route the selected path independently.
  static Future<String?> captureImage(BuildContext context) async {
    if (!cameraSupported) {
      showInfo(
        context,
        'Camera capture is only available on Android and iOS. '
        'Please use Gallery on this device.',
      );
      return null;
    }

    try {
      final hasPermission = await _ensureCameraPermission(context);
      if (!hasPermission || !context.mounted) return null;

      final XFile? photo = await _picker.pickImage(
        source: ImageSource.camera,
        imageQuality: 90,
        maxWidth: 1500,
        preferredCameraDevice: CameraDevice.rear,
      );
      if (!context.mounted || photo == null) return null;
      return photo.path;
    } catch (e) {
      if (!context.mounted) return null;
      _snack(context, 'Could not open camera: $e', AppColors.coral);
      return null;
    }
  }

  static Future<void> pickFromGallery(BuildContext context) async {
    if (!backendReachable) {
      showBackendError(context);
      return;
    }

    final path = await selectGalleryImage(context);
    if (!context.mounted || path == null) return;
    await _openPreview(context, path);
  }

  /// Pick a gallery image for either fish or shrimp analysis.
  static Future<String?> selectGalleryImage(BuildContext context) async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
        imageQuality: 90,
        maxWidth: 1500,
      );
      if (!context.mounted || image == null) return null;
      return image.path;
    } catch (e) {
      if (!context.mounted) return null;
      _snack(context, 'Could not open gallery: $e', AppColors.coral);
      return null;
    }
  }

  static Future<void> _openPreview(BuildContext context, String path) {
    return Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => ScanPreviewScreen(imagePath: path),
      ),
    );
  }
}
