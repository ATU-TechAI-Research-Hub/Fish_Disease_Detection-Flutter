import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';

import '../theme/app_theme.dart';
import '../widgets/bubble_background.dart';
import '../widgets/wave_clipper.dart';
import 'result_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();
  bool _isBusy = false;

  Future<bool> _ensureCameraPermission() async {
    if (!Platform.isAndroid && !Platform.isIOS) return true;

    var status = await Permission.camera.status;
    if (status.isGranted) return true;

    status = await Permission.camera.request();
    if (status.isGranted) return true;

    if (!mounted) return false;

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

  Future<void> _scanWithCamera() async {
    if (_isBusy) return;
    setState(() => _isBusy = true);

    try {
      final hasPermission = await _ensureCameraPermission();
      if (!hasPermission || !mounted) return;

      final XFile? photo = await _picker.pickImage(
        source: ImageSource.camera,
        imageQuality: 90,
        maxWidth: 1500,
        preferredCameraDevice: CameraDevice.rear,
      );
      if (!mounted || photo == null) return;

      await Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (_) => ResultScreen(imagePath: photo.path),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      _showError('Could not open camera: $e');
    } finally {
      if (mounted) setState(() => _isBusy = false);
    }
  }

  Future<void> _pickFromGallery() async {
    if (_isBusy) return;
    setState(() => _isBusy = true);

    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
        imageQuality: 90,
        maxWidth: 1500,
      );
      if (!mounted || image == null) return;

      await Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (_) => ResultScreen(imagePath: image.path),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      _showError('Could not open gallery: $e');
    } finally {
      if (mounted) setState(() => _isBusy = false);
    }
  }

  void _showError(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        backgroundColor: AppColors.coral,
        content: Text(msg, style: const TextStyle(color: Colors.white)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.surface,
      body: SingleChildScrollView(
        child: Column(
          children: [
            _buildWaveHeader(context),
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 32),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 20),
                  Text('Scan Your Fish',
                      style: Theme.of(context).textTheme.headlineSmall),
                  const SizedBox(height: 6),
                  Text(
                    'Choose a method to capture the fish image for AI disease analysis.',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                  const SizedBox(height: 20),
                  _buildScanButton(
                    icon: Icons.camera_alt_rounded,
                    title: _isBusy ? 'Opening...' : 'Take a Photo',
                    subtitle: 'Use camera to capture a live fish image',
                    colors: [AppColors.seaBlue, AppColors.aqua],
                    onTap: _isBusy ? null : _scanWithCamera,
                  ),
                  const SizedBox(height: 14),
                  _buildScanButton(
                    icon: Icons.photo_library_rounded,
                    title: _isBusy ? 'Opening...' : 'Upload from Gallery',
                    subtitle: 'Select an existing fish photo',
                    colors: [AppColors.teal, AppColors.seafoam],
                    onTap: _isBusy ? null : _pickFromGallery,
                  ),
                  const SizedBox(height: 32),
                  Text('How It Works',
                      style: Theme.of(context).textTheme.headlineSmall),
                  const SizedBox(height: 16),
                  _buildStep(
                    number: '1',
                    icon: Icons.add_a_photo_rounded,
                    title: 'Capture',
                    desc: 'Take a photo or pick from gallery',
                    color: AppColors.seaBlue,
                  ),
                  _buildStep(
                    number: '2',
                    icon: Icons.auto_awesome_rounded,
                    title: 'AI Analysis',
                    desc: 'CNN model classifies the disease',
                    color: AppColors.purple,
                    showLine: true,
                  ),
                  _buildStep(
                    number: '3',
                    icon: Icons.fact_check_rounded,
                    title: 'Get Results',
                    desc: 'View disease info and treatment',
                    color: AppColors.emerald,
                    showLine: true,
                    isLast: true,
                  ),
                  const SizedBox(height: 28),
                  _buildInfoRow(context),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildWaveHeader(BuildContext context) {
    return BubbleBackground(
      child: WaveHeader(
        height: 260,
        colors: const [AppColors.deepOcean, AppColors.seaBlue],
        child: SafeArea(
          bottom: false,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(24, 8, 24, 32),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      width: 40,
                      height: 40,
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha: 0.15),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Icon(Icons.set_meal_rounded,
                          color: Colors.white, size: 22),
                    ),
                    const SizedBox(width: 10),
                    const Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'AquaScan',
                          style: TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.w800,
                            color: Colors.white,
                            letterSpacing: -0.5,
                          ),
                        ),
                        Text(
                          'Fish Disease Detection',
                          style: TextStyle(
                            fontSize: 12,
                            color: Color(0xFFB0D9F0),
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
                const Spacer(),
                const Text(
                  'Identify Fish Diseases\nwith AI in Seconds',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.w800,
                    color: Colors.white,
                    height: 1.2,
                    letterSpacing: -0.3,
                  ),
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    _HeaderChip(
                        label: '7 Classes', icon: Icons.category_rounded),
                    const SizedBox(width: 8),
                    _HeaderChip(
                        label: '79% Accuracy', icon: Icons.verified_rounded),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildScanButton({
    required IconData icon,
    required String title,
    required String subtitle,
    required List<Color> colors,
    VoidCallback? onTap,
  }) {
    final isDisabled = onTap == null;
    return AnimatedOpacity(
      opacity: isDisabled ? 0.55 : 1.0,
      duration: const Duration(milliseconds: 200),
      child: GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: colors,
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(20),
          boxShadow: [
            BoxShadow(
              color: colors.first.withValues(alpha: 0.25),
              blurRadius: 16,
              offset: const Offset(0, 6),
            ),
          ],
        ),
        child: Row(
          children: [
            Container(
              width: 54,
              height: 54,
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Icon(icon, color: Colors.white, size: 28),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      fontSize: 17,
                      fontWeight: FontWeight.w700,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 3),
                  Text(
                    subtitle,
                    style: TextStyle(
                      fontSize: 13,
                      color: Colors.white.withValues(alpha: 0.85),
                    ),
                  ),
                ],
              ),
            ),
            Icon(Icons.arrow_forward_ios_rounded,
                color: Colors.white.withValues(alpha: 0.5), size: 18),
          ],
        ),
      ),
    ),
    );
  }

  Widget _buildStep({
    required String number,
    required IconData icon,
    required String title,
    required String desc,
    required Color color,
    bool showLine = false,
    bool isLast = false,
  }) {
    return IntrinsicHeight(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Column(
            children: [
              if (showLine)
                Container(
                  width: 2,
                  height: 12,
                  color: color.withValues(alpha: 0.2),
                ),
              Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: color.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(13),
                  border: Border.all(color: color.withValues(alpha: 0.3)),
                ),
                child: Icon(icon, color: color, size: 20),
              ),
              if (!isLast)
                Expanded(
                  child: Container(
                    width: 2,
                    color: color.withValues(alpha: 0.2),
                  ),
                ),
            ],
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Padding(
              padding: EdgeInsets.only(
                top: showLine ? 12 : 0,
                bottom: isLast ? 0 : 16,
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w700,
                      color: AppColors.textPrimary,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    desc,
                    style: TextStyle(
                      fontSize: 14,
                      color: AppColors.textSecondary,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildInfoRow(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.wave,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.aqua.withValues(alpha: 0.15)),
      ),
      child: Row(
        children: [
          Icon(Icons.info_outline_rounded,
              color: AppColors.seaBlue, size: 20),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              'Trained on 2,444 freshwater fish images from South Asian aquaculture. Detects 6 diseases + healthy fish.',
              style: TextStyle(
                fontSize: 13,
                color: AppColors.ocean,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _HeaderChip extends StatelessWidget {
  const _HeaderChip({required this.label, required this.icon});

  final String label;
  final IconData icon;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: AppColors.seafoam, size: 14),
          const SizedBox(width: 5),
          Text(
            label,
            style: TextStyle(
              color: Colors.white.withValues(alpha: 0.9),
              fontSize: 11,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}
