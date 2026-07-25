import 'package:flutter/material.dart';

import '../services/auth_service.dart';
import '../services/backend_status_service.dart';
import '../services/scan_flow.dart';
import '../theme/app_theme.dart';
import '../widgets/account_sheet.dart';
import '../widgets/backend_status_banner.dart';
import '../widgets/bubble_background.dart';
import '../widgets/disease_category_strip.dart';
import '../widgets/how_it_works_strip.dart';
import '../widgets/primary_action_card.dart';
import '../widgets/section_header.dart';
import '../widgets/stat_card.dart';
import '../widgets/wave_clipper.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _isBusy = false;

  @override
  void initState() {
    super.initState();
    BackendStatusService.instance.addListener(_onStatusChanged);
  }

  @override
  void dispose() {
    BackendStatusService.instance.removeListener(_onStatusChanged);
    super.dispose();
  }

  void _onStatusChanged() {
    if (mounted) setState(() {});
  }

  bool get _backendReachable => ScanFlow.backendReachable;

  Future<void> _runScan(Future<void> Function() action) async {
    if (_isBusy) return;
    setState(() => _isBusy = true);
    try {
      await action();
    } finally {
      if (mounted) setState(() => _isBusy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final reachable = _backendReachable;
    final busyLabel = _isBusy ? 'Opening…' : null;

    return Scaffold(
      body: CustomScrollView(
        slivers: [
          SliverToBoxAdapter(child: _buildHero(context)),
          SliverPadding(
            padding: const EdgeInsets.fromLTRB(20, 0, 20, 100),
            sliver: SliverList(
              delegate: SliverChildListDelegate([
                const BackendStatusBanner(),
                const SizedBox(height: 20),
                Row(
                  children: [
                    Expanded(
                      child: StatCard(
                        icon: Icons.category_rounded,
                        value: '7',
                        label: 'Classes',
                        accent: AppColors.seaBlue,
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: StatCard(
                        icon: Icons.psychology_rounded,
                        value: 'CNN',
                        label: 'AI Model',
                        accent: AppColors.purple,
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: StatCard(
                        icon: Icons.wifi_tethering_rounded,
                        value: reachable ? 'On' : 'Off',
                        label: 'Backend',
                        accent: reachable ? AppColors.emerald : AppColors.coral,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 28),
                const SectionHeader(
                  title: 'Start a scan',
                  subtitle:
                      'Capture or upload a clear fish photo for analysis.',
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    Expanded(
                      child: SizedBox(
                        height: 156,
                        child: PrimaryActionCard(
                          testKey: const Key('home_camera_action'),
                          icon: Icons.camera_alt_rounded,
                          title: busyLabel ?? 'Camera',
                          subtitle: ScanFlow.cameraSupported
                              ? 'Live capture'
                              : 'Mobile only',
                          colors: const [AppColors.seaBlue, AppColors.aqua],
                          onTap: (!_isBusy && reachable)
                              ? () => _runScan(
                                    () => ScanFlow.scanWithCamera(context),
                                  )
                              : null,
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: SizedBox(
                        height: 156,
                        child: PrimaryActionCard(
                          testKey: const Key('home_gallery_action'),
                          icon: Icons.photo_library_rounded,
                          title: busyLabel ?? 'Gallery',
                          subtitle: 'Saved images',
                          colors: const [AppColors.teal, AppColors.emerald],
                          onTap: (!_isBusy && reachable)
                              ? () => _runScan(
                                    () => ScanFlow.pickFromGallery(context),
                                  )
                              : null,
                        ),
                      ),
                    ),
                  ],
                ),
                if (!reachable) ...[
                  const SizedBox(height: 12),
                  _OfflineHint(),
                ],
                const SizedBox(height: 28),
                const SectionHeaderAccent(title: 'Detectable conditions'),
                const SizedBox(height: 12),
                const DiseaseCategoryStrip(),
                const SizedBox(height: 28),
                const SectionHeader(
                  title: 'How it works',
                  subtitle:
                      'Four quick steps from photo to treatment guidance.',
                ),
                const SizedBox(height: 14),
                const HowItWorksStrip(),
                const SizedBox(height: 24),
                _ResearchCard(),
                const SizedBox(height: 8),
                _PhotoTipsCard(),
              ]),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHero(BuildContext context) {
    return BubbleBackground(
      child: WaveHeader(
        height: 272,
        colors: const [AppColors.deepOcean, AppColors.ocean, AppColors.seaBlue],
        child: SafeArea(
          bottom: false,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    Container(
                      width: 44,
                      height: 44,
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: [
                            Colors.white.withValues(alpha: 0.22),
                            Colors.white.withValues(alpha: 0.08),
                          ],
                        ),
                        borderRadius: BorderRadius.circular(14),
                        border: Border.all(
                          color: Colors.white.withValues(alpha: 0.2),
                        ),
                      ),
                      child: const Icon(
                        Icons.set_meal_rounded,
                        color: Colors.white,
                        size: 24,
                      ),
                    ),
                    const SizedBox(width: 12),
                    const Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'AquaScan',
                            style: TextStyle(
                              fontSize: 22,
                              fontWeight: FontWeight.w800,
                              color: Colors.white,
                              letterSpacing: -0.5,
                            ),
                          ),
                          Text(
                            'Freshwater fish disease AI',
                            style: TextStyle(
                              fontSize: 12,
                              color: Color(0xFFB0D9F0),
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const BackendStatusBanner(compact: true),
                    const SizedBox(width: 8),
                    const _AccountAvatar(),
                  ],
                ),
                const Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Scan. Preview.\nDiagnose.',
                      style: TextStyle(
                        fontSize: 26,
                        fontWeight: FontWeight.w800,
                        color: Colors.white,
                        height: 1.15,
                        letterSpacing: -0.6,
                      ),
                    ),
                    SizedBox(height: 6),
                    Text(
                      '7-class CNN for South Asian freshwater aquaculture.',
                      style: TextStyle(
                        fontSize: 12,
                        color: Color(0xFFB8D9F0),
                        height: 1.35,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

/// Tappable avatar in the hero header — opens the account / settings sheet.
class _AccountAvatar extends StatelessWidget {
  const _AccountAvatar();

  @override
  Widget build(BuildContext context) {
    final user = AuthService.instance.user;
    return Semantics(
      button: true,
      label: 'Account and settings',
      child: GestureDetector(
        onTap: () => AccountSheet.show(context),
        child: CircleAvatar(
          radius: 18,
          backgroundColor: Colors.white.withValues(alpha: 0.18),
          foregroundImage:
              user?.photoUrl != null ? NetworkImage(user!.photoUrl!) : null,
          child: const Icon(
            Icons.person_rounded,
            color: Colors.white,
            size: 20,
          ),
        ),
      ),
    );
  }
}

class _OfflineHint extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.coral.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.coral.withValues(alpha: 0.2)),
      ),
      child: Row(
        children: [
          const Icon(Icons.cloud_off_rounded, color: AppColors.coral, size: 22),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              'Start the backend with run_backend.bat on port 8000.',
              style: TextStyle(
                fontSize: 13,
                color: context.textPrimary.withValues(alpha: 0.85),
                height: 1.35,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ResearchCard extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: context.tintedSurface,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.aqua.withValues(alpha: 0.15)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              color: AppColors.seaBlue.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(
              Icons.science_rounded,
              color: AppColors.seaBlue,
              size: 22,
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Text(
              'Based on Tamut et al., Aquac. J. 2025 — trained on 2,444 '
              'freshwater fish images. Model accuracy ~81% on the held-out test set.',
              style: TextStyle(
                fontSize: 13,
                color: context.isDarkMode ? AppColors.wave : AppColors.ocean,
                height: 1.45,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _PhotoTipsCard extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: context.surfaceCard,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: context.cardShadow,
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.tips_and_updates_rounded,
                color: AppColors.amber.withValues(alpha: 0.9),
                size: 22,
              ),
              const SizedBox(width: 8),
              Text(
                'Photo tips',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w800,
                    ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          ...[
            'Side view of the fish in clear water',
            'Good lighting — avoid heavy shadows',
            'Fill the frame; avoid hands or nets',
          ].map(
            (tip) => Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('• ', style: TextStyle(color: AppColors.teal)),
                  Expanded(
                    child: Text(
                      tip,
                      style: TextStyle(
                        fontSize: 13,
                        color: context.textSecondary,
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
    );
  }
}
