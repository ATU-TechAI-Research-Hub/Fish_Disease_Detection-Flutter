import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

class HowItWorksStrip extends StatelessWidget {
  const HowItWorksStrip({super.key});

  static const _steps = [
    _StepData(
      icon: Icons.add_a_photo_rounded,
      title: 'Capture',
      desc: 'Photo or gallery',
      color: AppColors.seaBlue,
    ),
    _StepData(
      icon: Icons.preview_rounded,
      title: 'Preview',
      desc: 'Confirm image',
      color: AppColors.teal,
    ),
    _StepData(
      icon: Icons.auto_awesome_rounded,
      title: 'Analyze',
      desc: 'CNN on device',
      color: AppColors.purple,
    ),
    _StepData(
      icon: Icons.medical_information_rounded,
      title: 'Treat',
      desc: 'Guidance & tips',
      color: AppColors.emerald,
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 132,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 2),
        itemCount: _steps.length,
        separatorBuilder: (_, __) => const SizedBox(width: 12),
        itemBuilder: (context, i) {
          final s = _steps[i];
          return Container(
            width: 128,
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: context.surfaceCard,
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: s.color.withValues(alpha: 0.12)),
              boxShadow: [
                BoxShadow(
                  color: context.cardShadow,
                  blurRadius: 10,
                  offset: const Offset(0, 4),
                ),
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  width: 36,
                  height: 36,
                  decoration: BoxDecoration(
                    color: s.color.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Icon(s.icon, color: s.color, size: 20),
                ),
                const Spacer(),
                Text(
                  s.title,
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w800,
                    color: context.textPrimary,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  s.desc,
                  style: TextStyle(
                    fontSize: 11,
                    color: context.textSecondary,
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}

class _StepData {
  const _StepData({
    required this.icon,
    required this.title,
    required this.desc,
    required this.color,
  });

  final IconData icon;
  final String title;
  final String desc;
  final Color color;
}
