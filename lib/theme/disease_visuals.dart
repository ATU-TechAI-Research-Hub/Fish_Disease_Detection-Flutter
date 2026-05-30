import 'package:flutter/material.dart';

import 'app_theme.dart';

/// Visual identity helpers for disease types (icons, colors, gradients).
class DiseaseVisuals {
  DiseaseVisuals._();

  static Color colorFor(String type) {
    return switch (type.toLowerCase()) {
      'bacterial' => const Color(0xFFEF4444),
      'fungal' => const Color(0xFFF59E0B),
      'parasitic' => const Color(0xFF8B5CF6),
      'viral' => const Color(0xFFEC4899),
      'healthy' => AppColors.emerald,
      _ => AppColors.seaBlue,
    };
  }

  static IconData iconFor(String type) {
    return switch (type.toLowerCase()) {
      'bacterial' => Icons.coronavirus_rounded,
      'fungal' => Icons.water_drop_rounded,
      'parasitic' => Icons.pest_control_rounded,
      'viral' => Icons.biotech_rounded,
      'healthy' => Icons.check_circle_rounded,
      _ => Icons.medical_services_rounded,
    };
  }

  static List<Color> gradientFor(String type) {
    final base = colorFor(type);
    return [base, base.withValues(alpha: 0.65)];
  }

  static String shortLabel(String type) {
    return switch (type.toLowerCase()) {
      'bacterial' => 'Bacterial',
      'fungal' => 'Fungal',
      'parasitic' => 'Parasitic',
      'viral' => 'Viral',
      'healthy' => 'Healthy',
      _ => type,
    };
  }
}

/// Quick reference for the seven detectable classes on the home screen.
class DiseaseCategoryPreview {
  const DiseaseCategoryPreview({
    required this.name,
    required this.type,
    required this.icon,
  });

  final String name;
  final String type;
  final IconData icon;

  static const List<DiseaseCategoryPreview> all = [
    DiseaseCategoryPreview(
      name: 'Red Disease',
      type: 'Bacterial',
      icon: Icons.circle,
    ),
    DiseaseCategoryPreview(
      name: 'Aeromoniasis',
      type: 'Bacterial',
      icon: Icons.circle,
    ),
    DiseaseCategoryPreview(
      name: 'Gill Disease',
      type: 'Bacterial',
      icon: Icons.circle,
    ),
    DiseaseCategoryPreview(
      name: 'Saprolegniasis',
      type: 'Fungal',
      icon: Icons.grass_rounded,
    ),
    DiseaseCategoryPreview(
      name: 'Healthy Fish',
      type: 'Healthy',
      icon: Icons.favorite_rounded,
    ),
    DiseaseCategoryPreview(
      name: 'Parasitic',
      type: 'Parasitic',
      icon: Icons.bubble_chart_rounded,
    ),
    DiseaseCategoryPreview(
      name: 'White Tail',
      type: 'Viral',
      icon: Icons.waves_rounded,
    ),
  ];
}
