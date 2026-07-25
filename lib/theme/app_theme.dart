import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Brand palette. Accent colors are shared by both themes; surface and text
/// colors have dark variants resolved through [AppThemeContext].
class AppColors {
  AppColors._();

  static const Color deepOcean = Color(0xFF021B33);
  static const Color ocean = Color(0xFF053B6A);
  static const Color seaBlue = Color(0xFF0969B2);
  static const Color aqua = Color(0xFF0EA5E9);
  static const Color teal = Color(0xFF14B8A6);
  static const Color seafoam = Color(0xFF6EE7B7);
  static const Color coral = Color(0xFFFF6B6B);
  static const Color sand = Color(0xFFFFF7ED);
  static const Color wave = Color(0xFFE0F2FE);
  static const Color waveLight = Color(0xFFF0F9FF);
  static const Color emerald = Color(0xFF10B981);
  static const Color amber = Color(0xFFF59E0B);
  static const Color purple = Color(0xFF8B5CF6);

  // Light surfaces / text
  static const Color surface = Color(0xFFF0F9FF);
  static const Color textPrimary = Color(0xFF0C1D2E);
  static const Color textSecondary = Color(0xFF64748B);
  static const Color cardBg = Color(0xFFFFFFFF);
  static const Color divider = Color(0xFFE2E8F0);

  // Dark surfaces / text
  static const Color surfaceDark = Color(0xFF0A1622);
  static const Color textPrimaryDark = Color(0xFFE8F1F8);
  static const Color textSecondaryDark = Color(0xFF9FB1C1);
  static const Color cardBgDark = Color(0xFF13232F);
  static const Color dividerDark = Color(0xFF23384A);
}

/// Theme-aware color lookups so widgets render correctly in light AND dark
/// mode without duplicating `Theme.of(context)` boilerplate everywhere.
extension AppThemeContext on BuildContext {
  bool get isDarkMode => Theme.of(this).brightness == Brightness.dark;

  /// Card / elevated surface background.
  Color get surfaceCard => isDarkMode ? AppColors.cardBgDark : AppColors.cardBg;

  /// Primary body text.
  Color get textPrimary =>
      isDarkMode ? AppColors.textPrimaryDark : AppColors.textPrimary;

  /// Secondary / muted text.
  Color get textSecondary =>
      isDarkMode ? AppColors.textSecondaryDark : AppColors.textSecondary;

  /// Subtle border / divider.
  Color get subtleBorder =>
      isDarkMode ? AppColors.dividerDark : AppColors.divider;

  /// Soft aqua-tinted background used for info panels.
  Color get tintedSurface =>
      isDarkMode ? AppColors.seaBlue.withValues(alpha: 0.14) : AppColors.wave;

  /// Soft shadow color for cards (transparent-ish in dark mode).
  Color get cardShadow => isDarkMode
      ? Colors.black.withValues(alpha: 0.35)
      : AppColors.deepOcean.withValues(alpha: 0.04);
}

/// Persists and exposes the user's theme-mode choice (system by default).
class ThemeController extends ChangeNotifier {
  ThemeController._();
  static final ThemeController instance = ThemeController._();

  static const String _prefKey = 'aquascan_theme_mode';

  ThemeMode _mode = ThemeMode.system;
  ThemeMode get mode => _mode;

  Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    _mode = switch (prefs.getString(_prefKey)) {
      'light' => ThemeMode.light,
      'dark' => ThemeMode.dark,
      _ => ThemeMode.system,
    };
    notifyListeners();
  }

  Future<void> setMode(ThemeMode mode) async {
    _mode = mode;
    notifyListeners();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefKey, mode.name);
  }

  String get label => switch (_mode) {
        ThemeMode.system => 'System',
        ThemeMode.light => 'Light',
        ThemeMode.dark => 'Dark',
      };
}

class AppTheme {
  AppTheme._();

  static ThemeData get light => _build(Brightness.light);
  static ThemeData get dark => _build(Brightness.dark);

  static ThemeData _build(Brightness brightness) {
    final isDark = brightness == Brightness.dark;

    final scaffoldBg = isDark ? AppColors.surfaceDark : AppColors.surface;
    final card = isDark ? AppColors.cardBgDark : AppColors.cardBg;
    final divider = isDark ? AppColors.dividerDark : AppColors.divider;
    final primaryText =
        isDark ? AppColors.textPrimaryDark : AppColors.textPrimary;
    final secondaryText =
        isDark ? AppColors.textSecondaryDark : AppColors.textSecondary;

    final colorScheme = ColorScheme.fromSeed(
      seedColor: AppColors.seaBlue,
      brightness: brightness,
      primary: isDark ? AppColors.aqua : AppColors.seaBlue,
      secondary: AppColors.teal,
      tertiary: AppColors.coral,
      surface: scaffoldBg,
    );

    return ThemeData(
      useMaterial3: true,
      brightness: brightness,
      colorScheme: colorScheme,
      scaffoldBackgroundColor: scaffoldBg,
      dividerColor: divider,
      appBarTheme: AppBarTheme(
        elevation: 0,
        scrolledUnderElevation: 0,
        centerTitle: true,
        backgroundColor: Colors.transparent,
        foregroundColor: primaryText,
        titleTextStyle: TextStyle(
          fontSize: 18,
          fontWeight: FontWeight.w700,
          color: primaryText,
          letterSpacing: -0.3,
        ),
      ),
      navigationBarTheme: NavigationBarThemeData(
        height: 72,
        backgroundColor: card,
        indicatorColor:
            AppColors.seaBlue.withValues(alpha: isDark ? 0.3 : 0.12),
        labelTextStyle: WidgetStateProperty.resolveWith((states) {
          final selected = states.contains(WidgetState.selected);
          return TextStyle(
            fontSize: 12,
            fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
            color: selected
                ? (isDark ? AppColors.aqua : AppColors.seaBlue)
                : secondaryText,
          );
        }),
        iconTheme: WidgetStateProperty.resolveWith((states) {
          final selected = states.contains(WidgetState.selected);
          return IconThemeData(
            color: selected
                ? (isDark ? AppColors.aqua : AppColors.seaBlue)
                : secondaryText,
          );
        }),
      ),
      cardTheme: CardThemeData(
        elevation: 0,
        color: card,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
        ),
        margin: EdgeInsets.zero,
      ),
      snackBarTheme: SnackBarThemeData(
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          elevation: 0,
          padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 24),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          textStyle: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w700,
            letterSpacing: 0.2,
          ),
        ),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 24),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          textStyle: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w700,
          ),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: card,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: BorderSide(color: divider),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: BorderSide(color: divider),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: AppColors.seaBlue, width: 1.5),
        ),
        hintStyle: TextStyle(color: secondaryText),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      ),
      textTheme: TextTheme(
        headlineLarge: TextStyle(
          fontSize: 28,
          fontWeight: FontWeight.w800,
          color: primaryText,
          letterSpacing: -0.5,
          height: 1.2,
        ),
        headlineMedium: TextStyle(
          fontSize: 24,
          fontWeight: FontWeight.w700,
          color: primaryText,
          letterSpacing: -0.3,
        ),
        headlineSmall: TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.w700,
          color: primaryText,
        ),
        titleLarge: TextStyle(
          fontSize: 18,
          fontWeight: FontWeight.w700,
          color: primaryText,
        ),
        titleMedium: TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w600,
          color: primaryText,
        ),
        bodyLarge: TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w400,
          color: secondaryText,
          height: 1.5,
        ),
        bodyMedium: TextStyle(
          fontSize: 14,
          fontWeight: FontWeight.w400,
          color: secondaryText,
          height: 1.5,
        ),
      ),
    );
  }
}
