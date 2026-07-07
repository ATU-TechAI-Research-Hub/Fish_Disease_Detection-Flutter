import 'package:flutter/material.dart';

import '../services/auth_service.dart';
import '../theme/app_theme.dart';

/// Appearance (light / dark / system) selector persisted across launches.
class _ThemeModeRow extends StatefulWidget {
  const _ThemeModeRow();

  @override
  State<_ThemeModeRow> createState() => _ThemeModeRowState();
}

class _ThemeModeRowState extends State<_ThemeModeRow> {
  @override
  Widget build(BuildContext context) {
    final controller = ThemeController.instance;
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(Icons.brightness_6_rounded,
            size: 18, color: context.textSecondary),
        const SizedBox(width: 10),
        Text('Appearance',
            style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: context.textSecondary,
            )),
        const SizedBox(width: 12),
        SegmentedButton<ThemeMode>(
          segments: const [
            ButtonSegment(
              value: ThemeMode.system,
              icon: Icon(Icons.settings_suggest_rounded, size: 16),
            ),
            ButtonSegment(
              value: ThemeMode.light,
              icon: Icon(Icons.light_mode_rounded, size: 16),
            ),
            ButtonSegment(
              value: ThemeMode.dark,
              icon: Icon(Icons.dark_mode_rounded, size: 16),
            ),
          ],
          selected: {controller.mode},
          onSelectionChanged: (selection) async {
            await controller.setMode(selection.first);
            if (mounted) setState(() {});
          },
          showSelectedIcon: false,
          style: const ButtonStyle(
            visualDensity: VisualDensity.compact,
          ),
        ),
      ],
    );
  }
}

/// Bottom sheet showing the signed-in account (or guest state) with sign-out.
class AccountSheet extends StatelessWidget {
  const AccountSheet({super.key});

  static Future<void> show(BuildContext context) {
    return showModalBottomSheet<void>(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (_) => const AccountSheet(),
    );
  }

  Future<void> _signOut(BuildContext context) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: const Text('Sign Out'),
        content: const Text(
          'You will return to the sign-in screen. Scan history for this '
          'session will be cleared.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: FilledButton.styleFrom(backgroundColor: AppColors.coral),
            child: const Text('Sign Out'),
          ),
        ],
      ),
    );
    if (confirmed == true) {
      await AuthService.instance.signOut();
      if (context.mounted) Navigator.of(context).pop();
    }
  }

  @override
  Widget build(BuildContext context) {
    final auth = AuthService.instance;
    final user = auth.user;
    final isGuest = auth.state == AuthState.guest;
    final theme = Theme.of(context);
    final bottom = MediaQuery.paddingOf(context).bottom;

    return Container(
      margin: const EdgeInsets.fromLTRB(16, 0, 16, 16),
      padding: EdgeInsets.fromLTRB(20, 12, 20, 20 + bottom),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerLowest,
        borderRadius: BorderRadius.circular(24),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 40,
            height: 4,
            decoration: BoxDecoration(
              color: theme.dividerColor,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(height: 20),
          CircleAvatar(
            radius: 32,
            backgroundColor: AppColors.seaBlue.withValues(alpha: 0.12),
            foregroundImage: (user?.photoUrl != null)
                ? NetworkImage(user!.photoUrl!)
                : null,
            child: Icon(
              isGuest ? Icons.person_outline_rounded : Icons.person_rounded,
              size: 32,
              color: AppColors.seaBlue,
            ),
          ),
          const SizedBox(height: 12),
          Text(
            isGuest ? 'Guest' : (user?.displayName ?? 'AquaScan User'),
            style: theme.textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.w800,
            ),
          ),
          if (!isGuest && (user?.email.isNotEmpty ?? false)) ...[
            const SizedBox(height: 4),
            Text(user!.email, style: theme.textTheme.bodyMedium),
          ],
          if (isGuest) ...[
            const SizedBox(height: 4),
            Text(
              'Sign in with Google to attach scans to an account.',
              textAlign: TextAlign.center,
              style: theme.textTheme.bodyMedium,
            ),
          ],
          const SizedBox(height: 20),
          const _ThemeModeRow(),
          const SizedBox(height: 16),
          FilledButton.icon(
            onPressed: () => _signOut(context),
            icon: const Icon(Icons.logout_rounded, size: 20),
            label: Text(isGuest ? 'Exit Guest Mode' : 'Sign Out'),
            style: FilledButton.styleFrom(
              backgroundColor: AppColors.coral,
              minimumSize: const Size.fromHeight(50),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(14),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
