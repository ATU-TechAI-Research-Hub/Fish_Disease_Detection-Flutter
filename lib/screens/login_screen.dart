import 'package:flutter/material.dart';

import '../services/auth_service.dart';
import '../theme/app_theme.dart';
import '../widgets/bubble_background.dart';

/// Sign-in / sign-up entry point.
///
/// Google Sign-In doubles as account creation, so a single screen covers
/// both flows. When Firebase isn't configured the Google button explains
/// that and guest mode remains available, keeping the app fully usable.
class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  bool _signingIn = false;

  Future<void> _signInWithGoogle() async {
    if (_signingIn) return;
    setState(() => _signingIn = true);
    try {
      await AuthService.instance.signInWithGoogle();
      // On success the root listener swaps to the app shell automatically.
    } on AuthException catch (e) {
      _showError(e.message);
    } finally {
      if (mounted) setState(() => _signingIn = false);
    }
  }

  Future<void> _continueAsGuest() async {
    await AuthService.instance.continueAsGuest();
  }

  void _showError(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        behavior: SnackBarBehavior.floating,
        backgroundColor: AppColors.coral,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        content: Text(message, style: const TextStyle(color: Colors.white)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final firebaseReady = AuthService.instance.isFirebaseAvailable;

    return Scaffold(
      body: BubbleBackground(
        child: Container(
          width: double.infinity,
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [
                AppColors.deepOcean,
                AppColors.ocean,
                AppColors.seaBlue,
              ],
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
            ),
          ),
          child: SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 28),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const Spacer(flex: 2),
                  Container(
                    width: 88,
                    height: 88,
                    alignment: Alignment.center,
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          Colors.white.withValues(alpha: 0.22),
                          Colors.white.withValues(alpha: 0.08),
                        ],
                      ),
                      borderRadius: BorderRadius.circular(26),
                      border: Border.all(
                        color: Colors.white.withValues(alpha: 0.25),
                      ),
                    ),
                    child: const Icon(
                      Icons.set_meal_rounded,
                      color: Colors.white,
                      size: 46,
                    ),
                  ),
                  const SizedBox(height: 24),
                  const Text(
                    'AquaScan',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 34,
                      fontWeight: FontWeight.w800,
                      color: Colors.white,
                      letterSpacing: -0.8,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'AI-powered freshwater fish disease detection\nfor South Asian aquaculture',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 14,
                      height: 1.45,
                      color: Colors.white.withValues(alpha: 0.75),
                    ),
                  ),
                  const Spacer(flex: 3),
                  _GoogleButton(
                    busy: _signingIn,
                    enabled: firebaseReady && !_signingIn,
                    onPressed: _signInWithGoogle,
                  ),
                  if (!firebaseReady) ...[
                    const SizedBox(height: 10),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 14,
                        vertical: 10,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha: 0.08),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Row(
                        children: [
                          Icon(
                            Icons.info_outline_rounded,
                            size: 18,
                            color: Colors.white.withValues(alpha: 0.7),
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: Text(
                              'Google Sign-In needs Firebase configuration '
                              '(see firebase_options.dart). Guest mode has '
                              'every feature.',
                              style: TextStyle(
                                fontSize: 12,
                                height: 1.35,
                                color: Colors.white.withValues(alpha: 0.7),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                  const SizedBox(height: 14),
                  OutlinedButton.icon(
                    onPressed: _signingIn ? null : _continueAsGuest,
                    icon: const Icon(Icons.person_outline_rounded),
                    label: const Text('Continue as Guest'),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: Colors.white,
                      minimumSize: const Size.fromHeight(52),
                      side: BorderSide(
                        color: Colors.white.withValues(alpha: 0.45),
                      ),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                      textStyle: const TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                  const SizedBox(height: 18),
                  Text(
                    'Signing in creates your account automatically.\n'
                    'Your scans stay on this device.',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 12,
                      height: 1.4,
                      color: Colors.white.withValues(alpha: 0.55),
                    ),
                  ),
                  const SizedBox(height: 20),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _GoogleButton extends StatelessWidget {
  const _GoogleButton({
    required this.busy,
    required this.enabled,
    required this.onPressed,
  });

  final bool busy;
  final bool enabled;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: 'Sign in with Google',
      child: FilledButton(
        onPressed: enabled ? onPressed : null,
        style: FilledButton.styleFrom(
          backgroundColor: Colors.white,
          disabledBackgroundColor: Colors.white.withValues(alpha: 0.55),
          foregroundColor: const Color(0xFF1F1F1F),
          minimumSize: const Size.fromHeight(54),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
        child: busy
            ? const SizedBox(
                width: 22,
                height: 22,
                child: CircularProgressIndicator(strokeWidth: 2.5),
              )
            : const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                mainAxisSize: MainAxisSize.min,
                children: [
                  _GoogleGlyph(),
                  SizedBox(width: 12),
                  Flexible(
                    child: Text(
                      'Sign in with Google',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                ],
              ),
      ),
    );
  }
}

/// Small multicolour "G" mark drawn with text (avoids bundling an asset).
class _GoogleGlyph extends StatelessWidget {
  const _GoogleGlyph();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 26,
      height: 26,
      alignment: Alignment.center,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: const Color(0xFFE0E0E0)),
      ),
      child: const Text(
        'G',
        style: TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w800,
          color: Color(0xFF4285F4),
        ),
      ),
    );
  }
}
