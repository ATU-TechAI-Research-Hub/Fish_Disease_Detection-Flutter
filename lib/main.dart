import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'screens/app_shell.dart';
import 'screens/login_screen.dart';
import 'services/auth_service.dart';
import 'services/backend_status_service.dart';
import 'services/scan_history_service.dart';
import 'theme/app_theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(statusBarColor: Colors.transparent),
  );

  await ThemeController.instance.init();
  await AuthService.instance.init();
  BackendStatusService.instance.start();

  runApp(const FishDiseaseApp());
}

class FishDiseaseApp extends StatelessWidget {
  const FishDiseaseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Listenable.merge(
        [ThemeController.instance, AuthService.instance],
      ),
      builder: (context, _) {
        return MaterialApp(
          title: 'AquaScan',
          debugShowCheckedModeBanner: false,
          theme: AppTheme.light,
          darkTheme: AppTheme.dark,
          themeMode: ThemeController.instance.mode,
          home: const _Root(),
        );
      },
    );
  }
}

/// Routes to the login screen or the main shell based on auth state.
class _Root extends StatefulWidget {
  const _Root();

  @override
  State<_Root> createState() => _RootState();
}

class _RootState extends State<_Root> {
  AuthState? _lastState;

  @override
  void initState() {
    super.initState();
    // _Root is constructed const, so it must subscribe itself instead of
    // relying on ancestors rebuilding it.
    AuthService.instance.addListener(_onAuthChanged);
  }

  @override
  void dispose() {
    AuthService.instance.removeListener(_onAuthChanged);
    super.dispose();
  }

  void _onAuthChanged() {
    if (mounted) setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final state = AuthService.instance.state;

    // Clear session-scoped scan history when the user signs out.
    if (_lastState != null &&
        _lastState != AuthState.signedOut &&
        state == AuthState.signedOut) {
      WidgetsBinding.instance.addPostFrameCallback(
        (_) => ScanHistoryService.instance.clear(),
      );
    }
    _lastState = state;

    return switch (state) {
      AuthState.initializing => const _BootSplash(),
      AuthState.signedOut => const LoginScreen(),
      AuthState.signedIn || AuthState.guest => const AppShell(),
    };
  }
}

class _BootSplash extends StatelessWidget {
  const _BootSplash();

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      backgroundColor: AppColors.deepOcean,
      body: Center(
        child: CircularProgressIndicator(color: AppColors.seafoam),
      ),
    );
  }
}
