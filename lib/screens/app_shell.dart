import 'package:flutter/material.dart';

import '../services/backend_status_service.dart';
import '../services/scan_flow.dart';
import '../theme/app_theme.dart';
import '../widgets/scan_options_sheet.dart';
import 'disease_library_screen.dart';
import 'home_screen.dart';
import 'scan_history_screen.dart';

class AppShell extends StatefulWidget {
  const AppShell({super.key});

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  int _currentIndex = 0;

  final List<Widget> _screens = const [
    HomeScreen(),
    DiseaseLibraryScreen(),
    ScanHistoryScreen(),
  ];

  @override
  void initState() {
    super.initState();
    BackendStatusService.instance.addListener(_onBackendChanged);
  }

  @override
  void dispose() {
    BackendStatusService.instance.removeListener(_onBackendChanged);
    super.dispose();
  }

  void _onBackendChanged() {
    if (mounted) setState(() {});
  }

  void _openScanSheet() {
    ScanOptionsSheet.show(
      context,
      cameraEnabled: ScanFlow.cameraSupported,
      onCamera: ScanFlow.backendReachable
          ? () {
              Navigator.pop(context);
              ScanFlow.scanWithCamera(context);
            }
          : null,
      onGallery: ScanFlow.backendReachable
          ? () {
              Navigator.pop(context);
              ScanFlow.pickFromGallery(context);
            }
          : null,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: _screens,
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _openScanSheet,
        backgroundColor: AppColors.seaBlue,
        foregroundColor: Colors.white,
        elevation: 4,
        icon: const Icon(Icons.document_scanner_outlined),
        label: const Text(
          'Scan',
          style: TextStyle(fontWeight: FontWeight.w800, letterSpacing: 0.2),
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (i) => setState(() => _currentIndex = i),
        backgroundColor: Colors.white,
        indicatorColor: AppColors.seaBlue.withValues(alpha: 0.12),
        labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.home_outlined),
            selectedIcon: Icon(Icons.home_rounded),
            label: 'Home',
          ),
          NavigationDestination(
            icon: Icon(Icons.menu_book_outlined),
            selectedIcon: Icon(Icons.menu_book_rounded),
            label: 'Library',
          ),
          NavigationDestination(
            icon: Icon(Icons.history_outlined),
            selectedIcon: Icon(Icons.history_rounded),
            label: 'History',
          ),
        ],
      ),
    );
  }
}
