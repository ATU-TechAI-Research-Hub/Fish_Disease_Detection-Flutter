import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:aquaculture/main.dart';
import 'package:aquaculture/services/auth_service.dart';
import 'package:aquaculture/theme/app_theme.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('App boots to login screen when signed out',
      (WidgetTester tester) async {
    tester.view.physicalSize = const Size(400, 900);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);

    SharedPreferences.setMockInitialValues({});
    await ThemeController.instance.init();
    await AuthService.instance.init();

    await tester.pumpWidget(const FishDiseaseApp());
    await tester.pump();

    expect(find.text('AquaScan'), findsOneWidget);
    expect(find.text('Continue as Guest'), findsOneWidget);

    // Guest sign-in leads to the home dashboard.
    await tester.tap(find.text('Continue as Guest'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 300));

    expect(find.text('Home'), findsOneWidget);
    expect(find.text('Library'), findsOneWidget);

    await AuthService.instance.signOut();
  });

  testWidgets('Guest session restores straight into home dashboard',
      (WidgetTester tester) async {
    tester.view.physicalSize = const Size(400, 900);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);

    SharedPreferences.setMockInitialValues({'aquascan_guest_mode': true});
    await ThemeController.instance.init();
    await AuthService.instance.init();

    await tester.pumpWidget(const FishDiseaseApp());
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 300));

    expect(find.text('AquaScan'), findsWidgets);
    expect(find.text('Scan'), findsOneWidget);
    expect(find.text('Home'), findsOneWidget);

    await tester.scrollUntilVisible(
      find.text('Start a scan'),
      120,
      scrollable: find.byType(Scrollable).first,
    );
    expect(find.text('Start a scan'), findsOneWidget);
    expect(find.byKey(const Key('home_camera_action')), findsOneWidget);

    await AuthService.instance.signOut();
  });
}
