import 'package:aquaculture/screens/shrimp_counting_screen.dart';
import 'package:aquaculture/theme/app_theme.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('shrimp flow starts with real capture and upload actions',
      (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        theme: AppTheme.light,
        home: const ShrimpCountingScreen(),
      ),
    );

    expect(find.text('Shrimp Counting & Analysis'), findsOneWidget);
    expect(find.byKey(const Key('shrimp_camera_action')), findsOneWidget);
    expect(find.byKey(const Key('shrimp_gallery_action')), findsOneWidget);
    expect(find.text('Analyze Shrimp'), findsNothing);
    expect(find.textContaining('white bowl'), findsOneWidget);
  });
}
