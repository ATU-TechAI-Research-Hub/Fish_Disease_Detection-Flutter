import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:aquaculture/main.dart';

void main() {
  testWidgets('App boots with home dashboard and scan FAB',
      (WidgetTester tester) async {
    tester.view.physicalSize = const Size(400, 900);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);

    await tester.pumpWidget(const FishDiseaseApp());
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 300));

    expect(find.text('AquaScan'), findsWidgets);
    expect(find.text('Scan'), findsOneWidget);
    expect(find.text('Home'), findsOneWidget);
    expect(find.text('Library'), findsOneWidget);

    await tester.scrollUntilVisible(
      find.text('Start a scan'),
      120,
      scrollable: find.byType(Scrollable).first,
    );
    expect(find.text('Start a scan'), findsOneWidget);
    expect(find.byKey(const Key('home_camera_action')), findsOneWidget);
  });
}
