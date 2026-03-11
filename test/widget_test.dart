import 'package:flutter_test/flutter_test.dart';

import 'package:aquaculture/main.dart';

void main() {
  testWidgets('Home screen shows image input actions', (WidgetTester tester) async {
    await tester.pumpWidget(const FishDiseaseApp());

    expect(find.text('Fish Disease Detection Prototype'), findsOneWidget);
    expect(find.text('Take Photo'), findsOneWidget);
    expect(find.text('Upload from Gallery'), findsOneWidget);
  });
}
