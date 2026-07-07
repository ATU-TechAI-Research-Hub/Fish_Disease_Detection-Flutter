import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:aquaculture/services/auth_service.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('AuthService (Firebase not configured — guest fallback)', () {
    late AuthService auth;

    setUp(() {
      SharedPreferences.setMockInitialValues({});
      auth = AuthService.forTest();
    });

    test('init degrades gracefully when Firebase is unavailable', () async {
      await auth.init();
      expect(auth.isFirebaseAvailable, isFalse);
      expect(auth.state, AuthState.signedOut);
      expect(auth.isAuthenticated, isFalse);
      expect(auth.user, isNull);
    });

    test('continueAsGuest enters and persists guest mode', () async {
      await auth.init();
      await auth.continueAsGuest();

      expect(auth.state, AuthState.guest);
      expect(auth.isAuthenticated, isTrue);

      // A new service instance (fresh app launch) restores the session.
      final restarted = AuthService.forTest();
      await restarted.init();
      expect(restarted.state, AuthState.guest);
    });

    test('signOut leaves guest mode and clears persistence', () async {
      await auth.init();
      await auth.continueAsGuest();
      await auth.signOut();

      expect(auth.state, AuthState.signedOut);
      expect(auth.isAuthenticated, isFalse);

      final restarted = AuthService.forTest();
      await restarted.init();
      expect(restarted.state, AuthState.signedOut);
    });

    test('signInWithGoogle throws a friendly error when not configured',
        () async {
      await auth.init();
      expect(
        () => auth.signInWithGoogle(),
        throwsA(isA<AuthException>().having(
          (e) => e.message,
          'message',
          contains('not configured'),
        )),
      );
    });

    test('state change notifications fire for listeners', () async {
      await auth.init();
      var notified = 0;
      auth.addListener(() => notified++);
      await auth.continueAsGuest();
      expect(notified, greaterThan(0));
    });
  });
}
