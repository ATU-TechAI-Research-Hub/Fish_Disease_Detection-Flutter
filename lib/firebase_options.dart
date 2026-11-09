// Placeholder Firebase configuration.
//
// Google Sign-In is fully wired in the app (see `AuthService`), but it needs
// YOUR Firebase project config to go live. Until then the app runs in
// guest mode and every other feature keeps working.
//
// To activate Google Sign-In:
//   1. Create a Firebase project at https://console.firebase.google.com
//   2. Enable the "Google" provider under Authentication → Sign-in method.
//   3. Register the Android app (package: com.example.aquaculture) and add
//      your debug SHA-1 (`cd android && ./gradlew signingReport`).
//   4. Install the FlutterFire CLI and run from the project root:
//        dart pub global activate flutterfire_cli
//        flutterfire configure
//      That command REPLACES this file with real per-platform options and
//      downloads android/app/google-services.json.
//
// `AuthService` calls `DefaultFirebaseOptions.currentPlatform` inside a
// try/catch, so the UnsupportedError below simply switches the app to
// guest-only mode instead of crashing.

import 'package:firebase_core/firebase_core.dart';

class DefaultFirebaseOptions {
  DefaultFirebaseOptions._();

  static FirebaseOptions get currentPlatform {
    throw UnsupportedError(
      'Firebase is not configured yet. Run `flutterfire configure` to '
      'generate real options (this file will be replaced).',
    );
  }
}
