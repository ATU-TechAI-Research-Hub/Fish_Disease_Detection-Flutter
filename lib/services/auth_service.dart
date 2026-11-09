import 'dart:async';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/foundation.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../firebase_options.dart';

/// Authentication states surfaced to the UI.
enum AuthState {
  /// `init()` has not completed yet (splash / boot).
  initializing,

  /// Nobody is signed in and guest mode is not active.
  signedOut,

  /// A Google account is signed in through Firebase.
  signedIn,

  /// The user chose to continue without an account.
  guest,
}

/// Immutable snapshot of the signed-in user shown in the UI.
class AuthUser {
  const AuthUser({
    required this.uid,
    required this.displayName,
    required this.email,
    this.photoUrl,
  });

  final String uid;
  final String displayName;
  final String email;
  final String? photoUrl;
}

/// Google Sign-In via Firebase Authentication with a guest-mode fallback.
///
/// Design goals:
///  * The app must keep working when Firebase has not been configured yet
///    (see `firebase_options.dart`): `isFirebaseAvailable` becomes false and
///    only guest mode is offered.
///  * Session persistence is handled by Firebase itself (secure, encrypted
///    per-platform storage). Guest mode persists via [SharedPreferences]
///    (it stores no sensitive data — just a boolean flag).
///  * All failures surface as [AuthException] with a user-friendly message.
class AuthService extends ChangeNotifier {
  AuthService._();
  static final AuthService instance = AuthService._();

  /// Fresh, isolated instance for unit tests (the singleton keeps state
  /// between tests otherwise).
  @visibleForTesting
  factory AuthService.forTest() => AuthService._();

  static const String _guestPrefKey = 'aquascan_guest_mode';

  AuthState _state = AuthState.initializing;
  AuthUser? _user;
  bool _firebaseAvailable = false;
  bool _busy = false;
  StreamSubscription<User?>? _authSub;

  AuthState get state => _state;
  AuthUser? get user => _user;
  bool get isFirebaseAvailable => _firebaseAvailable;
  bool get isBusy => _busy;
  bool get isAuthenticated =>
      _state == AuthState.signedIn || _state == AuthState.guest;

  /// Initialise Firebase (if configured) and restore any previous session.
  /// Never throws — a broken/missing Firebase config degrades to guest-only.
  Future<void> init() async {
    try {
      await Firebase.initializeApp(
        options: DefaultFirebaseOptions.currentPlatform,
      );
      _firebaseAvailable = true;
    } catch (e) {
      // UnsupportedError from the placeholder config, or a platform issue.
      _firebaseAvailable = false;
      debugPrint('AuthService: Firebase unavailable → guest-only mode ($e)');
    }

    if (_firebaseAvailable) {
      try {
        await GoogleSignIn.instance.initialize();
      } catch (e) {
        debugPrint('AuthService: GoogleSignIn.initialize failed: $e');
      }
      // Firebase restores persisted sessions automatically; mirror them.
      _authSub = FirebaseAuth.instance.authStateChanges().listen(_onUser);
      final current = FirebaseAuth.instance.currentUser;
      if (current != null) {
        _onUser(current);
        return;
      }
    }

    final prefs = await SharedPreferences.getInstance();
    _setState(
      prefs.getBool(_guestPrefKey) == true
          ? AuthState.guest
          : AuthState.signedOut,
    );
  }

  void _onUser(User? firebaseUser) {
    if (firebaseUser == null) {
      if (_state == AuthState.signedIn) _setState(AuthState.signedOut);
      return;
    }
    _user = AuthUser(
      uid: firebaseUser.uid,
      displayName: firebaseUser.displayName ?? 'AquaScan User',
      email: firebaseUser.email ?? '',
      photoUrl: firebaseUser.photoURL,
    );
    _setState(AuthState.signedIn);
  }

  /// Launch the Google account picker and sign in through Firebase.
  Future<void> signInWithGoogle() async {
    if (!_firebaseAvailable) {
      throw const AuthException(
        'Google Sign-In is not configured on this build yet. '
        'Continue as guest, or see firebase_options.dart for setup steps.',
      );
    }
    if (_busy) return;
    _setBusy(true);
    try {
      final account = await GoogleSignIn.instance.authenticate();
      final idToken = account.authentication.idToken;
      if (idToken == null) {
        throw const AuthException(
          'Google did not return an ID token. Please try again.',
        );
      }
      final credential = GoogleAuthProvider.credential(idToken: idToken);
      await FirebaseAuth.instance.signInWithCredential(credential);
      // `_onUser` fires via authStateChanges and updates state.
      final prefs = await SharedPreferences.getInstance();
      await prefs.remove(_guestPrefKey);
    } on GoogleSignInException catch (e) {
      if (e.code == GoogleSignInExceptionCode.canceled) {
        return; // User dismissed the picker — not an error.
      }
      throw AuthException('Google Sign-In failed: ${e.description ?? e.code}');
    } on FirebaseAuthException catch (e) {
      throw AuthException(_friendlyFirebaseMessage(e));
    } on AuthException {
      rethrow;
    } catch (e) {
      throw AuthException('Sign-in failed unexpectedly: $e');
    } finally {
      _setBusy(false);
    }
  }

  /// Use the app without an account. Persisted across restarts.
  Future<void> continueAsGuest() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_guestPrefKey, true);
    _user = null;
    _setState(AuthState.guest);
  }

  /// Sign out of Google/Firebase (or leave guest mode).
  Future<void> signOut() async {
    if (_firebaseAvailable) {
      try {
        await GoogleSignIn.instance.signOut();
      } catch (_) {
        // Google session may already be gone; Firebase signOut still runs.
      }
      try {
        await FirebaseAuth.instance.signOut();
      } catch (e) {
        debugPrint('AuthService: Firebase signOut failed: $e');
      }
    }
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_guestPrefKey);
    _user = null;
    _setState(AuthState.signedOut);
  }

  static String _friendlyFirebaseMessage(FirebaseAuthException e) {
    return switch (e.code) {
      'account-exists-with-different-credential' =>
        'This email is already linked to another sign-in method.',
      'network-request-failed' =>
        'Network error — check your internet connection and try again.',
      'user-disabled' => 'This account has been disabled.',
      'too-many-requests' =>
        'Too many attempts. Please wait a moment and try again.',
      _ => 'Authentication failed (${e.code}). Please try again.',
    };
  }

  void _setBusy(bool value) {
    _busy = value;
    notifyListeners();
  }

  void _setState(AuthState value) {
    _state = value;
    notifyListeners();
  }

  @override
  void dispose() {
    _authSub?.cancel();
    super.dispose();
  }
}

/// User-facing authentication error.
class AuthException implements Exception {
  const AuthException(this.message);
  final String message;

  @override
  String toString() => message;
}
