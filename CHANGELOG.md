# Changelog

All notable changes to AquaScan are documented here.

## [Unreleased] — feature/auth-ui-prediction-improvements (2026-07-07)

### Added — Authentication

- **Google Sign-In via Firebase Authentication** (`lib/services/auth_service.dart`):
  Firebase-backed Google login for Android/iOS/Web using `firebase_auth` +
  `google_sign_in` (v7 API). Sessions persist automatically through Firebase's
  secure per-platform token storage.
- **Guest mode fallback**: when Firebase has not been configured yet
  (`lib/firebase_options.dart` is a documented placeholder until you run
  `flutterfire configure`), the app degrades gracefully — every feature stays
  usable and the login screen explains what is missing. Guest mode persists
  across launches via `shared_preferences` (a non-sensitive boolean flag only).
- **Modern login screen** (`lib/screens/login_screen.dart`): ocean-gradient
  design with the animated bubble background, "Sign in with Google" and
  "Continue as Guest" actions, inline busy indicator, and friendly error
  snackbars (network failures, cancelled sign-in, disabled accounts, etc.).
- **Account sheet with logout** (`lib/widgets/account_sheet.dart`): tap the
  avatar in the home header to see the signed-in profile (name, email, photo)
  or guest status, switch appearance (system/light/dark), and sign out with a
  confirmation dialog. Signing out clears session scan history.
- Auth-gated app root (`lib/main.dart`): boot splash → login screen → main
  shell driven by a single `AuthState` enum.
- Android `minSdk` raised to 23 (Firebase Auth requirement).

### Added — Prediction pipeline

- **Client-side image quality validation** (`lib/services/image_quality.dart`):
  before upload, images are decoded locally and checked for low resolution
  (< 200 px short side), extreme aspect ratios, and suspiciously small files.
  Soft warnings appear on the preview screen so users can retake poor photos.
- **Backend pytest suite** (`backend/tests/test_api.py`, 13 tests): request
  validation (empty/oversized/non-image uploads, bad content types), health
  and model metadata, label-map ↔ disease-catalog consistency, top-3 ordering,
  **determinism check** (identical bytes → identical confidence), and
  preprocessing parity checks (150×150 float32 in [0, 1], grayscale/alpha
  handling, EXIF-safe decode).

Previously landed on this branch's parent (documented for completeness):
two-stage inference (MobileNetV2 fish-presence gate → disease CNN), softer
uncertainty gate (`confidence < 0.20 AND entropy > 1.90`), env-configurable
thresholds, top-3 predictions, confidence tiers, and centralized
preprocessing matching the training configuration exactly.

### Changed — UI/UX

- **Dark mode**: full dark theme (`AppTheme.dark`) with `ThemeMode.system`
  default and a persisted user override (system/light/dark) in the account
  sheet. All cards, sheets, tiles, and text swap through new theme-aware
  lookups (`context.surfaceCard`, `context.textPrimary`, ...), so both themes
  render correctly across Home, Library, History, Preview, Result, and Login.
- Navigation bar, snackbars, and input fields now themed centrally in
  `AppTheme` instead of per-screen hardcoded colors.
- Fixed a horizontal overflow in wide-font environments on the login button
  and a 4 px vertical overflow in the home hero.

### Added — Testing

- `test/auth_service_test.dart` (5 tests): graceful Firebase degradation,
  guest-mode persistence and restore, sign-out behaviour, friendly error on
  unconfigured Google sign-in, and change notifications.
- `test/widget_test.dart` rewritten (2 tests): boot-to-login flow with guest
  sign-in navigation, and session-restore straight to the home dashboard.
- `pytest` + `httpx` added to `backend/requirements.txt`.

### Notes / remaining work

- Google Sign-In is code-complete but needs a real Firebase project:
  run `flutterfire configure` (steps documented in `lib/firebase_options.dart`),
  add your SHA-1 for Android, and enable the Google provider. Until then the
  login screen offers guest mode and explains the missing configuration.
- Scan history remains in-memory (per session) by design; persisting it
  per-account (e.g. Firestore or local DB keyed by uid) is a natural follow-up.
