import 'package:flutter/material.dart';

import '../services/assistant_controller.dart';
import '../theme/app_theme.dart';
import 'assistant_panel.dart';

class AssistantOverlay extends StatefulWidget {
  const AssistantOverlay({
    required this.child,
    required this.enabled,
    super.key,
  });

  final Widget child;
  final bool enabled;

  @override
  State<AssistantOverlay> createState() => _AssistantOverlayState();
}

class _AssistantOverlayState extends State<AssistantOverlay> {
  late final OverlayEntry _entry = OverlayEntry(
    builder: (context) => ListenableBuilder(
      listenable: AssistantController.instance,
      builder: (context, _) => _buildContent(context),
    ),
  );

  @override
  void didUpdateWidget(covariant AssistantOverlay oldWidget) {
    super.didUpdateWidget(oldWidget);
    _entry.markNeedsBuild();
  }

  @override
  Widget build(BuildContext context) {
    return Overlay(initialEntries: [_entry]);
  }

  Widget _buildContent(BuildContext context) {
    final controller = AssistantController.instance;
    if (!widget.enabled) return widget.child;
    return Material(
      type: MaterialType.transparency,
      child: LayoutBuilder(builder: (context, constraints) {
        final media = MediaQuery.of(context);
        final wide = constraints.maxWidth >= 900;
        final keyboard = media.viewInsets.bottom;
        final panelWidth = wide ? 420.0 : constraints.maxWidth - 16;
        final usableHeight = constraints.maxHeight - keyboard;
        final panelBottom = keyboard + (keyboard > 0 ? 8 : (wide ? 12 : 74));
        final maxPanelHeight =
            (constraints.maxHeight - panelBottom - 8).clamp(120.0, 860.0);
        final desiredHeight = wide ? usableHeight - 24 : usableHeight * 0.76;
        final panelHeight =
            desiredHeight.clamp(120.0, maxPanelHeight).toDouble();

        return Stack(
          children: [
            widget.child,
            if (controller.isOpen)
              Positioned(
                right: wide ? 12 : 8,
                bottom: panelBottom,
                width: panelWidth,
                height: panelHeight,
                child: TweenAnimationBuilder<double>(
                  duration: const Duration(milliseconds: 220),
                  tween: Tween(begin: 0.96, end: 1),
                  curve: Curves.easeOutCubic,
                  builder: (context, scale, child) => Transform.scale(
                    scale: scale,
                    alignment: Alignment.bottomRight,
                    child: child,
                  ),
                  child: AssistantPanel(controller: controller),
                ),
              ),
            if (!controller.isOpen)
              Positioned(
                right: 18,
                bottom: 92 + keyboard,
                child: Semantics(
                  button: true,
                  label: 'Open Aquaculture AI Assistant',
                  child: _AssistantFloatingButton(
                    onPressed: controller.open,
                    hasPrediction: controller.currentPrediction != null,
                  ),
                ),
              ),
          ],
        );
      }),
    );
  }
}

class _AssistantFloatingButton extends StatelessWidget {
  const _AssistantFloatingButton({
    required this.onPressed,
    required this.hasPrediction,
  });

  final VoidCallback onPressed;
  final bool hasPrediction;

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: 'Ask AquaScan AI',
      child: InkWell(
        onTap: onPressed,
        customBorder: const CircleBorder(),
        child: Ink(
          width: 58,
          height: 58,
          decoration: BoxDecoration(
            gradient: const LinearGradient(
              colors: [AppColors.seaBlue, AppColors.teal],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            shape: BoxShape.circle,
            border: Border.all(
              color: context.isDarkMode
                  ? AppColors.seafoam.withValues(alpha: 0.35)
                  : Colors.white,
              width: 2,
            ),
            boxShadow: [
              BoxShadow(
                color: AppColors.deepOcean.withValues(alpha: 0.28),
                blurRadius: 18,
                offset: const Offset(0, 7),
              ),
            ],
          ),
          child: Stack(
            alignment: Alignment.center,
            children: [
              const Icon(
                Icons.auto_awesome_rounded,
                color: Colors.white,
                size: 27,
              ),
              if (hasPrediction)
                Positioned(
                  right: 2,
                  top: 2,
                  child: Container(
                    width: 13,
                    height: 13,
                    decoration: BoxDecoration(
                      color: AppColors.seafoam,
                      shape: BoxShape.circle,
                      border: Border.all(color: AppColors.deepOcean, width: 2),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
