import 'dart:math' as math;

import 'package:flutter/material.dart';

class BubbleBackground extends StatefulWidget {
  const BubbleBackground({required this.child, super.key});

  final Widget child;

  @override
  State<BubbleBackground> createState() => _BubbleBackgroundState();
}

class _BubbleBackgroundState extends State<BubbleBackground>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(seconds: 8),
      vsync: this,
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        widget.child,
        Positioned.fill(
          child: IgnorePointer(
            child: AnimatedBuilder(
              animation: _controller,
              builder: (context, _) {
                return CustomPaint(
                  painter: _BubblePainter(progress: _controller.value),
                );
              },
            ),
          ),
        ),
      ],
    );
  }
}

class _BubblePainter extends CustomPainter {
  _BubblePainter({required this.progress});

  final double progress;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..style = PaintingStyle.fill;

    final bubbles = [
      _Bubble(0.1, 0.15, 6, 0.06),
      _Bubble(0.3, 0.25, 8, 0.08),
      _Bubble(0.55, 0.12, 5, 0.05),
      _Bubble(0.7, 0.2, 7, 0.07),
      _Bubble(0.85, 0.18, 9, 0.04),
      _Bubble(0.45, 0.3, 4, 0.09),
      _Bubble(0.2, 0.35, 6, 0.06),
      _Bubble(0.9, 0.1, 5, 0.07),
    ];

    for (final b in bubbles) {
      final phase = (progress + b.phase) % 1.0;
      final y = size.height * (1.0 - phase);
      final x = size.width * b.xRatio +
          math.sin(phase * math.pi * 4) * 12;
      final alpha = (0.12 * (1.0 - (phase - 0.5).abs() * 2)).clamp(0.0, 0.12);
      paint.color = Colors.white.withValues(alpha: alpha);
      canvas.drawCircle(Offset(x, y), b.radius, paint);
    }
  }

  @override
  bool shouldRepaint(_BubblePainter oldDelegate) =>
      oldDelegate.progress != progress;
}

class _Bubble {
  const _Bubble(this.xRatio, this.phase, this.radius, this.speed);

  final double xRatio;
  final double phase;
  final double radius;
  final double speed;
}
