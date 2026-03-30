import 'dart:math' as math;

import 'package:flutter/material.dart';

class WaveClipper extends CustomClipper<Path> {
  WaveClipper({this.waveHeight = 20});

  final double waveHeight;

  @override
  Path getClip(Size size) {
    final path = Path();
    path.lineTo(0, size.height - waveHeight);

    for (double x = 0; x <= size.width; x++) {
      final y = size.height -
          waveHeight +
          math.sin(x / size.width * 2 * math.pi) * waveHeight * 0.5 +
          math.sin(x / size.width * 4 * math.pi) * waveHeight * 0.25;
      path.lineTo(x, y);
    }

    path.lineTo(size.width, 0);
    path.close();
    return path;
  }

  @override
  bool shouldReclip(WaveClipper oldClipper) =>
      oldClipper.waveHeight != waveHeight;
}

class WaveHeader extends StatelessWidget {
  const WaveHeader({
    required this.height,
    required this.colors,
    this.child,
    super.key,
  });

  final double height;
  final List<Color> colors;
  final Widget? child;

  @override
  Widget build(BuildContext context) {
    return ClipPath(
      clipper: WaveClipper(),
      child: Container(
        height: height,
        width: double.infinity,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: colors,
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: child,
      ),
    );
  }
}
