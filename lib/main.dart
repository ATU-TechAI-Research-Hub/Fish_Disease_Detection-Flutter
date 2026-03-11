import 'package:flutter/material.dart';

import 'screens/home_screen.dart';

void main() {
  runApp(const FishDiseaseApp());
}

class FishDiseaseApp extends StatelessWidget {
  const FishDiseaseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Freshwater Fish Disease Aquaculture in South Asia',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF0D8B9E)),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}
