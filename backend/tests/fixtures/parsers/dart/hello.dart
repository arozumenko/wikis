import 'package:flutter/material.dart';

class Greeter extends Base {
  String greet(String name) {
    return formatName(name);
  }

  String formatName(String n) {
    return n.toUpperCase();
  }
}

void standaloneHelper() {
  final g = Greeter();
  g.greet("world");
}
