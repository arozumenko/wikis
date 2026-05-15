package com.example

import scala.collection.mutable

class Greeter extends Base {
  def greet(name: String): Unit = {
    val formatted = formatName(name)
    println(formatted)
  }

  def formatName(name: String): String = name.toUpperCase
}

object Main {
  def run(): Unit = {
    new Greeter().greet("world")
  }
}
