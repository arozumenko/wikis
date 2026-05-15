package com.example

import kotlin.text.uppercase

class Greeter : Base() {
    fun greet(name: String) {
        val formatted = formatName(name)
        println(formatted)
    }

    fun formatName(name: String): String {
        return name.uppercase()
    }
}

fun standaloneHelper() {
    Greeter().greet("world")
}
