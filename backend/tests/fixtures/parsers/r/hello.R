library(stringr)

greet <- function(name) {
    format_name(name)
}

format_name <- function(n) {
    toupper(n)
}

standalone_helper <- function() {
    greet("world")
}

Greeter <- setRefClass("Greeter", methods = list())
