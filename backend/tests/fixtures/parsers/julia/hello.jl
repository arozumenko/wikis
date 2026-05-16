using LinearAlgebra

struct Greeter
end

function greet(g::Greeter, name)
    return format_name(name)
end

function format_name(n)
    return uppercase(n)
end

function standalone_helper()
    greet(Greeter(), "world")
end
