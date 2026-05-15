local m = require("utils")

Greeter = {}

function Greeter:greet(name)
    local formatted = self:format_name(name)
    print(formatted)
end

function Greeter:format_name(name)
    return m.upper(name)
end

local function standalone_helper()
    Greeter:greet("world")
end
