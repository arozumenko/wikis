require 'json'

class Greeter < Base
  def greet(name)
    formatted = format_name(name)
    puts(formatted)
  end

  def format_name(name)
    name.upcase
  end
end

def standalone_helper
  Greeter.new.greet("world")
end
