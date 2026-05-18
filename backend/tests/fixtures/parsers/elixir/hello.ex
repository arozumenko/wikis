defmodule Greeter do
  def greet(name) do
    format_name(name)
  end

  def format_name(n) do
    String.upcase(n)
  end
end

defmodule Main do
  def run do
    Greeter.greet("world")
  end
end
