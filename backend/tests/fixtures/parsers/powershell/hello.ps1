Import-Module Utils

class Greeter {
    [string] Greet([string]$name) {
        return $this.FormatName($name)
    }

    [string] FormatName([string]$n) {
        return $n.ToUpper()
    }
}

function StandaloneHelper {
    $g = [Greeter]::new()
    $g.Greet("world")
}
