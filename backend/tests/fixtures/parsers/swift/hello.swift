import Foundation

class Greeter: Base {
    func greet(_ name: String) -> String {
        return formatName(name)
    }

    func formatName(_ n: String) -> String {
        return n.uppercased()
    }
}

func standaloneHelper() {
    let g = Greeter()
    _ = g.greet("world")
}
