const std = @import("std");

const Greeter = struct {
    name: []const u8,
};

fn greet(name: []const u8) []const u8 {
    return formatName(name);
}

fn formatName(n: []const u8) []const u8 {
    return n;
}

fn standalone() void {
    _ = greet("world");
}
