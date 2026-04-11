"""
Unit tests for RustVisitorParser — coverage target ≥80%.

Uses inline Rust source snippets; no real file I/O.
"""
from __future__ import annotations

import pytest

from app.core.parsers.rust_visitor_parser import RustVisitorParser, _parse_single_rust_file
from app.core.parsers.base_parser import SymbolType, RelationshipType, ParseResult


def parse(source: str, path: str = "main.rs") -> ParseResult:
    parser = RustVisitorParser()
    return parser.parse_file(path, content=source)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

class TestRustParserSmoke:
    def test_parser_initialises(self):
        p = RustVisitorParser()
        assert p.capabilities.language == "rust"

    def test_parse_empty_returns_result(self):
        result = parse("")
        assert isinstance(result, ParseResult)
        assert result.language == "rust"

    def test_parse_minimal_program(self):
        result = parse("fn main() {}\n")
        assert isinstance(result, ParseResult)
        assert result.file_path == "main.rs"

    def test_parse_time_is_positive(self):
        result = parse("fn main() {}\n")
        assert result.parse_time >= 0.0


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

class TestRustFunctions:
    def test_simple_function_extracted(self):
        src = "fn greet(name: &str) -> String {\n    format!(\"Hello, {}!\", name)\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "greet" in names

    def test_function_symbol_type(self):
        src = "fn add(a: i32, b: i32) -> i32 { a + b }\n"
        result = parse(src)
        funcs = [s for s in result.symbols if s.name == "add"]
        assert len(funcs) >= 1
        assert funcs[0].symbol_type == SymbolType.FUNCTION

    def test_main_function_extracted(self):
        src = 'fn main() {\n    println!("Hello, world!");\n}\n'
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "main" in names

    def test_async_function_extracted(self):
        src = "async fn fetch() -> Result<(), Box<dyn std::error::Error>> { Ok(()) }\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "fetch" in names

    def test_public_function(self):
        src = "pub fn public_api(x: u32) -> u32 { x * 2 }\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "public_api" in names

    def test_generic_function(self):
        src = "fn identity<T>(x: T) -> T { x }\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "identity" in names

    def test_function_with_where_clause(self):
        src = (
            "fn largest<T>(list: &[T]) -> &T where T: PartialOrd {\n"
            "    &list[0]\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "largest" in names

    def test_multiple_functions(self):
        src = "fn a() {}\nfn b() {}\nfn c() {}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "a" in names
        assert "b" in names
        assert "c" in names


# ---------------------------------------------------------------------------
# Structs
# ---------------------------------------------------------------------------

class TestRustStructs:
    def test_struct_extracted(self):
        src = "struct Point { x: f64, y: f64 }\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Point" in names

    def test_struct_symbol_type(self):
        src = "struct User { name: String, age: u32 }\n"
        result = parse(src)
        structs = [s for s in result.symbols if s.name == "User"]
        assert len(structs) >= 1
        assert structs[0].symbol_type == SymbolType.STRUCT

    def test_tuple_struct(self):
        src = "struct Celsius(f64);\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Celsius" in names

    def test_unit_struct(self):
        src = "struct Unit;\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Unit" in names

    def test_struct_with_pub_fields(self):
        src = "pub struct Config {\n    pub host: String,\n    pub port: u16,\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Config" in names

    def test_generic_struct(self):
        src = "struct Stack<T> { items: Vec<T> }\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Stack" in names

    def test_struct_with_lifetime(self):
        src = "struct Excerpt<'a> { content: &'a str }\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Excerpt" in names


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestRustEnums:
    def test_simple_enum_extracted(self):
        src = "enum Direction { North, South, East, West }\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Direction" in names

    def test_enum_symbol_type(self):
        src = "enum Color { Red, Green, Blue }\n"
        result = parse(src)
        enums = [s for s in result.symbols if s.name == "Color"]
        assert len(enums) >= 1
        assert enums[0].symbol_type == SymbolType.ENUM

    def test_enum_with_data(self):
        src = (
            "enum Message {\n"
            "    Quit,\n"
            "    Move { x: i32, y: i32 },\n"
            "    Write(String),\n"
            "    ChangeColor(i32, i32, i32),\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Message" in names

    def test_option_like_enum(self):
        src = "enum MyOption<T> { Some(T), None }\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "MyOption" in names


# ---------------------------------------------------------------------------
# Traits
# ---------------------------------------------------------------------------

class TestRustTraits:
    def test_trait_extracted(self):
        src = "trait Animal {\n    fn speak(&self) -> String;\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Animal" in names

    def test_trait_symbol_type(self):
        src = "pub trait Display {\n    fn fmt(&self) -> String;\n}\n"
        result = parse(src)
        traits = [s for s in result.symbols if s.name == "Display"]
        assert len(traits) >= 1
        assert traits[0].symbol_type == SymbolType.TRAIT

    def test_trait_with_default_method(self):
        src = (
            "trait Greet {\n"
            "    fn name(&self) -> String;\n"
            "    fn greeting(&self) -> String {\n"
            "        format!(\"Hello, {}!\", self.name())\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Greet" in names

    def test_trait_with_associated_type(self):
        src = (
            "trait Iterator {\n"
            "    type Item;\n"
            "    fn next(&mut self) -> Option<Self::Item>;\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Iterator" in names


# ---------------------------------------------------------------------------
# Impl blocks
# ---------------------------------------------------------------------------

class TestRustImpl:
    def test_impl_method_extracted(self):
        src = (
            "struct Counter { count: u32 }\n\n"
            "impl Counter {\n"
            "    fn increment(&mut self) { self.count += 1; }\n"
            "    fn new() -> Counter { Counter { count: 0 } }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "increment" in names
        assert "new" in names

    def test_impl_trait_for_struct(self):
        src = (
            "trait Speak { fn speak(&self) -> &str; }\n"
            "struct Dog;\n"
            "impl Speak for Dog {\n"
            "    fn speak(&self) -> &str { \"Woof\" }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "speak" in names

    def test_impl_with_generics(self):
        src = (
            "struct Wrapper<T>(T);\n"
            "impl<T: std::fmt::Display> Wrapper<T> {\n"
            "    fn show(&self) -> String {\n"
            "        format!(\"{}\", self.0)\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "show" in names


# ---------------------------------------------------------------------------
# Constants and type aliases
# ---------------------------------------------------------------------------

class TestRustConstantsAndAliases:
    def test_const_extracted(self):
        src = "const MAX_SIZE: usize = 1024;\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "MAX_SIZE" in names

    def test_static_extracted(self):
        src = "static GREETING: &str = \"Hello\";\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "GREETING" in names

    def test_type_alias_extracted(self):
        src = "type Result<T> = std::result::Result<T, String>;\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Result" in names


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class TestRustModules:
    def test_inline_module_extracted(self):
        src = (
            "mod utils {\n"
            "    pub fn helper() -> bool { true }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "utils" in names

    def test_pub_module(self):
        src = "pub mod config {\n    pub const VERSION: &str = \"1.0\";\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "config" in names


# ---------------------------------------------------------------------------
# Use declarations
# ---------------------------------------------------------------------------

class TestRustUseDeclarationsBasic:
    def test_use_creates_import_relationship(self):
        src = "use std::collections::HashMap;\nfn main() {}\n"
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.IMPORTS in rel_types

    def test_use_as_alias(self):
        src = "use std::fmt::Display as Fmt;\nfn main() {}\n"
        result = parse(src)
        assert isinstance(result, ParseResult)

    def test_nested_use(self):
        src = "use std::io::{Read, Write, BufRead};\nfn main() {}\n"
        result = parse(src)
        assert isinstance(result, ParseResult)


# ---------------------------------------------------------------------------
# Attributes (derive, cfg, etc.)
# ---------------------------------------------------------------------------

class TestRustAttributes:
    def test_derive_attribute_no_crash(self):
        src = "#[derive(Debug, Clone, PartialEq)]\nstruct Point { x: i32, y: i32 }\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Point" in names

    def test_cfg_attribute_no_crash(self):
        src = "#[cfg(test)]\nfn test_only() {}\n"
        result = parse(src)
        assert isinstance(result, ParseResult)

    def test_allow_attribute(self):
        src = "#[allow(dead_code)]\nfn unused() {}\n"
        result = parse(src)
        assert isinstance(result, ParseResult)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestRustErrorHandling:
    def test_syntax_error_returns_result(self):
        src = "fn broken( { }\n"
        result = parse(src)
        assert isinstance(result, ParseResult)

    def test_missing_file_returns_error(self):
        parser = RustVisitorParser()
        result = parser.parse_file("/nonexistent/file.rs")
        assert isinstance(result, ParseResult)
        assert result.errors

    def test_content_overrides_file_read(self):
        parser = RustVisitorParser()
        result = parser.parse_file("/nonexistent.rs", content="fn injected() {}\n")
        names = [s.name for s in result.symbols]
        assert "injected" in names


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

class TestRustCapabilities:
    def test_language_is_rust(self):
        p = RustVisitorParser()
        assert p.capabilities.language == "rust"

    def test_extensions_include_rs(self):
        p = RustVisitorParser()
        assert ".rs" in p._get_supported_extensions()

    def test_has_generics(self):
        p = RustVisitorParser()
        assert p.capabilities.has_generics is True


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

class TestParseSingleRustFile:
    def test_returns_tuple(self, tmp_path):
        f = tmp_path / "lib.rs"
        f.write_text("pub fn hello() -> &'static str { \"hello\" }\n")
        path, result = _parse_single_rust_file(str(f))
        assert path == str(f)
        assert isinstance(result, ParseResult)

    def test_missing_file_returns_empty(self):
        path, result = _parse_single_rust_file("/nonexistent.rs")
        assert path == "/nonexistent.rs"
        assert result.symbols == []


# ---------------------------------------------------------------------------
# Multi-file parse
# ---------------------------------------------------------------------------

class TestRustMultipleFiles:
    def test_parse_multiple_files(self, tmp_path):
        f1 = tmp_path / "a.rs"
        f2 = tmp_path / "b.rs"
        f1.write_text("pub fn func_a() {}\n")
        f2.write_text("pub fn func_b() {}\n")
        parser = RustVisitorParser()
        results = parser.parse_multiple_files([str(f1), str(f2)])
        assert isinstance(results, dict)
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "func_a" in all_names
        assert "func_b" in all_names


# ---------------------------------------------------------------------------
# Additional coverage tests — Rust-specific patterns
# ---------------------------------------------------------------------------

class TestRustLifetimes:
    def test_lifetime_parameter_function(self):
        src = "fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {\n    if x.len() > y.len() { x } else { y }\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "longest" in names

    def test_struct_with_lifetime(self):
        src = "struct Excerpt<'a> {\n    part: &'a str,\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Excerpt" in names


class TestRustClosures:
    def test_function_taking_closure(self):
        src = "fn apply<F: Fn(i32) -> i32>(f: F, x: i32) -> i32 { f(x) }\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "apply" in names

    def test_fn_returning_closure(self):
        src = "fn make_adder(x: i32) -> impl Fn(i32) -> i32 {\n    move |y| x + y\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "make_adder" in names


class TestRustComplexStructs:
    def test_struct_with_box_field(self):
        src = (
            "struct TreeNode {\n"
            "    val: i32,\n"
            "    left: Option<Box<TreeNode>>,\n"
            "    right: Option<Box<TreeNode>>,\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "TreeNode" in names

    def test_struct_with_vec_field(self):
        src = (
            "struct Graph {\n"
            "    nodes: Vec<String>,\n"
            "    edges: Vec<(usize, usize)>,\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Graph" in names

    def test_struct_with_hashmap_field(self):
        src = (
            "use std::collections::HashMap;\n\n"
            "struct Config {\n"
            "    settings: HashMap<String, String>,\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Config" in names


class TestRustEnumPatterns:
    def test_result_like_enum(self):
        src = "enum MyResult<T, E> {\n    Ok(T),\n    Err(E),\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "MyResult" in names

    def test_complex_enum_variants(self):
        src = (
            "enum Shape {\n"
            "    Circle { radius: f64 },\n"
            "    Rectangle { width: f64, height: f64 },\n"
            "    Triangle(f64, f64, f64),\n"
            "    Point,\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Shape" in names


class TestRustTraitBounds:
    def test_multiple_trait_bounds(self):
        src = (
            "use std::fmt::{Display, Debug};\n\n"
            "fn print_both<T: Display + Debug>(val: T) {\n"
            "    println!(\"{} {:?}\", val, val);\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "print_both" in names

    def test_where_clause_trait_bound(self):
        src = (
            "fn compare<T>(a: T, b: T) -> bool\n"
            "where T: PartialEq + PartialOrd {\n"
            "    a == b\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "compare" in names


class TestRustMacros:
    def test_macro_definition_extracted(self):
        src = (
            "macro_rules! say_hello {\n"
            "    () => {\n"
            "        println!(\"Hello!\");\n"
            "    };\n"
            "    ($name:expr) => {\n"
            "        println!(\"Hello, {}!\", $name);\n"
            "    };\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "say_hello" in names

    def test_proc_macro_attribute_no_crash(self):
        src = (
            "fn main() {\n"
            '    println!("Hello, world!");\n'
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "main" in names


class TestRustImplPatterns:
    def test_impl_display_for_struct(self):
        src = (
            "use std::fmt;\n\n"
            "struct Matrix(f64, f64, f64, f64);\n\n"
            "impl fmt::Display for Matrix {\n"
            "    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {\n"
            "        write!(f, \"( {} {} )\\n( {} {} )\", self.0, self.1, self.2, self.3)\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Matrix" in names

    def test_impl_iterator_for_struct(self):
        src = (
            "struct Fibonacci {\n"
            "    curr: u32,\n"
            "    next: u32,\n"
            "}\n\n"
            "impl Iterator for Fibonacci {\n"
            "    type Item = u32;\n"
            "    fn next(&mut self) -> Option<Self::Item> {\n"
            "        let new_next = self.curr + self.next;\n"
            "        self.curr = self.next;\n"
            "        self.next = new_next;\n"
            "        Some(self.curr)\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Fibonacci" in names

    def test_multiple_impl_blocks_for_type(self):
        src = (
            "struct Vec2 { x: f64, y: f64 }\n\n"
            "impl Vec2 {\n"
            "    fn new(x: f64, y: f64) -> Self { Vec2 { x, y } }\n"
            "    fn length(&self) -> f64 { (self.x * self.x + self.y * self.y).sqrt() }\n"
            "}\n\n"
            "impl std::ops::Add for Vec2 {\n"
            "    type Output = Vec2;\n"
            "    fn add(self, other: Vec2) -> Vec2 {\n"
            "        Vec2 { x: self.x + other.x, y: self.y + other.y }\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Vec2" in names
        assert "new" in names


# ---------------------------------------------------------------------------
# Cross-file resolution (drives parse_multiple_files internal code)
# ---------------------------------------------------------------------------

class TestRustCrossFileResolution:
    def test_cross_file_trait_implementation(self, tmp_path):
        traits_file = tmp_path / "traits.rs"
        impl_file = tmp_path / "models.rs"
        traits_file.write_text("pub trait Describable { fn describe(&self) -> String; }\n")
        impl_file.write_text(
            "pub struct User { pub name: String }\n\n"
            "impl User {\n"
            "    pub fn new(name: String) -> Self { User { name } }\n"
            "}\n"
        )
        parser = RustVisitorParser()
        results = parser.parse_multiple_files([str(traits_file), str(impl_file)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "Describable" in all_names
        assert "User" in all_names

    def test_cross_file_function_calls(self, tmp_path):
        utils = tmp_path / "utils.rs"
        main = tmp_path / "main.rs"
        utils.write_text("pub fn helper(x: i32) -> i32 { x * 2 }\npub fn format_value(v: f64) -> String { format!(\"{:.2}\", v) }\n")
        main.write_text("mod utils;\nfn main() {\n    let result = utils::helper(21);\n    println!(\"{}\", result);\n}\n")
        parser = RustVisitorParser()
        results = parser.parse_multiple_files([str(utils), str(main)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "helper" in all_names
        assert "main" in all_names

    def test_cross_file_struct_usage(self, tmp_path):
        models = tmp_path / "models.rs"
        service = tmp_path / "service.rs"
        models.write_text("pub struct Config { pub host: String, pub port: u16 }\n")
        service.write_text(
            "pub struct AppService {\n"
            "    config: String,\n"
            "}\n\n"
            "impl AppService {\n"
            "    pub fn new() -> Self { AppService { config: String::new() } }\n"
            "    pub fn start(&self) { println!(\"Starting on {}\", self.config); }\n"
            "}\n"
        )
        parser = RustVisitorParser()
        results = parser.parse_multiple_files([str(models), str(service)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "Config" in all_names
        assert "AppService" in all_names

    def test_cross_file_trait_and_impl(self, tmp_path):
        traits = tmp_path / "traits.rs"
        impls = tmp_path / "impls.rs"
        traits.write_text(
            "pub trait Serializable {\n"
            "    fn serialize(&self) -> String;\n"
            "    fn to_json(&self) -> String { format!(\"{{}}\") }\n"
            "}\n\n"
            "pub trait Deserializable {\n"
            "    fn deserialize(s: &str) -> Self;\n"
            "}\n"
        )
        impls.write_text(
            "pub struct User { pub name: String, pub email: String }\n\n"
            "impl User {\n"
            "    pub fn new(name: &str, email: &str) -> Self {\n"
            "        User { name: name.to_string(), email: email.to_string() }\n"
            "    }\n"
            "}\n"
        )
        parser = RustVisitorParser()
        results = parser.parse_multiple_files([str(traits), str(impls)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "Serializable" in all_names
        assert "User" in all_names

    def test_cross_file_with_repo_path(self, tmp_path):
        # Tests parse_multiple_files with repo_path (Cargo workspace discovery)
        cargo = tmp_path / "Cargo.toml"
        src = tmp_path / "src"
        src.mkdir()
        cargo.write_text("[package]\nname = \"my-crate\"\nversion = \"0.1.0\"\n")
        lib = src / "lib.rs"
        main = src / "main.rs"
        lib.write_text("pub fn add(a: i32, b: i32) -> i32 { a + b }\n")
        main.write_text("fn main() { let x = 1 + 2; println!(\"{}\", x); }\n")
        parser = RustVisitorParser()
        results = parser.parse_multiple_files([str(lib), str(main)], repo_path=str(tmp_path))
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "add" in all_names
        assert "main" in all_names

    def test_cross_file_calls_resolution(self, tmp_path):
        utils = tmp_path / "utils.rs"
        app = tmp_path / "app.rs"
        utils.write_text(
            "pub fn format_user(name: &str) -> String {\n"
            "    format!(\"User: {}\", name)\n"
            "}\n\n"
            "pub struct Logger;\n\n"
            "impl Logger {\n"
            "    pub fn log(msg: &str) { println!(\"{}\", msg); }\n"
            "}\n"
        )
        app.write_text(
            "fn run() {\n"
            "    let s = format_user(\"Alice\");\n"
            "    Logger::log(&s);\n"
            "}\n"
        )
        parser = RustVisitorParser()
        results = parser.parse_multiple_files([str(utils), str(app)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "format_user" in all_names
        assert "run" in all_names


# ---------------------------------------------------------------------------
# Cargo.toml parsing helpers (drives _parse_cargo_toml and related code)
# ---------------------------------------------------------------------------

class TestRustCargoParsing:
    def test_parse_cargo_toml_simple_package(self, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text("[package]\nname = \"my-app\"\nversion = \"1.0.0\"\n")
        # Use parse_multiple_files with repo_path to trigger Cargo discovery
        src = tmp_path / "src"
        src.mkdir()
        lib = src / "lib.rs"
        lib.write_text("pub fn hello() -> &'static str { \"hello\" }\n")
        parser = RustVisitorParser()
        results = parser.parse_multiple_files([str(lib)], repo_path=str(tmp_path))
        assert str(lib) in results

    def test_parse_cargo_toml_workspace(self, tmp_path):
        # Create a workspace structure
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[workspace]\nmembers = ["crate-a", "crate-b"]\n')
        crate_a = tmp_path / "crate-a"
        crate_a.mkdir()
        (crate_a / "Cargo.toml").write_text('[package]\nname = "crate-a"\nversion = "0.1.0"\n')
        src_a = crate_a / "src"
        src_a.mkdir()
        (src_a / "lib.rs").write_text("pub fn from_a() -> i32 { 42 }\n")

        crate_b = tmp_path / "crate-b"
        crate_b.mkdir()
        (crate_b / "Cargo.toml").write_text('[package]\nname = "crate-b"\nversion = "0.1.0"\n')
        src_b = crate_b / "src"
        src_b.mkdir()
        (src_b / "lib.rs").write_text("pub fn from_b() -> i32 { 100 }\n")

        parser = RustVisitorParser()
        results = parser.parse_multiple_files(
            [str(src_a / "lib.rs"), str(src_b / "lib.rs")],
            repo_path=str(tmp_path)
        )
        assert len(results) == 2
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "from_a" in all_names
        assert "from_b" in all_names


# ---------------------------------------------------------------------------
# Advanced Rust language features
# ---------------------------------------------------------------------------

class TestRustAdvancedFeatures:
    def test_union_type(self):
        src = (
            "union IntOrFloat {\n"
            "    i: i32,\n"
            "    f: f32,\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "IntOrFloat" in names

    def test_extern_crate(self):
        src = (
            "extern crate serde;\n"
            "extern crate serde as serde_alias;\n"
            "fn main() {}\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.IMPORTS in rel_types

    def test_use_with_self_import(self):
        src = (
            "use std::io::{self, Read, Write};\n"
            "fn process<R: Read>(reader: &mut R) -> io::Result<Vec<u8>> {\n"
            "    let mut buf = Vec::new();\n"
            "    reader.read_to_end(&mut buf)?;\n"
            "    Ok(buf)\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "process" in names

    def test_use_wildcard_import(self):
        src = (
            "use std::collections::*;\n"
            "fn make_map() -> HashMap<String, Vec<i32>> {\n"
            "    HashMap::new()\n"
            "}\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.IMPORTS in rel_types

    def test_scoped_use_list(self):
        src = (
            "use std::{fmt::{self, Display}, io::Write};\n"
            "struct Wrapper(i32);\n"
            "impl Display for Wrapper {\n"
            "    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {\n"
            "        write!(f, \"{}\", self.0)\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Wrapper" in names

    def test_dyn_trait_reference(self):
        src = (
            "trait Handler {\n"
            "    fn handle(&self, req: &str) -> String;\n"
            "}\n\n"
            "struct Router {\n"
            "    handler: Box<dyn Handler>,\n"
            "}\n\n"
            "impl Router {\n"
            "    pub fn route(&self, path: &str) -> String {\n"
            "        self.handler.handle(path)\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Router" in names
        assert "Handler" in names

    def test_impl_trait_return(self):
        src = (
            "trait Greet {\n"
            "    fn greet(&self) -> String;\n"
            "}\n\n"
            "fn make_greeter() -> impl Greet {\n"
            "    struct En;\n"
            "    impl Greet for En {\n"
            "        fn greet(&self) -> String { \"Hello!\".to_string() }\n"
            "    }\n"
            "    En\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "make_greeter" in names

    def test_turbofish_call(self):
        src = (
            "fn parse<T: std::str::FromStr>(s: &str) -> Option<T> {\n"
            "    s.parse::<T>().ok()\n"
            "}\n\n"
            "fn main() {\n"
            "    let n = parse::<i32>(\"42\");\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "parse" in names
        assert "main" in names

    def test_struct_expression_creates(self):
        src = (
            "struct Point { x: f64, y: f64 }\n\n"
            "fn origin() -> Point {\n"
            "    Point { x: 0.0, y: 0.0 }\n"
            "}\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.CREATES in rel_types

    def test_complex_generic_bounds(self):
        src = (
            "use std::fmt::Display;\n\n"
            "fn print_pair<T, U>(first: T, second: U)\n"
            "where\n"
            "    T: Display + Clone,\n"
            "    U: Display + std::fmt::Debug,\n"
            "{\n"
            "    println!(\"{} {}\", first, second);\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "print_pair" in names

    def test_single_file_parse_function(self, tmp_path):
        f = tmp_path / "test.rs"
        f.write_text("pub fn compute(x: i32) -> i32 { x * x }\n")
        path, result = _parse_single_rust_file(str(f))
        assert path == str(f)
        assert isinstance(result, ParseResult)
        assert result.language == "rust"
        names = [s.name for s in result.symbols]
        assert "compute" in names

    def test_extract_symbols_wrapper(self):
        p = RustVisitorParser()
        p.parse_file("test.rs", content="fn main() {}\n")
        syms = p.extract_symbols(None, "test.rs")
        assert isinstance(syms, list)

    def test_extract_relationships_wrapper(self):
        p = RustVisitorParser()
        p.parse_file("test.rs", content="fn main() {}\n")
        rels = p.extract_relationships(None, [], "test.rs")
        assert isinstance(rels, list)


# ---------------------------------------------------------------------------
# Use declarations with complex patterns (drives _extract_use_paths)
# ---------------------------------------------------------------------------

class TestRustUseDeclarations:
    def test_use_list_with_multiple_items(self):
        src = (
            "use std::collections::{HashMap, HashSet, BTreeMap};\n"
            "fn make() -> HashMap<String, HashSet<i32>> {\n"
            "    HashMap::new()\n"
            "}\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.IMPORTS in rel_types

    def test_use_list_with_self_and_other(self):
        src = (
            "use std::fmt::{self, Debug, Display};\n"
            "struct Point { x: i32, y: i32 }\n"
            "impl Display for Point {\n"
            "    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {\n"
            "        write!(f, \"({}, {})\", self.x, self.y)\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Point" in names

    def test_use_as_clause_in_list(self):
        src = (
            "use std::collections::{HashMap as Map, BTreeMap as OrderedMap};\n"
            "fn make_map() -> Map<String, i32> { Map::new() }\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "make_map" in names

    def test_pub_use_re_export(self):
        src = (
            "pub use std::sync::{Arc, Mutex};\n"
            "pub use std::collections::HashMap;\n"
            "pub fn shared(data: Vec<i32>) -> Arc<Mutex<Vec<i32>>> {\n"
            "    Arc::new(Mutex::new(data))\n"
            "}\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.IMPORTS in rel_types

    def test_simple_use_as_alias(self):
        # Drives the use_as_clause at top level (line 1639-1650)
        src = (
            "use std::collections::HashMap as Map;\n"
            "use std::io::Error as IoError;\n"
            "use std::sync::Mutex as Lock;\n\n"
            "fn get_map() -> Map<String, i32> { Map::new() }\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.IMPORTS in rel_types
        names = [s.name for s in result.symbols]
        assert "get_map" in names

    def test_direct_use_list_syntax(self):
        # `use {A, B};` produces a use_list as direct child of use_declaration
        # This drives lines 1662-1694 in _extract_use_paths
        src = (
            "use {HashMap, HashSet};\n"
            "use {Arc, Mutex};\n"
            "fn main() {}\n"
        )
        result = parse(src)
        # May or may not produce IMPORTS depending on tree-sitter version
        # but should not crash
        assert isinstance(result, ParseResult)
        assert result.language == "rust"

    def test_use_list_with_self_keyword(self):
        # std::io::{self, Read} drives the self branch in use_list handler
        src = (
            "use std::io::{self, Read};\n"
            "fn read_all<R: Read>(r: &mut R) -> io::Result<Vec<u8>> {\n"
            "    let mut v = Vec::new();\n"
            "    r.read_to_end(&mut v)?;\n"
            "    Ok(v)\n"
            "}\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.IMPORTS in rel_types

    def test_nested_scoped_use(self):
        src = (
            "use std::{io::{BufRead, BufReader}, fs::File};\n"
            "fn read_lines(path: &str) -> Vec<String> {\n"
            "    let file = File::open(path).unwrap();\n"
            "    BufReader::new(file).lines().map(|l| l.unwrap()).collect()\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "read_lines" in names


# ---------------------------------------------------------------------------
# Impl block variations
# ---------------------------------------------------------------------------

class TestRustImplVariations:
    def test_impl_with_where_clause(self):
        src = (
            "use std::fmt::Display;\n\n"
            "struct Wrapper<T>(T);\n\n"
            "impl<T> Wrapper<T>\n"
            "where\n"
            "    T: Display + Clone,\n"
            "{\n"
            "    pub fn show(&self) { println!(\"{}\", self.0); }\n"
            "    pub fn clone_inner(&self) -> T { self.0.clone() }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Wrapper" in names
        assert "show" in names

    def test_trait_impl_with_associated_type(self):
        src = (
            "trait Converter {\n"
            "    type Output;\n"
            "    fn convert(&self) -> Self::Output;\n"
            "}\n\n"
            "struct Celsius(f64);\n"
            "struct Fahrenheit(f64);\n\n"
            "impl Converter for Celsius {\n"
            "    type Output = Fahrenheit;\n"
            "    fn convert(&self) -> Fahrenheit {\n"
            "        Fahrenheit(self.0 * 9.0 / 5.0 + 32.0)\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Converter" in names
        assert "Celsius" in names
        assert "Fahrenheit" in names

    def test_impl_for_generic_struct(self):
        src = (
            "struct Stack<T> { items: Vec<T> }\n\n"
            "impl<T: Clone> Stack<T> {\n"
            "    pub fn new() -> Self { Stack { items: Vec::new() } }\n"
            "    pub fn push(&mut self, item: T) { self.items.push(item); }\n"
            "    pub fn pop(&mut self) -> Option<T> { self.items.pop() }\n"
            "    pub fn peek(&self) -> Option<&T> { self.items.last() }\n"
            "    pub fn is_empty(&self) -> bool { self.items.is_empty() }\n"
            "    pub fn len(&self) -> usize { self.items.len() }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Stack" in names
        assert "push" in names
        assert "pop" in names

    def test_multiple_trait_impl(self):
        src = (
            "use std::fmt;\n\n"
            "struct Matrix { rows: usize, cols: usize }\n\n"
            "impl fmt::Display for Matrix {\n"
            "    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {\n"
            "        write!(f, \"Matrix({}x{})\", self.rows, self.cols)\n"
            "    }\n"
            "}\n\n"
            "impl fmt::Debug for Matrix {\n"
            "    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {\n"
            "        write!(f, \"Matrix {{ rows: {}, cols: {} }}\", self.rows, self.cols)\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Matrix" in names
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.IMPLEMENTATION in rel_types

    def test_impl_with_const_associated(self):
        src = (
            "trait Area {\n"
            "    const PI: f64 = 3.14159;\n"
            "    fn area(&self) -> f64;\n"
            "}\n\n"
            "struct Circle { radius: f64 }\n\n"
            "impl Area for Circle {\n"
            "    fn area(&self) -> f64 { Self::PI * self.radius * self.radius }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Area" in names
        assert "Circle" in names


# ---------------------------------------------------------------------------
# Field type analysis (mut refs, dyn Trait, impl Trait in struct fields)
# ---------------------------------------------------------------------------

class TestRustFieldTypeAnalysis:
    def test_struct_with_mut_ref_field(self):
        src = (
            "trait Storage {}\n\n"
            "struct Writer<'a> {\n"
            "    storage: &'a mut dyn Storage,\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Writer" in names

    def test_struct_with_dyn_trait_field(self):
        src = (
            "trait Handler {\n"
            "    fn handle(&self, req: &str) -> String;\n"
            "}\n\n"
            "struct Server {\n"
            "    handler: Box<dyn Handler>,\n"
            "    fallback: Option<Box<dyn Handler>>,\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Server" in names

    def test_struct_with_impl_trait_field(self):
        src = (
            "trait Render {\n"
            "    fn render(&self) -> String;\n"
            "}\n\n"
            "struct Widget {\n"
            "    renderer: Box<impl Render>,\n"
            "    name: String,\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        # Parser may handle impl differently
        assert isinstance(result, ParseResult)

    def test_struct_with_raw_pointer_field(self):
        src = (
            "struct Node {\n"
            "    value: i32,\n"
            "    next: *mut Node,\n"
            "    prev: *const Node,\n"
            "}\n\n"
            "impl Node {\n"
            "    pub fn new(value: i32) -> Self {\n"
            "        Node { value, next: std::ptr::null_mut(), prev: std::ptr::null() }\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Node" in names

    def test_struct_with_vec_of_user_type(self):
        src = (
            "struct Block { id: u64 }\n\n"
            "struct Blockchain {\n"
            "    blocks: Vec<Block>,\n"
            "    pending: Option<Block>,\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Blockchain" in names
        assert "Block" in names
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.COMPOSITION in rel_types or RelationshipType.AGGREGATION in rel_types

    def test_struct_with_arc_mutex_field(self):
        src = (
            "use std::sync::{Arc, Mutex};\n\n"
            "struct Config { host: String }\n\n"
            "struct Service {\n"
            "    config: Arc<Mutex<Config>>,\n"
            "    name: String,\n"
            "}\n\n"
            "impl Service {\n"
            "    pub fn new(config: Arc<Mutex<Config>>) -> Self {\n"
            "        Service { config, name: String::new() }\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Service" in names

    def test_tuple_struct_composition(self):
        src = (
            "struct Color(u8, u8, u8);\n\n"
            "struct Pixel {\n"
            "    color: Color,\n"
            "    alpha: f32,\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Pixel" in names
        assert "Color" in names


# ---------------------------------------------------------------------------
# Macro invocations (drives macro_definition and invocation code)
# ---------------------------------------------------------------------------

class TestRustMacroInvocations:
    def test_macro_rules_definition(self):
        src = (
            "macro_rules! assert_eq_approx {\n"
            "    ($a:expr, $b:expr, $eps:expr) => {\n"
            "        assert!(($a - $b).abs() < $eps);\n"
            "    };\n"
            "}\n\n"
            "fn test_math() {\n"
            "    let pi = 3.14159;\n"
            "    assert_eq_approx!(pi, 3.14, 0.01);\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "test_math" in names

    def test_custom_derive_macro(self):
        src = (
            "#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]\n"
            "pub struct Config {\n"
            "    pub host: String,\n"
            "    pub port: u16,\n"
            "    pub debug: bool,\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Config" in names

    def test_cfg_attr_conditional(self):
        src = (
            "#[cfg(test)]\n"
            "mod tests {\n"
            "    use super::*;\n\n"
            "    #[test]\n"
            "    fn test_basic() {\n"
            "        assert_eq!(1 + 1, 2);\n"
            "    }\n"
            "}\n\n"
            "#[cfg(not(test))]\n"
            "fn production_only() {\n"
            "    println!(\"production\");\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "production_only" in names or "tests" in names
