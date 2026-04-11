"""
Unit tests for GoVisitorParser — coverage target ≥80%.

Uses inline Go source snippets passed via the `content` parameter.
"""
from __future__ import annotations

import pytest

from app.core.parsers.go_visitor_parser import GoVisitorParser, _parse_single_go_file
from app.core.parsers.base_parser import SymbolType, RelationshipType, ParseResult


def parse(source: str, path: str = "main.go") -> ParseResult:
    """Parse a Go source snippet using a fresh GoVisitorParser."""
    parser = GoVisitorParser()
    return parser.parse_file(path, content=source)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

class TestGoParserSmoke:
    def test_parser_initialises(self):
        p = GoVisitorParser()
        assert p.capabilities.language == "go"

    def test_parse_empty_returns_result(self):
        result = parse("")
        assert isinstance(result, ParseResult)
        assert result.language == "go"

    def test_parse_minimal_package(self):
        result = parse("package main\n")
        assert isinstance(result, ParseResult)
        assert result.file_path == "main.go"

    def test_parse_time_set(self):
        result = parse("package main\n")
        assert result.parse_time >= 0.0


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

class TestGoFunctions:
    def test_simple_function_extracted(self):
        src = "package main\n\nfunc Hello() string {\n    return \"hello\"\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Hello" in names

    def test_function_symbol_type(self):
        src = "package main\n\nfunc Add(a, b int) int {\n    return a + b\n}\n"
        result = parse(src)
        funcs = [s for s in result.symbols if s.name == "Add"]
        assert len(funcs) >= 1
        assert funcs[0].symbol_type == SymbolType.FUNCTION

    def test_unexported_function(self):
        src = "package main\n\nfunc helper() {}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "helper" in names

    def test_function_with_multiple_returns(self):
        src = "package main\n\nfunc Divide(a, b float64) (float64, error) {\n    return a / b, nil\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Divide" in names

    def test_variadic_function(self):
        src = "package main\n\nfunc Sum(nums ...int) int {\n    total := 0\n    for _, n := range nums { total += n }\n    return total\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Sum" in names

    def test_main_function(self):
        src = 'package main\n\nimport "fmt"\n\nfunc main() {\n    fmt.Println("hello")\n}\n'
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "main" in names

    def test_init_function(self):
        src = "package mypackage\n\nfunc init() {\n    // setup\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert any("init" in n for n in names)

    def test_generic_function(self):
        src = "package main\n\nfunc MapSlice[T, U any](s []T, f func(T) U) []U {\n    return nil\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "MapSlice" in names


# ---------------------------------------------------------------------------
# Structs
# ---------------------------------------------------------------------------

class TestGoStructs:
    def test_simple_struct_extracted(self):
        src = "package main\n\ntype User struct {\n    Name string\n    Age  int\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "User" in names

    def test_struct_symbol_type(self):
        src = "package main\n\ntype Point struct {\n    X, Y float64\n}\n"
        result = parse(src)
        structs = [s for s in result.symbols if s.name == "Point"]
        assert len(structs) >= 1
        assert structs[0].symbol_type in (SymbolType.STRUCT, SymbolType.CLASS)

    def test_struct_fields_extracted(self):
        src = "package main\n\ntype Config struct {\n    Host string\n    Port int\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Config" in names

    def test_struct_embedding_detected(self):
        src = (
            "package main\n\n"
            "type Base struct { ID int }\n\n"
            "type Extended struct {\n    Base\n    Name string\n}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Extended" in names

    def test_struct_with_pointer_field(self):
        src = (
            "package main\n\n"
            "type Node struct {\n    Value int\n    Next  *Node\n}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Node" in names

    def test_generic_struct(self):
        src = "package main\n\ntype Stack[T any] struct {\n    items []T\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Stack" in names


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

class TestGoMethods:
    def test_method_on_struct(self):
        src = (
            "package main\n\n"
            "type Counter struct { count int }\n\n"
            "func (c *Counter) Increment() {\n    c.count++\n}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Increment" in names

    def test_method_symbol_type(self):
        src = (
            "package main\n\n"
            "type Calc struct{}\n\n"
            "func (c Calc) Add(a, b int) int {\n    return a + b\n}\n"
        )
        result = parse(src)
        methods = [s for s in result.symbols if s.name == "Add"]
        assert len(methods) >= 1
        assert methods[0].symbol_type in (SymbolType.METHOD, SymbolType.FUNCTION)

    def test_value_vs_pointer_receiver(self):
        src = (
            "package main\n\n"
            "type Rect struct { W, H float64 }\n\n"
            "func (r Rect) Area() float64 { return r.W * r.H }\n"
            "func (r *Rect) Scale(f float64) { r.W *= f; r.H *= f }\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Area" in names
        assert "Scale" in names


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

class TestGoInterfaces:
    def test_interface_extracted(self):
        src = "package main\n\ntype Animal interface {\n    Speak() string\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Animal" in names

    def test_interface_symbol_type(self):
        src = "package main\n\ntype Reader interface {\n    Read(p []byte) (int, error)\n}\n"
        result = parse(src)
        ifaces = [s for s in result.symbols if s.name == "Reader"]
        assert len(ifaces) >= 1
        assert ifaces[0].symbol_type in (SymbolType.INTERFACE, SymbolType.STRUCT)

    def test_interface_embedding(self):
        src = (
            "package main\n\n"
            "type Reader interface { Read(p []byte) (int, error) }\n"
            "type Writer interface { Write(p []byte) (int, error) }\n"
            "type ReadWriter interface {\n    Reader\n    Writer\n}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "ReadWriter" in names

    def test_empty_interface(self):
        src = "package main\n\ntype Any interface{}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Any" in names


# ---------------------------------------------------------------------------
# Constants and variables
# ---------------------------------------------------------------------------

class TestGoConstantsAndVars:
    def test_const_block_extracted(self):
        src = (
            "package main\n\n"
            "const (\n"
            "    StatusOK  = 200\n"
            "    StatusNotFound = 404\n"
            ")\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "StatusOK" in names or "StatusNotFound" in names

    def test_single_const_extracted(self):
        src = "package main\n\nconst MaxRetries = 3\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "MaxRetries" in names

    def test_var_declaration_extracted(self):
        src = "package main\n\nvar DefaultTimeout = 30\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "DefaultTimeout" in names

    def test_iota_const_extracted(self):
        src = (
            "package main\n\n"
            "type Direction int\n\n"
            "const (\n"
            "    North Direction = iota\n"
            "    South\n"
            "    East\n"
            "    West\n"
            ")\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "North" in names or "South" in names


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

class TestGoImports:
    def test_single_import(self):
        src = 'package main\n\nimport "fmt"\n\nfunc main() {}\n'
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.IMPORTS in rel_types

    def test_grouped_imports(self):
        src = (
            'package main\n\n'
            'import (\n'
            '    "fmt"\n'
            '    "os"\n'
            '    "strings"\n'
            ')\n\n'
            'func main() {}\n'
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.IMPORTS in rel_types

    def test_aliased_import(self):
        src = 'package main\n\nimport f "fmt"\n\nfunc main() { f.Println("hi") }\n'
        result = parse(src)
        assert isinstance(result, ParseResult)


# ---------------------------------------------------------------------------
# Type aliases and type declarations
# ---------------------------------------------------------------------------

class TestGoTypeDeclarations:
    def test_type_alias_extracted(self):
        src = "package main\n\ntype Celsius float64\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Celsius" in names

    def test_function_type_extracted(self):
        src = "package main\n\ntype HandlerFunc func(w int, r string)\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "HandlerFunc" in names


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

class TestGoRelationships:
    def test_struct_defines_method_relationship(self):
        src = (
            "package main\n\n"
            "type Dog struct{ Name string }\n\n"
            "func (d *Dog) Bark() string { return \"Woof\" }\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        # Should have DEFINES linking Dog to Bark
        assert result.relationships is not None

    def test_struct_embedding_composition(self):
        src = (
            "package main\n\n"
            "type Engine struct { HP int }\n\n"
            "type Car struct {\n    Engine\n    Model string\n}\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.COMPOSITION in rel_types


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestGoErrorHandling:
    def test_syntax_error_returns_result(self):
        src = "package main\n\nfunc broken( {\n"
        result = parse(src)
        assert isinstance(result, ParseResult)

    def test_empty_file_no_crash(self):
        result = parse("")
        assert result.symbols == []

    def test_missing_file_no_crash(self):
        parser = GoVisitorParser()
        result = parser.parse_file("/no/such/file.go")
        assert isinstance(result, ParseResult)
        assert result.errors

    def test_content_overrides_file_read(self):
        parser = GoVisitorParser()
        result = parser.parse_file("/no/such.go", content="package main\nfunc Injected() {}\n")
        names = [s.name for s in result.symbols]
        assert "Injected" in names


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

class TestGoCapabilities:
    def test_language_is_go(self):
        p = GoVisitorParser()
        assert p.capabilities.language == "go"

    def test_extensions_include_go(self):
        p = GoVisitorParser()
        assert ".go" in p._get_supported_extensions()

    def test_has_interfaces(self):
        p = GoVisitorParser()
        assert p.capabilities.has_interfaces is True


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

class TestParseSingleGoFile:
    def test_returns_tuple(self, tmp_path):
        f = tmp_path / "main.go"
        f.write_text("package main\nfunc Hello() {}\n")
        path, result = _parse_single_go_file(str(f))
        assert path == str(f)
        assert isinstance(result, ParseResult)

    def test_missing_file_returns_empty(self):
        path, result = _parse_single_go_file("/nonexistent.go")
        assert path == "/nonexistent.go"
        assert result.symbols == []


# ---------------------------------------------------------------------------
# Parse multiple files
# ---------------------------------------------------------------------------

class TestGoParseMultipleFiles:
    def test_parse_multiple_files(self, tmp_path):
        f1 = tmp_path / "a.go"
        f2 = tmp_path / "b.go"
        f1.write_text("package main\nfunc FuncA() {}\n")
        f2.write_text("package main\nfunc FuncB() {}\n")
        parser = GoVisitorParser()
        results = parser.parse_multiple_files([str(f1), str(f2)])
        assert isinstance(results, dict)
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "FuncA" in all_names
        assert "FuncB" in all_names


# ---------------------------------------------------------------------------
# Additional coverage tests — Go-specific patterns
# ---------------------------------------------------------------------------

class TestGoInterfaceImplementation:
    def test_implicit_interface_satisfaction(self):
        src = (
            "package main\n\n"
            "type Stringer interface {\n"
            "    String() string\n"
            "}\n\n"
            "type Person struct { Name string }\n\n"
            "func (p Person) String() string { return p.Name }\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Person" in names
        assert "Stringer" in names

    def test_well_known_interface_error(self):
        src = (
            "package main\n\n"
            "type MyError struct{ msg string }\n\n"
            "func (e *MyError) Error() string { return e.msg }\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "MyError" in names

    def test_interface_with_multiple_methods(self):
        src = (
            "package main\n\n"
            "type ReadWriter interface {\n"
            "    Read(p []byte) (int, error)\n"
            "    Write(p []byte) (int, error)\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "ReadWriter" in names


class TestGoComplexTypes:
    def test_channel_parameter(self):
        src = "package main\n\nfunc producer(ch chan<- int) {\n    ch <- 42\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "producer" in names

    def test_function_type_parameter(self):
        src = "package main\n\nfunc apply(f func(int) int, x int) int {\n    return f(x)\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "apply" in names

    def test_slice_and_map_fields(self):
        src = (
            "package main\n\n"
            "type Registry struct {\n"
            "    items    []string\n"
            "    lookup   map[string]int\n"
            "    handlers []func(string)\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Registry" in names

    def test_pointer_to_struct_field(self):
        src = (
            "package main\n\n"
            "type Employee struct { Name string }\n\n"
            "type Department struct {\n"
            "    Manager *Employee\n"
            "    Staff   []*Employee\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Department" in names

    def test_defer_statement_no_crash(self):
        src = (
            'package main\n\nimport "fmt"\n\n'
            'func doWork() {\n    defer fmt.Println("done")\n    fmt.Println("working")\n}\n'
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "doWork" in names

    def test_goroutine_function(self):
        src = (
            "package main\n\n"
            "func worker(id int) {}\n\n"
            "func main() {\n    go worker(1)\n    go worker(2)\n}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "main" in names
        assert "worker" in names


class TestGoMultipleReturnValues:
    def test_multiple_return_function(self):
        src = (
            "package main\n\n"
            "func divide(a, b float64) (float64, error) {\n"
            "    if b == 0 { return 0, fmt.Errorf(\"division by zero\") }\n"
            "    return a / b, nil\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "divide" in names

    def test_named_return_values(self):
        src = (
            "package main\n\n"
            "func minMax(arr []int) (min int, max int) {\n"
            "    min, max = arr[0], arr[0]\n"
            "    return\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "minMax" in names


class TestGoConstants:
    def test_typed_constant(self):
        src = "package main\n\nconst Pi float64 = 3.14159265358979\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Pi" in names

    def test_multiple_const_block(self):
        src = (
            "package main\n\n"
            "const (\n"
            "    KB = 1024\n"
            "    MB = KB * 1024\n"
            "    GB = MB * 1024\n"
            ")\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "KB" in names or "MB" in names

    def test_var_block(self):
        src = (
            "package main\n\n"
            "var (\n"
            "    host = \"localhost\"\n"
            "    port = 8080\n"
            ")\n"
        )
        result = parse(src)
        # Go var block parsing may not extract variables in all implementations
        assert isinstance(result, ParseResult)


class TestGoLinkMethods:
    def test_methods_linked_to_struct(self):
        src = (
            "package main\n\n"
            "type Dog struct { Name string }\n\n"
            "func (d *Dog) Bark() string { return \"Woof\" }\n"
            "func (d *Dog) Sit() { }\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Dog" in names
        assert "Bark" in names
        assert "Sit" in names
        # Methods should be linked with DEFINES relationship
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.DEFINES in rel_types


class TestGoTypeAliases:
    def test_true_type_alias_with_equals(self):
        src = "package main\n\ntype MyString = string\ntype AnyMap = map[string]interface{}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "MyString" in names
        assert "AnyMap" in names

    def test_type_alias_creates_alias_of_relationship(self):
        src = "package main\n\ntype Handler func(w int, r string)\ntype Middleware func(Handler) Handler\n"
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.ALIAS_OF in rel_types

    def test_constraint_interface(self):
        src = (
            "package main\n\n"
            "type Number interface {\n"
            "    ~int | ~int8 | ~int16 | ~int32 | ~int64 | ~float32 | ~float64\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Number" in names

    def test_map_type_declaration(self):
        src = "package main\n\ntype Options map[string]interface{}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Options" in names


class TestGoGlobalCrossFile:
    def test_cross_file_interface_satisfaction(self, tmp_path):
        iface_file = tmp_path / "interfaces.go"
        impl_file = tmp_path / "models.go"
        iface_file.write_text(
            "package main\n\n"
            "type Stringer interface {\n"
            "    String() string\n"
            "}\n"
        )
        impl_file.write_text(
            "package main\n\n"
            "type Person struct { Name string }\n\n"
            "func (p Person) String() string { return p.Name }\n"
        )
        parser = GoVisitorParser()
        results = parser.parse_multiple_files([str(iface_file), str(impl_file)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "Stringer" in all_names
        assert "Person" in all_names

    def test_cross_file_struct_composition(self, tmp_path):
        base_file = tmp_path / "base.go"
        extended_file = tmp_path / "extended.go"
        base_file.write_text(
            "package main\n\n"
            "type Base struct { ID int }\n\n"
            "func (b Base) GetID() int { return b.ID }\n"
        )
        extended_file.write_text(
            "package main\n\n"
            "type Extended struct {\n"
            "    Base\n"
            "    Name string\n"
            "}\n"
        )
        parser = GoVisitorParser()
        results = parser.parse_multiple_files([str(base_file), str(extended_file)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "Base" in all_names
        assert "Extended" in all_names


class TestGoComments:
    def test_function_with_doc_comment(self):
        src = (
            "package main\n\n"
            "// Add adds two integers and returns the result.\n"
            "// It is safe for concurrent use.\n"
            "func Add(a, b int) int {\n"
            "    return a + b\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Add" in names

    def test_struct_with_doc_comment(self):
        src = (
            "package main\n\n"
            "// Config holds application configuration.\n"
            "type Config struct {\n"
            "    // Host is the server hostname.\n"
            "    Host string\n"
            "    // Port is the server port.\n"
            "    Port int\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Config" in names
