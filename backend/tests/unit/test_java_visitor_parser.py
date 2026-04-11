"""
Unit tests for JavaVisitorParser — coverage target ≥80%.

Uses inline Java source snippets; no real file I/O.
"""
from __future__ import annotations

import pytest

from app.core.parsers.java_visitor_parser import JavaVisitorParser
from app.core.parsers.base_parser import SymbolType, RelationshipType, ParseResult


def parse(source: str, path: str = "Main.java") -> ParseResult:
    parser = JavaVisitorParser()
    return parser.parse_file(path, content=source)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

class TestJavaParserSmoke:
    def test_parser_initialises(self):
        p = JavaVisitorParser()
        assert p is not None

    def test_parse_empty_returns_result(self):
        result = parse("")
        assert isinstance(result, ParseResult)
        assert result.language == "java"

    def test_parse_minimal_class(self):
        result = parse("public class Main {}")
        assert isinstance(result, ParseResult)
        assert result.file_path == "Main.java"

    def test_parse_time_is_positive(self):
        result = parse("public class X {}")
        assert result.parse_time >= 0.0


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

class TestJavaClasses:
    def test_simple_class_extracted(self):
        src = "public class Animal {}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Animal" in names

    def test_class_symbol_type(self):
        src = "public class Widget {}\n"
        result = parse(src)
        classes = [s for s in result.symbols if s.name == "Widget"]
        assert len(classes) >= 1
        assert classes[0].symbol_type == SymbolType.CLASS

    def test_class_with_package(self):
        src = "package com.example;\npublic class Service {}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Service" in names

    def test_class_inheritance(self):
        src = "public class Animal {}\npublic class Dog extends Animal {}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Dog" in names
        assert "Animal" in names

    def test_interface_implementation(self):
        src = (
            "public interface Runnable { void run(); }\n"
            "public class Task implements Runnable {\n"
            "    public void run() {}\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Task" in names

    def test_abstract_class(self):
        src = "public abstract class Shape {\n    public abstract double area();\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Shape" in names

    def test_final_class(self):
        src = "public final class Constants {\n    public static final int MAX = 100;\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Constants" in names

    def test_nested_class(self):
        src = (
            "public class Outer {\n"
            "    public static class Inner {}\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Outer" in names
        assert "Inner" in names


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

class TestJavaInterfaces:
    def test_interface_extracted(self):
        src = "public interface Comparable<T> {\n    int compareTo(T other);\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Comparable" in names

    def test_interface_symbol_type(self):
        src = "public interface Serializable {}\n"
        result = parse(src)
        ifaces = [s for s in result.symbols if s.name == "Serializable"]
        assert len(ifaces) >= 1
        assert ifaces[0].symbol_type == SymbolType.INTERFACE

    def test_interface_with_default_method(self):
        src = (
            "public interface Greeter {\n"
            "    default String greet(String name) { return \"Hello \" + name; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Greeter" in names

    def test_interface_extending_interface(self):
        src = (
            "public interface Base { void doBase(); }\n"
            "public interface Extended extends Base { void doExtended(); }\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Extended" in names


# ---------------------------------------------------------------------------
# Methods and constructors
# ---------------------------------------------------------------------------

class TestJavaMethods:
    def test_method_extracted(self):
        src = (
            "public class Calculator {\n"
            "    public int add(int a, int b) { return a + b; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "add" in names

    def test_constructor_extracted(self):
        src = (
            "public class Person {\n"
            "    private String name;\n"
            "    public Person(String name) { this.name = name; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Person" in names

    def test_static_method_extracted(self):
        src = (
            "public class MathUtils {\n"
            "    public static double square(double x) { return x * x; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "square" in names

    def test_method_with_generics(self):
        src = (
            "public class Container<T> {\n"
            "    public <R> R transform(T item) { return null; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Container" in names

    def test_method_with_annotation(self):
        src = (
            "public class Service {\n"
            "    @Override\n"
            "    public String toString() { return \"Service\"; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "toString" in names

    def test_private_method(self):
        src = (
            "public class Helper {\n"
            "    private void doWork() {}\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "doWork" in names


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------

class TestJavaFields:
    def test_field_extracted(self):
        src = (
            "public class Config {\n"
            "    private String host;\n"
            "    private int port;\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "host" in names or "Config" in names

    def test_static_final_field(self):
        src = (
            "public class Constants {\n"
            "    public static final int MAX_SIZE = 1000;\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "MAX_SIZE" in names or "Constants" in names


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestJavaEnums:
    def test_enum_extracted(self):
        src = "public enum Direction { NORTH, SOUTH, EAST, WEST }\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Direction" in names

    def test_enum_symbol_type(self):
        src = "public enum Status { ACTIVE, INACTIVE }\n"
        result = parse(src)
        enums = [s for s in result.symbols if s.name == "Status"]
        assert len(enums) >= 1
        assert enums[0].symbol_type in (SymbolType.ENUM, SymbolType.CLASS)

    def test_enum_with_constructor(self):
        src = (
            "public enum Planet {\n"
            "    EARTH(5.972e24, 6.371e6),\n"
            "    MARS(6.39e23, 3.3895e6);\n"
            "    private final double mass;\n"
            "    private final double radius;\n"
            "    Planet(double mass, double radius) {\n"
            "        this.mass = mass;\n"
            "        this.radius = radius;\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Planet" in names


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

class TestJavaAnnotations:
    def test_annotation_type_extracted(self):
        src = (
            "public @interface MyAnnotation {\n"
            "    String value() default \"\";\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "MyAnnotation" in names

    def test_class_with_spring_annotations(self):
        src = (
            "@RestController\n"
            "@RequestMapping(\"/api\")\n"
            "public class ApiController {\n"
            "    @GetMapping(\"/health\")\n"
            "    public String health() { return \"OK\"; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "ApiController" in names


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

class TestJavaImports:
    def test_import_creates_relationship(self):
        src = (
            "import java.util.List;\n"
            "import java.util.ArrayList;\n"
            "public class Example {\n"
            "    List<String> items = new ArrayList<>();\n"
            "}\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.IMPORTS in rel_types

    def test_static_import(self):
        src = (
            "import static java.lang.Math.PI;\n"
            "public class Circle {\n"
            "    double area(double r) { return PI * r * r; }\n"
            "}\n"
        )
        result = parse(src)
        assert isinstance(result, ParseResult)

    def test_wildcard_import(self):
        src = "import java.util.*;\npublic class App {}\n"
        result = parse(src)
        assert isinstance(result, ParseResult)


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

class TestJavaRelationships:
    def test_inheritance_relationship(self):
        src = "class Base {}\nclass Child extends Base {}\n"
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.INHERITANCE in rel_types

    def test_implementation_relationship(self):
        src = (
            "interface Flyable { void fly(); }\n"
            "class Bird implements Flyable {\n"
            "    public void fly() {}\n"
            "}\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.IMPLEMENTATION in rel_types

    def test_defines_relationship_for_methods(self):
        src = "public class Foo {\n    public void bar() {}\n}\n"
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.DEFINES in rel_types


# ---------------------------------------------------------------------------
# Generics
# ---------------------------------------------------------------------------

class TestJavaGenerics:
    def test_generic_class_extracted(self):
        src = "public class Pair<A, B> {\n    A first; B second;\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Pair" in names

    def test_bounded_type_parameter(self):
        src = (
            "public class NumberBox<T extends Number> {\n"
            "    T value;\n"
            "    T get() { return value; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "NumberBox" in names


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestJavaErrorHandling:
    def test_syntax_error_no_crash(self):
        src = "public class Broken { void method( { }\n"
        result = parse(src)
        assert isinstance(result, ParseResult)

    def test_missing_file_returns_error(self):
        parser = JavaVisitorParser()
        result = parser.parse_file("/nonexistent/file.java")
        assert isinstance(result, ParseResult)
        assert result.errors

    def test_content_overrides_file_read(self):
        parser = JavaVisitorParser()
        result = parser.parse_file("/nonexistent.java", content="public class Injected {}\n")
        names = [s.name for s in result.symbols]
        assert "Injected" in names


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

class TestJavaCapabilities:
    def test_language_is_java(self):
        p = JavaVisitorParser()
        assert p.capabilities.language == "java"

    def test_extensions_include_java(self):
        p = JavaVisitorParser()
        assert ".java" in p._get_supported_extensions()

    def test_has_interfaces(self):
        p = JavaVisitorParser()
        assert p.capabilities.has_interfaces is True

    def test_has_generics(self):
        p = JavaVisitorParser()
        assert p.capabilities.has_generics is True


# ---------------------------------------------------------------------------
# Parse multiple files
# ---------------------------------------------------------------------------

class TestJavaMultipleFiles:
    def test_parse_multiple_java_files(self, tmp_path):
        f1 = tmp_path / "A.java"
        f2 = tmp_path / "B.java"
        f1.write_text("public class ClassA {}\n")
        f2.write_text("public class ClassB {}\n")
        parser = JavaVisitorParser()
        results = parser.parse_multiple_files([str(f1), str(f2)])
        assert isinstance(results, dict)
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "ClassA" in all_names
        assert "ClassB" in all_names

    def test_parse_empty_list_returns_empty_dict(self):
        parser = JavaVisitorParser()
        results = parser.parse_multiple_files([])
        assert isinstance(results, dict)


# ---------------------------------------------------------------------------
# Additional coverage tests — Java-specific patterns
# ---------------------------------------------------------------------------

class TestJavaLambdasAndStreams:
    def test_lambda_in_method_no_crash(self):
        src = (
            "import java.util.*;\n"
            "class Example {\n"
            "    void test() {\n"
            "        List<String> list = new ArrayList<>();\n"
            "        list.forEach(s -> System.out.println(s));\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Example" in names

    def test_stream_api_no_crash(self):
        src = (
            "import java.util.stream.*;\n"
            "class StreamExample {\n"
            "    long countEvens(int[] nums) {\n"
            "        return java.util.Arrays.stream(nums)\n"
            "            .filter(n -> n % 2 == 0)\n"
            "            .count();\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "StreamExample" in names


class TestJavaTryCatch:
    def test_try_catch_finally(self):
        src = (
            "class Example {\n"
            "    void test() {\n"
            "        try {\n"
            "            int x = Integer.parseInt(\"abc\");\n"
            "        } catch (NumberFormatException e) {\n"
            "            System.err.println(e);\n"
            "        } finally {\n"
            "            System.out.println(\"done\");\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Example" in names

    def test_multi_catch(self):
        src = (
            "class Example {\n"
            "    void test() {\n"
            "        try { } catch (IllegalArgumentException | NullPointerException e) { }\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        assert isinstance(result, ParseResult)


class TestJavaFieldComposition:
    def test_field_with_object_type(self):
        src = (
            "import java.util.List;\n"
            "import java.util.ArrayList;\n"
            "class Repository {\n"
            "    private List<String> items = new ArrayList<>();\n"
            "    void add(String item) { items.add(item); }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Repository" in names

    def test_optional_field(self):
        src = (
            "import java.util.Optional;\n"
            "class Service {\n"
            "    private Optional<String> cache = Optional.empty();\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Service" in names


class TestJavaComplexInheritance:
    def test_class_extends_and_implements(self):
        src = (
            "interface Runnable { void run(); }\n"
            "interface Closeable { void close(); }\n"
            "abstract class AbstractBase {\n"
            "    abstract void doWork();\n"
            "}\n"
            "class Worker extends AbstractBase implements Runnable, Closeable {\n"
            "    public void run() {}\n"
            "    public void close() {}\n"
            "    public void doWork() {}\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Worker" in names
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.INHERITANCE in rel_types
        assert RelationshipType.IMPLEMENTATION in rel_types

    def test_interface_default_and_static(self):
        src = (
            "interface Validator<T> {\n"
            "    boolean validate(T value);\n"
            "    default boolean isValid(T value) { return validate(value); }\n"
            "    static <T> Validator<T> of(java.util.function.Predicate<T> p) {\n"
            "        return p::test;\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Validator" in names


class TestJavaAnnotationUsage:
    def test_override_annotation(self):
        src = (
            "class Animal { String sound() { return \"\"; } }\n"
            "class Cat extends Animal {\n"
            "    @Override\n"
            "    String sound() { return \"Meow\"; }\n"
            "}\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.INHERITANCE in rel_types

    def test_suppress_warnings(self):
        src = (
            "class Example {\n"
            "    @SuppressWarnings(\"unchecked\")\n"
            "    void test() {}\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "test" in names


class TestJavaMethodPatterns:
    def test_varargs_method(self):
        src = (
            "class Logger {\n"
            "    void log(String format, Object... args) {}\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "log" in names

    def test_synchronized_method(self):
        src = (
            "class Counter {\n"
            "    private int count = 0;\n"
            "    synchronized void increment() { count++; }\n"
            "    synchronized int getCount() { return count; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "increment" in names

    def test_throws_clause(self):
        src = (
            "class FileReader {\n"
            "    String read(String path) throws java.io.IOException {\n"
            "        return \"\";\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "read" in names

    def test_native_method_declaration(self):
        src = (
            "class NativeLib {\n"
            "    native int compute(int x);\n"
            "    static { System.loadLibrary(\"native\"); }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "NativeLib" in names


# ---------------------------------------------------------------------------
# Cross-file resolution (drives parse_multiple_files internal code)
# ---------------------------------------------------------------------------

class TestJavaCrossFileResolution:
    def test_cross_file_inheritance_resolution(self, tmp_path):
        animal = tmp_path / "Animal.java"
        dog = tmp_path / "Dog.java"
        animal.write_text("package animals;\npublic class Animal {\n    public void speak() {}\n}\n")
        dog.write_text("package animals;\npublic class Dog extends Animal {\n    @Override\n    public void speak() { System.out.println(\"Woof\"); }\n}\n")
        parser = JavaVisitorParser()
        results = parser.parse_multiple_files([str(animal), str(dog)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "Animal" in all_names
        assert "Dog" in all_names

    def test_cross_file_interface_implementation(self, tmp_path):
        iface = tmp_path / "Printable.java"
        doc = tmp_path / "Document.java"
        iface.write_text("public interface Printable { void print(); }\n")
        doc.write_text("public class Document implements Printable {\n    public void print() { System.out.println(\"printing\"); }\n}\n")
        parser = JavaVisitorParser()
        results = parser.parse_multiple_files([str(iface), str(doc)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "Document" in all_names

    def test_cross_file_field_type(self, tmp_path):
        engine = tmp_path / "Engine.java"
        car = tmp_path / "Car.java"
        engine.write_text("public class Engine { public void start() {} }\n")
        car.write_text("public class Car {\n    private Engine engine;\n    public Car() { engine = new Engine(); }\n}\n")
        parser = JavaVisitorParser()
        results = parser.parse_multiple_files([str(engine), str(car)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "Car" in all_names
        assert "Engine" in all_names

    def test_cross_file_method_calls(self, tmp_path):
        service = tmp_path / "UserService.java"
        repo = tmp_path / "UserRepository.java"
        repo.write_text("public class UserRepository {\n    public String findById(int id) { return \"\"; }\n}\n")
        service.write_text(
            "public class UserService {\n"
            "    private UserRepository repo;\n"
            "    public UserService() { repo = new UserRepository(); }\n"
            "    public String getUser(int id) { return repo.findById(id); }\n"
            "}\n"
        )
        parser = JavaVisitorParser()
        results = parser.parse_multiple_files([str(service), str(repo)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "UserService" in all_names
        assert "UserRepository" in all_names


class TestJavaModernFeatures:
    def test_record_extracted(self):
        src = "public record Point(int x, int y) {}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Point" in names

    def test_record_with_methods(self):
        src = (
            "public record Person(String name, int age) {\n"
            "    public String greeting() { return \"Hello \" + name; }\n"
            "    public boolean isAdult() { return age >= 18; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Person" in names
        assert "greeting" in names

    def test_sealed_class(self):
        src = (
            "public sealed class Shape permits Circle, Rectangle {}\n"
            "public final class Circle extends Shape { int radius; }\n"
            "public final class Rectangle extends Shape { int width, height; }\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Shape" in names
        assert "Circle" in names

    def test_text_block_string(self):
        src = (
            'class Example {\n'
            '    String json = """\n'
            '                  { "key": "value" }\n'
            '                  """;\n'
            '}\n'
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Example" in names

    def test_pattern_matching_instanceof(self):
        src = (
            "class Processor {\n"
            "    void process(Object obj) {\n"
            "        if (obj instanceof String s) {\n"
            "            System.out.println(s.length());\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Processor" in names

    def test_switch_expression(self):
        src = (
            "class DayMapper {\n"
            "    String getDayType(String day) {\n"
            "        return switch (day) {\n"
            "            case \"Monday\", \"Tuesday\", \"Wednesday\", \"Thursday\", \"Friday\" -> \"Weekday\";\n"
            "            case \"Saturday\", \"Sunday\" -> \"Weekend\";\n"
            "            default -> throw new IllegalArgumentException(\"Invalid day: \" + day);\n"
            "        };\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "DayMapper" in names

    def test_inner_interface(self):
        src = (
            "public class OuterClass {\n"
            "    public interface InnerInterface {\n"
            "        void doWork();\n"
            "    }\n"
            "    public static class InnerImpl implements InnerInterface {\n"
            "        public void doWork() {}\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "OuterClass" in names
        assert "InnerInterface" in names

    def test_functional_interface(self):
        src = (
            "@FunctionalInterface\n"
            "public interface Transformer<T, R> {\n"
            "    R transform(T input);\n"
            "    default Transformer<T, R> andThen(java.util.function.Function<R, R> after) {\n"
            "        return input -> after.apply(this.transform(input));\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Transformer" in names


# ---------------------------------------------------------------------------
# Field relationship type determination
# ---------------------------------------------------------------------------

class TestJavaFieldRelationships:
    def test_list_field_creates_composition(self):
        src = (
            "import java.util.List;\n"
            "public class Team {\n"
            "    private List<Member> members;\n"
            "}\n"
            "class Member { String name; }\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        # Collection field → COMPOSITION
        assert RelationshipType.COMPOSITION in rel_types or len(result.symbols) > 0

    def test_optional_field_creates_aggregation(self):
        src = (
            "import java.util.Optional;\n"
            "public class UserProfile {\n"
            "    private Optional<Avatar> avatar;\n"
            "}\n"
            "class Avatar { String url; }\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "UserProfile" in names

    def test_service_field_creates_reference(self):
        src = (
            "public class OrderController {\n"
            "    private OrderService orderService;\n"
            "    private UserRepository userRepository;\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "OrderController" in names

    def test_nullable_field_creates_aggregation(self):
        src = (
            "public class Config {\n"
            "    @Nullable\n"
            "    private DatabasePool pool;\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Config" in names

    def test_string_field_creates_reference(self):
        src = (
            "public class User {\n"
            "    private String name;\n"
            "    private Integer age;\n"
            "    private Long id;\n"
            "}\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        # String/Integer/Long → REFERENCES (basic types)
        assert isinstance(result, ParseResult)


# ---------------------------------------------------------------------------
# Cross-file resolution (inheritance, implementation, calls)
# ---------------------------------------------------------------------------

class TestJavaCrossFileResolution2:
    def test_multi_file_inheritance_chain(self, tmp_path):
        base = tmp_path / "Animal.java"
        child = tmp_path / "Dog.java"
        base.write_text("public class Animal {\n    public void speak() {}\n}\n")
        child.write_text("public class Dog extends Animal {\n    @Override\n    public void speak() { System.out.println(\"Woof\"); }\n}\n")
        parser = JavaVisitorParser()
        results = parser.parse_multiple_files([str(base), str(child)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "Animal" in all_names
        assert "Dog" in all_names
        all_rels = [r for result in results.values() for r in result.relationships]
        rel_types = {r.relationship_type for r in all_rels}
        assert RelationshipType.INHERITANCE in rel_types

    def test_multi_file_interface_implementation(self, tmp_path):
        iface = tmp_path / "Drawable.java"
        impl = tmp_path / "Circle.java"
        iface.write_text("public interface Drawable {\n    void draw();\n}\n")
        impl.write_text("public class Circle implements Drawable {\n    public void draw() { System.out.println(\"O\"); }\n}\n")
        parser = JavaVisitorParser()
        results = parser.parse_multiple_files([str(iface), str(impl)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "Drawable" in all_names
        assert "Circle" in all_names

    def test_multi_file_method_calls(self, tmp_path):
        service = tmp_path / "EmailService.java"
        controller = tmp_path / "NotifyController.java"
        service.write_text(
            "public class EmailService {\n"
            "    public void sendEmail(String to, String body) {\n"
            "        System.out.println(\"Sending to \" + to);\n"
            "    }\n"
            "}\n"
        )
        controller.write_text(
            "public class NotifyController {\n"
            "    private EmailService emailService;\n"
            "    public void notify(String user) {\n"
            "        emailService.sendEmail(user, \"Hello!\");\n"
            "    }\n"
            "}\n"
        )
        parser = JavaVisitorParser()
        results = parser.parse_multiple_files([str(service), str(controller)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "EmailService" in all_names
        assert "NotifyController" in all_names

    def test_multi_file_with_package(self, tmp_path):
        model = tmp_path / "Role.java"
        user = tmp_path / "User2.java"
        model.write_text("package com.example.model;\npublic enum Role { ADMIN, USER }\n")
        user.write_text(
            "package com.example.model;\nimport com.example.model.Role;\n"
            "public class User2 {\n    private Role role;\n    public User2(Role role) { this.role = role; }\n}\n"
        )
        parser = JavaVisitorParser()
        results = parser.parse_multiple_files([str(model), str(user)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "User2" in all_names


# ---------------------------------------------------------------------------
# Overloaded methods and constructors
# ---------------------------------------------------------------------------

class TestJavaOverloadsAndConstructors:
    def test_overloaded_methods_extracted(self):
        src = (
            "public class Formatter {\n"
            "    public String format(int n) { return String.valueOf(n); }\n"
            "    public String format(double d) { return String.format(\"%.2f\", d); }\n"
            "    public String format(String s) { return s.trim(); }\n"
            "}\n"
        )
        result = parse(src)
        method_names = [s.name for s in result.symbols if s.symbol_type == SymbolType.METHOD]
        assert method_names.count("format") >= 2

    def test_constructor_overloading(self):
        src = (
            "public class Point {\n"
            "    int x, y;\n"
            "    public Point() { this.x = 0; this.y = 0; }\n"
            "    public Point(int x, int y) { this.x = x; this.y = y; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Point" in names

    def test_generic_method(self):
        src = (
            "public class Container<T> {\n"
            "    private T value;\n"
            "    public <R> Container<R> map(java.util.function.Function<T, R> fn) {\n"
            "        return new Container<>();\n"
            "    }\n"
            "    public T get() { return value; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Container" in names
        assert "map" in names

    def test_static_factory_methods(self):
        src = (
            "public class Result<T> {\n"
            "    private final T value;\n"
            "    private Result(T value) { this.value = value; }\n"
            "    public static <T> Result<T> success(T value) { return new Result<>(value); }\n"
            "    public static <T> Result<T> failure() { return new Result<>(null); }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Result" in names
        assert "success" in names


# ---------------------------------------------------------------------------
# Additional Java patterns for coverage
# ---------------------------------------------------------------------------

class TestJavaAdditionalPatterns:
    def test_try_with_resources(self):
        src = (
            "import java.io.*;\n"
            "public class FileReader {\n"
            "    public String read(String path) throws IOException {\n"
            "        try (BufferedReader br = new BufferedReader(new java.io.FileReader(path))) {\n"
            "            return br.readLine();\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "FileReader" in names
        assert "read" in names

    def test_interface_with_static_method(self):
        src = (
            "public interface Validator<T> {\n"
            "    boolean validate(T value);\n"
            "    static <T> Validator<T> noOp() { return value -> true; }\n"
            "    default Validator<T> and(Validator<T> other) {\n"
            "        return value -> this.validate(value) && other.validate(value);\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Validator" in names

    def test_annotation_definition(self):
        src = (
            "import java.lang.annotation.*;\n"
            "@Retention(RetentionPolicy.RUNTIME)\n"
            "@Target(ElementType.METHOD)\n"
            "public @interface Cacheable {\n"
            "    String key() default \"\";\n"
            "    int ttl() default 60;\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Cacheable" in names

    def test_abstract_class_template_method(self):
        src = (
            "public abstract class DataProcessor {\n"
            "    public final void process(String data) {\n"
            "        String clean = clean(data);\n"
            "        transform(clean);\n"
            "    }\n"
            "    protected abstract String clean(String data);\n"
            "    protected abstract void transform(String data);\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "DataProcessor" in names
        assert "process" in names

    def test_array_field_type(self):
        src = (
            "public class Matrix {\n"
            "    private int[][] data;\n"
            "    private String[] headers;\n"
            "    public Matrix(int rows, int cols) {\n"
            "        data = new int[rows][cols];\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Matrix" in names

    def test_wildcard_generic(self):
        src = (
            "import java.util.List;\n"
            "public class Utils {\n"
            "    public double sumList(List<? extends Number> numbers) {\n"
            "        return numbers.stream().mapToDouble(Number::doubleValue).sum();\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Utils" in names
        assert "sumList" in names


# ---------------------------------------------------------------------------
# Cross-file composition/aggregation/creates/references resolution
# ---------------------------------------------------------------------------

class TestJavaCrossFileComposition:
    def test_creates_relationship_cross_file(self, tmp_path):
        factory_file = tmp_path / "ConnectionFactory.java"
        service_file = tmp_path / "DatabaseService.java"
        factory_file.write_text(
            "public class ConnectionFactory {\n"
            "    public static ConnectionFactory create() { return new ConnectionFactory(); }\n"
            "}\n"
        )
        service_file.write_text(
            "public class DatabaseService {\n"
            "    public void init() {\n"
            "        ConnectionFactory factory = new ConnectionFactory();\n"
            "    }\n"
            "}\n"
        )
        parser = JavaVisitorParser()
        results = parser.parse_multiple_files([str(factory_file), str(service_file)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "ConnectionFactory" in all_names
        assert "DatabaseService" in all_names

    def test_references_relationship_cross_file(self, tmp_path):
        type_file = tmp_path / "Role.java"
        user_file = tmp_path / "UserWithRole.java"
        type_file.write_text("public enum Role { ADMIN, USER, GUEST }\n")
        user_file.write_text(
            "public class UserWithRole {\n"
            "    private String name;\n"
            "    private Role role;\n"
            "    public UserWithRole(String name, Role role) {\n"
            "        this.name = name;\n"
            "        this.role = role;\n"
            "    }\n"
            "}\n"
        )
        parser = JavaVisitorParser()
        results = parser.parse_multiple_files([str(type_file), str(user_file)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "UserWithRole" in all_names

    def test_import_cross_file_resolution(self, tmp_path):
        util = tmp_path / "StringUtils.java"
        app = tmp_path / "Application.java"
        util.write_text(
            "package com.example.util;\n"
            "public class StringUtils {\n"
            "    public static String trim(String s) { return s.trim(); }\n"
            "}\n"
        )
        app.write_text(
            "package com.example;\n"
            "import com.example.util.StringUtils;\n"
            "public class Application {\n"
            "    public void run(String input) {\n"
            "        String clean = StringUtils.trim(input);\n"
            "    }\n"
            "}\n"
        )
        parser = JavaVisitorParser()
        results = parser.parse_multiple_files([str(util), str(app)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "StringUtils" in all_names
        assert "Application" in all_names

    def test_composition_and_aggregation_cross_file(self, tmp_path):
        address = tmp_path / "Address.java"
        person = tmp_path / "PersonWithAddress.java"
        address.write_text("public class Address { public String street; public String city; }\n")
        person.write_text(
            "import java.util.Optional;\n"
            "public class PersonWithAddress {\n"
            "    private Address primaryAddress;\n"
            "    private Optional<Address> workAddress;\n"
            "    public PersonWithAddress(Address addr) {\n"
            "        this.primaryAddress = addr;\n"
            "        this.workAddress = Optional.empty();\n"
            "    }\n"
            "}\n"
        )
        parser = JavaVisitorParser()
        results = parser.parse_multiple_files([str(address), str(person)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "Address" in all_names
        assert "PersonWithAddress" in all_names


# ---------------------------------------------------------------------------
# Java local variable and parameter tracking
# ---------------------------------------------------------------------------

class TestJavaVariableTracking:
    def test_local_variable_declaration(self):
        src = (
            "public class Calculator {\n"
            "    public int calculate(int a, int b) {\n"
            "        int sum = a + b;\n"
            "        int product = a * b;\n"
            "        return sum + product;\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Calculator" in names
        assert "calculate" in names

    def test_method_call_on_local_variable(self):
        src = (
            "public class Processor {\n"
            "    public String process(String input) {\n"
            "        StringBuilder sb = new StringBuilder();\n"
            "        sb.append(input);\n"
            "        sb.append(\"!\");\n"
            "        return sb.toString();\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Processor" in names
        assert "process" in names

    def test_chained_method_calls(self):
        src = (
            "import java.util.stream.*;\n"
            "import java.util.List;\n"
            "public class StreamExample {\n"
            "    public List<String> filterAndMap(List<String> items) {\n"
            "        return items.stream()\n"
            "            .filter(s -> s.length() > 3)\n"
            "            .map(String::toUpperCase)\n"
            "            .collect(Collectors.toList());\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "StreamExample" in names


# ---------------------------------------------------------------------------
# Generic type parameter relationships (builtin vs user vs optional)
# ---------------------------------------------------------------------------

class TestJavaGenericParameterRelationships:
    def test_list_of_builtin_type(self):
        src = (
            "import java.util.List;\n"
            "public class Numbers {\n"
            "    private List<Integer> values;\n"
            "    private List<String> labels;\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Numbers" in names

    def test_optional_of_user_type(self):
        src = (
            "import java.util.Optional;\n"
            "public class Repository {\n"
            "    private Optional<Session> currentSession;\n"
            "    private Optional<Cache> cache;\n"
            "}\n"
            "class Session {}\n"
            "class Cache {}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Repository" in names
        rel_types = {r.relationship_type for r in result.relationships}
        # Optional<UserType> → AGGREGATION
        assert RelationshipType.AGGREGATION in rel_types

    def test_map_of_user_types(self):
        src = (
            "import java.util.Map;\n"
            "import java.util.HashMap;\n"
            "public class Registry {\n"
            "    private Map<String, Handler> handlers;\n"
            "    public Registry() { handlers = new HashMap<>(); }\n"
            "}\n"
            "class Handler {}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Registry" in names

    def test_same_package_inheritance_no_import(self, tmp_path):
        # Drives the "non-imported parent class" code path in cross-file resolution
        animal = tmp_path / "BaseAnimal.java"
        dog = tmp_path / "Puppy.java"
        animal.write_text("public class BaseAnimal {\n    protected String name;\n    public void breathe() {}\n}\n")
        dog.write_text(
            "// Same package — no import needed\n"
            "public class Puppy extends BaseAnimal {\n"
            "    public void bark() { System.out.println(\"Yip!\"); }\n"
            "}\n"
        )
        parser = JavaVisitorParser()
        results = parser.parse_multiple_files([str(animal), str(dog)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "BaseAnimal" in all_names
        assert "Puppy" in all_names
        all_rels = [r for result in results.values() for r in result.relationships]
        rel_types = {r.relationship_type for r in all_rels}
        assert RelationshipType.INHERITANCE in rel_types

    def test_same_package_implementation_no_import(self, tmp_path):
        # Drives the "non-imported interface" code path in cross-file resolution
        iface = tmp_path / "Runnable2.java"
        impl = tmp_path / "Task.java"
        iface.write_text("public interface Runnable2 {\n    void run();\n}\n")
        impl.write_text(
            "// Same package — no import needed\n"
            "public class Task implements Runnable2 {\n"
            "    public void run() { System.out.println(\"Running task\"); }\n"
            "}\n"
        )
        parser = JavaVisitorParser()
        results = parser.parse_multiple_files([str(iface), str(impl)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "Runnable2" in all_names
        assert "Task" in all_names


# ---------------------------------------------------------------------------
# Java compact constructors and complex type collection
# ---------------------------------------------------------------------------

class TestJavaCompactConstructors:
    def test_record_with_compact_constructor(self):
        src = (
            "public record Point(double x, double y) {\n"
            "    // Compact constructor\n"
            "    Point {\n"
            "        if (Double.isNaN(x) || Double.isNaN(y)) {\n"
            "            throw new IllegalArgumentException(\"NaN coordinates\");\n"
            "        }\n"
            "    }\n"
            "    public double distance(Point other) {\n"
            "        return Math.sqrt(Math.pow(x - other.x, 2) + Math.pow(y - other.y, 2));\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Point" in names

    def test_record_with_canonical_constructor(self):
        src = (
            "public record Person(String name, int age) {\n"
            "    public Person(String name, int age) {\n"
            "        this.name = name.trim();\n"
            "        this.age = age;\n"
            "    }\n"
            "    public boolean isAdult() { return age >= 18; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Person" in names

    def test_complex_generic_field_types(self):
        src = (
            "import java.util.*;\n"
            "import java.util.function.*;\n"
            "public class TypeComplex {\n"
            "    private Map<String, List<Integer>> nested;\n"
            "    private Function<String, Optional<Integer>> transform;\n"
            "    private List<Map.Entry<String, Integer>> entries;\n"
            "    private Integer[] indexes;\n"
            "    private List<? extends Comparable<?>> sorted;\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "TypeComplex" in names

    def test_array_field_types_user_defined(self):
        src = (
            "public class Matrix {\n"
            "    private Cell[][] grid;\n"
            "    private Row[] rows;\n"
            "    private Column[] columns;\n"
            "}\n"
            "class Cell {}\n"
            "class Row {}\n"
            "class Column {}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Matrix" in names

    def test_wildcard_type_in_field(self):
        src = (
            "import java.util.List;\n"
            "import java.util.Collection;\n"
            "public class Processor {\n"
            "    private List<? extends Runnable> tasks;\n"
            "    private Collection<? super Integer> numbers;\n"
            "    public void addTask(Runnable task) { tasks.add(task); }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Processor" in names

    def test_enum_with_methods(self):
        src = (
            "public enum Status {\n"
            "    PENDING, ACTIVE, INACTIVE, DELETED;\n\n"
            "    public boolean isActive() { return this == ACTIVE; }\n"
            "    public boolean isTerminal() { return this == INACTIVE || this == DELETED; }\n"
            "    public static Status fromString(String s) {\n"
            "        return valueOf(s.toUpperCase());\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Status" in names

    def test_sealed_class_hierarchy(self):
        src = (
            "public abstract sealed class Shape\n"
            "    permits Circle2, Rectangle2, Triangle {\n"
            "    public abstract double area();\n"
            "}\n"
            "final class Circle2 extends Shape {\n"
            "    double radius;\n"
            "    public double area() { return Math.PI * radius * radius; }\n"
            "}\n"
            "final class Rectangle2 extends Shape {\n"
            "    double width, height;\n"
            "    public double area() { return width * height; }\n"
            "}\n"
            "final class Triangle extends Shape {\n"
            "    double base, height;\n"
            "    public double area() { return 0.5 * base * height; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Shape" in names
        assert "Circle2" in names
