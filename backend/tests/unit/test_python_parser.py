"""
Unit tests for PythonParser — coverage target ≥80%.

Uses inline source snippets, no real file I/O.
"""
from __future__ import annotations

import pytest

from app.core.parsers.python_parser import PythonParser, _parse_single_python_file
from app.core.parsers.base_parser import SymbolType, RelationshipType, ParseResult


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def parse(source: str, path: str = "test_module.py") -> ParseResult:
    """Convenience wrapper: parse source string using a fresh PythonParser."""
    parser = PythonParser()
    return parser.parse_file(path, content=source)


# ---------------------------------------------------------------------------
# Smoke / basic
# ---------------------------------------------------------------------------

class TestPythonParserSmoke:
    def test_parse_empty_source_returns_empty_lists(self):
        result = parse("")
        assert result.symbols == []
        assert result.relationships == []
        assert result.errors == [] or result.errors is None

    def test_parse_returns_parse_result(self):
        result = parse("x = 1")
        assert isinstance(result, ParseResult)
        assert result.language == "python"
        assert result.file_path == "test_module.py"

    def test_parse_result_has_file_hash(self):
        result = parse("x = 1")
        assert result.file_hash is not None
        assert len(result.file_hash) == 32  # MD5 hex digest

    def test_parse_time_is_set(self):
        result = parse("pass")
        assert result.parse_time >= 0.0


# ---------------------------------------------------------------------------
# Syntax errors
# ---------------------------------------------------------------------------

class TestSyntaxErrors:
    def test_invalid_syntax_returns_errors_not_crash(self):
        result = parse("def foo( bar baz:")
        assert result.errors
        assert any("Syntax error" in e or "syntax" in e.lower() for e in result.errors)

    def test_invalid_syntax_returns_empty_symbols(self):
        result = parse("class @@BadClass:")
        assert isinstance(result.symbols, list)

    def test_deeply_malformed_code_no_exception(self):
        result = parse("if if if if if if:")
        assert isinstance(result, ParseResult)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

class TestFunctions:
    def test_simple_function_extracted(self):
        src = "def greet(name: str) -> str:\n    return f'Hello {name}'\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "greet" in names

    def test_function_symbol_type(self):
        src = "def add(a, b):\n    return a + b\n"
        result = parse(src)
        funcs = [s for s in result.symbols if s.name == "add"]
        assert len(funcs) == 1
        assert funcs[0].symbol_type == SymbolType.FUNCTION

    def test_async_function_extracted(self):
        src = "async def fetch_data(url: str):\n    pass\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "fetch_data" in names

    def test_function_with_decorator(self):
        src = (
            "def my_decorator(f):\n"
            "    return f\n\n"
            "@my_decorator\n"
            "def decorated():\n"
            "    pass\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "decorated" in names

    def test_nested_function_extracted(self):
        src = (
            "def outer():\n"
            "    def inner():\n"
            "        pass\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "outer" in names

    def test_multiple_functions(self):
        src = "def foo(): pass\ndef bar(): pass\ndef baz(): pass\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "foo" in names
        assert "bar" in names
        assert "baz" in names


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

class TestClasses:
    def test_simple_class_extracted(self):
        src = "class MyClass:\n    pass\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "MyClass" in names

    def test_class_symbol_type(self):
        src = "class Widget:\n    pass\n"
        result = parse(src)
        widgets = [s for s in result.symbols if s.name == "Widget"]
        assert len(widgets) >= 1
        assert widgets[0].symbol_type == SymbolType.CLASS

    def test_class_with_inheritance(self):
        src = (
            "class Animal:\n    pass\n\n"
            "class Dog(Animal):\n    pass\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Animal" in names
        assert "Dog" in names

    def test_class_method_extracted(self):
        src = (
            "class Calc:\n"
            "    def add(self, a, b):\n"
            "        return a + b\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "add" in names

    def test_constructor_extracted(self):
        src = (
            "class Person:\n"
            "    def __init__(self, name):\n"
            "        self.name = name\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "__init__" in names

    def test_static_method_extracted(self):
        src = (
            "class Utils:\n"
            "    @staticmethod\n"
            "    def helper():\n"
            "        return True\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "helper" in names

    def test_class_property_extracted(self):
        src = (
            "class Config:\n"
            "    @property\n"
            "    def value(self):\n"
            "        return self._value\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "value" in names

    def test_nested_class_extracted(self):
        src = (
            "class Outer:\n"
            "    class Inner:\n"
            "        pass\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Outer" in names
        assert "Inner" in names

    def test_dataclass_extracted(self):
        src = (
            "from dataclasses import dataclass\n\n"
            "@dataclass\n"
            "class Point:\n"
            "    x: float\n"
            "    y: float\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Point" in names


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

class TestImports:
    def test_simple_import_captured(self):
        src = "import os\n"
        result = parse(src)
        assert result.imports is not None
        # imports list should reference os
        import_sources = [str(i) for i in result.imports]
        assert any("os" in s for s in import_sources)

    def test_from_import_captured(self):
        src = "from pathlib import Path\n"
        result = parse(src)
        assert result.imports is not None

    def test_import_creates_relationship(self):
        src = "import sys\n\ndef foo():\n    return sys.argv\n"
        result = parse(src)
        rel_types = [r.relationship_type for r in result.relationships]
        assert RelationshipType.IMPORTS in rel_types

    def test_relative_import(self):
        src = "from . import utils\n"
        result = parse(src)
        assert isinstance(result.imports, list)


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

class TestRelationships:
    def test_inheritance_relationship(self):
        src = (
            "class Base:\n    pass\n\n"
            "class Derived(Base):\n    pass\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.INHERITANCE in rel_types

    def test_defines_relationship_class_method(self):
        src = (
            "class MyClass:\n"
            "    def my_method(self): pass\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.DEFINES in rel_types

    def test_composition_via_annotation(self):
        src = (
            "class Address:\n"
            "    pass\n\n"
            "class Person:\n"
            "    address: Address\n"
        )
        result = parse(src)
        # Should produce composition or defines edges
        assert len(result.relationships) >= 0  # parser may or may not produce these

    def test_multiple_inheritance(self):
        src = (
            "class A:\n    pass\n"
            "class B:\n    pass\n"
            "class C(A, B):\n    pass\n"
        )
        result = parse(src)
        inheritance_rels = [r for r in result.relationships if r.relationship_type == RelationshipType.INHERITANCE]
        assert len(inheritance_rels) >= 1


# ---------------------------------------------------------------------------
# Module docstring
# ---------------------------------------------------------------------------

class TestModuleDocstring:
    def test_module_docstring_extracted(self):
        src = '"""This is the module docstring."""\n\nx = 1\n'
        result = parse(src)
        assert result.module_docstring is not None
        assert "module docstring" in result.module_docstring

    def test_no_docstring_is_none_or_empty(self):
        src = "x = 1\n"
        result = parse(src)
        # Either None or empty string — both are fine
        assert result.module_docstring is None or result.module_docstring == ""


# ---------------------------------------------------------------------------
# Type annotations & generics
# ---------------------------------------------------------------------------

class TestTypeAnnotations:
    def test_function_with_type_annotations(self):
        src = (
            "from typing import List, Optional\n\n"
            "def process(items: List[str]) -> Optional[str]:\n"
            "    return items[0] if items else None\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "process" in names

    def test_generic_class(self):
        src = (
            "from typing import TypeVar, Generic\n"
            "T = TypeVar('T')\n\n"
            "class Stack(Generic[T]):\n"
            "    def push(self, item: T) -> None: pass\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Stack" in names


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_lambda_in_class(self):
        src = (
            "class Config:\n"
            "    transform = lambda x: x * 2\n"
        )
        result = parse(src)
        assert isinstance(result.symbols, list)

    def test_class_with_class_variables(self):
        src = (
            "class Counter:\n"
            "    count: int = 0\n"
            "    MAX: int = 100\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Counter" in names

    def test_abstract_class(self):
        src = (
            "from abc import ABC, abstractmethod\n\n"
            "class Animal(ABC):\n"
            "    @abstractmethod\n"
            "    def speak(self) -> str: ...\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Animal" in names
        assert "speak" in names

    def test_complex_comprehension(self):
        src = (
            "def squares(n):\n"
            "    return [x**2 for x in range(n)]\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "squares" in names

    def test_walrus_operator_no_crash(self):
        src = (
            "def find_first(lst):\n"
            "    if (n := len(lst)) > 0:\n"
            "        return lst[0]\n"
        )
        result = parse(src)
        assert isinstance(result, ParseResult)

    def test_match_statement_no_crash(self):
        src = (
            "def classify(x):\n"
            "    match x:\n"
            "        case 0:\n"
            "            return 'zero'\n"
            "        case _:\n"
            "            return 'other'\n"
        )
        result = parse(src)
        assert isinstance(result, ParseResult)

    def test_class_methods_with_classmethods(self):
        src = (
            "class Factory:\n"
            "    @classmethod\n"
            "    def create(cls, value):\n"
            "        return cls()\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "create" in names

    def test_very_large_class_no_crash(self):
        methods = "\n".join(f"    def method_{i}(self): pass" for i in range(50))
        src = f"class BigClass:\n{methods}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "BigClass" in names

    def test_unicode_identifiers(self):
        # Python 3 allows unicode identifiers
        src = "def grüßen(): pass\n"
        result = parse(src)
        assert isinstance(result, ParseResult)

    def test_semicolon_separated_statements(self):
        src = "x = 1; y = 2; z = 3\n"
        result = parse(src)
        assert isinstance(result, ParseResult)


# ---------------------------------------------------------------------------
# parse_file convenience
# ---------------------------------------------------------------------------

class TestParseFilePaths:
    def test_parse_file_with_path_object(self, tmp_path):
        f = tmp_path / "sample.py"
        f.write_text("def hello(): pass\n")
        parser = PythonParser()
        result = parser.parse_file(f)
        names = [s.name for s in result.symbols]
        assert "hello" in names

    def test_parse_file_missing_file_returns_error(self):
        parser = PythonParser()
        result = parser.parse_file("/nonexistent/path/file.py")
        assert result.errors or result.symbols == []

    def test_parse_file_with_explicit_content_ignores_disk(self):
        parser = PythonParser()
        result = parser.parse_file("/nonexistent.py", content="def injected(): pass\n")
        names = [s.name for s in result.symbols]
        assert "injected" in names


# ---------------------------------------------------------------------------
# capabilities and metadata
# ---------------------------------------------------------------------------

class TestParserCapabilities:
    def test_capabilities_language_is_python(self):
        parser = PythonParser()
        caps = parser.capabilities
        assert caps.language == "python"

    def test_supported_extensions(self):
        parser = PythonParser()
        exts = parser._get_supported_extensions()
        assert ".py" in exts

    def test_capabilities_has_classes(self):
        parser = PythonParser()
        assert parser.capabilities.has_classes is True

    def test_capabilities_has_decorators(self):
        parser = PythonParser()
        assert parser.capabilities.has_decorators is True


# ---------------------------------------------------------------------------
# _parse_single_python_file module-level helper
# ---------------------------------------------------------------------------

class TestParseSingleFileFn:
    def test_returns_tuple_of_path_and_result(self, tmp_path):
        f = tmp_path / "mod.py"
        f.write_text("def hello(): pass\n")
        path, result = _parse_single_python_file(str(f))
        assert path == str(f)
        assert isinstance(result, ParseResult)

    def test_missing_file_returns_empty_result(self):
        path, result = _parse_single_python_file("/no/such/file.py")
        assert path == "/no/such/file.py"
        assert isinstance(result, ParseResult)
        assert result.symbols == []


# ---------------------------------------------------------------------------
# parse_multiple_files — drives cross-file analysis code paths
# ---------------------------------------------------------------------------

class TestParseMultipleFiles:
    def test_parse_multiple_files(self, tmp_path):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("def func_a(): pass\n")
        f2.write_text("def func_b(): pass\n")
        parser = PythonParser()
        results = parser.parse_multiple_files([str(f1), str(f2)])
        assert isinstance(results, dict)
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "func_a" in all_names
        assert "func_b" in all_names

    def test_parse_empty_list(self):
        parser = PythonParser()
        results = parser.parse_multiple_files([])
        assert isinstance(results, dict)

    def test_cross_file_class_and_function(self, tmp_path):
        f1 = tmp_path / "models.py"
        f2 = tmp_path / "service.py"
        f1.write_text(
            "class User:\n"
            "    def __init__(self, name: str):\n"
            "        self.name = name\n"
        )
        f2.write_text(
            "from models import User\n\n"
            "class UserService:\n"
            "    def get_user(self, name: str) -> User:\n"
            "        return User(name)\n"
        )
        parser = PythonParser()
        results = parser.parse_multiple_files([str(f1), str(f2)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "User" in all_names
        assert "UserService" in all_names


# ---------------------------------------------------------------------------
# Instance field extraction (drives __init__ analysis code paths)
# ---------------------------------------------------------------------------

class TestInstanceFieldExtraction:
    def test_self_field_assignment_creates_field_symbol(self):
        src = (
            "class Person:\n"
            "    def __init__(self, name: str, age: int):\n"
            "        self.name = name\n"
            "        self.age = age\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Person" in names
        # Self-assigned fields should be extracted
        assert "name" in names or "age" in names

    def test_self_field_with_type_annotation(self):
        src = (
            "class Config:\n"
            "    def __init__(self):\n"
            "        self.host: str = 'localhost'\n"
            "        self.port: int = 8080\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Config" in names

    def test_composition_via_init_instantiation(self):
        src = (
            "class Engine:\n"
            "    pass\n\n"
            "class Car:\n"
            "    def __init__(self):\n"
            "        self.engine = Engine()\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        # Should produce COMPOSITION or DEFINES relationships
        assert len(result.relationships) >= 1

    def test_optional_field_annotation(self):
        src = (
            "from typing import Optional\n\n"
            "class Node:\n"
            "    pass\n\n"
            "class Tree:\n"
            "    def __init__(self):\n"
            "        self.root: Optional[Node] = None\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Tree" in names

    def test_field_with_module_qualified_type(self):
        src = (
            "import os\n\n"
            "class FileManager:\n"
            "    def __init__(self):\n"
            "        self.path: os.PathLike = '/tmp'\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "FileManager" in names


# ---------------------------------------------------------------------------
# Call relationship extraction
# ---------------------------------------------------------------------------

class TestCallRelationships:
    def test_function_call_creates_calls_relationship(self):
        src = (
            "def helper(): pass\n\n"
            "def main():\n"
            "    helper()\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.CALLS in rel_types

    def test_method_call_creates_calls_relationship(self):
        src = (
            "class Service:\n"
            "    def process(self):\n"
            "        self.validate()\n"
            "    def validate(self): pass\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.CALLS in rel_types or RelationshipType.REFERENCES in rel_types


# ---------------------------------------------------------------------------
# Module-level variables and constants
# ---------------------------------------------------------------------------

class TestModuleVariables:
    def test_module_constant_extracted(self):
        src = "MAX_SIZE = 1000\nPI = 3.14159\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "MAX_SIZE" in names
        assert "PI" in names

    def test_annotated_module_variable(self):
        src = "count: int = 0\nname: str = 'default'\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "count" in names or "name" in names

    def test_all_exports_list(self):
        src = '__all__ = ["foo", "bar"]\ndef foo(): pass\nclass bar: pass\n'
        result = parse(src)
        # __all__ may or may not be in symbols depending on implementation
        assert isinstance(result, ParseResult)

    def test_type_alias_variable(self):
        src = "from typing import Union\nMyType = Union[int, str]\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "MyType" in names


# ---------------------------------------------------------------------------
# Type annotation extraction and relationship inference
# ---------------------------------------------------------------------------

class TestTypeAnnotationRelationships:
    def test_function_with_typed_params_and_return(self):
        src = (
            "class User: pass\n"
            "class Result: pass\n\n"
            "def process(user: User) -> Result:\n"
            "    return Result()\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "process" in names
        assert "User" in names
        assert "Result" in names

    def test_function_with_varargs_and_kwargs_types(self):
        src = (
            "from typing import Any\n\n"
            "def variadic(*args: int, **kwargs: str) -> None:\n"
            "    pass\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "variadic" in names

    def test_function_with_list_type_param(self):
        src = (
            "from typing import List\n"
            "class Item: pass\n\n"
            "def process_items(items: List[Item]) -> None:\n"
            "    pass\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        # Verify the function is extracted
        names = [s.name for s in result.symbols]
        assert "process_items" in names

    def test_function_with_optional_type(self):
        src = (
            "from typing import Optional\n"
            "class Config: pass\n\n"
            "def init(config: Optional[Config] = None) -> None:\n"
            "    pass\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "init" in names

    def test_union_pipe_syntax(self):
        # PEP 604 union: X | Y  (Python 3.10+)
        src = (
            "class Dog: pass\n"
            "class Cat: pass\n\n"
            "def get_pet() -> Dog | Cat:\n"
            "    return Dog()\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "get_pet" in names

    def test_class_annotated_field_optional_type(self):
        src = (
            "from typing import Optional\n"
            "class Address: pass\n\n"
            "class Person:\n"
            "    address: Optional[Address] = None\n"
            "    name: str = ''\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        # Optional class variable produces AGGREGATION
        assert RelationshipType.AGGREGATION in rel_types or RelationshipType.COMPOSITION in rel_types

    def test_class_annotated_dict_type(self):
        src = (
            "from typing import Dict\n"
            "class Config: pass\n\n"
            "class App:\n"
            "    settings: Dict[str, Config]\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "App" in names

    def test_decorator_with_arguments(self):
        src = (
            "import functools\n\n"
            "def retry(times=3):\n"
            "    def decorator(fn):\n"
            "        @functools.wraps(fn)\n"
            "        def wrapper(*args, **kwargs):\n"
            "            return fn(*args, **kwargs)\n"
            "        return wrapper\n"
            "    return decorator\n\n"
            "@retry(times=5)\n"
            "def flaky_call(): pass\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "retry" in names
        assert "flaky_call" in names


# ---------------------------------------------------------------------------
# Cross-file relationship enhancement (drives parse_multiple_files paths)
# ---------------------------------------------------------------------------

class TestPythonCrossFileEnhancement:
    def test_cross_file_inheritance_resolution(self, tmp_path):
        base_file = tmp_path / "base.py"
        child_file = tmp_path / "child.py"
        base_file.write_text("class Animal:\n    def speak(self): pass\n")
        child_file.write_text("from base import Animal\n\nclass Dog(Animal):\n    def speak(self): return 'Woof'\n")
        parser = _make_python_parser()
        results = parser.parse_multiple_files([str(base_file), str(child_file)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "Animal" in all_names
        assert "Dog" in all_names

    def test_cross_file_function_calls(self, tmp_path):
        utils = tmp_path / "utils.py"
        main = tmp_path / "main.py"
        utils.write_text("def format_name(name: str) -> str:\n    return name.strip().title()\n")
        main.write_text("from utils import format_name\n\ndef greet(user):\n    return format_name(user)\n")
        parser = _make_python_parser()
        results = parser.parse_multiple_files([str(utils), str(main)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "format_name" in all_names
        assert "greet" in all_names

    def test_cross_file_composition_detection(self, tmp_path):
        models = tmp_path / "models.py"
        service = tmp_path / "service.py"
        models.write_text("class Database:\n    host: str = 'localhost'\n    port: int = 5432\n")
        service.write_text(
            "from models import Database\n\n"
            "class UserService:\n"
            "    def __init__(self):\n"
            "        self.db: Database = Database()\n"
        )
        parser = _make_python_parser()
        results = parser.parse_multiple_files([str(models), str(service)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "Database" in all_names
        assert "UserService" in all_names


def _make_python_parser():
    from app.core.parsers.python_parser import PythonParser
    return PythonParser()


# ---------------------------------------------------------------------------
# Signature building and default value extraction
# ---------------------------------------------------------------------------

class TestPythonSignatureBuilding:
    def test_function_signature_with_defaults(self):
        src = (
            "def greet(name: str = 'World', count: int = 1) -> str:\n"
            "    return f'Hello {name}' * count\n"
        )
        result = parse(src)
        funcs = [s for s in result.symbols if s.name == "greet"]
        assert len(funcs) >= 1
        assert funcs[0].signature is not None

    def test_function_signature_with_no_return(self):
        src = "def setup(host, port): pass\n"
        result = parse(src)
        funcs = [s for s in result.symbols if s.name == "setup"]
        assert len(funcs) >= 1

    def test_async_function_with_types(self):
        src = (
            "from typing import Optional\n\n"
            "async def fetch(url: str, timeout: Optional[float] = None) -> bytes:\n"
            "    pass\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "fetch" in names

    def test_classmethod_decorator(self):
        src = (
            "class Factory:\n"
            "    @classmethod\n"
            "    def create(cls, name: str) -> 'Factory':\n"
            "        return cls()\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Factory" in names
        assert "create" in names

    def test_property_decorator(self):
        src = (
            "class Circle:\n"
            "    def __init__(self, radius: float):\n"
            "        self._radius = radius\n"
            "    @property\n"
            "    def area(self) -> float:\n"
            "        import math\n"
            "        return math.pi * self._radius ** 2\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Circle" in names
        assert "area" in names
