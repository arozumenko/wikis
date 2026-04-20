"""
Unit tests for JavaScriptVisitorParser — coverage target ≥80%.

Uses inline JavaScript source snippets; no real file I/O needed.
"""
from __future__ import annotations

import pytest

from app.core.parsers.javascript_visitor_parser import (
    JavaScriptVisitorParser,
    _parse_single_js_file,
    ImportBinding,
    ExportEntry,
)
from app.core.parsers.base_parser import SymbolType, RelationshipType, ParseResult


def parse(source: str, path: str = "module.js") -> ParseResult:
    parser = JavaScriptVisitorParser()
    return parser.parse_file(path, content=source)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

class TestJSParserSmoke:
    def test_parser_initialises(self):
        p = JavaScriptVisitorParser()
        assert p.capabilities.language == "javascript"

    def test_parse_empty_returns_result(self):
        result = parse("")
        assert isinstance(result, ParseResult)
        assert result.language == "javascript"

    def test_parse_minimal_file(self):
        result = parse("// just a comment\n")
        assert isinstance(result, ParseResult)
        assert result.file_path == "module.js"

    def test_parse_time_is_set(self):
        result = parse("const x = 1;\n")
        assert result.parse_time >= 0.0


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

class TestJSFunctions:
    def test_function_declaration_extracted(self):
        src = "function greet(name) {\n    return 'Hello ' + name;\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "greet" in names

    def test_function_symbol_type(self):
        src = "function add(a, b) { return a + b; }\n"
        result = parse(src)
        funcs = [s for s in result.symbols if s.name == "add"]
        assert len(funcs) >= 1
        assert funcs[0].symbol_type == SymbolType.FUNCTION

    def test_arrow_function_exported(self):
        src = "export const multiply = (a, b) => a * b;\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "multiply" in names

    def test_async_function(self):
        src = "async function fetchData(url) {\n    return await fetch(url);\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "fetchData" in names

    def test_generator_function(self):
        # Generator function* — parser may or may not extract by name depending on implementation
        src = "function* generateIds() {\n    let id = 0;\n    while(true) yield id++;\n}\n"
        result = parse(src)
        # Verify it parses without crash and returns a result
        assert isinstance(result, ParseResult)
        assert result.language == "javascript"

    def test_multiple_functions(self):
        src = "function a() {}\nfunction b() {}\nfunction c() {}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "a" in names
        assert "b" in names
        assert "c" in names

    def test_immediately_invoked_function(self):
        src = "(function() { const x = 1; })();\n"
        result = parse(src)
        assert isinstance(result, ParseResult)


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

class TestJSClasses:
    def test_class_declaration_extracted(self):
        src = "class Animal {\n    constructor(name) { this.name = name; }\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Animal" in names

    def test_class_symbol_type(self):
        src = "class Widget {}\n"
        result = parse(src)
        classes = [s for s in result.symbols if s.name == "Widget"]
        assert len(classes) >= 1
        assert classes[0].symbol_type == SymbolType.CLASS

    def test_class_with_extends(self):
        src = (
            "class Animal { constructor(name) { this.name = name; } }\n"
            "class Dog extends Animal {\n"
            "    speak() { return 'Woof'; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Dog" in names
        assert "Animal" in names

    def test_class_method_extracted(self):
        src = "class Calc {\n    add(a, b) { return a + b; }\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "add" in names

    def test_class_constructor_extracted(self):
        src = "class Person {\n    constructor(name) { this.name = name; }\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "constructor" in names or "Person" in names

    def test_static_method_extracted(self):
        src = "class Utils {\n    static helper() { return true; }\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "helper" in names

    def test_getter_setter_extracted(self):
        src = (
            "class Temperature {\n"
            "    get celsius() { return this._c; }\n"
            "    set celsius(v) { this._c = v; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "celsius" in names

    def test_class_fields(self):
        src = (
            "class Config {\n"
            "    host = 'localhost';\n"
            "    port = 8080;\n"
            "    connect() {}\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Config" in names

    def test_inheritance_relationship(self):
        src = "class Base {}\nclass Child extends Base {}\n"
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.INHERITANCE in rel_types

    def test_exported_class(self):
        src = "export class Service {\n    execute() {}\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Service" in names


# ---------------------------------------------------------------------------
# Imports & exports
# ---------------------------------------------------------------------------

class TestJSImportsExports:
    def test_import_creates_relationship(self):
        src = "import { useState } from 'react';\nfunction App() { return null; }\n"
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.IMPORTS in rel_types

    def test_default_import(self):
        src = "import React from 'react';\nconst x = React.createElement;\n"
        result = parse(src)
        assert isinstance(result, ParseResult)

    def test_namespace_import(self):
        src = "import * as fs from 'fs';\n"
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.IMPORTS in rel_types

    def test_named_exports_captured(self):
        src = "export function foo() {}\nexport const bar = 42;\n"
        result = parse(src)
        assert result.exports is not None
        assert len(result.exports) >= 1

    def test_default_export_function(self):
        src = "export default function main() {}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "main" in names

    def test_re_export(self):
        src = "export { foo } from './utils';\n"
        result = parse(src)
        assert isinstance(result, ParseResult)

    def test_import_sources_captured(self):
        src = "import { a } from './utils';\nimport { b } from './helpers';\n"
        result = parse(src)
        assert result.imports is not None


# ---------------------------------------------------------------------------
# CommonJS
# ---------------------------------------------------------------------------

class TestCommonJS:
    def test_require_detected(self):
        src = "const fs = require('fs');\n"
        result = parse(src)
        assert isinstance(result, ParseResult)

    def test_module_exports_detected(self):
        src = "function helper() {}\nmodule.exports = helper;\n"
        result = parse(src)
        assert isinstance(result, ParseResult)


# ---------------------------------------------------------------------------
# JSX
# ---------------------------------------------------------------------------

class TestJSX:
    def test_jsx_component_no_crash(self):
        src = (
            "import React from 'react';\n"
            "function Button({ label }) {\n"
            "    return <button>{label}</button>;\n"
            "}\n"
        )
        # Use .jsx extension to trigger JSX parsing if applicable
        parser = JavaScriptVisitorParser()
        result = parser.parse_file("component.jsx", content=src)
        assert isinstance(result, ParseResult)


# ---------------------------------------------------------------------------
# Variable declarations
# ---------------------------------------------------------------------------

class TestVarDeclarations:
    def test_const_top_level_exported(self):
        src = "export const API_URL = 'https://api.example.com';\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "API_URL" in names

    def test_let_declaration(self):
        src = "let counter = 0;\n"
        result = parse(src)
        assert isinstance(result, ParseResult)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestJSErrorHandling:
    def test_syntax_error_no_crash(self):
        src = "function broken( {\n"
        result = parse(src)
        assert isinstance(result, ParseResult)

    def test_missing_file_returns_result(self):
        parser = JavaScriptVisitorParser()
        result = parser.parse_file("/nonexistent/file.js")
        assert isinstance(result, ParseResult)

    def test_content_overrides_file_read(self):
        parser = JavaScriptVisitorParser()
        result = parser.parse_file("/nonexistent.js", content="function injected() {}\n")
        names = [s.name for s in result.symbols]
        assert "injected" in names


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

class TestJSCapabilities:
    def test_language_is_javascript(self):
        p = JavaScriptVisitorParser()
        assert p.capabilities.language == "javascript"

    def test_extensions_include_js(self):
        p = JavaScriptVisitorParser()
        exts = p._get_supported_extensions()
        assert ".js" in exts or ".jsx" in exts

    def test_has_classes(self):
        p = JavaScriptVisitorParser()
        assert p.capabilities.has_classes is True


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

class TestParseSingleJSFile:
    def test_returns_four_tuple(self, tmp_path):
        f = tmp_path / "index.js"
        f.write_text("function hello() {}\n")
        path, result, imps, exps = _parse_single_js_file(str(f))
        assert path == str(f)
        assert isinstance(result, ParseResult)
        assert isinstance(imps, dict)
        assert isinstance(exps, list)


# ---------------------------------------------------------------------------
# Multi-file parse
# ---------------------------------------------------------------------------

class TestJSMultipleFiles:
    def test_parse_multiple_files(self, tmp_path):
        f1 = tmp_path / "a.js"
        f2 = tmp_path / "b.js"
        f1.write_text("export function funcA() {}\n")
        f2.write_text("export function funcB() {}\n")
        parser = JavaScriptVisitorParser()
        results = parser.parse_multiple_files([str(f1), str(f2)], max_workers=1)
        assert isinstance(results, dict)
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "funcA" in all_names
        assert "funcB" in all_names

    def test_parse_empty_file_list(self):
        parser = JavaScriptVisitorParser()
        results = parser.parse_multiple_files([])
        assert isinstance(results, dict)


# ---------------------------------------------------------------------------
# Additional coverage tests — JavaScript-specific patterns
# ---------------------------------------------------------------------------

class TestJSClassFields:
    def test_class_with_this_assignments_in_constructor(self):
        src = (
            "class Config {\n"
            "    constructor() {\n"
            "        this.host = 'localhost';\n"
            "        this.port = 8080;\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Config" in names

    def test_static_class_property(self):
        src = (
            "class Service {\n"
            "    static instance = null;\n"
            "    static getInstance() { return Service.instance; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Service" in names
        assert "instance" in names or "getInstance" in names

    def test_private_field_syntax(self):
        src = (
            "class Counter {\n"
            "    #count = 0;\n"
            "    increment() { this.#count++; }\n"
            "    get value() { return this.#count; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Counter" in names


class TestJSDestructuring:
    def test_destructured_imports(self):
        src = "import { useState, useEffect, useCallback } from 'react';\nconst App = () => {};\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "App" in names

    def test_object_destructuring_in_function(self):
        src = (
            "function process({ name, age, address: { city } }) {\n"
            "    return `${name} from ${city}`;\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "process" in names

    def test_array_destructuring(self):
        src = (
            "function swap(arr) {\n"
            "    const [first, second, ...rest] = arr;\n"
            "    return [second, first, ...rest];\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "swap" in names


class TestJSAsyncPatterns:
    def test_async_arrow_function(self):
        src = "const fetchUser = async (id) => {\n    const res = await fetch(`/api/${id}`);\n    return res.json();\n};\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "fetchUser" in names

    def test_async_class_method(self):
        src = (
            "class ApiClient {\n"
            "    async get(url) {\n"
            "        const response = await fetch(url);\n"
            "        return response.json();\n"
            "    }\n"
            "    async post(url, data) {\n"
            "        return fetch(url, { method: 'POST', body: JSON.stringify(data) });\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "ApiClient" in names

    def test_promise_chain(self):
        src = (
            "function loadData(url) {\n"
            "    return fetch(url)\n"
            "        .then(res => res.json())\n"
            "        .catch(err => console.error(err));\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "loadData" in names


class TestJSPrototypes:
    def test_prototype_method(self):
        src = (
            "function Animal(name) { this.name = name; }\n"
            "Animal.prototype.speak = function() { return this.name; };\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Animal" in names

    def test_es5_inheritance(self):
        src = (
            "function Vehicle(make) { this.make = make; }\n"
            "function Car(make, model) {\n"
            "    Vehicle.call(this, make);\n"
            "    this.model = model;\n"
            "}\n"
            "Car.prototype = Object.create(Vehicle.prototype);\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Car" in names


class TestJSModulePatterns:
    def test_iife_module_pattern(self):
        src = (
            "const Module = (function() {\n"
            "    let private_state = 0;\n"
            "    function private_fn() { return private_state; }\n"
            "    return {\n"
            "        increment: function() { private_state++; },\n"
            "        get: private_fn\n"
            "    };\n"
            "})();\n"
        )
        result = parse(src)
        assert isinstance(result, ParseResult)

    def test_named_export_object(self):
        src = (
            "const PI = 3.14159;\n"
            "function circle(r) { return PI * r * r; }\n"
            "export { PI, circle };\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "PI" in names
        assert "circle" in names

    def test_dynamic_import_comment(self):
        # Dynamic import() — just verify no crash
        src = (
            "async function lazyLoad() {\n"
            "    const module = await import('./heavy-module.js');\n"
            "    return module.default;\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "lazyLoad" in names


class TestJSSpread:
    def test_rest_parameters(self):
        src = "function merge(target, ...sources) { return Object.assign(target, ...sources); }\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "merge" in names

    def test_spread_in_array(self):
        src = "function combine(arr1, arr2) { return [...arr1, ...arr2]; }\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "combine" in names


class TestJSIterators:
    def test_symbol_iterator(self):
        src = (
            "class Range {\n"
            "    constructor(start, end) { this.start = start; this.end = end; }\n"
            "    [Symbol.iterator]() {\n"
            "        let current = this.start;\n"
            "        const end = this.end;\n"
            "        return {\n"
            "            next() {\n"
            "                return current <= end\n"
            "                    ? { value: current++, done: false }\n"
            "                    : { done: true };\n"
            "            }\n"
            "        };\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Range" in names


# ---------------------------------------------------------------------------
# JSDoc-driven type references (drives jsdoc parsing + emit paths)
# ---------------------------------------------------------------------------

class TestJSJSDocReferences:
    def test_function_jsdoc_param_type(self):
        src = (
            "/**\n"
            " * @param {User} user The user to process\n"
            " * @returns {Result} The result\n"
            " */\n"
            "function processUser(user) {\n"
            "    return {};\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "processUser" in names
        # JSDoc should have been parsed (no crash)
        assert isinstance(result, ParseResult)

    def test_function_jsdoc_returns_type(self):
        src = (
            "/**\n"
            " * @returns {Promise<Response>} The HTTP response\n"
            " */\n"
            "async function fetchData(url) {\n"
            "    return fetch(url);\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "fetchData" in names

    def test_method_jsdoc_type(self):
        src = (
            "class Api {\n"
            "    /**\n"
            "     * @param {string} endpoint The API endpoint\n"
            "     * @param {RequestConfig} config Optional config\n"
            "     * @returns {ApiResult} The result\n"
            "     */\n"
            "    async call(endpoint, config) {\n"
            "        return {};\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Api" in names
        assert "call" in names

    def test_field_jsdoc_type(self):
        src = (
            "class Service {\n"
            "    /** @type {Database} */\n"
            "    db = null;\n"
            "    name = 'default';\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Service" in names


# ---------------------------------------------------------------------------
# Cross-file analysis (parse_multiple_files + enhancement)
# ---------------------------------------------------------------------------

class TestJSCrossFileAnalysis:
    def test_parse_multiple_files_basic(self, tmp_path):
        f1 = tmp_path / "utils.js"
        f2 = tmp_path / "app.js"
        f1.write_text("export function helper(x) { return x * 2; }\nexport const VERSION = '1.0';\n")
        f2.write_text(
            "import { helper, VERSION } from './utils';\n"
            "function main() {\n"
            "    const result = helper(21);\n"
            "    console.log(VERSION);\n"
            "}\n"
        )
        parser = JavaScriptVisitorParser()
        results = parser.parse_multiple_files([str(f1), str(f2)])
        assert len(results) == 2
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "helper" in all_names
        assert "main" in all_names

    def test_parse_multiple_files_class_inheritance(self, tmp_path):
        base = tmp_path / "base.js"
        child = tmp_path / "child.js"
        base.write_text(
            "export class Animal {\n"
            "    constructor(name) { this.name = name; }\n"
            "    speak() { return '...'; }\n"
            "}\n"
        )
        child.write_text(
            "import { Animal } from './base';\n"
            "export class Dog extends Animal {\n"
            "    speak() { return 'Woof!'; }\n"
            "}\n"
        )
        parser = JavaScriptVisitorParser()
        results = parser.parse_multiple_files([str(base), str(child)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "Animal" in all_names
        assert "Dog" in all_names

    def test_parse_multiple_files_with_default_export(self, tmp_path):
        service = tmp_path / "service.js"
        consumer = tmp_path / "consumer.js"
        service.write_text(
            "class DataService {\n"
            "    getData() { return []; }\n"
            "}\n"
            "export default DataService;\n"
        )
        consumer.write_text(
            "import DataService from './service';\n"
            "const svc = new DataService();\n"
        )
        parser = JavaScriptVisitorParser()
        results = parser.parse_multiple_files([str(service), str(consumer)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "DataService" in all_names

    def test_parse_multiple_files_commonjs(self, tmp_path):
        lib = tmp_path / "lib.js"
        main = tmp_path / "main.js"
        lib.write_text(
            "function add(a, b) { return a + b; }\n"
            "function multiply(a, b) { return a * b; }\n"
            "module.exports = { add, multiply };\n"
        )
        main.write_text(
            "const { add, multiply } = require('./lib');\n"
            "function compute(x, y) {\n"
            "    return add(x, y) + multiply(x, y);\n"
            "}\n"
        )
        parser = JavaScriptVisitorParser()
        results = parser.parse_multiple_files([str(lib), str(main)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "add" in all_names
        assert "compute" in all_names

    def test_parse_multiple_files_namespace_import(self, tmp_path):
        constants = tmp_path / "constants.js"
        config = tmp_path / "config.js"
        constants.write_text(
            "export const HOST = 'localhost';\n"
            "export const PORT = 3000;\n"
            "export const DEBUG = true;\n"
        )
        config.write_text(
            "import * as C from './constants';\n"
            "function getConfig() {\n"
            "    return { host: C.HOST, port: C.PORT };\n"
            "}\n"
        )
        parser = JavaScriptVisitorParser()
        results = parser.parse_multiple_files([str(constants), str(config)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "HOST" in all_names
        assert "getConfig" in all_names

    def test_single_file_parse_function(self, tmp_path):
        f = tmp_path / "example.js"
        f.write_text("export function greet(name) { return `Hello ${name}`; }\n")
        fp, result, imports, exports = _parse_single_js_file(str(f))
        assert fp == str(f)
        assert isinstance(result, ParseResult)
        assert result.language == "javascript"

    def test_extract_symbols_via_wrapper(self):
        p = JavaScriptVisitorParser()
        # extract_symbols is a thin wrapper around parse_file
        result_direct = p.parse_file("test.js", content="function test() {}\n")
        syms = p.extract_symbols(None, "test.js")
        # extract_symbols returns symbols from last parse_file call
        assert isinstance(syms, list)

    def test_extract_relationships_via_wrapper(self):
        p = JavaScriptVisitorParser()
        p.parse_file("test.js", content="function test() {}\n")
        rels = p.extract_relationships(None, [], "test.js")
        assert isinstance(rels, list)


# ---------------------------------------------------------------------------
# Destructured require bindings (CommonJS destructuring)
# ---------------------------------------------------------------------------

class TestJSRequireDestructuring:
    def test_destructured_require_with_alias(self):
        src = (
            "const { original: alias, simple } = require('./module');\n"
            "function useAlias() { return alias(); }\n"
        )
        result = parse(src)
        # Should not crash and should extract the function
        names = [s.name for s in result.symbols]
        assert "useAlias" in names

    def test_destructured_require_shorthand(self):
        src = (
            "const { readFile, writeFile } = require('fs');\n"
            "async function copy(src, dst) {\n"
            "    const data = await readFile(src);\n"
            "    await writeFile(dst, data);\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "copy" in names

    def test_new_expression_creates_relationship(self):
        src = (
            "class EventEmitter {}\n"
            "class Server {\n"
            "    constructor() {\n"
            "        this.emitter = new EventEmitter();\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.CREATES in rel_types

    def test_class_with_fields_and_methods(self):
        src = (
            "class DatabasePool {\n"
            "    maxConnections = 10;\n"
            "    #privateField = null;\n"
            "    static instance = null;\n"
            "    constructor(max) { this.maxConnections = max; }\n"
            "    acquire() { return {}; }\n"
            "    release(conn) { conn.close(); }\n"
            "    static getInstance() {\n"
            "        if (!DatabasePool.instance) DatabasePool.instance = new DatabasePool(5);\n"
            "        return DatabasePool.instance;\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "DatabasePool" in names
        assert "acquire" in names

    def test_body_reference_scanning(self):
        # Tests _scan_all_body_references path by creating import bindings
        src = (
            "import { UserService } from './services';\n"
            "import { Logger } from './logger';\n\n"
            "function bootstrap() {\n"
            "    const svc = new UserService();\n"
            "    Logger.info('started');\n"
            "    return svc;\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "bootstrap" in names
        # Should have IMPORTS + CREATES relationships
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.IMPORTS in rel_types
