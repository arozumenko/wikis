"""
Unit tests for TypeScriptEnhancedParser — coverage target ≥80%.

Uses inline TypeScript/TSX source snippets; no real file I/O.
"""
from __future__ import annotations

import pytest

from app.core.parsers.typescript_enhanced_parser import TypeScriptEnhancedParser
from app.core.parsers.base_parser import SymbolType, RelationshipType, ParseResult


def parse(source: str, path: str = "module.ts") -> ParseResult:
    parser = TypeScriptEnhancedParser()
    return parser.parse_file(path, content=source)


def parse_tsx(source: str, path: str = "component.tsx") -> ParseResult:
    parser = TypeScriptEnhancedParser()
    return parser.parse_file(path, content=source)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

class TestTSParserSmoke:
    def test_parser_initialises(self):
        p = TypeScriptEnhancedParser()
        assert p is not None

    def test_parse_empty_returns_result(self):
        result = parse("")
        assert isinstance(result, ParseResult)
        assert result.language == "typescript"

    def test_parse_minimal_file(self):
        result = parse("// TypeScript file\n")
        assert isinstance(result, ParseResult)
        assert result.file_path == "module.ts"

    def test_parse_time_is_set(self):
        result = parse("const x = 1;\n")
        assert result.parse_time >= 0.0

    def test_file_hash_is_set(self):
        result = parse("const x = 1;\n")
        assert result.file_hash is not None
        assert len(result.file_hash) == 32


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

class TestTSFunctions:
    def test_function_declaration_extracted(self):
        src = "function greet(name: string): string {\n    return `Hello ${name}`;\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "greet" in names

    def test_function_symbol_type(self):
        src = "function add(a: number, b: number): number { return a + b; }\n"
        result = parse(src)
        funcs = [s for s in result.symbols if s.name == "add"]
        assert len(funcs) >= 1
        assert funcs[0].symbol_type == SymbolType.FUNCTION

    def test_async_function_extracted(self):
        src = "async function fetchData(url: string): Promise<Response> {\n    return fetch(url);\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "fetchData" in names

    def test_arrow_function_const_extracted(self):
        src = "const multiply = (a: number, b: number): number => a * b;\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "multiply" in names

    def test_generic_function(self):
        src = "function identity<T>(x: T): T { return x; }\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "identity" in names

    def test_optional_parameter(self):
        src = "function greet(name?: string): string { return name ?? 'World'; }\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "greet" in names

    def test_exported_function(self):
        src = "export function publicApi(x: number): number { return x * 2; }\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "publicApi" in names

    def test_default_exported_function(self):
        src = "export default function main(): void {}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "main" in names


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

class TestTSClasses:
    def test_class_extracted(self):
        src = "class Animal {\n    name: string;\n    constructor(name: string) { this.name = name; }\n}\n"
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
            "class Animal { name: string = ''; }\n"
            "class Dog extends Animal {\n"
            "    speak(): string { return 'Woof'; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Dog" in names
        assert "Animal" in names

    def test_class_implements_interface(self):
        src = (
            "interface Printable { print(): void; }\n"
            "class Document implements Printable {\n"
            "    print(): void { console.log('printing'); }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Document" in names

    def test_abstract_class(self):
        src = (
            "abstract class Shape {\n"
            "    abstract area(): number;\n"
            "    toString(): string { return 'Shape'; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Shape" in names

    def test_class_method_extracted(self):
        src = "class Calc {\n    add(a: number, b: number): number { return a + b; }\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "add" in names

    def test_constructor_extracted(self):
        src = (
            "class Config {\n"
            "    constructor(private host: string, private port: number) {}\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Config" in names

    def test_static_method_extracted(self):
        src = (
            "class Factory {\n"
            "    static create<T>(): T | null { return null; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "create" in names

    def test_accessor_extracted(self):
        src = (
            "class Temperature {\n"
            "    private _celsius: number = 0;\n"
            "    get celsius(): number { return this._celsius; }\n"
            "    set celsius(v: number) { this._celsius = v; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "celsius" in names

    def test_class_with_decorators(self):
        src = (
            "@Injectable()\n"
            "class UserService {\n"
            "    @Inject()\n"
            "    private db: any;\n"
            "    getUser(id: number) { return null; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "UserService" in names

    def test_generic_class(self):
        src = (
            "class Stack<T> {\n"
            "    private items: T[] = [];\n"
            "    push(item: T): void { this.items.push(item); }\n"
            "    pop(): T | undefined { return this.items.pop(); }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Stack" in names


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

class TestTSInterfaces:
    def test_interface_extracted(self):
        src = "interface User {\n    id: number;\n    name: string;\n    email?: string;\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "User" in names

    def test_interface_symbol_type(self):
        src = "interface Serializable { serialize(): string; }\n"
        result = parse(src)
        ifaces = [s for s in result.symbols if s.name == "Serializable"]
        assert len(ifaces) >= 1
        assert ifaces[0].symbol_type == SymbolType.INTERFACE

    def test_interface_extending_interface(self):
        src = (
            "interface Animal { name: string; }\n"
            "interface Pet extends Animal { owner: string; }\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Pet" in names

    def test_interface_with_method_signature(self):
        src = (
            "interface Repository<T> {\n"
            "    findById(id: number): Promise<T | null>;\n"
            "    save(entity: T): Promise<T>;\n"
            "    delete(id: number): Promise<void>;\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Repository" in names

    def test_exported_interface(self):
        src = "export interface Config {\n    host: string;\n    port: number;\n}\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Config" in names


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

class TestTSTypeAliases:
    def test_simple_type_alias_extracted(self):
        src = "type ID = string | number;\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "ID" in names

    def test_object_type_alias(self):
        src = "type Point = { x: number; y: number };\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Point" in names

    def test_generic_type_alias(self):
        src = "type Optional<T> = T | null | undefined;\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Optional" in names

    def test_union_type_alias(self):
        src = "type Status = 'active' | 'inactive' | 'pending';\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Status" in names

    def test_intersection_type_alias(self):
        src = (
            "type HasId = { id: number };\n"
            "type HasName = { name: string };\n"
            "type Entity = HasId & HasName;\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Entity" in names


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestTSEnums:
    def test_numeric_enum_extracted(self):
        src = "enum Direction { North, South, East, West }\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Direction" in names

    def test_string_enum_extracted(self):
        src = (
            "enum Color {\n"
            "    Red = 'RED',\n"
            "    Green = 'GREEN',\n"
            "    Blue = 'BLUE',\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Color" in names

    def test_const_enum_extracted(self):
        src = "const enum Weekday { Monday = 1, Tuesday, Wednesday, Thursday, Friday }\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Weekday" in names

    def test_enum_symbol_type(self):
        src = "enum Status { Active, Inactive }\n"
        result = parse(src)
        enums = [s for s in result.symbols if s.name == "Status"]
        assert len(enums) >= 1
        assert enums[0].symbol_type == SymbolType.ENUM


# ---------------------------------------------------------------------------
# Imports / exports
# ---------------------------------------------------------------------------

class TestTSImports:
    def test_named_import_creates_relationship(self):
        src = "import { useState, useEffect } from 'react';\nfunction Component() { return null; }\n"
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.IMPORTS in rel_types

    def test_default_import(self):
        src = "import React from 'react';\n"
        result = parse(src)
        assert isinstance(result, ParseResult)

    def test_namespace_import(self):
        src = "import * as _ from 'lodash';\n"
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.IMPORTS in rel_types

    def test_type_import(self):
        src = "import type { User } from './types';\n"
        result = parse(src)
        assert isinstance(result, ParseResult)

    def test_re_export(self):
        src = "export { default as Button } from './Button';\n"
        result = parse(src)
        assert isinstance(result, ParseResult)


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

class TestTSRelationships:
    def test_inheritance_relationship(self):
        src = "class Base {}\nclass Child extends Base {}\n"
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.INHERITANCE in rel_types

    def test_defines_relationship(self):
        src = "class Foo {\n    bar(): void {}\n}\n"
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.DEFINES in rel_types

    def test_implementation_relationship(self):
        src = (
            "interface Printable { print(): void; }\n"
            "class Doc implements Printable {\n"
            "    print(): void {}\n"
            "}\n"
        )
        result = parse(src)
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.IMPLEMENTATION in rel_types


# ---------------------------------------------------------------------------
# TSX / React components
# ---------------------------------------------------------------------------

class TestTSXComponents:
    def test_tsx_component_extracted(self):
        src = (
            "import React from 'react';\n\n"
            "interface Props {\n"
            "    label: string;\n"
            "    onClick: () => void;\n"
            "}\n\n"
            "export function Button({ label, onClick }: Props) {\n"
            "    return <button onClick={onClick}>{label}</button>;\n"
            "}\n"
        )
        result = parse_tsx(src)
        names = [s.name for s in result.symbols]
        assert "Button" in names

    def test_tsx_class_component(self):
        src = (
            "import React, { Component } from 'react';\n\n"
            "class Counter extends Component {\n"
            "    state = { count: 0 };\n"
            "    render() {\n"
            "        return <div>{this.state.count}</div>;\n"
            "    }\n"
            "}\n"
        )
        result = parse_tsx(src)
        names = [s.name for s in result.symbols]
        assert "Counter" in names


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

class TestTSDecorators:
    def test_class_decorator_no_crash(self):
        src = (
            "function sealed(constructor: Function) {}\n\n"
            "@sealed\n"
            "class BugReport {\n"
            "    type = 'report';\n"
            "    title: string;\n"
            "    constructor(t: string) { this.title = t; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "BugReport" in names

    def test_method_decorator_no_crash(self):
        src = (
            "function log(target: any, key: string, descriptor: PropertyDescriptor) {}\n\n"
            "class Greeter {\n"
            "    @log\n"
            "    greet(name: string) { return `Hello ${name}`; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Greeter" in names


# ---------------------------------------------------------------------------
# Advanced TypeScript features
# ---------------------------------------------------------------------------

class TestTSAdvancedFeatures:
    def test_mapped_type_alias(self):
        src = "type Readonly<T> = { readonly [K in keyof T]: T[K] };\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Readonly" in names

    def test_conditional_type_alias(self):
        src = "type IsString<T> = T extends string ? true : false;\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "IsString" in names

    def test_namespace_extracted(self):
        src = (
            "namespace Validation {\n"
            "    export interface StringValidator {\n"
            "        isAcceptable(s: string): boolean;\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Validation" in names or "StringValidator" in names

    def test_module_augmentation(self):
        src = (
            "declare module 'express' {\n"
            "    interface Request {\n"
            "        user?: any;\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        assert isinstance(result, ParseResult)

    def test_overloaded_function_signatures(self):
        src = (
            "function call(x: number): number;\n"
            "function call(x: string): string;\n"
            "function call(x: any): any { return x; }\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "call" in names


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestTSErrorHandling:
    def test_syntax_error_no_crash(self):
        src = "class Broken { method( { }\n"
        result = parse(src)
        assert isinstance(result, ParseResult)

    def test_missing_file_returns_error(self):
        parser = TypeScriptEnhancedParser()
        result = parser.parse_file("/nonexistent/file.ts")
        assert isinstance(result, ParseResult)

    def test_content_overrides_file_read(self):
        parser = TypeScriptEnhancedParser()
        result = parser.parse_file("/nonexistent.ts", content="function injected(): void {}\n")
        names = [s.name for s in result.symbols]
        assert "injected" in names


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

class TestTSCapabilities:
    def test_language_is_typescript(self):
        p = TypeScriptEnhancedParser()
        assert p.capabilities.language == "typescript"

    def test_extensions_include_ts(self):
        p = TypeScriptEnhancedParser()
        exts = p._get_supported_extensions()
        assert ".ts" in exts

    def test_extensions_include_tsx(self):
        p = TypeScriptEnhancedParser()
        exts = p._get_supported_extensions()
        assert ".tsx" in exts

    def test_has_classes(self):
        p = TypeScriptEnhancedParser()
        assert p.capabilities.has_classes is True

    def test_interface_symbol_type_extracted(self):
        # TS parser extracts interface symbols even if has_interfaces capability is False
        src = "interface Printable { print(): void; }\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Printable" in names


# ---------------------------------------------------------------------------
# Multi-file parse
# ---------------------------------------------------------------------------

class TestTSMultipleFiles:
    def test_parse_multiple_ts_files(self, tmp_path):
        f1 = tmp_path / "a.ts"
        f2 = tmp_path / "b.ts"
        f1.write_text("export function funcA(): void {}\n")
        f2.write_text("export function funcB(): void {}\n")
        parser = TypeScriptEnhancedParser()
        results = parser.parse_multiple_files([str(f1), str(f2)])
        assert isinstance(results, dict)
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "funcA" in all_names
        assert "funcB" in all_names

    def test_parse_empty_list_returns_empty_dict(self):
        parser = TypeScriptEnhancedParser()
        results = parser.parse_multiple_files([])
        assert isinstance(results, dict)

    def test_parse_tsx_file_in_multi_parse(self, tmp_path):
        f = tmp_path / "comp.tsx"
        f.write_text("export function MyComp() { return null; }\n")
        parser = TypeScriptEnhancedParser()
        results = parser.parse_multiple_files([str(f)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "MyComp" in all_names


# ---------------------------------------------------------------------------
# Additional coverage tests — TypeScript-specific patterns
# ---------------------------------------------------------------------------

class TestTSAdvancedTypes:
    def test_template_literal_type(self):
        src = 'type EventName = `on${string}`;\ntype CSSProperty = `${string}-${string}`;\n'
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "EventName" in names

    def test_infer_in_conditional_type(self):
        src = "type ReturnType<T> = T extends (...args: any[]) => infer R ? R : never;\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "ReturnType" in names

    def test_recursive_type_alias(self):
        src = "type JSONValue = string | number | boolean | null | JSONValue[] | { [key: string]: JSONValue };\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "JSONValue" in names

    def test_const_assertion(self):
        src = "const colors = ['red', 'green', 'blue'] as const;\ntype Color = typeof colors[number];\n"
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "colors" in names or "Color" in names

    def test_keyof_typeof_combination(self):
        src = (
            "const config = {\n"
            "    host: 'localhost',\n"
            "    port: 8080,\n"
            "    debug: false,\n"
            "} as const;\n"
            "type ConfigKey = keyof typeof config;\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "config" in names or "ConfigKey" in names

    def test_utility_types_usage(self):
        src = (
            "interface User { id: number; name: string; email: string; }\n"
            "type PartialUser = Partial<User>;\n"
            "type RequiredUser = Required<User>;\n"
            "type ReadonlyUser = Readonly<User>;\n"
            "type UserPick = Pick<User, 'id' | 'name'>;\n"
            "type UserOmit = Omit<User, 'email'>;\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "PartialUser" in names
        assert "RequiredUser" in names


class TestTSClassAdvanced:
    def test_abstract_class_with_abstract_methods(self):
        src = (
            "abstract class Animal {\n"
            "    abstract sound(): string;\n"
            "    abstract move(distance: number): void;\n"
            "    describe(): string {\n"
            "        return `I make a ${this.sound()} sound.`;\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Animal" in names
        assert "sound" in names

    def test_class_with_private_members(self):
        src = (
            "class BankAccount {\n"
            "    private balance: number;\n"
            "    readonly accountNumber: string;\n"
            "    constructor(accountNum: string, initialBalance: number) {\n"
            "        this.accountNumber = accountNum;\n"
            "        this.balance = initialBalance;\n"
            "    }\n"
            "    deposit(amount: number): void { this.balance += amount; }\n"
            "    withdraw(amount: number): boolean {\n"
            "        if (amount > this.balance) return false;\n"
            "        this.balance -= amount;\n"
            "        return true;\n"
            "    }\n"
            "    getBalance(): number { return this.balance; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "BankAccount" in names
        assert "deposit" in names

    def test_class_implements_multiple_interfaces(self):
        src = (
            "interface Serializable { serialize(): string; }\n"
            "interface Comparable<T> { compareTo(other: T): number; }\n"
            "class Product implements Serializable, Comparable<Product> {\n"
            "    constructor(public name: string, public price: number) {}\n"
            "    serialize(): string { return JSON.stringify(this); }\n"
            "    compareTo(other: Product): number { return this.price - other.price; }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Product" in names
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.IMPLEMENTATION in rel_types


class TestTSModuleSystem:
    def test_es_module_imports_and_exports(self):
        src = (
            "import type { User } from './models';\n"
            "import { createUser, deleteUser } from './api';\n"
            "import * as utils from './utils';\n\n"
            "export interface UserService {\n"
            "    getUser(id: number): Promise<User>;\n"
            "    createUser(data: Partial<User>): Promise<User>;\n"
            "}\n\n"
            "export { createUser, deleteUser };\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "UserService" in names
        rel_types = {r.relationship_type for r in result.relationships}
        assert RelationshipType.IMPORTS in rel_types

    def test_barrel_export_pattern(self):
        src = (
            "export { Button } from './Button';\n"
            "export { Input } from './Input';\n"
            "export { Modal } from './Modal';\n"
            "export type { ButtonProps } from './Button';\n"
        )
        result = parse(src)
        assert isinstance(result, ParseResult)

    def test_ambient_module_declaration(self):
        src = (
            "declare module '*.svg' {\n"
            "    const content: string;\n"
            "    export default content;\n"
            "}\n"
        )
        result = parse(src)
        assert isinstance(result, ParseResult)


class TestTSGenericConstraints:
    def test_extends_constraint(self):
        src = (
            "function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {\n"
            "    return obj[key];\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "getProperty" in names

    def test_class_with_generic_extends(self):
        src = (
            "class Repository<T extends { id: number }> {\n"
            "    private items: T[] = [];\n"
            "    add(item: T): void { this.items.push(item); }\n"
            "    findById(id: number): T | undefined {\n"
            "        return this.items.find(item => item.id === id);\n"
            "    }\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "Repository" in names

    def test_conditional_return_type(self):
        src = (
            "function unwrap<T>(value: T | null | undefined): T {\n"
            "    if (value === null || value === undefined) throw new Error('Unexpected null');\n"
            "    return value;\n"
            "}\n"
        )
        result = parse(src)
        names = [s.name for s in result.symbols]
        assert "unwrap" in names


class TestTSCrossFileResolution:
    def test_cross_file_class_usage(self, tmp_path):
        models = tmp_path / "models.ts"
        service = tmp_path / "service.ts"
        models.write_text(
            "export class User {\n"
            "    constructor(public id: number, public name: string) {}\n"
            "    toString(): string { return `User(${this.id}: ${this.name})`; }\n"
            "}\n"
        )
        service.write_text(
            "import { User } from './models';\n\n"
            "export class UserService {\n"
            "    private users: User[] = [];\n"
            "    addUser(user: User): void { this.users.push(user); }\n"
            "    getUser(id: number): User | undefined {\n"
            "        return this.users.find(u => u.id === id);\n"
            "    }\n"
            "}\n"
        )
        parser = TypeScriptEnhancedParser()
        results = parser.parse_multiple_files([str(models), str(service)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "User" in all_names
        assert "UserService" in all_names

    def test_cross_file_interface_implementation(self, tmp_path):
        interfaces = tmp_path / "interfaces.ts"
        impl = tmp_path / "impl.ts"
        interfaces.write_text(
            "export interface Repository<T> {\n"
            "    findById(id: number): Promise<T | null>;\n"
            "    save(entity: T): Promise<T>;\n"
            "    delete(id: number): Promise<void>;\n"
            "}\n"
        )
        impl.write_text(
            "import { Repository } from './interfaces';\n\n"
            "interface User { id: number; name: string; }\n\n"
            "export class InMemoryUserRepository implements Repository<User> {\n"
            "    private users: User[] = [];\n"
            "    async findById(id: number): Promise<User | null> {\n"
            "        return this.users.find(u => u.id === id) ?? null;\n"
            "    }\n"
            "    async save(user: User): Promise<User> {\n"
            "        this.users.push(user);\n"
            "        return user;\n"
            "    }\n"
            "    async delete(id: number): Promise<void> {\n"
            "        this.users = this.users.filter(u => u.id !== id);\n"
            "    }\n"
            "}\n"
        )
        parser = TypeScriptEnhancedParser()
        results = parser.parse_multiple_files([str(interfaces), str(impl)])
        all_names = [s.name for r in results.values() for s in r.symbols]
        assert "Repository" in all_names
        assert "InMemoryUserRepository" in all_names
