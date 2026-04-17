"""
Clean C# visitor-based parser using Tree-sitter for Wiki Toolkit Graph Enhancement.

This parser implements a visitor pattern for C# AST traversal, focusing on
extracting symbols and relationships for cross-file analysis. Modelled after
the Java visitor parser but extended with C#-specific constructs: properties,
events, indexers, operator overloads, records, delegates, file-scoped namespaces,
attributes, nullable types, extension methods, and the base_list heuristic for
distinguishing INHERITANCE vs. IMPLEMENTATION.

Key Features:
- Real tree-sitter AST parsing (no mocks)
- Simple visitor pattern (no cursor API)
- Scope management for major entities only
- Cross-file relationship extraction
- C#-specific: properties, events, delegates, records, attributes,
  file-scoped namespaces, nullable types, extension methods, operator overloads
"""

import logging
import time
from typing import List, Dict, Optional, Set, Union, Tuple
from pathlib import Path

from tree_sitter import Node
from tree_sitter_language_pack import get_language, get_parser

from .base_parser import (
    BaseParser, Symbol, Relationship, ParseResult, LanguageCapabilities,
    SymbolType, RelationshipType, Scope, Position, Range, parser_registry
)

logger = logging.getLogger(__name__)

# Optional thinking emitter
try:  # pragma: no cover
    from tools import this  # type: ignore
except Exception:  # pragma: no cover
    this = None  # type: ignore


# ---------------------------------------------------------------------------
# C# built-in / BCL types – used to suppress noisy REFERENCES edges
# ---------------------------------------------------------------------------
_CSHARP_BUILTIN_TYPES: frozenset = frozenset({
    # Primitive / keyword types
    'byte', 'sbyte', 'short', 'ushort', 'int', 'uint', 'long', 'ulong',
    'float', 'double', 'decimal', 'char', 'bool', 'string', 'object', 'void',
    'nint', 'nuint', 'dynamic', 'var',
    # Wrapper / boxed equivalents
    'Byte', 'SByte', 'Int16', 'UInt16', 'Int32', 'UInt32', 'Int64', 'UInt64',
    'Single', 'Double', 'Decimal', 'Char', 'Boolean', 'String', 'Object', 'Void',
    # Common BCL
    'Task', 'ValueTask', 'Action', 'Func', 'Predicate', 'EventHandler',
    'IDisposable', 'IAsyncDisposable', 'IEnumerable', 'IEnumerator',
    'ICollection', 'IList', 'IDictionary', 'IReadOnlyList', 'IReadOnlyCollection',
    'IReadOnlyDictionary', 'ISet', 'IComparer', 'IEqualityComparer',
    'List', 'Dictionary', 'HashSet', 'SortedSet', 'Queue', 'Stack',
    'LinkedList', 'SortedDictionary', 'SortedList', 'ConcurrentDictionary',
    'ConcurrentBag', 'ConcurrentQueue', 'ConcurrentStack',
    'Array', 'Span', 'ReadOnlySpan', 'Memory', 'ReadOnlyMemory',
    'Nullable', 'Tuple', 'ValueTuple', 'KeyValuePair',
    'Type', 'Enum', 'Attribute', 'Exception', 'EventArgs', 'Delegate',
    'IObservable', 'IObserver',
    'DateTime', 'DateTimeOffset', 'TimeSpan', 'DateOnly', 'TimeOnly',
    'Guid', 'Uri', 'Version', 'BigInteger',
    'IOException', 'ArgumentException', 'ArgumentNullException',
    'InvalidOperationException', 'NotImplementedException', 'NotSupportedException',
    'NullReferenceException', 'IndexOutOfRangeException', 'OverflowException',
    'CancellationToken', 'CancellationTokenSource',
    'StringBuilder', 'Regex', 'Match',
    'Stream', 'MemoryStream', 'FileStream', 'StreamReader', 'StreamWriter',
    'HttpClient', 'HttpRequestMessage', 'HttpResponseMessage',
    'JsonSerializer', 'JsonElement', 'JsonDocument',
    'ILogger', 'ILoggerFactory',
    'IServiceProvider', 'IServiceCollection', 'IConfiguration',
    'IOptions', 'IOptionsMonitor', 'IOptionsSnapshot',
})

# Collection types that indicate COMPOSITION (strong ownership of contained items)
_COLLECTION_TYPES: frozenset = frozenset({
    'List', 'Dictionary', 'HashSet', 'SortedSet', 'Queue', 'Stack',
    'LinkedList', 'SortedDictionary', 'SortedList',
    'IList', 'ICollection', 'IDictionary', 'ISet',
    'IReadOnlyList', 'IReadOnlyCollection', 'IReadOnlyDictionary',
    'ConcurrentDictionary', 'ConcurrentBag', 'ConcurrentQueue',
    'ConcurrentStack', 'Array',
})

# Nullable/optional wrappers indicating AGGREGATION (weak ownership)
_NULLABLE_WRAPPERS: frozenset = frozenset({
    'Nullable',
})


def _parse_single_csharp_file(file_path: str) -> Tuple[str, ParseResult]:
    """Parse a single C# file – used by parallel workers."""
    try:
        parser = CSharpVisitorParser()
        result = parser.parse_file(file_path)
        return file_path, result
    except Exception as e:
        logger.warning(f"Failed to parse {file_path}: {e}")
        return file_path, ParseResult(
            file_path=file_path,
            language="csharp",
            symbols=[],
            relationships=[]
        )


class CSharpVisitorParser(BaseParser):
    """C# parser using visitor pattern with real Tree-sitter."""

    def __init__(self):
        super().__init__("csharp")
        self.parser = None
        self.ts_language = None
        self._current_file = ""
        self._current_content = ""
        self._symbols: List[Symbol] = []
        self._relationships: List[Relationship] = []
        self._scope_stack: List[str] = []

        # Cross-file resolution
        self._current_namespace = ""
        self._current_usings: Dict[str, str] = {}   # simple_name → full qualified
        self._field_types: Dict[str, str] = {}
        self._variable_types: Dict[str, str] = {}
        self._parameter_types: Dict[str, str] = {}

        # Global symbol tables (populated by parse_multiple_files)
        self._global_class_locations: Dict[str, str] = {}
        self._global_method_locations: Dict[str, List[Tuple[str, str]]] = {}

        self._setup_tree_sitter()

    # ==================================================================
    # BaseParser abstract methods
    # ==================================================================

    def _define_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            language="csharp",
            supported_symbols={
                SymbolType.CLASS, SymbolType.INTERFACE, SymbolType.STRUCT,
                SymbolType.METHOD, SymbolType.CONSTRUCTOR, SymbolType.FIELD,
                SymbolType.PROPERTY, SymbolType.VARIABLE, SymbolType.CONSTANT,
                SymbolType.ENUM, SymbolType.NAMESPACE, SymbolType.TYPE_ALIAS,
                SymbolType.PARAMETER, SymbolType.FUNCTION, SymbolType.MODULE,
            },
            supported_relationships={
                RelationshipType.INHERITANCE, RelationshipType.IMPLEMENTATION,
                RelationshipType.CALLS, RelationshipType.IMPORTS,
                RelationshipType.COMPOSITION, RelationshipType.CREATES,
                RelationshipType.DEFINES, RelationshipType.ANNOTATES,
                RelationshipType.REFERENCES, RelationshipType.AGGREGATION,
                RelationshipType.OVERRIDES,
            },
            supports_ast_parsing=True,
            supports_type_inference=True,
            supports_cross_file_analysis=True,
            has_classes=True,
            has_interfaces=True,
            has_namespaces=True,
            has_generics=True,
            has_decorators=False,
            has_annotations=True,
            max_file_size_mb=50.0,
            typical_parse_time_ms=25.0,
        )

    def _get_supported_extensions(self) -> Set[str]:
        return {'.cs'}

    # ==================================================================
    # Tree-sitter setup
    # ==================================================================

    def _setup_tree_sitter(self) -> bool:
        try:
            self.ts_language = get_language('csharp')
            self.parser = get_parser('csharp')
            return True
        except Exception as e:
            logger.error(f"Failed to initialise tree-sitter C# parser: {e}")
            return False

    # ==================================================================
    # parse_file  (single file entry-point)
    # ==================================================================

    def parse_file(self, file_path: Union[str, Path], content: Optional[str] = None) -> ParseResult:
        start_time = time.time()
        file_path = str(file_path)
        self._current_file = file_path

        # Reset per-file state
        self._symbols = []
        self._relationships = []
        self._scope_stack = []
        self._current_namespace = ""
        self._current_usings = {}
        self._field_types = {}
        self._variable_types = {}
        self._parameter_types = {}

        try:
            if content is None:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

            self._current_content = content

            if self.parser and self.ts_language:
                tree = self.parser.parse(bytes(content, "utf8"))
                self.visit_node(tree.root_node)
            else:
                return ParseResult(
                    file_path=file_path, language="csharp",
                    symbols=[], relationships=[],
                    errors=["Tree-sitter C# not available"]
                )

            # Pass 2: DEFINES relationships (class → members)
            defines = self._extract_defines_relationships(self._symbols, file_path)
            self._relationships.extend(defines)

            parse_time = time.time() - start_time
            return ParseResult(
                file_path=file_path,
                language="csharp",
                symbols=self._symbols,
                relationships=self._relationships,
                parse_time=parse_time,
            )
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return ParseResult(
                file_path=file_path, language="csharp",
                symbols=[], relationships=[],
                parse_time=0.0, errors=[f"File not found: {file_path}"]
            )
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return ParseResult(
                file_path=file_path, language="csharp",
                symbols=[], relationships=[],
                parse_time=0.0, errors=[str(e)]
            )

    # Convenience wrappers required by BaseParser
    def parse_code(self, content: str, file_path: str = "<string>") -> ParseResult:
        return self.parse_file(file_path, content=content)

    def extract_symbols(self, file_path: str, content: str) -> List[Symbol]:
        return self.parse_code(content, file_path).symbols

    def extract_relationships(self, file_path: str, content: str, symbols: List[Symbol]) -> List[Relationship]:
        return self.parse_code(content, file_path).relationships

    # ==================================================================
    # VISITOR PATTERN
    # ==================================================================

    def visit_node(self, node: Node) -> None:
        if not node:
            return

        dispatch = {
            'namespace_declaration': self._handle_namespace_declaration,
            'file_scoped_namespace_declaration': self._handle_file_scoped_namespace,
            'using_directive': self._handle_using_directive,
            'class_declaration': self._handle_class_declaration,
            'struct_declaration': self._handle_struct_declaration,
            'interface_declaration': self._handle_interface_declaration,
            'enum_declaration': self._handle_enum_declaration,
            'record_declaration': self._handle_record_declaration,
            'delegate_declaration': self._handle_delegate_declaration,
            'method_declaration': self._handle_method_declaration,
            'constructor_declaration': self._handle_constructor_declaration,
            'destructor_declaration': self._handle_destructor_declaration,
            'property_declaration': self._handle_property_declaration,
            'field_declaration': self._handle_field_declaration,
            'event_field_declaration': self._handle_event_field_declaration,
            'indexer_declaration': self._handle_indexer_declaration,
            'operator_declaration': self._handle_operator_declaration,
            'invocation_expression': self._handle_invocation_expression,
            'object_creation_expression': self._handle_object_creation,
        }

        handler = dispatch.get(node.type)
        if handler:
            handler(node)
        else:
            self._visit_children(node)

    def _visit_children(self, node: Node) -> None:
        if hasattr(node, 'children'):
            for child in node.children:
                self.visit_node(child)

    # ==================================================================
    # SCOPE MANAGEMENT
    # ==================================================================

    def _enter_scope(self, name: str) -> None:
        self._scope_stack.append(name)

    def _exit_scope(self) -> None:
        if self._scope_stack:
            self._scope_stack.pop()

    def _get_current_scope_path(self) -> str:
        return '.'.join(self._scope_stack) if self._scope_stack else ""

    def _build_qualified_name(self, name: str) -> str:
        scope = self._get_current_scope_path()
        return f"{scope}.{name}" if scope else name

    # ==================================================================
    # UTILITY HELPERS
    # ==================================================================

    def _get_node_text(self, node: Node) -> str:
        if not node or not hasattr(node, 'text'):
            return ""
        return node.text.decode('utf-8')

    def _byte_to_line_col(self, byte_offset: int) -> Tuple[int, int]:
        content = self._current_content
        lines_before = content[:byte_offset].count('\n')
        line_start = content.rfind('\n', 0, byte_offset) + 1
        column = byte_offset - line_start
        return lines_before + 1, column

    def _create_position(self, node: Node) -> Position:
        line, col = self._byte_to_line_col(node.start_byte)
        return Position(line=line, column=col)

    def _create_range(self, node: Node) -> Range:
        sl, sc = self._byte_to_line_col(node.start_byte)
        el, ec = self._byte_to_line_col(node.end_byte)
        return Range(start=Position(line=sl, column=sc),
                     end=Position(line=el, column=ec))

    def _find_child_by_type(self, node: Node, child_type: str) -> Optional[Node]:
        if not hasattr(node, 'children'):
            return None
        for child in node.children:
            if child.type == child_type:
                return child
        return None

    def _find_children_by_types(self, node: Node, child_types: Set[str]) -> List[Node]:
        if not hasattr(node, 'children'):
            return []
        return [c for c in node.children if c.type in child_types]

    def _extract_node_source(self, node: Node) -> Optional[str]:
        if not node or not self._current_content:
            return None
        try:
            src = self._current_content.encode('utf-8')
            return src[node.start_byte:node.end_byte].decode('utf-8')
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Modifier extraction
    # ------------------------------------------------------------------

    def _extract_modifiers_from_node(self, node: Node) -> Dict[str, bool]:
        """Extract all modifier keywords from a declaration node.
        
        Returns a dict with boolean flags and a 'visibility' string.
        """
        mods: Dict[str, bool] = {
            'is_static': False, 'is_abstract': False, 'is_sealed': False,
            'is_virtual': False, 'is_override': False, 'is_readonly': False,
            'is_const': False, 'is_async': False, 'is_partial': False,
            'is_extern': False, 'is_new': False, 'is_volatile': False,
            'is_unsafe': False,
        }
        visibility = 'internal'  # C# default access

        for child in node.children:
            if child.type == 'modifier':
                kw = self._get_node_text(child).strip()
                if kw == 'public':
                    visibility = 'public'
                elif kw == 'private':
                    visibility = 'private'
                elif kw == 'protected':
                    visibility = 'protected'
                elif kw == 'internal':
                    visibility = 'internal'
                elif kw == 'static':
                    mods['is_static'] = True
                elif kw == 'abstract':
                    mods['is_abstract'] = True
                elif kw == 'sealed':
                    mods['is_sealed'] = True
                elif kw == 'virtual':
                    mods['is_virtual'] = True
                elif kw == 'override':
                    mods['is_override'] = True
                elif kw == 'readonly':
                    mods['is_readonly'] = True
                elif kw == 'const':
                    mods['is_const'] = True
                elif kw == 'async':
                    mods['is_async'] = True
                elif kw == 'partial':
                    mods['is_partial'] = True
                elif kw == 'extern':
                    mods['is_extern'] = True
                elif kw == 'new':
                    mods['is_new'] = True
                elif kw == 'volatile':
                    mods['is_volatile'] = True
                elif kw == 'unsafe':
                    mods['is_unsafe'] = True

        mods['visibility'] = visibility
        return mods

    # ------------------------------------------------------------------
    # Type helpers
    # ------------------------------------------------------------------

    def _extract_type_name(self, node: Node) -> Optional[str]:
        """Extract the simple type name from a type AST node."""
        if not node:
            return None

        if node.type == 'predefined_type':
            return self._get_node_text(node)
        elif node.type == 'identifier':
            return self._get_node_text(node)
        elif node.type == 'qualified_name':
            return self._get_node_text(node)
        elif node.type == 'nullable_type':
            # e.g. string? → underlying type is first child
            for child in node.children:
                if child.type != '?':
                    return self._extract_type_name(child)
        elif node.type == 'generic_name':
            # GenericName = identifier + type_argument_list
            id_node = self._find_child_by_type(node, 'identifier')
            if id_node:
                return self._get_node_text(id_node)
        elif node.type == 'array_type':
            for child in node.children:
                if child.type in ('predefined_type', 'identifier', 'qualified_name',
                                  'generic_name', 'nullable_type'):
                    return self._extract_type_name(child)
        elif node.type == 'tuple_type':
            return 'ValueTuple'

        return self._get_node_text(node) if node else None

    def _is_nullable_type_node(self, node: Node) -> bool:
        """Check whether a type node represents a nullable type (T?)."""
        return node is not None and node.type == 'nullable_type'

    def _extract_return_type(self, node: Node) -> Optional[str]:
        """Extract return type from a method/property/operator declaration.
        
        In the C# tree-sitter grammar the return type appears as a child
        of the declaration node, *before* the identifier.  It can be a
        ``predefined_type``, ``identifier``, ``qualified_name``,
        ``generic_name``, ``nullable_type``, ``array_type`` or ``tuple_type``.
        """
        type_node_types = {
            'predefined_type', 'identifier', 'qualified_name', 'generic_name',
            'nullable_type', 'array_type', 'tuple_type',
        }
        # Walk children; the first type-like node before the method name is the
        # return type.  We skip 'modifier' and 'attribute_list' children.
        for child in node.children:
            if child.type in type_node_types:
                return self._extract_type_name(child)
        return None

    def _extract_parameter_types(self, param_list_node: Node) -> List[str]:
        """Extract parameter types from parameter_list or bracketed_parameter_list."""
        types: List[str] = []
        for child in param_list_node.children:
            if child.type == 'parameter':
                ptype = self._get_parameter_type(child)
                if ptype:
                    types.append(ptype)
        return types

    def _get_parameter_type(self, param_node: Node) -> Optional[str]:
        """Get the type string of a single parameter node."""
        type_node_types = {
            'predefined_type', 'identifier', 'qualified_name', 'generic_name',
            'nullable_type', 'array_type', 'tuple_type',
        }
        for child in param_node.children:
            if child.type in type_node_types:
                return self._extract_type_name(child)
        return None

    def _get_parameter_name(self, param_node: Node) -> Optional[str]:
        """Get the name of a single parameter node (the identifier child)."""
        # In C# tree-sitter, parameter children are: [modifier*] type identifier [= default]
        # The identifier is usually the last identifier child
        ids = [c for c in param_node.children if c.type == 'identifier']
        type_node_types = {
            'predefined_type', 'qualified_name', 'generic_name',
            'nullable_type', 'array_type', 'tuple_type',
        }
        # If the type is an identifier too, we need the *second* identifier
        has_type_id = any(c.type == 'identifier' for c in param_node.children
                         if c == param_node.children[0] or c.type in type_node_types)
        if len(ids) >= 2:
            return self._get_node_text(ids[-1])
        elif len(ids) == 1:
            # Check if any non-identifier type node exists
            has_explicit_type = any(c.type in type_node_types for c in param_node.children)
            if has_explicit_type:
                return self._get_node_text(ids[0])
            else:
                # single identifier is probably the name (type inferred?)
                return self._get_node_text(ids[0])
        return None

    def _is_builtin_type(self, type_name: str) -> bool:
        if not type_name:
            return True
        # Strip generic suffix if any  (e.g. "List" from "List<Foo>")
        base = type_name.split('<')[0].split('[')[0].split('?')[0].strip()
        if '.' in base:
            # Fully-qualified BCL checks
            bcl_prefixes = ('System.', 'Microsoft.', 'Newtonsoft.', 'NLog.')
            if any(base.startswith(p) for p in bcl_prefixes):
                return True
            base = base.rsplit('.', 1)[-1]
        return base in _CSHARP_BUILTIN_TYPES

    def _is_interface_name(self, name: str) -> bool:
        """Heuristic: C# interfaces conventionally start with 'I' followed by uppercase."""
        if not name:
            return False
        # Strip generic parameters
        base = name.split('<')[0]
        return len(base) >= 2 and base[0] == 'I' and base[1].isupper()

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def _extract_generic_type_parameters(self, node: Node) -> List[str]:
        """Extract type parameter names from type_parameter_list."""
        params: List[str] = []
        tp_list = self._find_child_by_type(node, 'type_parameter_list')
        if tp_list:
            for child in tp_list.children:
                if child.type == 'type_parameter':
                    id_node = self._find_child_by_type(child, 'identifier')
                    if id_node:
                        params.append(self._get_node_text(id_node))
        return params

    def _extract_generic_constraints(self, node: Node) -> List[Dict]:
        """Extract generic constraint clauses (where T : class, new())."""
        constraints: List[Dict] = []
        for child in node.children:
            if child.type == 'type_parameter_constraints_clause':
                clause = {'parameter': '', 'constraints': []}
                for cc in child.children:
                    if cc.type == 'identifier':
                        clause['parameter'] = self._get_node_text(cc)
                    elif cc.type == 'type_parameter_constraint':
                        clause['constraints'].append(self._get_node_text(cc))
                constraints.append(clause)
        return constraints

    # ------------------------------------------------------------------
    # Attribute helpers
    # ------------------------------------------------------------------

    def _extract_attributes(self, node: Node) -> List[str]:
        """Extract attribute names from attribute_list children of a node."""
        attrs: List[str] = []
        for child in node.children:
            if child.type == 'attribute_list':
                for attr_child in child.children:
                    if attr_child.type == 'attribute':
                        id_node = self._find_child_by_type(attr_child, 'identifier')
                        if id_node:
                            attrs.append(self._get_node_text(id_node))
        return attrs

    def _emit_annotates(self, attrs: List[str], target_symbol: str) -> None:
        """Emit ANNOTATES relationships for a list of attribute names."""
        for attr in attrs:
            self._relationships.append(Relationship(
                source_symbol=attr,
                target_symbol=target_symbol,
                relationship_type=RelationshipType.ANNOTATES,
                source_file=self._current_file,
                target_file=self._current_file,
            ))

    # ==================================================================
    # NODE HANDLERS – Namespaces & Usings
    # ==================================================================

    def _handle_namespace_declaration(self, node: Node) -> None:
        """Handle block-scoped namespace: namespace A.B.C { ... }"""
        ns_name = None
        for child in node.children:
            if child.type in ('identifier', 'qualified_name'):
                ns_name = self._get_node_text(child)
                break

        if not ns_name:
            self._visit_children(node)
            return

        self._current_namespace = ns_name
        self._enter_scope(ns_name)
        self._visit_children(node)
        self._exit_scope()

    def _handle_file_scoped_namespace(self, node: Node) -> None:
        """Handle file-scoped namespace: namespace MyApp;"""
        ns_name = None
        for child in node.children:
            if child.type in ('identifier', 'qualified_name'):
                ns_name = self._get_node_text(child)
                break

        if ns_name:
            self._current_namespace = ns_name
            self._enter_scope(ns_name)
            # File-scoped – do NOT pop; covers entire file
            # Visit remaining siblings (they come after in compilation_unit)

    def _handle_using_directive(self, node: Node) -> None:
        """Handle using directives (imports).
        
        Patterns:
          using System.IO;
          using Dict = System.Collections.Generic.Dictionary;
          global using System;
        """
        is_global = False
        alias_name = None
        import_path = None

        for child in node.children:
            if child.type == 'global':
                is_global = True
            elif child.type == 'identifier' and import_path is None:
                # Might be alias or simple namespace
                # If followed by '=' it's an alias
                next_sib = child.next_sibling
                if next_sib and self._get_node_text(next_sib) == '=':
                    alias_name = self._get_node_text(child)
                else:
                    import_path = self._get_node_text(child)
            elif child.type == 'qualified_name':
                import_path = self._get_node_text(child)

        if not import_path:
            return

        # Track for type resolution
        if alias_name:
            self._current_usings[alias_name] = import_path
        else:
            # Store simple name → full path mapping
            parts = import_path.split('.')
            if parts:
                self._current_usings[parts[-1]] = import_path

        # Emit IMPORTS relationship
        source = self._get_current_scope_path() or Path(self._current_file).stem
        annotations = {}
        if is_global:
            annotations['global'] = True
        if alias_name:
            annotations['alias'] = alias_name

        self._relationships.append(Relationship(
            source_symbol=source,
            target_symbol=import_path,
            relationship_type=RelationshipType.IMPORTS,
            source_file=self._current_file,
            target_file=self._current_file,
            annotations=annotations if annotations else None,
        ))

    # ==================================================================
    # NODE HANDLERS – Type Declarations
    # ==================================================================

    def _handle_class_declaration(self, node: Node) -> None:
        class_name = None
        for child in node.children:
            if child.type == 'identifier':
                class_name = self._get_node_text(child)
                break

        if not class_name:
            return

        mods = self._extract_modifiers_from_node(node)
        attrs = self._extract_attributes(node)
        generic_params = self._extract_generic_type_parameters(node)
        constraints = self._extract_generic_constraints(node)

        parent_symbol = self._get_current_scope_path() if self._scope_stack else None
        qualified_name = self._build_qualified_name(class_name)

        metadata = {k: v for k, v in mods.items() if v and k != 'visibility'}
        if generic_params:
            metadata['generic_parameters'] = generic_params
        if constraints:
            metadata['generic_constraints'] = constraints

        symbol = Symbol(
            name=class_name,
            symbol_type=SymbolType.CLASS,
            file_path=self._current_file,
            range=self._create_range(node),
            scope=Scope.CLASS,
            parent_symbol=parent_symbol,
            full_name=qualified_name,
            visibility=mods.get('visibility', 'internal'),
            is_static=mods.get('is_static', False),
            is_abstract=mods.get('is_abstract', False),
            source_text=self._extract_node_source(node),
            metadata=metadata if metadata else None,
        )
        self._symbols.append(symbol)

        # Attributes → ANNOTATES
        self._emit_annotates(attrs, qualified_name)

        # Inheritance / implementation via base_list
        self._extract_base_list(node, symbol, is_interface=False)

        # Generic constraint types → REFERENCES
        self._emit_constraint_references(constraints, qualified_name)

        self._enter_scope(class_name)
        self._visit_children(node)
        self._exit_scope()

    def _handle_struct_declaration(self, node: Node) -> None:
        struct_name = None
        for child in node.children:
            if child.type == 'identifier':
                struct_name = self._get_node_text(child)
                break

        if not struct_name:
            return

        mods = self._extract_modifiers_from_node(node)
        attrs = self._extract_attributes(node)
        generic_params = self._extract_generic_type_parameters(node)

        parent_symbol = self._get_current_scope_path() if self._scope_stack else None
        qualified_name = self._build_qualified_name(struct_name)

        metadata = {k: v for k, v in mods.items() if v and k != 'visibility'}
        if generic_params:
            metadata['generic_parameters'] = generic_params

        symbol = Symbol(
            name=struct_name,
            symbol_type=SymbolType.STRUCT,
            file_path=self._current_file,
            range=self._create_range(node),
            scope=Scope.CLASS,
            parent_symbol=parent_symbol,
            full_name=qualified_name,
            visibility=mods.get('visibility', 'internal'),
            is_static=mods.get('is_static', False),
            source_text=self._extract_node_source(node),
            metadata=metadata if metadata else None,
        )
        self._symbols.append(symbol)

        self._emit_annotates(attrs, qualified_name)
        # Structs can only implement interfaces (no class inheritance)
        self._extract_base_list(node, symbol, is_interface=False, is_struct=True)

        self._enter_scope(struct_name)
        self._visit_children(node)
        self._exit_scope()

    def _handle_interface_declaration(self, node: Node) -> None:
        iface_name = None
        for child in node.children:
            if child.type == 'identifier':
                iface_name = self._get_node_text(child)
                break

        if not iface_name:
            return

        mods = self._extract_modifiers_from_node(node)
        attrs = self._extract_attributes(node)
        generic_params = self._extract_generic_type_parameters(node)

        parent_symbol = self._get_current_scope_path() if self._scope_stack else None
        qualified_name = self._build_qualified_name(iface_name)

        metadata = {k: v for k, v in mods.items() if v and k != 'visibility'}
        if generic_params:
            metadata['generic_parameters'] = generic_params

        symbol = Symbol(
            name=iface_name,
            symbol_type=SymbolType.INTERFACE,
            file_path=self._current_file,
            range=self._create_range(node),
            scope=Scope.CLASS,
            parent_symbol=parent_symbol,
            full_name=qualified_name,
            visibility=mods.get('visibility', 'internal'),
            source_text=self._extract_node_source(node),
            metadata=metadata if metadata else None,
        )
        self._symbols.append(symbol)

        self._emit_annotates(attrs, qualified_name)
        # Interfaces can only extend other interfaces → INHERITANCE
        self._extract_base_list(node, symbol, is_interface=True)

        self._enter_scope(iface_name)
        self._visit_children(node)
        self._exit_scope()

    def _handle_enum_declaration(self, node: Node) -> None:
        enum_name = None
        for child in node.children:
            if child.type == 'identifier':
                enum_name = self._get_node_text(child)
                break

        if not enum_name:
            return

        mods = self._extract_modifiers_from_node(node)
        attrs = self._extract_attributes(node)

        parent_symbol = self._get_current_scope_path() if self._scope_stack else None
        qualified_name = self._build_qualified_name(enum_name)

        symbol = Symbol(
            name=enum_name,
            symbol_type=SymbolType.ENUM,
            file_path=self._current_file,
            range=self._create_range(node),
            scope=Scope.CLASS,
            parent_symbol=parent_symbol,
            full_name=qualified_name,
            visibility=mods.get('visibility', 'internal'),
            source_text=self._extract_node_source(node),
        )
        self._symbols.append(symbol)
        self._emit_annotates(attrs, qualified_name)

        # Extract enum members as FIELD symbols
        self._enter_scope(enum_name)
        member_list = self._find_child_by_type(node, 'enum_member_declaration_list')
        if member_list:
            for child in member_list.children:
                if child.type == 'enum_member_declaration':
                    member_id = self._find_child_by_type(child, 'identifier')
                    if member_id:
                        member_name = self._get_node_text(member_id)
                        member_qn = self._build_qualified_name(member_name)
                        mem_sym = Symbol(
                            name=member_name,
                            symbol_type=SymbolType.FIELD,
                            file_path=self._current_file,
                            range=self._create_range(child),
                            scope=Scope.CLASS,
                            parent_symbol=qualified_name,
                            full_name=member_qn,
                            visibility='public',
                            is_static=True,
                            source_text=self._extract_node_source(child),
                            metadata={'is_enum_member': True},
                        )
                        self._symbols.append(mem_sym)
        self._exit_scope()

    def _handle_record_declaration(self, node: Node) -> None:
        record_name = None
        is_record_struct = False

        for child in node.children:
            if child.type == 'struct':
                is_record_struct = True
            elif child.type == 'identifier':
                record_name = self._get_node_text(child)
                break

        if not record_name:
            return

        mods = self._extract_modifiers_from_node(node)
        attrs = self._extract_attributes(node)
        generic_params = self._extract_generic_type_parameters(node)

        parent_symbol = self._get_current_scope_path() if self._scope_stack else None
        qualified_name = self._build_qualified_name(record_name)

        sym_type = SymbolType.STRUCT if is_record_struct else SymbolType.CLASS
        metadata = {k: v for k, v in mods.items() if v and k != 'visibility'}
        metadata['is_record'] = True
        if is_record_struct:
            metadata['is_record_struct'] = True
        if generic_params:
            metadata['generic_parameters'] = generic_params

        symbol = Symbol(
            name=record_name,
            symbol_type=sym_type,
            file_path=self._current_file,
            range=self._create_range(node),
            scope=Scope.CLASS,
            parent_symbol=parent_symbol,
            full_name=qualified_name,
            visibility=mods.get('visibility', 'internal'),
            source_text=self._extract_node_source(node),
            metadata=metadata,
        )
        self._symbols.append(symbol)
        self._emit_annotates(attrs, qualified_name)
        self._extract_base_list(node, symbol, is_interface=False,
                                is_struct=is_record_struct)

        # Record positional parameters → PROPERTY symbols
        self._enter_scope(record_name)
        param_list = self._find_child_by_type(node, 'parameter_list')
        if param_list:
            for child in param_list.children:
                if child.type == 'parameter':
                    ptype = self._get_parameter_type(child)
                    pname = self._get_parameter_name(child)
                    if pname:
                        prop_qn = self._build_qualified_name(pname)
                        prop = Symbol(
                            name=pname,
                            symbol_type=SymbolType.PROPERTY,
                            file_path=self._current_file,
                            range=self._create_range(child),
                            scope=Scope.CLASS,
                            parent_symbol=qualified_name,
                            full_name=prop_qn,
                            visibility='public',
                            return_type=ptype,
                            source_text=self._extract_node_source(child),
                            metadata={'is_record_parameter': True},
                        )
                        self._symbols.append(prop)

                        # Type relationship for record parameter
                        if ptype and not self._is_builtin_type(ptype):
                            self._relationships.append(Relationship(
                                source_symbol=prop_qn,
                                target_symbol=ptype,
                                relationship_type=RelationshipType.COMPOSITION,
                                source_file=self._current_file,
                                target_file=self._current_file,
                            ))

        # Visit declaration_list if present (methods, properties, etc.)
        self._visit_children(node)
        self._exit_scope()

    def _handle_delegate_declaration(self, node: Node) -> None:
        delegate_name = None
        return_type = None

        for child in node.children:
            if child.type in ('predefined_type', 'identifier', 'qualified_name',
                              'generic_name', 'nullable_type', 'array_type'):
                if delegate_name is None:
                    # First type-like child is return type, but 'identifier' could
                    # also be the delegate name. We need to distinguish.
                    # Pattern: modifier* delegate return_type identifier parameter_list ;
                    # After 'delegate' keyword: return_type identifier
                    pass
            if child.type == 'identifier' and delegate_name is not None:
                # Already have a name candidate — this second identifier is the real name
                pass

        # Re-parse more carefully: after 'delegate' keyword, first type-ish is return, then identifier is name
        saw_delegate = False
        for child in node.children:
            if child.type == 'delegate':
                saw_delegate = True
                continue
            if saw_delegate:
                if return_type is None and child.type in ('predefined_type', 'generic_name',
                                                           'qualified_name', 'nullable_type',
                                                           'array_type', 'tuple_type'):
                    return_type = self._extract_type_name(child)
                elif return_type is None and child.type == 'identifier':
                    # Could be return type (user type) — peek ahead
                    # If the next non-trivial sibling is also an identifier, this is return type
                    next_meaningful = child.next_named_sibling
                    if next_meaningful and next_meaningful.type == 'identifier':
                        return_type = self._get_node_text(child)
                    elif next_meaningful and next_meaningful.type == 'parameter_list':
                        # No return type before name (shouldn't happen but defensive)
                        delegate_name = self._get_node_text(child)
                    else:
                        return_type = self._get_node_text(child)
                elif child.type == 'identifier' and delegate_name is None:
                    delegate_name = self._get_node_text(child)

        if not delegate_name:
            return

        mods = self._extract_modifiers_from_node(node)
        parent_symbol = self._get_current_scope_path() if self._scope_stack else None
        qualified_name = self._build_qualified_name(delegate_name)

        symbol = Symbol(
            name=delegate_name,
            symbol_type=SymbolType.TYPE_ALIAS,
            file_path=self._current_file,
            range=self._create_range(node),
            scope=Scope.CLASS,
            parent_symbol=parent_symbol,
            full_name=qualified_name,
            visibility=mods.get('visibility', 'internal'),
            return_type=return_type,
            source_text=self._extract_node_source(node),
            metadata={'is_delegate': True},
        )
        self._symbols.append(symbol)

    # ==================================================================
    # NODE HANDLERS – Members
    # ==================================================================

    def _handle_method_declaration(self, node: Node) -> None:
        method_name = None
        return_type = None

        mods = self._extract_modifiers_from_node(node)
        attrs = self._extract_attributes(node)

        # In C# tree-sitter: modifier* return_type identifier parameter_list block|;|arrow
        # Return type can be predefined_type, identifier (user type), generic_name, etc.
        # The method name is the identifier that follows the return type.
        type_candidates = ('predefined_type', 'qualified_name', 'generic_name',
                           'nullable_type', 'array_type', 'tuple_type')

        found_return = False
        for child in node.children:
            if child.type == 'modifier' or child.type == 'attribute_list':
                continue
            if not found_return and child.type in type_candidates:
                return_type = self._extract_type_name(child)
                found_return = True
            elif child.type == 'identifier':
                if not found_return:
                    # Two consecutive identifiers: first=return type, second=name
                    # But we might not have seen return type yet
                    # Peek ahead
                    nxt = child.next_named_sibling
                    if nxt and nxt.type == 'identifier':
                        return_type = self._get_node_text(child)
                        found_return = True
                    elif nxt and nxt.type == 'parameter_list':
                        method_name = self._get_node_text(child)
                        break
                    else:
                        return_type = self._get_node_text(child)
                        found_return = True
                else:
                    method_name = self._get_node_text(child)
                    break

        if not method_name:
            return

        # Extract parameters
        param_list = self._find_child_by_type(node, 'parameter_list')
        parameter_types = self._extract_parameter_types(param_list) if param_list else []

        # Track parameter types for call resolution
        if param_list:
            for child in param_list.children:
                if child.type == 'parameter':
                    ptype = self._get_parameter_type(child)
                    pname = self._get_parameter_name(child)
                    if ptype and pname:
                        self._parameter_types[pname] = ptype

        parent_symbol = self._get_current_scope_path() if self._scope_stack else None
        qualified_name = self._build_qualified_name(method_name)

        metadata = {k: v for k, v in mods.items() if v and k != 'visibility'}

        # Detect extension method (first param has 'this' modifier)
        is_extension = self._is_extension_method(param_list) if param_list else False
        if is_extension:
            metadata['is_extension'] = True

        symbol = Symbol(
            name=method_name,
            symbol_type=SymbolType.METHOD,
            file_path=self._current_file,
            range=self._create_range(node),
            scope=Scope.FUNCTION,
            parent_symbol=parent_symbol,
            full_name=qualified_name,
            visibility=mods.get('visibility', 'internal'),
            is_static=mods.get('is_static', False),
            is_abstract=mods.get('is_abstract', False),
            is_async=mods.get('is_async', False),
            return_type=return_type,
            parameter_types=parameter_types,
            source_text=self._extract_node_source(node),
            metadata=metadata if metadata else None,
        )
        self._symbols.append(symbol)

        # Attributes → ANNOTATES
        self._emit_annotates(attrs, qualified_name)

        # OVERRIDES for methods with 'override' modifier
        if mods.get('is_override') and parent_symbol:
            parent_class_super = self._find_superclass(parent_symbol)
            if parent_class_super:
                self._relationships.append(Relationship(
                    source_symbol=qualified_name,
                    target_symbol=f"{parent_class_super}.{method_name}",
                    relationship_type=RelationshipType.OVERRIDES,
                    source_file=self._current_file,
                    target_file=self._current_file,
                    annotations={'override_keyword': True},
                ))

        # Return type → REFERENCES
        if return_type and return_type != 'void' and not self._is_builtin_type(return_type):
            self._relationships.append(Relationship(
                source_symbol=qualified_name,
                target_symbol=return_type,
                relationship_type=RelationshipType.REFERENCES,
                source_file=self._current_file,
                target_file=self._current_file,
            ))

        # Parameter types → REFERENCES
        if param_list:
            self._emit_parameter_references(param_list, qualified_name)

        # Visit body for calls / creates
        self._enter_scope(method_name)
        self._visit_children(node)
        self._exit_scope()

        # Clean up parameter types (scoped to this method)
        if param_list:
            for child in param_list.children:
                if child.type == 'parameter':
                    pname = self._get_parameter_name(child)
                    if pname:
                        self._parameter_types.pop(pname, None)

    def _handle_constructor_declaration(self, node: Node) -> None:
        if not self._scope_stack:
            return

        ctor_name = None
        for child in node.children:
            if child.type == 'identifier':
                ctor_name = self._get_node_text(child)
                break

        if not ctor_name:
            ctor_name = self._scope_stack[-1]

        mods = self._extract_modifiers_from_node(node)
        parent_symbol = self._get_current_scope_path() if self._scope_stack else None
        qualified_name = self._build_qualified_name(ctor_name)

        param_list = self._find_child_by_type(node, 'parameter_list')
        parameter_types = self._extract_parameter_types(param_list) if param_list else []

        symbol = Symbol(
            name=ctor_name,
            symbol_type=SymbolType.CONSTRUCTOR,
            file_path=self._current_file,
            range=self._create_range(node),
            scope=Scope.FUNCTION,
            parent_symbol=parent_symbol,
            full_name=qualified_name,
            visibility=mods.get('visibility', 'internal'),
            is_static=mods.get('is_static', False),
            parameter_types=parameter_types,
            source_text=self._extract_node_source(node),
        )
        self._symbols.append(symbol)

        # Constructor initializer: : this(...) or : base(...)
        ctor_init = self._find_child_by_type(node, 'constructor_initializer')
        if ctor_init:
            for child in ctor_init.children:
                if child.type == 'this':
                    self._relationships.append(Relationship(
                        source_symbol=qualified_name,
                        target_symbol=qualified_name,
                        relationship_type=RelationshipType.CALLS,
                        source_file=self._current_file,
                        target_file=self._current_file,
                        annotations={'constructor_chain': 'this'},
                    ))
                elif child.type == 'base':
                    parent_class_super = self._find_superclass(parent_symbol)
                    target = f"{parent_class_super}.{parent_class_super.rsplit('.', 1)[-1]}" if parent_class_super else "base.<init>"
                    self._relationships.append(Relationship(
                        source_symbol=qualified_name,
                        target_symbol=target,
                        relationship_type=RelationshipType.CALLS,
                        source_file=self._current_file,
                        target_file=self._current_file,
                        annotations={'constructor_chain': 'base'},
                    ))

        self._enter_scope('<init>')
        self._visit_children(node)
        self._exit_scope()

    def _handle_destructor_declaration(self, node: Node) -> None:
        if not self._scope_stack:
            return

        parent_symbol = self._get_current_scope_path() if self._scope_stack else None
        dtor_name = '~Finalize'
        qualified_name = self._build_qualified_name(dtor_name)

        symbol = Symbol(
            name=dtor_name,
            symbol_type=SymbolType.METHOD,
            file_path=self._current_file,
            range=self._create_range(node),
            scope=Scope.FUNCTION,
            parent_symbol=parent_symbol,
            full_name=qualified_name,
            visibility='protected',
            source_text=self._extract_node_source(node),
            metadata={'is_destructor': True},
        )
        self._symbols.append(symbol)

        self._enter_scope(dtor_name)
        self._visit_children(node)
        self._exit_scope()

    def _handle_property_declaration(self, node: Node) -> None:
        prop_name = None
        prop_type = None
        is_nullable = False

        mods = self._extract_modifiers_from_node(node)
        attrs = self._extract_attributes(node)

        # Pattern: modifier* type identifier accessor_list|arrow|;
        type_candidates = ('predefined_type', 'qualified_name', 'generic_name',
                           'nullable_type', 'array_type', 'tuple_type')

        found_type = False
        for child in node.children:
            if child.type == 'modifier' or child.type == 'attribute_list':
                continue
            if not found_type and child.type in type_candidates:
                prop_type = self._extract_type_name(child)
                is_nullable = self._is_nullable_type_node(child)
                found_type = True
            elif not found_type and child.type == 'identifier':
                # identifier as type followed by identifier as name?
                nxt = child.next_named_sibling
                if nxt and nxt.type == 'identifier':
                    prop_type = self._get_node_text(child)
                    found_type = True
                elif nxt and nxt.type in ('accessor_list', 'arrow_expression_clause'):
                    prop_name = self._get_node_text(child)
                    break
                else:
                    prop_type = self._get_node_text(child)
                    found_type = True
            elif child.type == 'identifier' and found_type:
                prop_name = self._get_node_text(child)
                break

        if not prop_name:
            return

        parent_symbol = self._get_current_scope_path() if self._scope_stack else None
        qualified_name = self._build_qualified_name(prop_name)

        metadata = {k: v for k, v in mods.items() if v and k != 'visibility'}
        if is_nullable:
            metadata['is_nullable'] = True

        # Detect init-only accessors
        accessor_list = self._find_child_by_type(node, 'accessor_list')
        if accessor_list:
            for acc in accessor_list.children:
                if acc.type == 'accessor_declaration':
                    acc_text = self._get_node_text(acc).strip()
                    if acc_text.startswith('init'):
                        metadata['has_init_accessor'] = True

        symbol = Symbol(
            name=prop_name,
            symbol_type=SymbolType.PROPERTY,
            file_path=self._current_file,
            range=self._create_range(node),
            scope=Scope.CLASS,
            parent_symbol=parent_symbol,
            full_name=qualified_name,
            visibility=mods.get('visibility', 'internal'),
            is_static=mods.get('is_static', False),
            is_abstract=mods.get('is_abstract', False),
            return_type=prop_type,
            source_text=self._extract_node_source(node),
            metadata=metadata if metadata else None,
        )
        self._symbols.append(symbol)
        self._emit_annotates(attrs, qualified_name)

        # Type → COMPOSITION / AGGREGATION / REFERENCES
        if prop_type and not self._is_builtin_type(prop_type):
            rel_type = self._determine_field_relationship_type(prop_type, is_nullable)
            self._relationships.append(Relationship(
                source_symbol=qualified_name,
                target_symbol=prop_type,
                relationship_type=rel_type,
                source_file=self._current_file,
                target_file=self._current_file,
            ))

        # Visit expression body if present (for call analysis)
        self._visit_children(node)

    def _handle_field_declaration(self, node: Node) -> None:
        mods = self._extract_modifiers_from_node(node)
        attrs = self._extract_attributes(node)

        # field_declaration → modifier* variable_declaration → type variable_declarator+
        var_decl = self._find_child_by_type(node, 'variable_declaration')
        if not var_decl:
            return

        field_type = None
        is_nullable = False

        for child in var_decl.children:
            if child.type in ('predefined_type', 'identifier', 'qualified_name',
                              'generic_name', 'nullable_type', 'array_type', 'tuple_type'):
                field_type = self._extract_type_name(child)
                is_nullable = self._is_nullable_type_node(child)
                break

        parent_symbol = self._get_current_scope_path() if self._scope_stack else None

        for child in var_decl.children:
            if child.type == 'variable_declarator':
                field_name = None
                id_node = self._find_child_by_type(child, 'identifier')
                if id_node:
                    field_name = self._get_node_text(id_node)
                else:
                    field_name = self._get_node_text(child).split('=')[0].strip()

                if not field_name:
                    continue

                qualified_name = self._build_qualified_name(field_name)

                # Determine symbol type
                sym_type = SymbolType.CONSTANT if mods.get('is_const') else SymbolType.FIELD

                metadata = {k: v for k, v in mods.items() if v and k != 'visibility'}
                if is_nullable:
                    metadata['is_nullable'] = True

                symbol = Symbol(
                    name=field_name,
                    symbol_type=sym_type,
                    file_path=self._current_file,
                    range=self._create_range(node),
                    scope=Scope.CLASS,
                    parent_symbol=parent_symbol,
                    full_name=qualified_name,
                    visibility=mods.get('visibility', 'internal'),
                    is_static=mods.get('is_static', False),
                    return_type=field_type,
                    source_text=self._extract_node_source(node),
                    metadata=metadata if metadata else None,
                )
                self._symbols.append(symbol)
                self._emit_annotates(attrs, qualified_name)

                # Track field type for call resolution
                if field_type:
                    self._field_types[field_name] = field_type

                # Type → relationship
                if field_type and not self._is_builtin_type(field_type):
                    rel_type = self._determine_field_relationship_type(field_type, is_nullable)
                    self._relationships.append(Relationship(
                        source_symbol=qualified_name,
                        target_symbol=field_type,
                        relationship_type=rel_type,
                        source_file=self._current_file,
                        target_file=self._current_file,
                    ))

        # Visit initializers for calls
        self._visit_children(node)

    def _handle_event_field_declaration(self, node: Node) -> None:
        mods = self._extract_modifiers_from_node(node)

        var_decl = self._find_child_by_type(node, 'variable_declaration')
        if not var_decl:
            return

        event_type = None
        for child in var_decl.children:
            if child.type in ('identifier', 'qualified_name', 'generic_name'):
                event_type = self._extract_type_name(child)
                break

        parent_symbol = self._get_current_scope_path() if self._scope_stack else None

        for child in var_decl.children:
            if child.type == 'variable_declarator':
                id_node = self._find_child_by_type(child, 'identifier')
                if not id_node:
                    continue
                event_name = self._get_node_text(id_node)
                qualified_name = self._build_qualified_name(event_name)

                symbol = Symbol(
                    name=event_name,
                    symbol_type=SymbolType.FIELD,
                    file_path=self._current_file,
                    range=self._create_range(node),
                    scope=Scope.CLASS,
                    parent_symbol=parent_symbol,
                    full_name=qualified_name,
                    visibility=mods.get('visibility', 'internal'),
                    is_static=mods.get('is_static', False),
                    return_type=event_type,
                    source_text=self._extract_node_source(node),
                    metadata={'is_event': True},
                )
                self._symbols.append(symbol)

    def _handle_indexer_declaration(self, node: Node) -> None:
        mods = self._extract_modifiers_from_node(node)

        # Return type
        indexer_type = self._extract_return_type(node)

        # Parameters
        bp_list = self._find_child_by_type(node, 'bracketed_parameter_list')
        param_types = self._extract_parameter_types(bp_list) if bp_list else []

        parent_symbol = self._get_current_scope_path() if self._scope_stack else None
        qualified_name = self._build_qualified_name('this[]')

        symbol = Symbol(
            name='this[]',
            symbol_type=SymbolType.PROPERTY,
            file_path=self._current_file,
            range=self._create_range(node),
            scope=Scope.CLASS,
            parent_symbol=parent_symbol,
            full_name=qualified_name,
            visibility=mods.get('visibility', 'internal'),
            return_type=indexer_type,
            parameter_types=param_types,
            source_text=self._extract_node_source(node),
            metadata={'is_indexer': True},
        )
        self._symbols.append(symbol)

        self._visit_children(node)

    def _handle_operator_declaration(self, node: Node) -> None:
        mods = self._extract_modifiers_from_node(node)

        # Pattern: modifier* type operator op parameter_list block
        op_symbol = None
        return_type = None

        type_candidates = ('predefined_type', 'qualified_name', 'generic_name',
                           'nullable_type', 'array_type', 'tuple_type')
        found_operator_kw = False
        found_return = False

        for child in node.children:
            if child.type == 'modifier' or child.type == 'attribute_list':
                continue
            if not found_return and child.type in type_candidates:
                return_type = self._extract_type_name(child)
                found_return = True
            elif not found_return and child.type == 'identifier':
                return_type = self._get_node_text(child)
                found_return = True
            elif child.type == 'operator':
                found_operator_kw = True
            elif found_operator_kw and op_symbol is None:
                op_symbol = self._get_node_text(child)

        if not op_symbol:
            return

        op_name = f"operator {op_symbol}"
        parent_symbol = self._get_current_scope_path() if self._scope_stack else None
        qualified_name = self._build_qualified_name(op_name)

        param_list = self._find_child_by_type(node, 'parameter_list')
        param_types = self._extract_parameter_types(param_list) if param_list else []

        symbol = Symbol(
            name=op_name,
            symbol_type=SymbolType.METHOD,
            file_path=self._current_file,
            range=self._create_range(node),
            scope=Scope.FUNCTION,
            parent_symbol=parent_symbol,
            full_name=qualified_name,
            visibility=mods.get('visibility', 'internal'),
            is_static=True,  # operators are always static
            return_type=return_type,
            parameter_types=param_types,
            source_text=self._extract_node_source(node),
            metadata={'is_operator': True},
        )
        self._symbols.append(symbol)

    # ==================================================================
    # NODE HANDLERS – Expressions (relationships only)
    # ==================================================================

    def _handle_invocation_expression(self, node: Node) -> None:
        """Handle method calls: obj.Method() or Method()."""
        # invocation_expression children:
        #   - member_access_expression or identifier (the thing being called)
        #   - argument_list
        callee = None
        for child in node.children:
            if child.type == 'member_access_expression':
                callee = self._get_node_text(child)
                break
            elif child.type == 'identifier':
                callee = self._get_node_text(child)
                break

        if not callee:
            self._visit_children(node)
            return

        source = self._get_current_scope_path() or Path(self._current_file).stem

        # Try to resolve dotted callee (e.g. obj.Method -> Type.Method)
        if '.' in callee:
            parts = callee.rsplit('.', 1)
            obj_name = parts[0].split('.')[-1]  # last part of object chain
            method_name = parts[1]
            obj_type = self._resolve_object_type(obj_name)
            if obj_type:
                callee = f"{obj_type}.{method_name}"

        self._relationships.append(Relationship(
            source_symbol=source,
            target_symbol=callee,
            relationship_type=RelationshipType.CALLS,
            source_file=self._current_file,
            target_file=self._current_file,
        ))

        # Visit children (nested invocations, argument expressions)
        self._visit_children(node)

    def _handle_object_creation(self, node: Node) -> None:
        """Handle `new Type()` expressions."""
        type_name = None
        for child in node.children:
            if child.type in ('identifier', 'qualified_name', 'generic_name'):
                type_name = self._extract_type_name(child)
                break

        if type_name and not self._is_builtin_type(type_name):
            source = self._get_current_scope_path() or Path(self._current_file).stem
            self._relationships.append(Relationship(
                source_symbol=source,
                target_symbol=type_name,
                relationship_type=RelationshipType.CREATES,
                source_file=self._current_file,
                target_file=self._current_file,
                confidence=0.95,
                annotations={'creation_type': 'new'},
            ))

        self._visit_children(node)

    # ==================================================================
    # RELATIONSHIP HELPERS
    # ==================================================================

    def _extract_base_list(self, node: Node, symbol: Symbol,
                           is_interface: bool = False,
                           is_struct: bool = False) -> None:
        """Extract INHERITANCE and IMPLEMENTATION from base_list.
        
        Heuristic for classes:
          - In C# a class can inherit from at most one class.
          - Items in base_list starting with 'I' + uppercase → IMPLEMENTATION
          - The first non-interface item → INHERITANCE
        For interfaces: all base_list items → INHERITANCE (interface extending interface)
        For structs: all items → IMPLEMENTATION (structs only implement interfaces)
        """
        base_list = self._find_child_by_type(node, 'base_list')
        if not base_list:
            return

        source_qn = symbol.full_name or symbol.name

        for child in base_list.children:
            if child.type in ('identifier', 'qualified_name', 'generic_name'):
                base_name = self._extract_type_name(child)
                if not base_name:
                    continue

                if is_interface or is_struct:
                    # Interface extending interface → INHERITANCE
                    # Struct implementing interface → IMPLEMENTATION
                    rel_type = RelationshipType.INHERITANCE if is_interface else RelationshipType.IMPLEMENTATION
                else:
                    # Class: use I-prefix heuristic
                    rel_type = (RelationshipType.IMPLEMENTATION
                                if self._is_interface_name(base_name)
                                else RelationshipType.INHERITANCE)

                self._relationships.append(Relationship(
                    source_symbol=source_qn,
                    target_symbol=base_name,
                    relationship_type=rel_type,
                    source_file=self._current_file,
                    target_file=self._current_file,
                ))

    def _emit_constraint_references(self, constraints: List[Dict], source_qn: str) -> None:
        """Emit REFERENCES for type constraint types."""
        for clause in constraints:
            for ctype in clause.get('constraints', []):
                if ctype in ('class', 'struct', 'new()', 'notnull', 'unmanaged', 'default'):
                    continue  # keywords, not types
                if not self._is_builtin_type(ctype):
                    self._relationships.append(Relationship(
                        source_symbol=source_qn,
                        target_symbol=ctype,
                        relationship_type=RelationshipType.REFERENCES,
                        source_file=self._current_file,
                        target_file=self._current_file,
                        annotations={'context': 'generic_constraint'},
                    ))

    def _emit_parameter_references(self, param_list: Node, method_qn: str) -> None:
        """Emit REFERENCES for method parameter types (non-builtin)."""
        for child in param_list.children:
            if child.type == 'parameter':
                ptype = self._get_parameter_type(child)
                pname = self._get_parameter_name(child)
                if ptype and not self._is_builtin_type(ptype):
                    self._relationships.append(Relationship(
                        source_symbol=method_qn,
                        target_symbol=ptype,
                        relationship_type=RelationshipType.REFERENCES,
                        source_file=self._current_file,
                        target_file=self._current_file,
                        annotations={'context': 'parameter', 'param_name': pname or 'unknown'},
                    ))

    def _find_superclass(self, class_qualified_name: str) -> Optional[str]:
        """Find the superclass from existing INHERITANCE relationships."""
        for rel in self._relationships:
            if (rel.relationship_type == RelationshipType.INHERITANCE
                    and rel.source_symbol == class_qualified_name):
                return rel.target_symbol
        return None

    def _determine_field_relationship_type(self, type_name: str,
                                            is_nullable: bool = False) -> RelationshipType:
        """Determine COMPOSITION vs AGGREGATION vs REFERENCES for a field type."""
        base = type_name.split('<')[0].split('[')[0].strip()

        if is_nullable or base in _NULLABLE_WRAPPERS:
            return RelationshipType.AGGREGATION
        if base in _COLLECTION_TYPES:
            return RelationshipType.COMPOSITION
        # Default: single object field → COMPOSITION
        return RelationshipType.COMPOSITION

    def _is_extension_method(self, param_list: Node) -> bool:
        """Check if the first parameter has the 'this' modifier (extension method)."""
        if not param_list or not hasattr(param_list, 'children'):
            return False
        for child in param_list.children:
            if child.type == 'parameter':
                # Check for 'this' keyword child
                text = self._get_node_text(child)
                if text.lstrip().startswith('this '):
                    return True
                return False  # only check first parameter
        return False

    def _resolve_object_type(self, name: str) -> Optional[str]:
        """Resolve a variable/field/param name to its type for call resolution."""
        if name in self._field_types:
            return self._field_types[name].split('.')[-1] if '.' in self._field_types[name] else self._field_types[name]
        if name in self._variable_types:
            return self._variable_types[name].split('.')[-1] if '.' in self._variable_types[name] else self._variable_types[name]
        if name in self._parameter_types:
            return self._parameter_types[name].split('.')[-1] if '.' in self._parameter_types[name] else self._parameter_types[name]
        return None

    # ==================================================================
    # PASS 2: DEFINES RELATIONSHIPS
    # ==================================================================

    def _extract_defines_relationships(self, symbols: List[Symbol],
                                        file_path: str) -> List[Relationship]:
        """
        Extract DEFINES relationships (class/struct/interface → members).
        
        Rules:
        - CLASS → METHOD, CONSTRUCTOR, FIELD, PROPERTY, TYPE_ALIAS
        - STRUCT → same
        - INTERFACE → METHOD, PROPERTY
        - ENUM → FIELD (enum members)
        """
        relationships: List[Relationship] = []

        # Build parent→children index
        parent_index: Dict[str, List[Symbol]] = {}
        for sym in symbols:
            if sym.parent_symbol:
                parent_index.setdefault(sym.parent_symbol, []).append(sym)

        definable_containers = {SymbolType.CLASS, SymbolType.STRUCT,
                                SymbolType.INTERFACE, SymbolType.ENUM}
        definable_members = {SymbolType.METHOD, SymbolType.CONSTRUCTOR,
                             SymbolType.FIELD, SymbolType.PROPERTY,
                             SymbolType.CONSTANT, SymbolType.TYPE_ALIAS}

        for sym in symbols:
            if sym.symbol_type not in definable_containers:
                continue

            members = parent_index.get(sym.full_name, [])
            if not members:
                members = parent_index.get(sym.name, [])

            for member in members:
                if member.symbol_type not in definable_members:
                    continue

                # Build target symbol name
                if member.symbol_type == SymbolType.CONSTRUCTOR:
                    param_str = ', '.join(member.parameter_types) if member.parameter_types else ''
                    target = f"{sym.full_name}.<init>({param_str})"
                else:
                    # Detect overloads
                    same_name = [m for m in members
                                 if m.symbol_type == SymbolType.METHOD
                                 and m.name == member.name]
                    if member.symbol_type == SymbolType.METHOD and len(same_name) > 1:
                        param_str = ', '.join(member.parameter_types) if member.parameter_types else ''
                        target = f"{member.full_name}({param_str})"
                    else:
                        target = member.full_name

                annotations = {
                    'member_type': self._classify_member(member),
                    'is_static': member.is_static if hasattr(member, 'is_static') else False,
                    'visibility': member.visibility if hasattr(member, 'visibility') else 'internal',
                }
                if member.symbol_type == SymbolType.METHOD and hasattr(member, 'parameter_types'):
                    annotations['parameter_types'] = member.parameter_types or []

                relationships.append(Relationship(
                    source_symbol=sym.full_name,
                    target_symbol=target,
                    relationship_type=RelationshipType.DEFINES,
                    source_file=file_path,
                    target_file=file_path,
                    annotations=annotations,
                ))

        return relationships

    def _classify_member(self, sym: Symbol) -> str:
        if sym.symbol_type == SymbolType.CONSTRUCTOR:
            return 'constructor'
        if sym.symbol_type == SymbolType.FIELD:
            return 'field'
        if sym.symbol_type == SymbolType.PROPERTY:
            return 'property'
        if sym.symbol_type == SymbolType.CONSTANT:
            return 'constant'
        if sym.symbol_type == SymbolType.METHOD:
            return 'method'
        if sym.symbol_type == SymbolType.TYPE_ALIAS:
            return 'delegate'
        return 'unknown'

    # ==================================================================
    # MULTI-FILE PARSING & CROSS-FILE RESOLUTION
    # ==================================================================

    def parse_multiple_files(self, file_paths: List[str],
                              max_workers: int = 4) -> Dict[str, ParseResult]:
        """Parse multiple C# files and resolve cross-file relationships."""
        import concurrent.futures

        results: Dict[str, ParseResult] = {}
        total = len(file_paths)

        logger.info(f"Parsing {total} C# files for cross-file analysis with {max_workers} workers")
        if this and getattr(this, 'module', None):
            this.module.invocation_thinking(
                f"I am on phase parsing\nStarting C# multi-file parsing: {total} files\n"
                "Reasoning: Collect type/method symbols & usings to fuel cross-file resolution.\n"
                "Next: Parallel parse, then construct global symbol registry."
            )

        results = self._parse_files_parallel(file_paths, max_workers)

        if this and getattr(this, 'module', None):
            succeeded = sum(1 for r in results.values() if r.symbols)
            pct = int((succeeded / total) * 100) if total else 100
            this.module.invocation_thinking(
                f"I am on phase parsing\nRaw C# parsing complete: {succeeded}/{total} files ({pct}%)\n"
                "Next: Build global symbol tables."
            )

        # Build global symbol registry
        for fp, result in results.items():
            if result.symbols:
                self._extract_global_symbols(fp, result)

        # Enhance relationships with cross-file targets
        logger.info("Enhancing C# relationships with cross-file resolution")
        for fp, result in results.items():
            if result.symbols:
                self._enhance_cross_file_relationships(fp, result)

        return results

    def _parse_files_parallel(self, file_paths: List[str],
                               max_workers: int) -> Dict[str, ParseResult]:
        import concurrent.futures

        results: Dict[str, ParseResult] = {}
        total = len(file_paths)
        batch_size = 500

        for batch_start in range(0, total, batch_size):
            batch = file_paths[batch_start:batch_start + batch_size]
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_parse_single_csharp_file, fp): fp
                               for fp in batch}
                    for future in concurrent.futures.as_completed(futures):
                        fp = futures[future]
                        try:
                            parsed_fp, result = future.result(timeout=60)
                            results[parsed_fp] = result
                        except Exception as e:
                            logger.error(f"Failed to parse {fp}: {e}")
                            results[fp] = ParseResult(
                                file_path=fp, language="csharp",
                                symbols=[], relationships=[],
                                errors=[str(e)]
                            )
            except Exception as e:
                logger.error(f"Batch executor failed: {e}")
                for fp in batch:
                    if fp not in results:
                        results[fp] = ParseResult(
                            file_path=fp, language="csharp",
                            symbols=[], relationships=[],
                            errors=[str(e)]
                        )
        return results

    def _extract_global_symbols(self, file_path: str, result: ParseResult) -> None:
        for sym in result.symbols:
            if sym.symbol_type in (SymbolType.CLASS, SymbolType.INTERFACE,
                                   SymbolType.STRUCT, SymbolType.ENUM):
                self._global_class_locations[sym.name] = file_path
                # Store methods of this type
                class_methods = [s for s in result.symbols
                                 if s.symbol_type == SymbolType.METHOD
                                 and s.parent_symbol
                                 and sym.name in s.parent_symbol]
                for method in class_methods:
                    self._global_method_locations.setdefault(method.name, []).append(
                        (sym.name, file_path))

    def _enhance_cross_file_relationships(self, file_path: str,
                                            result: ParseResult) -> None:
        enhanced = []
        usings: Dict[str, str] = {}
        for rel in result.relationships:
            if rel.relationship_type == RelationshipType.IMPORTS and '.' in rel.target_symbol:
                parts = rel.target_symbol.split('.')
                usings[parts[-1]] = rel.target_symbol

        for rel in result.relationships:
            enhanced.append(self._enhance_relationship(rel, file_path, usings))

        result.relationships.clear()
        result.relationships.extend(enhanced)

    def _enhance_relationship(self, rel: Relationship, source_file: str,
                               usings: Dict[str, str]) -> Relationship:
        target = rel.target_symbol
        target_class = target.split('.')[0] if '.' in target else target

        if rel.relationship_type in (RelationshipType.CREATES, RelationshipType.INHERITANCE,
                                     RelationshipType.IMPLEMENTATION):
            target_file = self._global_class_locations.get(target_class)
            if target_file and target_file != source_file:
                return Relationship(
                    source_symbol=rel.source_symbol,
                    target_symbol=rel.target_symbol,
                    relationship_type=rel.relationship_type,
                    source_file=source_file,
                    target_file=target_file,
                    source_range=rel.source_range,
                    confidence=rel.confidence,
                    annotations=rel.annotations,
                )

        elif rel.relationship_type == RelationshipType.CALLS:
            if '.' in target:
                parts = target.rsplit('.', 1)
                class_name = parts[0]
                target_file = self._global_class_locations.get(class_name)
                if target_file and target_file != source_file:
                    return Relationship(
                        source_symbol=rel.source_symbol,
                        target_symbol=target,
                        relationship_type=RelationshipType.CALLS,
                        source_file=source_file,
                        target_file=target_file,
                    )
            else:
                if target in self._global_method_locations:
                    for cls_name, fp in self._global_method_locations[target]:
                        if fp != source_file:
                            return Relationship(
                                source_symbol=rel.source_symbol,
                                target_symbol=f"{cls_name}.{target}",
                                relationship_type=RelationshipType.CALLS,
                                source_file=source_file,
                                target_file=fp,
                            )

        elif rel.relationship_type == RelationshipType.IMPORTS:
            if '.' in target:
                parts = target.split('.')
                class_name = parts[-1]
                target_file = self._global_class_locations.get(class_name)
                if target_file and target_file != source_file:
                    return Relationship(
                        source_symbol=rel.source_symbol,
                        target_symbol=target,
                        relationship_type=RelationshipType.IMPORTS,
                        source_file=source_file,
                        target_file=target_file,
                    )

        elif rel.relationship_type == RelationshipType.REFERENCES:
            target_file = self._global_class_locations.get(target_class)
            if target_file and target_file != source_file:
                return Relationship(
                    source_symbol=rel.source_symbol,
                    target_symbol=target,
                    relationship_type=RelationshipType.REFERENCES,
                    source_file=source_file,
                    target_file=target_file,
                )

        return rel
