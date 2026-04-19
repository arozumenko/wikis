"""
Go Visitor-based Parser using Tree-sitter for DeepWiki Graph Enhancement.

Implements a visitor pattern for Go AST traversal, extracting symbols and
relationships for cross-file analysis. Follows the JavaVisitorParser pattern.

Key Go-specific features:
- Struct ↔ method linkage via receivers (methods are top-level, not nested)
- Struct embedding → COMPOSITION (not inheritance)
- Interface embedding → INHERITANCE (the only INHERITANCE in Go)
- Implicit interface satisfaction via method-set subset matching
- Visibility from capitalization (Exported vs unexported)
- Multiple init() functions per package
- Generics (type parameters on functions, structs, interfaces)

Relationship semantics:
- INHERITANCE is used ONLY for interface embedding (method set extension)
- COMPOSITION is used for struct embedding (anonymous fields) + value-type fields
- AGGREGATION is used for pointer-type fields
- IMPLEMENTATION is used for implicit interface satisfaction (structural typing)
"""

import logging
import time
import concurrent.futures
from typing import List, Dict, Optional, Set, Union, Tuple, Any
from pathlib import Path

from tree_sitter import Node
from tree_sitter_language_pack import get_language, get_parser

from .base_parser import (
    BaseParser, Symbol, Relationship, ParseResult, LanguageCapabilities,
    SymbolType, RelationshipType, Scope, Position, Range, parser_registry
)

logger = logging.getLogger(__name__)

# Optional thinking emitter (platform integration)
try:  # pragma: no cover
    from tools import this  # type: ignore
except Exception:  # pragma: no cover
    this = None  # type: ignore


# ============================================================================
# Go-specific Constants
# ============================================================================

GO_BUILTIN_TYPES = frozenset({
    'int', 'int8', 'int16', 'int32', 'int64',
    'uint', 'uint8', 'uint16', 'uint32', 'uint64', 'uintptr',
    'float32', 'float64', 'complex64', 'complex128',
    'string', 'byte', 'rune', 'bool',
    'error', 'any', 'comparable',
})

GO_BUILTIN_FUNCTIONS = frozenset({
    'make', 'new', 'len', 'cap', 'append', 'copy', 'delete',
    'close', 'panic', 'recover', 'print', 'println',
    'complex', 'real', 'imag', 'clear', 'min', 'max',
})

# Well-known stdlib interfaces for implicit implementation detection.
# Maps interface_name → frozenset of method signatures (just names for now).
WELL_KNOWN_INTERFACES: Dict[str, frozenset] = {
    'error': frozenset({'Error'}),
    'fmt.Stringer': frozenset({'String'}),
    'io.Reader': frozenset({'Read'}),
    'io.Writer': frozenset({'Write'}),
    'io.Closer': frozenset({'Close'}),
    'io.ReadWriter': frozenset({'Read', 'Write'}),
    'io.ReadCloser': frozenset({'Read', 'Close'}),
    'io.WriteCloser': frozenset({'Write', 'Close'}),
    'io.ReadWriteCloser': frozenset({'Read', 'Write', 'Close'}),
    'sort.Interface': frozenset({'Len', 'Less', 'Swap'}),
    'encoding.TextMarshaler': frozenset({'MarshalText'}),
    'encoding.TextUnmarshaler': frozenset({'UnmarshalText'}),
    'encoding.BinaryMarshaler': frozenset({'MarshalBinary'}),
    'encoding.BinaryUnmarshaler': frozenset({'UnmarshalBinary'}),
    'json.Marshaler': frozenset({'MarshalJSON'}),
    'json.Unmarshaler': frozenset({'UnmarshalJSON'}),
    'context.Context': frozenset({'Deadline', 'Done', 'Err', 'Value'}),
    'http.Handler': frozenset({'ServeHTTP'}),
}


def _parse_single_go_file(file_path: str) -> Tuple[str, ParseResult]:
    """Parse a single Go file — used by parallel workers (module-level for pickling)."""
    try:
        parser = GoVisitorParser()
        result = parser.parse_file(file_path)
        return file_path, result
    except Exception as e:
        logger.warning(f"Failed to parse {file_path}: {e}")
        return file_path, ParseResult(
            file_path=file_path,
            language="go",
            symbols=[],
            relationships=[]
        )


class GoVisitorParser(BaseParser):
    """Go parser using visitor pattern with real Tree-sitter."""

    def __init__(self):
        super().__init__("go")
        self.parser = None
        self.ts_language = None

        # Per-file state (reset on each parse_file call)
        self._current_file: str = ""
        self._current_content: str = ""
        self._symbols: List[Symbol] = []
        self._relationships: List[Relationship] = []

        # Go-specific per-file state
        self._current_package: str = ""
        self._current_imports: Dict[str, str] = {}       # short_name → full_path
        self._file_method_receivers: Dict[str, Set[str]] = {}  # type_name → {method_names}
        self._file_type_names: Set[str] = set()           # types defined in current file
        self._init_counter: int = 0                        # for disambiguating multiple init()

        # Global registries for cross-file analysis (populated by parse_multiple_files)
        self._global_type_registry: Dict[str, str] = {}          # type_name → file_path
        self._global_function_registry: Dict[str, str] = {}      # func_name → file_path
        self._global_method_registry: Dict[str, Dict[str, str]] = {}  # type_name → {method_name → file_path}
        self._global_interface_methods: Dict[str, Set[str]] = {}  # interface_name → {method_names}

        self._setup_tree_sitter()

    # ========================================================================
    # BaseParser abstract method implementations
    # ========================================================================

    def _define_capabilities(self) -> LanguageCapabilities:
        """Define Go parser capabilities."""
        return LanguageCapabilities(
            language="go",
            supported_symbols={
                SymbolType.FUNCTION, SymbolType.METHOD,
                SymbolType.STRUCT, SymbolType.INTERFACE,
                SymbolType.FIELD, SymbolType.CONSTANT,
                SymbolType.VARIABLE, SymbolType.TYPE_ALIAS,
                SymbolType.ENUM,
            },
            supported_relationships={
                RelationshipType.IMPORTS, RelationshipType.CALLS,
                RelationshipType.COMPOSITION, RelationshipType.AGGREGATION,
                RelationshipType.INHERITANCE,
                RelationshipType.IMPLEMENTATION,
                RelationshipType.CREATES, RelationshipType.DEFINES,
                RelationshipType.REFERENCES, RelationshipType.ALIAS_OF,
            },
            supports_ast_parsing=True,
            supports_type_inference=True,
            supports_cross_file_analysis=True,
            has_classes=False,
            has_interfaces=True,
            has_namespaces=False,
            has_generics=True,
            has_decorators=False,
            has_annotations=False,
            max_file_size_mb=50.0,
            typical_parse_time_ms=20.0,
        )

    def _get_supported_extensions(self) -> Set[str]:
        """Get supported Go file extensions."""
        return {'.go'}

    def _setup_tree_sitter(self) -> bool:
        """Setup real tree-sitter Go parser."""
        try:
            self.ts_language = get_language('go')
            self.parser = get_parser('go')
            return True
        except Exception as e:
            logger.error(f"Failed to initialize tree-sitter Go parser: {e}")
            return False

    def extract_symbols(self, ast_node: Any, file_path: str) -> List[Symbol]:
        """Extract symbols from AST — delegated to parse_file visitor."""
        return self._symbols

    def extract_relationships(self, ast_node: Any, symbols: List[Symbol], file_path: str) -> List[Relationship]:
        """Extract relationships from AST — delegated to parse_file visitor."""
        return self._relationships

    # ========================================================================
    # Main parse entry point
    # ========================================================================

    def parse_file(self, file_path: Union[str, Path], content: Optional[str] = None) -> ParseResult:
        """Parse a Go file using tree-sitter visitor pattern."""
        start_time = time.time()
        file_path = str(file_path)
        self._current_file = file_path

        # Reset per-file state
        self._symbols = []
        self._relationships = []
        self._current_package = ""
        self._current_imports = {}
        self._file_method_receivers = {}
        self._file_type_names = set()
        self._init_counter = 0

        try:
            if content is None:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()

            self._current_content = content

            if not self.parser or not self.ts_language:
                return ParseResult(
                    file_path=file_path, language="go",
                    symbols=[], relationships=[],
                    errors=["Tree-sitter Go parser not available"]
                )

            tree = self.parser.parse(bytes(content, "utf8"))
            self._visit_top_level(tree.root_node)

            # Post-parse: link methods to structs within the same file
            self._link_same_file_methods()

            parse_time = time.time() - start_time
            return ParseResult(
                file_path=file_path,
                language="go",
                symbols=self._symbols,
                relationships=self._relationships,
                imports=list(self._current_imports.values()),
                parse_time=parse_time,
            )

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return ParseResult(
                file_path=file_path, language="go",
                symbols=[], relationships=[],
                errors=[f"File not found: {file_path}"]
            )
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}", exc_info=True)
            return ParseResult(
                file_path=file_path, language="go",
                symbols=[], relationships=[],
                errors=[str(e)]
            )

    # ========================================================================
    # Visitor Pattern — Top-level dispatch
    # ========================================================================

    def _visit_top_level(self, node: Node) -> None:
        """Visit top-level nodes in a Go source file."""
        if not node:
            return

        for child in node.children:
            ntype = child.type
            if ntype == 'package_clause':
                self._handle_package_clause(child)
            elif ntype == 'import_declaration':
                self._handle_import_declaration(child)
            elif ntype == 'type_declaration':
                self._handle_type_declaration(child)
            elif ntype == 'function_declaration':
                self._handle_function_declaration(child)
            elif ntype == 'method_declaration':
                self._handle_method_declaration(child)
            elif ntype == 'const_declaration':
                self._handle_const_declaration(child)
            elif ntype == 'var_declaration':
                self._handle_var_declaration(child)
            # comments are handled lazily via _get_preceding_comment

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _get_node_text(self, node: Optional[Node]) -> str:
        """Get text content of a node."""
        if not node or not hasattr(node, 'text'):
            return ""
        try:
            return node.text.decode('utf8')
        except Exception:
            return ""

    def _get_source_text(self, node: Node) -> str:
        """Get the full source text of a node."""
        return self._get_node_text(node)

    def _make_range(self, node: Node) -> Range:
        """Create a Range from a tree-sitter node."""
        return Range(
            start=Position(line=node.start_point[0] + 1, column=node.start_point[1]),
            end=Position(line=node.end_point[0] + 1, column=node.end_point[1]),
        )

    def _get_visibility(self, name: str) -> str:
        """Determine visibility from Go naming convention (capitalized = public)."""
        if not name:
            return "private"
        return "public" if name[0].isupper() else "private"

    def _find_child_by_type(self, node: Node, child_type: str) -> Optional[Node]:
        """Find the first child of the given type."""
        for child in node.children:
            if child.type == child_type:
                return child
        return None

    def _find_children_by_type(self, node: Node, child_type: str) -> List[Node]:
        """Find all children of the given type."""
        return [c for c in node.children if c.type == child_type]

    def _get_preceding_comment(self, node: Node) -> Optional[str]:
        """Get the comment immediately preceding a node (doc comment)."""
        prev = node.prev_named_sibling
        if prev and prev.type == 'comment':
            text = self._get_node_text(prev)
            # Strip // prefix
            if text.startswith('//'):
                return text[2:].strip()
            # Strip /* */ prefix
            if text.startswith('/*') and text.endswith('*/'):
                return text[2:-2].strip()
        return None

    def _make_relationship(
        self,
        source: str,
        target: str,
        rel_type: RelationshipType,
        node: Optional[Node] = None,
        target_file: Optional[str] = None,
        confidence: float = 1.0,
        annotations: Optional[Dict[str, Any]] = None,
    ) -> Relationship:
        """Create a Relationship with proper source file and range."""
        source_range = self._make_range(node) if node else Range(
            start=Position(line=1, column=0), end=Position(line=1, column=0)
        )
        return Relationship(
            source_symbol=source,
            target_symbol=target,
            relationship_type=rel_type,
            source_file=self._current_file,
            target_file=target_file or self._current_file,
            source_range=source_range,
            confidence=confidence,
            annotations=annotations or {},
        )

    # ========================================================================
    # Phase 1: Symbol Extraction Handlers
    # ========================================================================

    def _handle_package_clause(self, node: Node) -> None:
        """Handle package declaration — sets _current_package, no symbol emitted."""
        pkg_id = self._find_child_by_type(node, 'package_identifier')
        if pkg_id:
            self._current_package = self._get_node_text(pkg_id)

    def _handle_import_declaration(self, node: Node) -> None:
        """Handle import declaration — extract import relationships."""
        # Single import: import "fmt"
        # Group import: import ( ... )
        spec_list = self._find_child_by_type(node, 'import_spec_list')
        if spec_list:
            for spec in self._find_children_by_type(spec_list, 'import_spec'):
                self._handle_import_spec(spec)
        else:
            # Single import spec directly under import_declaration
            spec = self._find_child_by_type(node, 'import_spec')
            if spec:
                self._handle_import_spec(spec)

    def _handle_import_spec(self, node: Node) -> None:
        """Handle a single import spec."""
        # import_spec can contain:
        #   - interpreted_string_literal (just path)
        #   - package_identifier + interpreted_string_literal (aliased)
        #   - dot + interpreted_string_literal (dot import)
        #   - blank_identifier + interpreted_string_literal (blank import)
        path_node = self._find_child_by_type(node, 'interpreted_string_literal')
        if not path_node:
            return

        import_path = self._get_node_text(path_node).strip('"')

        # Determine the short name (alias or last segment of path)
        alias_node = self._find_child_by_type(node, 'package_identifier')
        dot_node = self._find_child_by_type(node, '.')
        blank_node = self._find_child_by_type(node, 'blank_identifier')

        if alias_node:
            short_name = self._get_node_text(alias_node)
        elif dot_node:
            short_name = '.'  # dot import
        elif blank_node:
            short_name = '_'  # blank import (side-effect only)
        else:
            # Default: last segment of import path
            short_name = import_path.rsplit('/', 1)[-1] if '/' in import_path else import_path

        self._current_imports[short_name] = import_path

        # Emit IMPORTS relationship
        self._relationships.append(self._make_relationship(
            source=Path(self._current_file).stem,
            target=import_path,
            rel_type=RelationshipType.IMPORTS,
            node=node,
            annotations={'alias': short_name if alias_node else None,
                         'is_dot': dot_node is not None,
                         'is_blank': blank_node is not None},
        ))

    def _handle_type_declaration(self, node: Node) -> None:
        """Handle type declaration — dispatch to struct, interface, alias, or named type."""
        for child in node.children:
            if child.type == 'type_spec':
                self._handle_type_spec(child, node)
            elif child.type == 'type_alias':
                self._handle_type_alias(child, node)

    def _handle_type_spec(self, spec_node: Node, decl_node: Node) -> None:
        """Handle type_spec: type Name <underlying_type>."""
        name_node = self._find_child_by_type(spec_node, 'type_identifier')
        if not name_node:
            return
        name = self._get_node_text(name_node)

        # Determine what kind of type this is based on the underlying type node
        struct_node = self._find_child_by_type(spec_node, 'struct_type')
        interface_node = self._find_child_by_type(spec_node, 'interface_type')

        if struct_node:
            self._handle_struct_type(name, spec_node, struct_node, decl_node)
        elif interface_node:
            self._handle_interface_type(name, spec_node, interface_node, decl_node)
        else:
            # Named type (e.g., `type UserID int64`, `type Direction int`)
            self._handle_named_type(name, spec_node, decl_node)

    def _handle_struct_type(self, name: str, spec_node: Node, struct_node: Node, decl_node: Node) -> None:
        """Handle struct type definition — emit STRUCT symbol + FIELD children."""
        docstring = self._get_preceding_comment(decl_node)
        source_text = self._get_source_text(decl_node)
        type_params = self._extract_type_parameters(spec_node)

        # Build signature
        sig = f"type {name}"
        if type_params:
            sig += f"[{', '.join(f'{n} {c}' for n, c in type_params)}]"
        sig += " struct"

        sym = Symbol(
            name=name,
            symbol_type=SymbolType.STRUCT,
            scope=Scope.GLOBAL,
            range=self._make_range(decl_node),
            file_path=self._current_file,
            full_name=name,
            visibility=self._get_visibility(name),
            docstring=docstring,
            source_text=source_text,
            signature=sig,
            metadata={'type_parameters': type_params} if type_params else {},
        )
        self._symbols.append(sym)
        self._file_type_names.add(name)

        # Extract fields
        field_list = self._find_child_by_type(struct_node, 'field_declaration_list')
        if field_list:
            self._extract_struct_fields(name, field_list)

    def _extract_struct_fields(self, struct_name: str, field_list_node: Node) -> None:
        """Extract fields from a struct's field_declaration_list."""
        for field_decl in self._find_children_by_type(field_list_node, 'field_declaration'):
            field_id = self._find_child_by_type(field_decl, 'field_identifier')

            if field_id:
                # Named field: e.g., `Host string`, `Logger *Logger`
                field_name = self._get_node_text(field_id)
                field_type_str = self._extract_field_type(field_decl)
                is_pointer = self._is_pointer_type(field_decl)

                # Extract struct tag if present
                tag_node = (self._find_child_by_type(field_decl, 'raw_string_literal') or
                            self._find_child_by_type(field_decl, 'interpreted_string_literal'))
                struct_tag = self._get_node_text(tag_node).strip('`"') if tag_node else None

                field_sym = Symbol(
                    name=field_name,
                    symbol_type=SymbolType.FIELD,
                    scope=Scope.CLASS,
                    range=self._make_range(field_decl),
                    file_path=self._current_file,
                    parent_symbol=struct_name,
                    full_name=f"{struct_name}.{field_name}",
                    visibility=self._get_visibility(field_name),
                    source_text=self._get_source_text(field_decl),
                    metadata={
                        'field_type': field_type_str,
                        'is_pointer': is_pointer,
                        'struct_tag': struct_tag,
                    },
                )
                self._symbols.append(field_sym)

                # DEFINES: struct → field
                self._relationships.append(self._make_relationship(
                    source=struct_name,
                    target=f"{struct_name}.{field_name}",
                    rel_type=RelationshipType.DEFINES,
                    node=field_decl,
                    annotations={'member_type': 'field'},
                ))

                # Field type relationships (skip builtins)
                base_type = self._strip_pointer_slice(field_type_str)
                if base_type and base_type not in GO_BUILTIN_TYPES:
                    if is_pointer:
                        self._relationships.append(self._make_relationship(
                            source=struct_name, target=base_type,
                            rel_type=RelationshipType.AGGREGATION,
                            node=field_decl,
                        ))
                    else:
                        self._relationships.append(self._make_relationship(
                            source=struct_name, target=base_type,
                            rel_type=RelationshipType.COMPOSITION,
                            node=field_decl,
                        ))
            else:
                # Embedded field (no field_identifier): `Handler` or `io.Reader`
                type_node = (self._find_child_by_type(field_decl, 'type_identifier') or
                             self._find_child_by_type(field_decl, 'qualified_type') or
                             self._find_child_by_type(field_decl, 'pointer_type'))

                if type_node:
                    embed_name = self._extract_embedded_name(type_node)
                    if embed_name:
                        field_sym = Symbol(
                            name=embed_name,
                            symbol_type=SymbolType.FIELD,
                            scope=Scope.CLASS,
                            range=self._make_range(field_decl),
                            file_path=self._current_file,
                            parent_symbol=struct_name,
                            full_name=f"{struct_name}.{embed_name}",
                            visibility=self._get_visibility(embed_name),
                            source_text=self._get_source_text(field_decl),
                            metadata={'is_embedded': True},
                        )
                        self._symbols.append(field_sym)

                        # Struct embedding → COMPOSITION (NOT inheritance)
                        self._relationships.append(self._make_relationship(
                            source=struct_name, target=embed_name,
                            rel_type=RelationshipType.COMPOSITION,
                            node=field_decl,
                            annotations={'is_embedding': True},
                        ))

                        # DEFINES: struct → embedded field
                        self._relationships.append(self._make_relationship(
                            source=struct_name,
                            target=f"{struct_name}.{embed_name}",
                            rel_type=RelationshipType.DEFINES,
                            node=field_decl,
                            annotations={'member_type': 'embedded_field'},
                        ))

    def _handle_interface_type(self, name: str, spec_node: Node, iface_node: Node, decl_node: Node) -> None:
        """Handle interface type definition — emit INTERFACE + METHOD children."""
        docstring = self._get_preceding_comment(decl_node)
        source_text = self._get_source_text(decl_node)
        type_params = self._extract_type_parameters(spec_node)

        sig = f"type {name}"
        if type_params:
            sig += f"[{', '.join(f'{n} {c}' for n, c in type_params)}]"
        sig += " interface"

        sym = Symbol(
            name=name,
            symbol_type=SymbolType.INTERFACE,
            scope=Scope.GLOBAL,
            range=self._make_range(decl_node),
            file_path=self._current_file,
            full_name=name,
            visibility=self._get_visibility(name),
            docstring=docstring,
            source_text=source_text,
            signature=sig,
            metadata={'type_parameters': type_params} if type_params else {},
        )
        self._symbols.append(sym)
        self._file_type_names.add(name)

        # Extract interface methods and embedded types
        self._extract_interface_members(name, iface_node)

    def _extract_interface_members(self, iface_name: str, iface_node: Node) -> None:
        """Extract methods and embedded interfaces from an interface body."""
        for child in iface_node.children:
            if child.type == 'method_elem':
                # Interface method signature
                method_name_node = self._find_child_by_type(child, 'field_identifier')
                if method_name_node:
                    method_name = self._get_node_text(method_name_node)
                    params = self._extract_parameters_from_list(
                        self._find_child_by_type(child, 'parameter_list'))

                    # Return type(s) — could be a single type_identifier or a parameter_list
                    return_types = self._extract_return_types_from_method_elem(child)

                    sig_parts = [f"{method_name}({', '.join(f'{n} {t}' for n, t in params)})"]
                    if return_types:
                        sig_parts.append(' '.join(return_types))

                    method_sym = Symbol(
                        name=method_name,
                        symbol_type=SymbolType.METHOD,
                        scope=Scope.FUNCTION,
                        range=self._make_range(child),
                        file_path=self._current_file,
                        parent_symbol=iface_name,
                        full_name=f"{iface_name}.{method_name}",
                        visibility=self._get_visibility(method_name),
                        is_abstract=True,
                        source_text=self._get_source_text(child),
                        signature=' '.join(sig_parts),
                        parameter_types=[t for _, t in params],
                        return_type=', '.join(return_types) if return_types else None,
                        metadata={'is_abstract': True},
                    )
                    self._symbols.append(method_sym)

                    # DEFINES: interface → method
                    self._relationships.append(self._make_relationship(
                        source=iface_name,
                        target=f"{iface_name}.{method_name}",
                        rel_type=RelationshipType.DEFINES,
                        node=child,
                        annotations={'member_type': 'interface_method'},
                    ))

            elif child.type == 'type_elem':
                # Embedded interface: `Reader`, `io.Writer`
                type_id = self._find_child_by_type(child, 'type_identifier')
                qualified = self._find_child_by_type(child, 'qualified_type')
                embedded_name = None

                if qualified:
                    embedded_name = self._get_node_text(qualified)
                elif type_id:
                    embedded_name = self._get_node_text(type_id)

                if embedded_name:
                    # Interface embedding → INHERITANCE (the only use of INHERITANCE in Go)
                    self._relationships.append(self._make_relationship(
                        source=iface_name,
                        target=embedded_name,
                        rel_type=RelationshipType.INHERITANCE,
                        node=child,
                        annotations={'is_interface_embedding': True},
                    ))

    def _handle_function_declaration(self, node: Node) -> None:
        """Handle standalone function declaration."""
        name_node = self._find_child_by_type(node, 'identifier')
        if not name_node:
            return
        name = self._get_node_text(name_node)

        # Handle multiple init() per package
        if name == 'init':
            self._init_counter += 1
            if self._init_counter > 1:
                name = f"init_L{node.start_point[0] + 1}"

        docstring = self._get_preceding_comment(node)
        source_text = self._get_source_text(node)
        type_params = self._extract_type_parameters(node)

        # Extract parameters and return types
        param_lists = self._find_children_by_type(node, 'parameter_list')
        params = self._extract_parameters_from_list(param_lists[0]) if param_lists else []
        return_types = self._extract_return_types_from_param_lists(param_lists)

        # Build signature
        sig = f"func {name}"
        if type_params:
            sig += f"[{', '.join(f'{n} {c}' for n, c in type_params)}]"
        sig += f"({', '.join(f'{n} {t}' for n, t in params)})"
        if return_types:
            if len(return_types) == 1:
                sig += f" {return_types[0]}"
            else:
                sig += f" ({', '.join(return_types)})"

        sym = Symbol(
            name=name,
            symbol_type=SymbolType.FUNCTION,
            scope=Scope.GLOBAL,
            range=self._make_range(node),
            file_path=self._current_file,
            full_name=name,
            visibility=self._get_visibility(name),
            docstring=docstring,
            source_text=source_text,
            signature=sig,
            parameter_types=[t for _, t in params],
            return_type=', '.join(return_types) if return_types else None,
            metadata={
                'type_parameters': type_params,
            } if type_params else {},
        )
        self._symbols.append(sym)

        # Walk function body for calls and creates
        body = self._find_child_by_type(node, 'block')
        if body:
            self._walk_body(body, source_symbol=name)

    def _handle_method_declaration(self, node: Node) -> None:
        """Handle method declaration (function with receiver)."""
        # Method name is field_identifier (not identifier)
        name_node = self._find_child_by_type(node, 'field_identifier')
        if not name_node:
            return
        method_name = self._get_node_text(name_node)

        # Extract receiver info: first parameter_list is the receiver
        receiver_type, is_pointer = self._extract_receiver_info(node)
        if not receiver_type:
            return

        docstring = self._get_preceding_comment(node)
        source_text = self._get_source_text(node)

        # Parameters: second parameter_list
        param_lists = self._find_children_by_type(node, 'parameter_list')
        # param_lists[0] = receiver, param_lists[1] = params, param_lists[2] = returns (optional)
        params = self._extract_parameters_from_list(param_lists[1]) if len(param_lists) > 1 else []
        return_types = self._extract_return_types_from_param_lists(param_lists, skip_first=2)

        # Also check for a simple return type (just a type_identifier after the param list)
        if not return_types:
            # Look for type_identifier directly after parameter lists
            for child in node.children:
                if child.type in ('type_identifier', 'pointer_type', 'qualified_type',
                                  'slice_type', 'map_type', 'channel_type', 'array_type'):
                    return_types = [self._get_node_text(child)]
                    break

        full_name = f"{receiver_type}.{method_name}"

        sig = f"func ({receiver_type}) {method_name}({', '.join(f'{n} {t}' for n, t in params)})"
        if return_types:
            if len(return_types) == 1:
                sig += f" {return_types[0]}"
            else:
                sig += f" ({', '.join(return_types)})"

        sym = Symbol(
            name=method_name,
            symbol_type=SymbolType.METHOD,
            scope=Scope.FUNCTION,
            range=self._make_range(node),
            file_path=self._current_file,
            parent_symbol=receiver_type,
            full_name=full_name,
            visibility=self._get_visibility(method_name),
            docstring=docstring,
            source_text=source_text,
            signature=sig,
            parameter_types=[t for _, t in params],
            return_type=', '.join(return_types) if return_types else None,
            metadata={
                'is_pointer_receiver': is_pointer,
                'receiver_type': receiver_type,
            },
        )
        self._symbols.append(sym)

        # Track for cross-file linkage
        self._file_method_receivers.setdefault(receiver_type, set()).add(method_name)

        # Walk method body for calls and creates
        body = self._find_child_by_type(node, 'block')
        if body:
            self._walk_body(body, source_symbol=full_name)

    def _handle_const_declaration(self, node: Node) -> None:
        """Handle const declaration — single or grouped, with iota detection."""
        # Check for iota group
        const_specs = self._find_children_by_type(node, 'const_spec')
        if not const_specs:
            return

        has_iota = False
        iota_type = None

        # Check first const_spec for iota
        first_spec = const_specs[0]
        expr_list = self._find_child_by_type(first_spec, 'expression_list')
        if expr_list:
            for child in expr_list.children:
                if child.type == 'iota':
                    has_iota = True
                    break

        # If iota, check for explicit type
        if has_iota:
            type_node = self._find_child_by_type(first_spec, 'type_identifier')
            if type_node:
                iota_type = self._get_node_text(type_node)

        if has_iota and len(const_specs) > 1:
            # Emit as ENUM group
            enum_name = iota_type or self._get_node_text(
                self._find_child_by_type(first_spec, 'identifier')
            ) + "_group"
            docstring = self._get_preceding_comment(node)

            enum_sym = Symbol(
                name=enum_name,
                symbol_type=SymbolType.ENUM,
                scope=Scope.GLOBAL,
                range=self._make_range(node),
                file_path=self._current_file,
                full_name=enum_name,
                visibility=self._get_visibility(enum_name),
                docstring=docstring,
                source_text=self._get_source_text(node),
                signature=f"const ({enum_name} = iota ...)",
                metadata={'is_iota_group': True},
            )
            self._symbols.append(enum_sym)
            self._file_type_names.add(enum_name)

        # Emit individual constants
        for spec in const_specs:
            id_node = self._find_child_by_type(spec, 'identifier')
            if not id_node:
                continue
            const_name = self._get_node_text(id_node)

            const_sym = Symbol(
                name=const_name,
                symbol_type=SymbolType.CONSTANT,
                scope=Scope.GLOBAL,
                range=self._make_range(spec),
                file_path=self._current_file,
                full_name=const_name,
                visibility=self._get_visibility(const_name),
                source_text=self._get_source_text(spec),
                metadata={
                    'iota_type': iota_type,
                    'is_iota': has_iota,
                },
            )
            self._symbols.append(const_sym)

            # If part of enum group, link constant to enum
            if has_iota and len(const_specs) > 1:
                self._relationships.append(self._make_relationship(
                    source=enum_name,
                    target=const_name,
                    rel_type=RelationshipType.DEFINES,
                    node=spec,
                    annotations={'member_type': 'enum_value'},
                ))

    def _handle_var_declaration(self, node: Node) -> None:
        """Handle package-level variable declaration."""
        for var_spec in self._find_children_by_type(node, 'var_spec'):
            id_node = self._find_child_by_type(var_spec, 'identifier')
            if not id_node:
                continue
            var_name = self._get_node_text(id_node)

            var_sym = Symbol(
                name=var_name,
                symbol_type=SymbolType.VARIABLE,
                scope=Scope.GLOBAL,
                range=self._make_range(var_spec),
                file_path=self._current_file,
                full_name=var_name,
                visibility=self._get_visibility(var_name),
                source_text=self._get_source_text(var_spec),
            )
            self._symbols.append(var_sym)

    def _handle_type_alias(self, alias_node: Node, decl_node: Node) -> None:
        """Handle type alias: `type StringAlias = string`."""
        name_node = self._find_child_by_type(alias_node, 'type_identifier')
        if not name_node:
            return
        name = self._get_node_text(name_node)

        # Get the target type (the type after '=')
        # type_alias children: type_identifier, '=', type_identifier
        target_type = None
        found_eq = False
        for child in alias_node.children:
            if child.type == '=':
                found_eq = True
            elif found_eq and child.type in ('type_identifier', 'qualified_type',
                                              'pointer_type', 'slice_type', 'map_type'):
                target_type = self._get_node_text(child)
                break

        docstring = self._get_preceding_comment(decl_node)

        sym = Symbol(
            name=name,
            symbol_type=SymbolType.TYPE_ALIAS,
            scope=Scope.GLOBAL,
            range=self._make_range(decl_node),
            file_path=self._current_file,
            full_name=name,
            visibility=self._get_visibility(name),
            docstring=docstring,
            source_text=self._get_source_text(decl_node),
            signature=f"type {name} = {target_type or '?'}",
            metadata={'target_type': target_type, 'is_true_alias': True},
        )
        self._symbols.append(sym)
        self._file_type_names.add(name)

        # ALIAS_OF relationship
        if target_type and target_type not in GO_BUILTIN_TYPES:
            self._relationships.append(self._make_relationship(
                source=name, target=target_type,
                rel_type=RelationshipType.ALIAS_OF,
                node=alias_node,
            ))

    def _handle_named_type(self, name: str, spec_node: Node, decl_node: Node) -> None:
        """Handle named type: `type UserID int64`, `type Direction int`."""
        # Get underlying type
        underlying = None
        name_found = False
        for child in spec_node.children:
            if child.type == 'type_identifier':
                if not name_found:
                    name_found = True  # skip the name itself
                else:
                    underlying = self._get_node_text(child)
                    break
            elif name_found and child.type in ('qualified_type', 'pointer_type',
                                                'slice_type', 'map_type', 'struct_type',
                                                'interface_type', 'function_type',
                                                'channel_type', 'array_type'):
                underlying = self._get_node_text(child)
                break

        docstring = self._get_preceding_comment(decl_node)

        sym = Symbol(
            name=name,
            symbol_type=SymbolType.TYPE_ALIAS,
            scope=Scope.GLOBAL,
            range=self._make_range(decl_node),
            file_path=self._current_file,
            full_name=name,
            visibility=self._get_visibility(name),
            docstring=docstring,
            source_text=self._get_source_text(decl_node),
            signature=f"type {name} {underlying or '?'}",
            metadata={'target_type': underlying, 'is_true_alias': False},
        )
        self._symbols.append(sym)
        self._file_type_names.add(name)

        # ALIAS_OF relationship to the underlying type
        if underlying and underlying not in GO_BUILTIN_TYPES:
            self._relationships.append(self._make_relationship(
                source=name, target=underlying,
                rel_type=RelationshipType.ALIAS_OF,
                node=spec_node,
            ))

    # ========================================================================
    # Phase 2: Relationship Extraction — Body Walking
    # ========================================================================

    def _walk_body(self, node: Node, source_symbol: str) -> None:
        """Recursively walk a function/method body to extract calls and creates."""
        if not node:
            return

        ntype = node.type

        if ntype == 'call_expression':
            self._handle_call_expression(node, source_symbol)

        elif ntype == 'composite_literal':
            self._handle_composite_literal(node, source_symbol)

        elif ntype == 'type_assertion_expression':
            self._handle_type_assertion(node, source_symbol)

        elif ntype == 'go_statement':
            # `go func_call()` — extract the inner call with metadata
            for child in node.children:
                if child.type == 'call_expression':
                    self._handle_call_expression(child, source_symbol,
                                                  annotations={'is_goroutine': True})
                else:
                    self._walk_body(child, source_symbol)
            return

        elif ntype == 'defer_statement':
            # `defer func_call()` — extract the inner call with metadata
            for child in node.children:
                if child.type == 'call_expression':
                    self._handle_call_expression(child, source_symbol,
                                                  annotations={'is_deferred': True})
                else:
                    self._walk_body(child, source_symbol)
            return

        # Recurse into children
        for child in node.children:
            self._walk_body(child, source_symbol)

    def _handle_call_expression(self, node: Node, source_symbol: str,
                                 annotations: Optional[Dict[str, Any]] = None) -> None:
        """Handle call_expression to extract CALLS relationships."""
        func_node = node.children[0] if node.children else None
        if not func_node:
            return

        target = None
        call_annotations = annotations.copy() if annotations else {}

        if func_node.type == 'identifier':
            # Direct call: process(), NewServer()
            target = self._get_node_text(func_node)
        elif func_node.type == 'selector_expression':
            # Qualified call: fmt.Println() or obj.Method()
            left = func_node.children[0] if func_node.children else None
            right = self._find_child_by_type(func_node, 'field_identifier')
            if left and right:
                left_name = self._get_node_text(left)
                right_name = self._get_node_text(right)

                if left_name in self._current_imports:
                    # Package call: fmt.Println
                    target = f"{left_name}.{right_name}"
                    call_annotations['is_package_call'] = True
                else:
                    # Method call: obj.Method() → receiver_type.Method
                    target = f"{left_name}.{right_name}"
                    call_annotations['is_method_call'] = True
        elif func_node.type == 'parenthesized_expression':
            # e.g., (func() { ... })()
            return

        if target and target not in GO_BUILTIN_FUNCTIONS:
            # Strip package name for builtin-like calls
            base_name = target.split('.')[-1] if '.' in target else target
            if base_name not in GO_BUILTIN_FUNCTIONS:
                self._relationships.append(self._make_relationship(
                    source=source_symbol,
                    target=target,
                    rel_type=RelationshipType.CALLS,
                    node=node,
                    annotations=call_annotations,
                ))

    def _handle_composite_literal(self, node: Node, source_symbol: str) -> None:
        """Handle composite literal to extract CREATES relationships."""
        # composite_literal: type_identifier { ... } or qualified_type { ... }
        type_node = node.children[0] if node.children else None
        if not type_node:
            return

        target = None
        if type_node.type == 'type_identifier':
            target = self._get_node_text(type_node)
        elif type_node.type == 'qualified_type':
            target = self._get_node_text(type_node)
        elif type_node.type in ('slice_type', 'map_type', 'array_type'):
            # e.g., []int{1,2,3} — not an architectural relationship
            return

        if target and target not in GO_BUILTIN_TYPES:
            self._relationships.append(self._make_relationship(
                source=source_symbol,
                target=target,
                rel_type=RelationshipType.CREATES,
                node=node,
            ))

    def _handle_type_assertion(self, node: Node, source_symbol: str) -> None:
        """Handle type assertion to extract REFERENCES relationships."""
        # type_assertion_expression: expr . ( type )
        # Look for the asserted type
        for child in node.children:
            if child.type == 'type_identifier':
                target = self._get_node_text(child)
                if target and target not in GO_BUILTIN_TYPES:
                    self._relationships.append(self._make_relationship(
                        source=source_symbol,
                        target=target,
                        rel_type=RelationshipType.REFERENCES,
                        node=node,
                    ))
                break

    # ========================================================================
    # Same-file method ↔ struct linkage
    # ========================================================================

    def _link_same_file_methods(self) -> None:
        """Link methods to their receiver structs when both are in the same file."""
        for receiver_type, method_names in self._file_method_receivers.items():
            if receiver_type in self._file_type_names:
                for method_name in method_names:
                    # Emit DEFINES: struct → method (same file)
                    self._relationships.append(self._make_relationship(
                        source=receiver_type,
                        target=f"{receiver_type}.{method_name}",
                        rel_type=RelationshipType.DEFINES,
                        annotations={'member_type': 'method', 'cross_file': False},
                    ))

    # ========================================================================
    # Phase 3: Cross-File Analysis
    # ========================================================================

    def parse_multiple_files(self, file_paths: List[str], max_workers: int = 4) -> Dict[str, ParseResult]:
        """
        Parse multiple Go files and resolve cross-file relationships.

        3-pass algorithm:
        1. Parallel single-file parsing
        2. Build global registries
        3. Cross-file relationship enhancement (receiver linkage + implicit interfaces)
        """
        results: Dict[str, ParseResult] = {}
        total = len(file_paths)

        logger.info(f"Parsing {total} Go files for cross-file analysis with {max_workers} workers")
        if this and getattr(this, 'module', None):
            this.module.invocation_thinking(
                f"I am on phase parsing\nStarting Go multi-file parsing: {total} files\n"
                f"Reasoning: Collect struct/interface/method symbols & imports for cross-file resolution.\n"
                f"Next: Parallel parse then build global registries."
            )

        # === Pass 1: Parallel single-file parsing ===
        results = self._parse_files_parallel(file_paths, max_workers)

        if this and getattr(this, 'module', None):
            succeeded = sum(1 for r in results.values() if r.symbols)
            pct = int((succeeded / total) * 100) if total else 100
            this.module.invocation_thinking(
                f"I am on phase parsing\nGo raw parsing complete: {succeeded}/{total} files ({pct}%)\n"
                f"Next: Build global type + method registries."
            )

        # === Pass 2: Build global registries ===
        self._global_type_registry.clear()
        self._global_function_registry.clear()
        self._global_method_registry.clear()
        self._global_interface_methods.clear()

        for file_path, result in results.items():
            if not result.symbols:
                continue
            for sym in result.symbols:
                if sym.symbol_type in (SymbolType.STRUCT, SymbolType.INTERFACE,
                                       SymbolType.ENUM, SymbolType.TYPE_ALIAS):
                    self._global_type_registry[sym.name] = file_path
                elif sym.symbol_type == SymbolType.FUNCTION:
                    self._global_function_registry[sym.name] = file_path
                elif sym.symbol_type == SymbolType.METHOD and sym.parent_symbol:
                    self._global_method_registry.setdefault(
                        sym.parent_symbol, {}
                    )[sym.name] = file_path

                # Collect interface method sets
                if sym.symbol_type == SymbolType.INTERFACE:
                    iface_methods = set()
                    for other in result.symbols:
                        if (other.symbol_type == SymbolType.METHOD and
                                other.parent_symbol == sym.name and
                                other.metadata.get('is_abstract')):
                            iface_methods.add(other.name)
                    if iface_methods:
                        self._global_interface_methods[sym.name] = iface_methods

        if this and getattr(this, 'module', None):
            this.module.invocation_thinking(
                f"I am on phase parsing\nGlobal registries: {len(self._global_type_registry)} types, "
                f"{len(self._global_function_registry)} functions, "
                f"{len(self._global_method_registry)} receiver types\n"
                f"Next: Cross-file relationship enhancement."
            )

        # === Pass 3: Cross-file relationship enhancement ===
        self._link_methods_to_structs_cross_file(results)
        self._detect_implicit_interfaces(results)
        self._resolve_cross_file_calls(results)

        if this and getattr(this, 'module', None):
            this.module.invocation_thinking(
                f"I am on phase parsing\nGo parsing complete: {len(results)}/{total} files\n"
                f"Cross-file analysis done."
            )

        return results

    def _parse_files_parallel(self, file_paths: List[str], max_workers: int) -> Dict[str, ParseResult]:
        """Parse multiple Go files in parallel using ThreadPoolExecutor with batching."""
        results: Dict[str, ParseResult] = {}
        total = len(file_paths)
        batch_size = 500

        num_batches = (total + batch_size - 1) // batch_size
        if num_batches > 1:
            logger.info(f"Processing {total} Go files in {num_batches} batches of up to {batch_size}")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            batch_files = file_paths[start_idx:end_idx]

            if num_batches > 1:
                logger.info(f"Go batch {batch_idx + 1}/{num_batches}: files {start_idx + 1}-{end_idx}")

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_file = {}
                    for fp in batch_files:
                        try:
                            future = executor.submit(_parse_single_go_file, fp)
                            future_to_file[future] = fp
                        except Exception as e:
                            logger.error(f"Failed to submit Go file {fp}: {e}")
                            results[fp] = ParseResult(
                                file_path=fp, language="go",
                                symbols=[], relationships=[],
                                errors=[f"Submit failed: {str(e)}"]
                            )

                    for future in concurrent.futures.as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            parsed_path, result = future.result(timeout=60)
                            results[parsed_path] = result

                            completed = len(results)
                            if completed % max(1, int(total / 10)) == 0:
                                logger.info(f"Parsed {completed}/{total} Go files...")
                        except concurrent.futures.TimeoutError:
                            logger.error(f"Timeout parsing Go file (>60s): {file_path}")
                            results[file_path] = ParseResult(
                                file_path=file_path, language="go",
                                symbols=[], relationships=[],
                                errors=["Timeout during parsing"]
                            )
                        except Exception as e:
                            logger.error(f"Failed to parse Go file {file_path}: {e}")
                            results[file_path] = ParseResult(
                                file_path=file_path, language="go",
                                symbols=[], relationships=[],
                                errors=[str(e)]
                            )
            except Exception as e:
                logger.error(f"Go batch {batch_idx + 1} executor failed: {e}")
                for fp in batch_files:
                    if fp not in results:
                        results[fp] = ParseResult(
                            file_path=fp, language="go",
                            symbols=[], relationships=[],
                            errors=[f"Batch failed: {str(e)}"]
                        )

        return results

    def _link_methods_to_structs_cross_file(self, results: Dict[str, ParseResult]) -> None:
        """Create DEFINES edges for methods whose receiver struct is in a different file."""
        for file_path, result in results.items():
            for sym in result.symbols:
                if sym.symbol_type != SymbolType.METHOD or not sym.parent_symbol:
                    continue

                receiver_type = sym.parent_symbol
                struct_file = self._global_type_registry.get(receiver_type)

                if struct_file and struct_file != file_path:
                    # Cross-file: method defined in different file than its receiver type
                    result.relationships.append(Relationship(
                        source_symbol=receiver_type,
                        target_symbol=f"{receiver_type}.{sym.name}",
                        relationship_type=RelationshipType.DEFINES,
                        source_file=struct_file,
                        target_file=file_path,
                        source_range=sym.range,
                        confidence=1.0,
                        annotations={
                            'member_type': 'method',
                            'cross_file': True,
                            'is_pointer_receiver': sym.metadata.get('is_pointer_receiver', False),
                        },
                    ))

    def _detect_implicit_interfaces(self, results: Dict[str, ParseResult]) -> None:
        """
        Detect implicit interface implementations via method-set subset matching.
        
        For each interface, check if any struct/type has all of its methods.
        Also checks against well-known stdlib interfaces.
        """
        # Build struct → method set from global method registry
        struct_methods: Dict[str, Set[str]] = {}
        for type_name, methods in self._global_method_registry.items():
            struct_methods[type_name] = set(methods.keys())

        # Check repo-local interfaces
        for iface_name, iface_methods in self._global_interface_methods.items():
            if not iface_methods:
                continue
            for struct_name, struct_meths in struct_methods.items():
                if struct_name == iface_name:
                    continue
                if iface_methods.issubset(struct_meths):
                    # Find which file has the struct to attach the relationship
                    struct_file = self._global_type_registry.get(struct_name)
                    iface_file = self._global_type_registry.get(iface_name)
                    if struct_file and struct_file in results:
                        results[struct_file].relationships.append(Relationship(
                            source_symbol=struct_name,
                            target_symbol=iface_name,
                            relationship_type=RelationshipType.IMPLEMENTATION,
                            source_file=struct_file,
                            target_file=iface_file,
                            source_range=Range(
                                start=Position(line=1, column=0),
                                end=Position(line=1, column=0),
                            ),
                            confidence=0.9,  # structural, not declared
                            annotations={'implicit': True},
                        ))

        # Check well-known stdlib interfaces
        for iface_full_name, iface_methods in WELL_KNOWN_INTERFACES.items():
            iface_short = iface_full_name.split('.')[-1] if '.' in iface_full_name else iface_full_name
            for struct_name, struct_meths in struct_methods.items():
                if iface_methods.issubset(struct_meths):
                    struct_file = self._global_type_registry.get(struct_name)
                    if struct_file and struct_file in results:
                        results[struct_file].relationships.append(Relationship(
                            source_symbol=struct_name,
                            target_symbol=iface_full_name,
                            relationship_type=RelationshipType.IMPLEMENTATION,
                            source_file=struct_file,
                            target_file=None,
                            source_range=Range(
                                start=Position(line=1, column=0),
                                end=Position(line=1, column=0),
                            ),
                            confidence=0.85,  # stdlib match
                            annotations={'implicit': True, 'stdlib': True},
                        ))

    def _resolve_cross_file_calls(self, results: Dict[str, ParseResult]) -> None:
        """Resolve cross-file CALLS relationships using global registries."""
        for file_path, result in results.items():
            for rel in result.relationships:
                if rel.relationship_type != RelationshipType.CALLS:
                    continue
                if rel.target_file and rel.target_file != file_path:
                    continue  # Already resolved

                target = rel.target_symbol
                if '.' in target:
                    # Could be pkg.Func or Type.Method
                    parts = target.split('.', 1)
                    left, right = parts[0], parts[1]

                    # Check if it's a known type's method
                    if left in self._global_method_registry:
                        method_file = self._global_method_registry[left].get(right)
                        if method_file:
                            rel.target_file = method_file
                            rel.target_symbol = f"{left}.{right}"
                    # Check if left is a package import
                    elif left in result.imports or left == self._current_package:
                        # Cross-package call — can't resolve file without module resolution
                        pass
                else:
                    # Simple function call
                    func_file = self._global_function_registry.get(target)
                    if func_file:
                        rel.target_file = func_file

    # ========================================================================
    # Helper Methods — Type Extraction
    # ========================================================================

    def _extract_receiver_info(self, method_node: Node) -> Tuple[Optional[str], bool]:
        """Extract receiver type name and pointer status from a method declaration."""
        # First parameter_list is the receiver
        param_lists = self._find_children_by_type(method_node, 'parameter_list')
        if not param_lists:
            return None, False

        receiver_list = param_lists[0]
        # Look for parameter_declaration inside
        param_decl = self._find_child_by_type(receiver_list, 'parameter_declaration')
        if not param_decl:
            return None, False

        # Check for pointer_type or type_identifier
        pointer_type = self._find_child_by_type(param_decl, 'pointer_type')
        if pointer_type:
            type_id = self._find_child_by_type(pointer_type, 'type_identifier')
            return (self._get_node_text(type_id), True) if type_id else (None, False)

        type_id = self._find_child_by_type(param_decl, 'type_identifier')
        if type_id:
            return self._get_node_text(type_id), False

        return None, False

    def _extract_parameters_from_list(self, param_list: Optional[Node]) -> List[Tuple[str, str]]:
        """Extract (name, type) pairs from a parameter_list node."""
        if not param_list:
            return []

        params: List[Tuple[str, str]] = []
        for param_decl in self._find_children_by_type(param_list, 'parameter_declaration'):
            # Collect all identifiers and the type
            names = []
            type_str = None
            for child in param_decl.children:
                if child.type == 'identifier':
                    names.append(self._get_node_text(child))
                elif child.type in ('type_identifier', 'pointer_type', 'qualified_type',
                                     'slice_type', 'map_type', 'channel_type',
                                     'array_type', 'function_type', 'interface_type',
                                     'struct_type', 'variadic_parameter_declaration'):
                    type_str = self._get_node_text(child)
                elif child.type == 'variadic_argument':
                    type_str = self._get_node_text(child)

            if not type_str:
                # Unnamed params (e.g., just type_identifier)
                for child in param_decl.children:
                    if child.type == 'type_identifier':
                        type_str = self._get_node_text(child)
                        break

            if names and type_str:
                for n in names:
                    params.append((n, type_str))
            elif type_str:
                params.append(('', type_str))
            elif names:
                # Type might be same as previous param (Go allows `a, b int`)
                # Just store with empty type for now
                for n in names:
                    params.append((n, ''))

        # Handle variadic parameter declarations
        for variadic in self._find_children_by_type(param_list, 'variadic_parameter_declaration'):
            id_node = self._find_child_by_type(variadic, 'identifier')
            var_name = self._get_node_text(id_node) if id_node else ''
            var_type = self._get_node_text(variadic)
            params.append((var_name, var_type))

        return params

    def _extract_return_types_from_param_lists(self, param_lists: List[Node], skip_first: int = 1) -> List[str]:
        """Extract return types from the parameter lists after the main param list."""
        # In Go AST:
        # function_declaration: func name (params) (returns)
        # The returns can be a parameter_list or a single type_identifier
        if len(param_lists) <= skip_first:
            return []

        return_list = param_lists[skip_first]
        # It's a parameter_list containing parameter_declarations
        types = []
        for param_decl in self._find_children_by_type(return_list, 'parameter_declaration'):
            for child in param_decl.children:
                if child.type in ('type_identifier', 'pointer_type', 'qualified_type',
                                   'slice_type', 'map_type', 'channel_type',
                                   'array_type', 'function_type', 'interface_type',
                                   'struct_type'):
                    types.append(self._get_node_text(child))
        return types

    def _extract_return_types_from_method_elem(self, method_elem: Node) -> List[str]:
        """Extract return types from an interface method_elem."""
        types = []
        # After the parameter_list(s), remaining type nodes are return types
        past_params = False
        for child in method_elem.children:
            if child.type == 'parameter_list':
                past_params = True
                continue
            if past_params and child.type in ('type_identifier', 'pointer_type',
                                               'qualified_type', 'slice_type',
                                               'map_type', 'channel_type',
                                               'array_type', 'function_type'):
                types.append(self._get_node_text(child))
        return types

    def _extract_type_parameters(self, node: Node) -> List[Tuple[str, str]]:
        """Extract type parameters from a generic declaration."""
        type_param_list = self._find_child_by_type(node, 'type_parameter_list')
        if not type_param_list:
            return []

        params: List[Tuple[str, str]] = []
        for tpd in self._find_children_by_type(type_param_list, 'type_parameter_declaration'):
            name_node = self._find_child_by_type(tpd, 'identifier')
            constraint_node = self._find_child_by_type(tpd, 'type_constraint')

            name = self._get_node_text(name_node) if name_node else ''
            constraint = ''
            if constraint_node:
                type_id = self._find_child_by_type(constraint_node, 'type_identifier')
                constraint = self._get_node_text(type_id) if type_id else self._get_node_text(constraint_node)

            params.append((name, constraint))

        return params

    def _extract_field_type(self, field_decl: Node) -> str:
        """Extract the type string from a field declaration."""
        for child in field_decl.children:
            if child.type in ('type_identifier', 'pointer_type', 'qualified_type',
                               'slice_type', 'map_type', 'channel_type',
                               'array_type', 'function_type', 'interface_type',
                               'struct_type'):
                return self._get_node_text(child)
        return ""

    def _is_pointer_type(self, field_decl: Node) -> bool:
        """Check if a field declaration has a pointer type."""
        return self._find_child_by_type(field_decl, 'pointer_type') is not None

    def _extract_embedded_name(self, type_node: Node) -> Optional[str]:
        """Extract the name from an embedded field's type node."""
        if type_node.type == 'type_identifier':
            return self._get_node_text(type_node)
        elif type_node.type == 'qualified_type':
            # e.g., io.Reader → use Reader as the embedded name
            type_id = self._find_child_by_type(type_node, 'type_identifier')
            return self._get_node_text(type_id) if type_id else None
        elif type_node.type == 'pointer_type':
            inner = self._find_child_by_type(type_node, 'type_identifier')
            return self._get_node_text(inner) if inner else None
        return None

    def _strip_pointer_slice(self, type_str: str) -> str:
        """Strip pointer (*), slice ([]), and map prefixes from a type string."""
        s = type_str.strip()
        while s.startswith('*') or s.startswith('['):
            if s.startswith('*'):
                s = s[1:]
            elif s.startswith('[]'):
                s = s[2:]
            elif s.startswith('['):
                # array type [N]T — skip to ]
                bracket_end = s.find(']')
                if bracket_end >= 0:
                    s = s[bracket_end + 1:]
                else:
                    break
        if s.startswith('map['):
            # map[K]V — extract V
            bracket_count = 0
            for i, ch in enumerate(s):
                if ch == '[':
                    bracket_count += 1
                elif ch == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        s = s[i + 1:]
                        break
        return s.strip()


# Register parser
parser_registry.register_parser(GoVisitorParser())
