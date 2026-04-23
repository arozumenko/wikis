"""
Rust Visitor-based Parser using Tree-sitter for DeepWiki Graph Enhancement.

Implements a visitor pattern for Rust AST traversal, extracting symbols and
relationships for cross-file and cross-crate analysis. Follows the GoVisitorParser pattern.

Key Rust-specific features:
- `impl` blocks are containers, not symbols — methods extracted with parent_symbol
- `Self` type resolution in impl blocks (struct expressions, return types)
- `dyn Trait` → REFERENCES(dispatch=dynamic)
- `impl Trait` → REFERENCES(dispatch=static)
- Generic bounds + where clauses → REFERENCES(dispatch=generic_bound/where_bound)
- Struct expressions (struct literals) → CREATES
- Cargo workspace resolution for cross-crate graph connectivity
- `extern crate ... as alias` → alias resolution
- `pub use crate::*` glob re-exports → re-export chain tracking

Relationship semantics:
- INHERITANCE is used ONLY for trait supertraits (trait Sub: Base)
- COMPOSITION is used for owned struct fields (value types)
- AGGREGATION is used for reference/Box/Rc/Arc struct fields
- IMPLEMENTATION is used for `impl Trait for Struct` (explicit, not structural)
- CREATES is used for struct expressions (struct literals) including Self { ... }
"""

import logging
import time
import concurrent.futures
from dataclasses import dataclass
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
# Rust-specific Constants
# ============================================================================

RUST_BUILTIN_TYPES = frozenset({
    # Primitives
    'bool', 'char',
    'i8', 'i16', 'i32', 'i64', 'i128', 'isize',
    'u8', 'u16', 'u32', 'u64', 'u128', 'usize',
    'f32', 'f64',
    'str', 'String',
    # Unit and never
    '()', '!',
    # Smart pointers (keep the inner type, skip the wrapper)
    'Box', 'Rc', 'Arc', 'Cell', 'RefCell', 'Mutex', 'RwLock',
    # Collections (keep inner types)
    'Vec', 'HashMap', 'HashSet', 'BTreeMap', 'BTreeSet',
    'Option', 'Result',
    # Fn traits
    'Fn', 'FnMut', 'FnOnce',
    # Other common std types
    'Pin', 'Future', 'Iterator',
    'Send', 'Sync', 'Sized', 'Copy', 'Clone',
    'Debug', 'Display', 'Default',
    'Drop', 'Deref', 'DerefMut',
    'From', 'Into', 'TryFrom', 'TryInto',
    'AsRef', 'AsMut',
    'PartialEq', 'Eq', 'PartialOrd', 'Ord',
    'Hash',
    'ToString',
    'Self',  # resolved contextually, not a real external type
})

RUST_BUILTIN_MACROS = frozenset({
    'println', 'print', 'eprintln', 'eprint',
    'format', 'write', 'writeln',
    'vec', 'todo', 'unimplemented', 'unreachable',
    'panic', 'assert', 'assert_eq', 'assert_ne',
    'debug_assert', 'debug_assert_eq', 'debug_assert_ne',
    'cfg', 'env', 'option_env', 'concat', 'stringify',
    'include', 'include_str', 'include_bytes',
    'file', 'line', 'column', 'module_path',
    'compile_error', 'matches',
})

# Types that indicate reference/pointer semantics for AGGREGATION vs COMPOSITION
RUST_REFERENCE_WRAPPERS = frozenset({
    'Box', 'Rc', 'Arc', 'Weak', 'Ref', 'RefMut',
    'MutexGuard', 'RwLockReadGuard', 'RwLockWriteGuard',
})


# ============================================================================
# Cargo workspace data types
# ============================================================================

@dataclass
class CrateInfo:
    """Metadata about a crate in a Cargo workspace."""
    name: str                # Original name from Cargo.toml (e.g., "my-core")
    normalized_name: str     # Use-path name (e.g., "my_core")
    path: str                # Absolute path to crate root directory
    src_root: str            # Absolute path to src/ directory
    is_proc_macro: bool      # Whether this is a proc-macro crate


def _parse_cargo_toml(path: Path) -> dict:
    """Parse Cargo.toml — try tomllib first, tomli fallback, then regex."""
    try:
        import tomllib  # Python 3.11+
        with open(path, 'rb') as f:
            return tomllib.load(f)
    except ImportError:
        pass
    try:
        import tomli  # pip install tomli
        with open(path, 'rb') as f:
            return tomli.load(f)
    except (ImportError, Exception):
        pass
    return _parse_cargo_toml_regex(path)


def _parse_cargo_toml_regex(path: Path) -> dict:
    """Regex fallback for minimal Cargo.toml parsing."""
    import re
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return {}

    result: Dict[str, Any] = {}

    # Detect [workspace]
    if re.search(r'^\[workspace\]', text, re.MULTILINE):
        ws: Dict[str, Any] = {}
        # Extract members = [...]
        m = re.search(r'members\s*=\s*\[(.*?)\]', text, re.DOTALL)
        if m:
            members_str = m.group(1)
            ws['members'] = re.findall(r'"([^"]+)"', members_str)
        result['workspace'] = ws

    # Detect [package]
    pkg_match = re.search(r'^\[package\]', text, re.MULTILINE)
    if pkg_match:
        pkg: Dict[str, Any] = {}
        # Extract name
        nm = re.search(r'name\s*=\s*"([^"]+)"', text[pkg_match.start():])
        if nm:
            pkg['name'] = nm.group(1)
        result['package'] = pkg

    # Detect [lib] proc-macro
    if re.search(r'proc[_-]macro\s*=\s*true', text):
        result.setdefault('lib', {})['proc-macro'] = True

    return result


def _is_proc_macro(cargo_data: dict) -> bool:
    """Check if Cargo.toml describes a proc-macro crate."""
    return cargo_data.get('lib', {}).get('proc-macro', False)


def _parse_single_rust_file(file_path: str) -> Tuple[str, ParseResult]:
    """Parse a single Rust file — used by parallel workers (module-level for pickling)."""
    try:
        parser = RustVisitorParser()
        result = parser.parse_file(file_path)
        return file_path, result
    except Exception as e:
        logger.warning(f"Failed to parse {file_path}: {e}")
        return file_path, ParseResult(
            file_path=file_path,
            language="rust",
            symbols=[],
            relationships=[]
        )


class RustVisitorParser(BaseParser):
    """Rust parser using visitor pattern with real Tree-sitter."""

    def __init__(self):
        super().__init__("rust")
        self.parser = None
        self.ts_language = None

        # Per-file state (reset on each parse_file call)
        self._current_file: str = ""
        self._current_content: str = ""
        self._symbols: List[Symbol] = []
        self._relationships: List[Relationship] = []

        # Rust-specific per-file state
        self._current_impl_target: Optional[str] = None    # Current impl block target type
        self._current_impl_trait: Optional[str] = None      # Current impl trait (for trait impls)
        self._file_type_names: Set[str] = set()             # Types defined in current file
        self._file_trait_names: Set[str] = set()            # Traits defined in current file
        self._file_impl_methods: Dict[str, Set[str]] = {}   # type_name → {method_names} in this file
        self._pending_derives: List[str] = []               # Derives from #[derive(...)] for next item
        self._pending_attributes: List[Dict[str, Any]] = [] # Other attributes for next item
        self._current_imports: Dict[str, str] = {}          # short_name → full_path
        self._extern_crate_aliases: Dict[str, str] = {}     # alias → normalized_crate_name

        # Global registries for cross-file analysis (populated by parse_multiple_files)
        self._global_type_registry: Dict[str, str] = {}          # type_name → file_path
        self._global_function_registry: Dict[str, str] = {}      # func_name → file_path
        self._global_method_registry: Dict[str, Dict[str, str]] = {}  # type_name → {method → file_path}
        self._global_trait_methods: Dict[str, Set[str]] = {}     # trait_name → {method_names}

        # Cargo workspace state (populated by parse_multiple_files pass 0)
        self._crate_map: Dict[str, CrateInfo] = {}          # normalized_crate_name → CrateInfo
        self._file_to_crate: Dict[str, str] = {}            # file_path → normalized_crate_name

        self._setup_tree_sitter()

    # ========================================================================
    # BaseParser abstract method implementations
    # ========================================================================

    def _define_capabilities(self) -> LanguageCapabilities:
        """Define Rust parser capabilities."""
        return LanguageCapabilities(
            language="rust",
            supported_symbols={
                SymbolType.FUNCTION, SymbolType.METHOD,
                SymbolType.STRUCT, SymbolType.TRAIT,
                SymbolType.ENUM, SymbolType.FIELD,
                SymbolType.CONSTANT, SymbolType.TYPE_ALIAS,
                SymbolType.MODULE, SymbolType.MACRO,
            },
            supported_relationships={
                RelationshipType.IMPORTS, RelationshipType.CALLS,
                RelationshipType.COMPOSITION, RelationshipType.AGGREGATION,
                RelationshipType.INHERITANCE,
                RelationshipType.IMPLEMENTATION,
                RelationshipType.CREATES, RelationshipType.DEFINES,
                RelationshipType.REFERENCES, RelationshipType.ALIAS_OF,
                RelationshipType.ANNOTATES,
            },
            supports_ast_parsing=True,
            supports_type_inference=True,
            supports_cross_file_analysis=True,
            has_classes=False,
            has_interfaces=True,  # traits are like interfaces
            has_namespaces=True,  # modules
            has_generics=True,
            has_decorators=False,
            has_annotations=True,  # #[derive] etc.
            max_file_size_mb=50.0,
            typical_parse_time_ms=20.0,
        )

    def _get_supported_extensions(self) -> Set[str]:
        """Get supported Rust file extensions."""
        return {'.rs'}

    def _setup_tree_sitter(self) -> bool:
        """Setup real tree-sitter Rust parser."""
        try:
            self.ts_language = get_language('rust')
            self.parser = get_parser('rust')
            return True
        except Exception as e:
            logger.error(f"Failed to initialize tree-sitter Rust parser: {e}")
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
        """Parse a Rust file using tree-sitter visitor pattern."""
        start_time = time.time()
        file_path = str(file_path)
        self._current_file = file_path

        # Reset per-file state
        self._symbols = []
        self._relationships = []
        self._current_impl_target = None
        self._current_impl_trait = None
        self._file_type_names = set()
        self._file_trait_names = set()
        self._file_impl_methods = {}
        self._pending_derives = []
        self._pending_attributes = []
        self._current_imports = {}
        self._extern_crate_aliases = {}

        try:
            if content is None:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()

            self._current_content = content

            if not self.parser or not self.ts_language:
                return ParseResult(
                    file_path=file_path, language="rust",
                    symbols=[], relationships=[],
                    errors=["Tree-sitter Rust parser not available"]
                )

            tree = self.parser.parse(bytes(content, "utf8"))
            self._visit_top_level(tree.root_node)

            parse_time = time.time() - start_time
            return ParseResult(
                file_path=file_path,
                language="rust",
                symbols=self._symbols,
                relationships=self._relationships,
                imports=list(self._current_imports.values()),
                parse_time=parse_time,
            )

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return ParseResult(
                file_path=file_path, language="rust",
                symbols=[], relationships=[],
                errors=[f"File not found: {file_path}"]
            )
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}", exc_info=True)
            return ParseResult(
                file_path=file_path, language="rust",
                symbols=[], relationships=[],
                errors=[str(e)]
            )

    # ========================================================================
    # Visitor Pattern — Top-level dispatch
    # ========================================================================

    def _visit_top_level(self, node: Node) -> None:
        """Visit top-level nodes in a Rust source file."""
        if not node:
            return

        for child in node.children:
            ntype = child.type
            if ntype == 'attribute_item':
                self._handle_attribute(child)
            elif ntype == 'struct_item':
                self._handle_struct_item(child)
            elif ntype == 'enum_item':
                self._handle_enum_item(child)
            elif ntype == 'trait_item':
                self._handle_trait_item(child)
            elif ntype == 'impl_item':
                self._handle_impl_item(child)
            elif ntype == 'function_item':
                self._handle_function_item(child, parent_symbol=None)
            elif ntype == 'const_item':
                self._handle_const_item(child)
            elif ntype == 'static_item':
                self._handle_static_item(child)
            elif ntype == 'type_item':
                self._handle_type_item(child)
            elif ntype == 'mod_item':
                self._handle_mod_item(child)
            elif ntype == 'use_declaration':
                self._handle_use_declaration(child)
            elif ntype == 'macro_definition':
                self._handle_macro_definition(child)
            elif ntype == 'union_item':
                self._handle_union_item(child)
            elif ntype == 'extern_crate_declaration':
                self._handle_extern_crate(child)
            else:
                # Consume pending attributes/derives if unused
                self._pending_derives = []
                self._pending_attributes = []

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

    def _find_child_by_type(self, node: Node, child_type: str) -> Optional[Node]:
        """Find the first child of the given type."""
        for child in node.children:
            if child.type == child_type:
                return child
        return None

    def _find_children_by_type(self, node: Node, child_type: str) -> List[Node]:
        """Find all children of the given type."""
        return [c for c in node.children if c.type == child_type]

    def _find_child_by_field(self, node: Node, field_name: str) -> Optional[Node]:
        """Find child node by field name."""
        return node.child_by_field_name(field_name)

    def _get_visibility(self, node: Node) -> str:
        """Extract visibility from a Rust item node."""
        vis_node = self._find_child_by_type(node, 'visibility_modifier')
        if not vis_node:
            return "private"
        vis_text = self._get_node_text(vis_node)
        if vis_text == 'pub':
            return "public"
        elif 'crate' in vis_text:
            return "crate"
        elif 'super' in vis_text:
            return "protected"
        elif 'in' in vis_text:
            return "restricted"
        return "public"

    def _get_preceding_comment(self, node: Node) -> Optional[str]:
        """Get the doc comment preceding a node (/// or /** */)."""
        comments = []
        prev = node.prev_named_sibling
        while prev and prev.type in ('line_comment', 'block_comment'):
            text = self._get_node_text(prev)
            if text.startswith('///'):
                comments.insert(0, text[3:].strip())
            elif text.startswith('//!'):
                comments.insert(0, text[3:].strip())
            elif text.startswith('/**') and text.endswith('*/'):
                comments.insert(0, text[3:-2].strip())
            else:
                break  # Not a doc comment, stop
            prev = prev.prev_named_sibling
        return '\n'.join(comments) if comments else None

    def _extract_function_modifiers(self, node: Node) -> Dict[str, bool]:
        """Extract async/unsafe/extern modifiers from a function."""
        modifiers: Dict[str, bool] = {}
        mod_node = self._find_child_by_type(node, 'function_modifiers')
        if mod_node:
            mod_text = self._get_node_text(mod_node)
            if 'async' in mod_text:
                modifiers['is_async'] = True
            if 'unsafe' in mod_text:
                modifiers['is_unsafe'] = True
            if 'extern' in mod_text:
                modifiers['is_extern'] = True
        return modifiers

    def _extract_type_parameters(self, node: Node) -> List[Tuple[str, str]]:
        """Extract type parameters from a generic item. Returns [(name, bound), ...]."""
        params = []
        tp_node = self._find_child_by_type(node, 'type_parameters')
        if not tp_node:
            return params
        for child in tp_node.children:
            if child.type == 'type_identifier':
                params.append((self._get_node_text(child), ''))
            elif child.type == 'constrained_type_parameter':
                name_node = self._find_child_by_type(child, 'type_identifier')
                bounds_node = self._find_child_by_type(child, 'trait_bounds')
                name = self._get_node_text(name_node) if name_node else ''
                bounds = self._get_node_text(bounds_node) if bounds_node else ''
                params.append((name, bounds))
            elif child.type == 'lifetime':
                params.append((self._get_node_text(child), 'lifetime'))
        return params

    def _extract_lifetimes(self, node: Node) -> List[str]:
        """Extract lifetime parameters from a type_parameters node."""
        lifetimes = []
        tp_node = self._find_child_by_type(node, 'type_parameters')
        if tp_node:
            for child in tp_node.children:
                if child.type == 'lifetime_parameter':
                    # lifetime_parameter → lifetime → ' + identifier
                    lt_node = self._find_child_by_type(child, 'lifetime')
                    if lt_node:
                        lt = self._get_node_text(lt_node)
                        if lt.startswith("'"):
                            lifetimes.append(lt[1:])  # strip leading '
                elif child.type == 'lifetime':
                    lt = self._get_node_text(child)
                    if lt.startswith("'"):
                        lifetimes.append(lt[1:])  # strip leading '
        return lifetimes

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

    def _consume_pending_metadata(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Consume and return pending derives and attributes, resetting them."""
        derives = self._pending_derives
        attrs = self._pending_attributes
        self._pending_derives = []
        self._pending_attributes = []
        return derives, attrs

    # ========================================================================
    # Phase 1: Symbol Extraction Handlers
    # ========================================================================

    def _handle_attribute(self, node: Node) -> None:
        """Handle #[...] attributes — extract derives and metadata for next item."""
        # attribute_item → [ attr ] where attr contains identifier + token_tree
        attr_node = self._find_child_by_type(node, 'attribute')
        if not attr_node:
            # Try getting the meta_item or path directly
            attr_node = node

        # Try to find the attribute name
        # In tree-sitter-rust: attribute_item > "[" > attribute > "]"
        # attribute can contain: identifier, or path (like derive), and token_tree
        attr_text = self._get_node_text(node)

        if 'derive' in attr_text:
            # Extract derive traits from #[derive(Trait1, Trait2)]
            # token_tree contains the parenthesized list
            token_tree = None
            for child in self._iter_descendants(node):
                if child.type == 'token_tree':
                    token_tree = child
                    break
            if token_tree:
                tt_text = self._get_node_text(token_tree)
                # Strip parens and split by comma
                inner = tt_text.strip('()')
                derives = [d.strip() for d in inner.split(',') if d.strip()]
                self._pending_derives.extend(derives)
        else:
            # Store other attributes as metadata
            self._pending_attributes.append({
                'text': attr_text,
                'range': self._make_range(node),
            })

    def _iter_descendants(self, node: Node):
        """Yield all descendant nodes recursively."""
        for child in node.children:
            yield child
            yield from self._iter_descendants(child)

    def _handle_struct_item(self, node: Node) -> None:
        """Handle struct definition — emit STRUCT symbol + FIELD children + DEFINES edges."""
        name_node = self._find_child_by_type(node, 'type_identifier')
        if not name_node:
            return
        name = self._get_node_text(name_node)

        derives, attrs = self._consume_pending_metadata()
        docstring = self._get_preceding_comment(node)
        source_text = self._get_source_text(node)
        visibility = self._get_visibility(node)
        type_params = self._extract_type_parameters(node)
        lifetimes = self._extract_lifetimes(node)

        # Build signature
        sig = f"struct {name}"
        if type_params:
            sig += f"<{', '.join(f'{n}: {c}' if c and c != 'lifetime' else n for n, c in type_params)}>"

        metadata: Dict[str, Any] = {}
        if type_params:
            metadata['type_parameters'] = type_params
        if lifetimes:
            metadata['lifetimes'] = lifetimes
        if derives:
            metadata['derives'] = derives
        if attrs:
            metadata['attributes'] = attrs

        sym = Symbol(
            name=name,
            symbol_type=SymbolType.STRUCT,
            scope=Scope.GLOBAL,
            range=self._make_range(node),
            file_path=self._current_file,
            full_name=name,
            visibility=visibility,
            docstring=docstring,
            source_text=source_text,
            signature=sig,
            metadata=metadata,
        )
        self._symbols.append(sym)
        self._file_type_names.add(name)

        # Emit ANNOTATES edges for derives
        for derive in derives:
            if derive not in RUST_BUILTIN_TYPES:
                self._relationships.append(self._make_relationship(
                    source=derive,
                    target=name,
                    rel_type=RelationshipType.ANNOTATES,
                    node=node,
                    annotations={'derives': derives},
                ))

        # Extract fields — named struct fields
        field_list = self._find_child_by_type(node, 'field_declaration_list')
        if field_list:
            self._extract_struct_fields(name, field_list, node)

        # Tuple struct fields
        ordered_list = self._find_child_by_type(node, 'ordered_field_declaration_list')
        if ordered_list:
            self._extract_tuple_struct_fields(name, ordered_list, node)

    def _extract_struct_fields(self, struct_name: str, field_list_node: Node, decl_node: Node) -> None:
        """Extract named fields from a struct's field_declaration_list."""
        for field_decl in self._find_children_by_type(field_list_node, 'field_declaration'):
            field_name_node = self._find_child_by_type(field_decl, 'field_identifier')
            if not field_name_node:
                continue
            field_name = self._get_node_text(field_name_node)
            field_type_str = self._extract_field_type_string(field_decl)
            visibility = self._get_visibility(field_decl)

            field_sym = Symbol(
                name=field_name,
                symbol_type=SymbolType.FIELD,
                scope=Scope.CLASS,
                range=self._make_range(field_decl),
                file_path=self._current_file,
                full_name=f"{struct_name}.{field_name}",
                parent_symbol=struct_name,
                visibility=visibility,
                source_text=self._get_source_text(field_decl),
                metadata={'field_type': field_type_str},
            )
            self._symbols.append(field_sym)

            # Emit DEFINES edge: struct → field
            self._relationships.append(self._make_relationship(
                source=struct_name,
                target=f"{struct_name}.{field_name}",
                rel_type=RelationshipType.DEFINES,
                node=field_decl,
                annotations={'member_type': 'field'},
            ))

            # Emit COMPOSITION/AGGREGATION for non-builtin field types
            self._emit_field_type_relationships(struct_name, field_type_str, field_decl)

    def _extract_tuple_struct_fields(self, struct_name: str, ordered_list: Node, decl_node: Node) -> None:
        """Extract tuple struct fields (positional)."""
        idx = 0
        for child in ordered_list.children:
            if child.type in ('type_identifier', 'generic_type', 'reference_type',
                              'scoped_type_identifier', 'primitive_type', 'tuple_type',
                              'array_type', 'dynamic_type', 'abstract_type'):
                field_name = str(idx)
                field_type_str = self._get_node_text(child)

                field_sym = Symbol(
                    name=field_name,
                    symbol_type=SymbolType.FIELD,
                    scope=Scope.CLASS,
                    range=self._make_range(child),
                    file_path=self._current_file,
                    full_name=f"{struct_name}.{field_name}",
                    parent_symbol=struct_name,
                    visibility=self._get_visibility(ordered_list),
                    metadata={'field_type': field_type_str, 'is_tuple_field': True},
                )
                self._symbols.append(field_sym)

                self._relationships.append(self._make_relationship(
                    source=struct_name,
                    target=f"{struct_name}.{field_name}",
                    rel_type=RelationshipType.DEFINES,
                    node=child,
                    annotations={'member_type': 'field', 'is_tuple_field': True},
                ))

                self._emit_field_type_relationships(struct_name, field_type_str, child)
                idx += 1

    def _extract_field_type_string(self, field_decl: Node) -> str:
        """Extract the type string from a field declaration."""
        # The type is typically the last named child that isn't the field name or visibility
        for child in reversed(field_decl.children):
            if child.type in ('type_identifier', 'generic_type', 'reference_type',
                              'scoped_type_identifier', 'primitive_type', 'tuple_type',
                              'array_type', 'dynamic_type', 'abstract_type',
                              'function_type', 'macro_invocation'):
                return self._get_node_text(child)
        return ""

    def _emit_field_type_relationships(self, struct_name: str, field_type_str: str, node: Node) -> None:
        """Emit COMPOSITION or AGGREGATION based on type analysis."""
        if not field_type_str:
            return

        # Determine the core type name (strip references, generics, etc.)
        core_type = self._extract_core_type_name(field_type_str)
        if not core_type or core_type in RUST_BUILTIN_TYPES:
            return

        # Determine if it's a reference/pointer type → AGGREGATION, else → COMPOSITION
        is_reference = (
            field_type_str.startswith('&') or
            field_type_str.startswith("*") or
            any(field_type_str.startswith(w) for w in RUST_REFERENCE_WRAPPERS)
        )

        rel_type = RelationshipType.AGGREGATION if is_reference else RelationshipType.COMPOSITION
        self._relationships.append(self._make_relationship(
            source=struct_name,
            target=core_type,
            rel_type=rel_type,
            node=node,
        ))

    def _extract_core_type_name(self, type_str: str) -> str:
        """Extract the core type name from a type expression.
        
        Box<Config>     → Config
        &dyn Animal     → Animal
        Rc<Vec<Node>>   → Node (but we extract the outermost non-wrapper)
        &str            → str
        Option<Config>  → Config
        Vec<Config>     → Config
        HashMap<K, V>   → '' (multiple type args, skip)
        """
        type_str = type_str.strip()

        # Strip references
        while type_str.startswith('&') or type_str.startswith("*"):
            type_str = type_str.lstrip('&*').strip()
            if type_str.startswith('mut '):
                type_str = type_str[4:].strip()

        # Strip 'dyn '
        if type_str.startswith('dyn '):
            type_str = type_str[4:].strip()

        # Strip 'impl '
        if type_str.startswith('impl '):
            type_str = type_str[5:].strip()

        # Handle wrapper types: Box<T>, Rc<T>, Arc<T>, Option<T>, Vec<T>
        if '<' in type_str:
            outer = type_str[:type_str.index('<')].strip()
            inner = type_str[type_str.index('<') + 1:type_str.rindex('>')].strip() if '>' in type_str else ''
            if outer in RUST_BUILTIN_TYPES and inner:
                # Unwrap one level: Box<Config> → Config, Vec<Node> → Node
                # But skip multi-arg generics like HashMap<K, V>
                if ',' not in inner:
                    return self._extract_core_type_name(inner)
            return outer

        # Handle scoped types: path::to::Type → Type
        if '::' in type_str:
            type_str = type_str.rsplit('::', 1)[-1]

        return type_str

    def _handle_enum_item(self, node: Node) -> None:
        """Handle enum definition — emit ENUM symbol + variant FIELD children."""
        name_node = self._find_child_by_type(node, 'type_identifier')
        if not name_node:
            return
        name = self._get_node_text(name_node)

        derives, attrs = self._consume_pending_metadata()
        docstring = self._get_preceding_comment(node)
        source_text = self._get_source_text(node)
        visibility = self._get_visibility(node)
        type_params = self._extract_type_parameters(node)

        sig = f"enum {name}"
        if type_params:
            sig += f"<{', '.join(f'{n}: {c}' if c and c != 'lifetime' else n for n, c in type_params)}>"

        metadata: Dict[str, Any] = {}
        if type_params:
            metadata['type_parameters'] = type_params
        if derives:
            metadata['derives'] = derives
        if attrs:
            metadata['attributes'] = attrs

        sym = Symbol(
            name=name,
            symbol_type=SymbolType.ENUM,
            scope=Scope.GLOBAL,
            range=self._make_range(node),
            file_path=self._current_file,
            full_name=name,
            visibility=visibility,
            docstring=docstring,
            source_text=source_text,
            signature=sig,
            metadata=metadata,
        )
        self._symbols.append(sym)
        self._file_type_names.add(name)

        # Emit ANNOTATES edges for derives
        for derive in derives:
            if derive not in RUST_BUILTIN_TYPES:
                self._relationships.append(self._make_relationship(
                    source=derive, target=name,
                    rel_type=RelationshipType.ANNOTATES,
                    node=node, annotations={'derives': derives},
                ))

        # Extract variants
        variant_list = self._find_child_by_type(node, 'enum_variant_list')
        if variant_list:
            for variant in self._find_children_by_type(variant_list, 'enum_variant'):
                self._handle_enum_variant(name, variant)

    def _handle_enum_variant(self, enum_name: str, node: Node) -> None:
        """Handle a single enum variant — emit as FIELD with DEFINES edge."""
        name_node = self._find_child_by_type(node, 'identifier')
        if not name_node:
            return
        variant_name = self._get_node_text(name_node)

        # Determine variant form
        field_list = self._find_child_by_type(node, 'field_declaration_list')
        ordered_list = self._find_child_by_type(node, 'ordered_field_declaration_list')

        variant_kind = 'unit'
        if field_list:
            variant_kind = 'struct'
        elif ordered_list:
            variant_kind = 'tuple'

        field_sym = Symbol(
            name=variant_name,
            symbol_type=SymbolType.FIELD,
            scope=Scope.CLASS,
            range=self._make_range(node),
            file_path=self._current_file,
            full_name=f"{enum_name}.{variant_name}",
            parent_symbol=enum_name,
            source_text=self._get_source_text(node),
            metadata={'variant_kind': variant_kind},
        )
        self._symbols.append(field_sym)

        # DEFINES edge: enum → variant
        self._relationships.append(self._make_relationship(
            source=enum_name,
            target=f"{enum_name}.{variant_name}",
            rel_type=RelationshipType.DEFINES,
            node=node,
            annotations={'member_type': 'variant', 'variant_kind': variant_kind},
        ))

        # Extract field types from struct-like and tuple variants for COMPOSITION/AGGREGATION
        if field_list:
            for field_decl in self._find_children_by_type(field_list, 'field_declaration'):
                ftype = self._extract_field_type_string(field_decl)
                self._emit_field_type_relationships(enum_name, ftype, field_decl)
        elif ordered_list:
            for child in ordered_list.children:
                if child.type in ('type_identifier', 'generic_type', 'reference_type',
                                  'scoped_type_identifier', 'primitive_type'):
                    ftype = self._get_node_text(child)
                    self._emit_field_type_relationships(enum_name, ftype, child)

    def _handle_trait_item(self, node: Node) -> None:
        """Handle trait definition — emit TRAIT symbol + method signatures + supertrait edges."""
        name_node = self._find_child_by_type(node, 'type_identifier')
        if not name_node:
            return
        name = self._get_node_text(name_node)

        derives, attrs = self._consume_pending_metadata()
        docstring = self._get_preceding_comment(node)
        source_text = self._get_source_text(node)
        visibility = self._get_visibility(node)
        type_params = self._extract_type_parameters(node)

        sig = f"trait {name}"
        if type_params:
            sig += f"<{', '.join(f'{n}: {c}' if c and c != 'lifetime' else n for n, c in type_params)}>"

        metadata: Dict[str, Any] = {}
        if type_params:
            metadata['type_parameters'] = type_params

        sym = Symbol(
            name=name,
            symbol_type=SymbolType.TRAIT,
            scope=Scope.GLOBAL,
            range=self._make_range(node),
            file_path=self._current_file,
            full_name=name,
            visibility=visibility,
            docstring=docstring,
            source_text=source_text,
            signature=sig,
            metadata=metadata,
        )
        self._symbols.append(sym)
        self._file_trait_names.add(name)

        # Extract supertraits → INHERITANCE edges
        # trait Sub: Base + OtherBase { ... }
        # tree-sitter: trait_bounds after the trait name
        bounds_node = self._find_child_by_type(node, 'trait_bounds')
        if bounds_node:
            for bound_child in bounds_node.children:
                if bound_child.type == 'type_identifier':
                    supertrait = self._get_node_text(bound_child)
                    if supertrait and supertrait not in RUST_BUILTIN_TYPES:
                        self._relationships.append(self._make_relationship(
                            source=name, target=supertrait,
                            rel_type=RelationshipType.INHERITANCE,
                            node=bound_child,
                            annotations={'supertrait': True},
                        ))
                elif bound_child.type == 'scoped_type_identifier':
                    supertrait = self._get_node_text(bound_child)
                    if supertrait and supertrait not in RUST_BUILTIN_TYPES:
                        self._relationships.append(self._make_relationship(
                            source=name, target=supertrait,
                            rel_type=RelationshipType.INHERITANCE,
                            node=bound_child,
                            annotations={'supertrait': True},
                        ))

        # Extract trait methods from declaration_list
        decl_list = self._find_child_by_type(node, 'declaration_list')
        if decl_list:
            for child in decl_list.children:
                if child.type == 'function_signature_item':
                    # Abstract method — signature only, no body
                    self._handle_trait_method_signature(name, child, is_abstract=True)
                elif child.type == 'function_item':
                    # Default method — has a body
                    self._handle_function_item(child, parent_symbol=name, is_trait_default=True)
                elif child.type == 'type_item':
                    # Associated type
                    self._handle_type_item(child, parent_symbol=name)
                elif child.type == 'const_item':
                    # Associated constant
                    self._handle_const_item(child, parent_symbol=name)

    def _handle_trait_method_signature(self, trait_name: str, node: Node, is_abstract: bool = True) -> None:
        """Handle a method signature inside a trait (abstract method)."""
        name_node = self._find_child_by_type(node, 'identifier')
        if not name_node:
            # Try function_identifier for some tree-sitter versions
            name_node = self._find_child_by_type(node, 'name')
        if not name_node:
            return
        method_name = self._get_node_text(name_node)
        modifiers = self._extract_function_modifiers(node)

        # Extract return type
        ret_type = self._extract_return_type(node)

        # Extract parameter types
        param_types = self._extract_parameter_types(node)

        sig = self._build_function_signature(method_name, node, ret_type)

        method_sym = Symbol(
            name=method_name,
            symbol_type=SymbolType.METHOD,
            scope=Scope.CLASS,
            range=self._make_range(node),
            file_path=self._current_file,
            full_name=f"{trait_name}.{method_name}",
            parent_symbol=trait_name,
            visibility="public",  # trait methods are always public
            is_abstract=is_abstract,
            is_async=modifiers.get('is_async', False),
            source_text=self._get_source_text(node),
            signature=sig,
            return_type=ret_type,
            parameter_types=param_types,
            metadata={'is_abstract': is_abstract, **modifiers},
        )
        self._symbols.append(method_sym)

        # DEFINES edge: trait → method
        self._relationships.append(self._make_relationship(
            source=trait_name,
            target=f"{trait_name}.{method_name}",
            rel_type=RelationshipType.DEFINES,
            node=node,
            annotations={'member_type': 'method', 'is_abstract': is_abstract},
        ))

        # Extract type references from signature (dyn Trait, impl Trait, generic bounds)
        self._extract_signature_type_references(f"{trait_name}.{method_name}", node)

    def _handle_impl_item(self, node: Node) -> None:
        """Handle impl block — extract methods and emit DEFINES/IMPLEMENTATION edges.
        
        impl blocks are NOT emitted as symbols. They are containers for methods.
        There are two forms:
        1. Inherent impl:  impl Foo { ... }
        2. Trait impl:     impl Trait for Foo { ... }
        """
        # Determine impl target type and optional trait
        impl_target = None
        impl_trait = None

        # Look at type_identifier / generic_type children
        # For `impl Trait for Struct`, tree-sitter uses child_by_field_name
        type_node = node.child_by_field_name('type')
        trait_node = node.child_by_field_name('trait')

        if type_node:
            impl_target = self._get_node_text(type_node)
            # Strip generics: Foo<T> → Foo
            if '<' in impl_target:
                impl_target = impl_target[:impl_target.index('<')]
        if trait_node:
            impl_trait = self._get_node_text(trait_node)
            if '<' in impl_trait:
                impl_trait = impl_trait[:impl_trait.index('<')]

        if not impl_target:
            # Fallback: look for type_identifier children
            type_ids = self._find_children_by_type(node, 'type_identifier')
            # For `impl Foo`, there's one type_identifier
            # For `impl Trait for Foo`, there're two + 'for' keyword
            has_for = any(c.type == 'for' or self._get_node_text(c) == 'for'
                         for c in node.children)
            if has_for and len(type_ids) >= 2:
                impl_trait = self._get_node_text(type_ids[0])
                impl_target = self._get_node_text(type_ids[-1])
            elif type_ids:
                impl_target = self._get_node_text(type_ids[-1])

            # Try generic_type as well
            if not impl_target:
                generic_types = self._find_children_by_type(node, 'generic_type')
                if generic_types:
                    gt = self._get_node_text(generic_types[-1])
                    impl_target = gt.split('<')[0] if '<' in gt else gt

        if not impl_target:
            return

        # Set impl context for Self resolution
        saved_impl_target = self._current_impl_target
        saved_impl_trait = self._current_impl_trait
        self._current_impl_target = impl_target
        self._current_impl_trait = impl_trait

        # Emit IMPLEMENTATION edge for trait impls
        if impl_trait:
            self._relationships.append(self._make_relationship(
                source=impl_target,
                target=impl_trait,
                rel_type=RelationshipType.IMPLEMENTATION,
                node=node,
                annotations={
                    'trait_name': impl_trait,
                    'target': impl_target,
                    'impl_kind': 'trait',
                },
            ))

        # Extract methods from declaration_list
        decl_list = self._find_child_by_type(node, 'declaration_list')
        if decl_list:
            for child in decl_list.children:
                if child.type == 'function_item':
                    impl_kind = 'trait' if impl_trait else 'inherent'
                    self._handle_function_item(
                        child,
                        parent_symbol=impl_target,
                        impl_kind=impl_kind,
                        impl_trait=impl_trait,
                    )
                elif child.type == 'type_item':
                    self._handle_type_item(child, parent_symbol=impl_target)
                elif child.type == 'const_item':
                    self._handle_const_item(child, parent_symbol=impl_target)
                elif child.type == 'attribute_item':
                    self._handle_attribute(child)

        # Restore impl context
        self._current_impl_target = saved_impl_target
        self._current_impl_trait = saved_impl_trait

    def _handle_function_item(
        self,
        node: Node,
        parent_symbol: Optional[str] = None,
        impl_kind: Optional[str] = None,
        impl_trait: Optional[str] = None,
        is_trait_default: bool = False,
    ) -> None:
        """Handle function/method definition — emit FUNCTION or METHOD symbol."""
        name_node = self._find_child_by_type(node, 'identifier')
        if not name_node:
            return
        func_name = self._get_node_text(name_node)

        derives, attrs = self._consume_pending_metadata()
        docstring = self._get_preceding_comment(node)
        source_text = self._get_source_text(node)
        visibility = self._get_visibility(node)
        modifiers = self._extract_function_modifiers(node)
        type_params = self._extract_type_parameters(node)

        # Determine if this is a method (inside impl/trait) or a free function
        is_method = parent_symbol is not None
        sym_type = SymbolType.METHOD if is_method else SymbolType.FUNCTION
        scope = Scope.CLASS if is_method else Scope.GLOBAL

        # Check for self parameter
        has_self = self._has_self_parameter(node)
        is_static_method = is_method and not has_self

        # Extract return type
        ret_type = self._extract_return_type(node)
        param_types = self._extract_parameter_types(node)

        sig = self._build_function_signature(func_name, node, ret_type)

        full_name = f"{parent_symbol}.{func_name}" if parent_symbol else func_name

        metadata: Dict[str, Any] = {**modifiers}
        if type_params:
            metadata['type_parameters'] = type_params
        if impl_kind:
            metadata['impl_kind'] = impl_kind
        if impl_trait:
            metadata['impl_trait'] = impl_trait
        if is_trait_default:
            metadata['is_default'] = True
        if is_static_method:
            metadata['is_static_method'] = True

        sym = Symbol(
            name=func_name,
            symbol_type=sym_type,
            scope=scope,
            range=self._make_range(node),
            file_path=self._current_file,
            full_name=full_name,
            parent_symbol=parent_symbol,
            visibility=visibility if not is_method else "public",  # trait/impl methods default public
            is_static=is_static_method,
            is_async=modifiers.get('is_async', False),
            docstring=docstring,
            source_text=source_text,
            signature=sig,
            return_type=ret_type,
            parameter_types=param_types,
            metadata=metadata,
        )
        self._symbols.append(sym)

        # Track methods for same-file impl linkage
        if parent_symbol:
            self._file_impl_methods.setdefault(parent_symbol, set()).add(func_name)

            # Emit DEFINES edge: struct/trait → method
            self._relationships.append(self._make_relationship(
                source=parent_symbol,
                target=full_name,
                rel_type=RelationshipType.DEFINES,
                node=node,
                annotations={
                    'member_type': 'method',
                    'impl_kind': impl_kind or ('default' if is_trait_default else None),
                    'cross_file': False,
                },
            ))

        # Extract type references from signature
        self._extract_signature_type_references(full_name, node)

        # Walk function body for calls, creates, etc.
        body = self._find_child_by_type(node, 'block')
        if body:
            self._walk_body(body, full_name)

    def _handle_const_item(self, node: Node, parent_symbol: Optional[str] = None) -> None:
        """Handle const item — emit CONSTANT symbol."""
        name_node = self._find_child_by_type(node, 'identifier')
        if not name_node:
            return
        name = self._get_node_text(name_node)

        derives, attrs = self._consume_pending_metadata()
        visibility = self._get_visibility(node)
        docstring = self._get_preceding_comment(node)
        source_text = self._get_source_text(node)

        full_name = f"{parent_symbol}.{name}" if parent_symbol else name

        sym = Symbol(
            name=name,
            symbol_type=SymbolType.CONSTANT,
            scope=Scope.CLASS if parent_symbol else Scope.GLOBAL,
            range=self._make_range(node),
            file_path=self._current_file,
            full_name=full_name,
            parent_symbol=parent_symbol,
            visibility=visibility,
            docstring=docstring,
            source_text=source_text,
        )
        self._symbols.append(sym)

        if parent_symbol:
            self._relationships.append(self._make_relationship(
                source=parent_symbol,
                target=full_name,
                rel_type=RelationshipType.DEFINES,
                node=node,
                annotations={'member_type': 'constant'},
            ))

    def _handle_static_item(self, node: Node) -> None:
        """Handle static item — emit CONSTANT symbol with is_static metadata."""
        name_node = self._find_child_by_type(node, 'identifier')
        if not name_node:
            return
        name = self._get_node_text(name_node)

        self._consume_pending_metadata()
        visibility = self._get_visibility(node)
        docstring = self._get_preceding_comment(node)
        source_text = self._get_source_text(node)

        sym = Symbol(
            name=name,
            symbol_type=SymbolType.CONSTANT,
            scope=Scope.GLOBAL,
            range=self._make_range(node),
            file_path=self._current_file,
            full_name=name,
            visibility=visibility,
            is_static=True,
            docstring=docstring,
            source_text=source_text,
            metadata={'is_static': True},
        )
        self._symbols.append(sym)

    def _handle_type_item(self, node: Node, parent_symbol: Optional[str] = None) -> None:
        """Handle type alias — emit TYPE_ALIAS symbol + ALIAS_OF edge."""
        name_node = self._find_child_by_type(node, 'type_identifier')
        if not name_node:
            return
        name = self._get_node_text(name_node)

        self._consume_pending_metadata()
        visibility = self._get_visibility(node)
        docstring = self._get_preceding_comment(node)
        source_text = self._get_source_text(node)

        full_name = f"{parent_symbol}.{name}" if parent_symbol else name

        sym = Symbol(
            name=name,
            symbol_type=SymbolType.TYPE_ALIAS,
            scope=Scope.CLASS if parent_symbol else Scope.GLOBAL,
            range=self._make_range(node),
            file_path=self._current_file,
            full_name=full_name,
            parent_symbol=parent_symbol,
            visibility=visibility,
            docstring=docstring,
            source_text=source_text,
        )
        self._symbols.append(sym)

        if parent_symbol:
            self._relationships.append(self._make_relationship(
                source=parent_symbol,
                target=full_name,
                rel_type=RelationshipType.DEFINES,
                node=node,
                annotations={'member_type': 'type'},
            ))

        # Extract the aliased type for ALIAS_OF edge
        # type Alias = OriginalType;
        # The right-hand side type is typically the last type node
        for child in reversed(node.children):
            if child.type in ('type_identifier', 'generic_type', 'scoped_type_identifier',
                              'reference_type', 'tuple_type', 'array_type',
                              'function_type', 'dynamic_type', 'abstract_type'):
                if child != name_node:
                    target = self._get_node_text(child)
                    core = self._extract_core_type_name(target)
                    # Check both full path and last segment against builtins
                    core_last = core.rsplit('::', 1)[-1] if '::' in core else core
                    is_builtin = core in RUST_BUILTIN_TYPES or core_last in RUST_BUILTIN_TYPES
                    if core and not is_builtin and core != name:
                        self._relationships.append(self._make_relationship(
                            source=name, target=core,
                            rel_type=RelationshipType.ALIAS_OF,
                            node=child,
                        ))
                    if child.type == 'generic_type':
                        # Also scan type_arguments for non-builtin, non-type-param types
                        # Collect type parameter names from this type alias
                        tp_idents = set()
                        tp_node = self._find_child_by_type(node, 'type_parameters')
                        if tp_node:
                            for tp in tp_node.children:
                                if tp.type == 'type_parameter':
                                    ti = self._find_child_by_type(tp, 'type_identifier')
                                    if ti:
                                        tp_idents.add(self._get_node_text(ti))
                        type_args = self._find_child_by_type(child, 'type_arguments')
                        if type_args:
                            for arg in type_args.children:
                                if arg.type == 'type_identifier':
                                    arg_name = self._get_node_text(arg)
                                    if (arg_name not in RUST_BUILTIN_TYPES
                                            and arg_name != name
                                            and arg_name not in tp_idents):
                                        self._relationships.append(self._make_relationship(
                                            source=name, target=arg_name,
                                            rel_type=RelationshipType.ALIAS_OF,
                                            node=arg,
                                        ))
                    break

    def _handle_mod_item(self, node: Node) -> None:
        """Handle mod declaration — emit MODULE symbol."""
        name_node = self._find_child_by_type(node, 'identifier')
        if not name_node:
            return
        name = self._get_node_text(name_node)

        self._consume_pending_metadata()
        visibility = self._get_visibility(node)
        docstring = self._get_preceding_comment(node)

        # Check if it's inline (has declaration_list) or external (just `mod foo;`)
        decl_list = self._find_child_by_type(node, 'declaration_list')
        is_inline = decl_list is not None

        sym = Symbol(
            name=name,
            symbol_type=SymbolType.MODULE,
            scope=Scope.GLOBAL,
            range=self._make_range(node),
            file_path=self._current_file,
            full_name=name,
            visibility=visibility,
            docstring=docstring,
            source_text=self._get_source_text(node),
            metadata={'is_inline': is_inline},
        )
        self._symbols.append(sym)

        # If inline, visit children
        if decl_list:
            for child in decl_list.children:
                ntype = child.type
                if ntype == 'struct_item':
                    self._handle_struct_item(child)
                elif ntype == 'enum_item':
                    self._handle_enum_item(child)
                elif ntype == 'trait_item':
                    self._handle_trait_item(child)
                elif ntype == 'impl_item':
                    self._handle_impl_item(child)
                elif ntype == 'function_item':
                    self._handle_function_item(child, parent_symbol=None)
                elif ntype == 'const_item':
                    self._handle_const_item(child)
                elif ntype == 'static_item':
                    self._handle_static_item(child)
                elif ntype == 'type_item':
                    self._handle_type_item(child)
                elif ntype == 'use_declaration':
                    self._handle_use_declaration(child)
                elif ntype == 'macro_definition':
                    self._handle_macro_definition(child)
                elif ntype == 'attribute_item':
                    self._handle_attribute(child)

    def _handle_use_declaration(self, node: Node) -> None:
        """Handle use declaration — extract import paths."""
        # use std::io::Read;
        # use std::io::{Read, Write};
        # use std::io::*;
        # use crate::types::Config;
        # pub use other_crate::SomeType;
        visibility = self._get_visibility(node)
        is_pub = visibility == "public"

        self._extract_use_paths(node, prefix="", is_pub=is_pub)

    def _extract_use_paths(self, node: Node, prefix: str = "", is_pub: bool = False) -> None:
        """Recursively extract use paths from a use_declaration."""
        for child in node.children:
            if child.type == 'scoped_identifier':
                path = self._get_node_text(child)
                full_path = f"{prefix}{path}" if prefix else path
                short_name = path.rsplit('::', 1)[-1] if '::' in path else path
                self._current_imports[short_name] = full_path
                self._relationships.append(self._make_relationship(
                    source=Path(self._current_file).stem,
                    target=full_path,
                    rel_type=RelationshipType.IMPORTS,
                    node=child,
                    annotations={
                        'path': full_path,
                        're_export': is_pub,
                    },
                ))
            elif child.type == 'use_as_clause':
                # use foo::bar as baz;
                path_node = self._find_child_by_type(child, 'scoped_identifier') or \
                            self._find_child_by_type(child, 'identifier')
                alias_node = self._find_child_by_type(child, 'identifier')
                # The last identifier is the alias
                identifiers = self._find_children_by_type(child, 'identifier')
                if path_node and len(identifiers) >= 1:
                    path = self._get_node_text(path_node)
                    alias = self._get_node_text(identifiers[-1])
                    full_path = f"{prefix}{path}" if prefix else path
                    self._current_imports[alias] = full_path
                    self._relationships.append(self._make_relationship(
                        source=Path(self._current_file).stem,
                        target=full_path,
                        rel_type=RelationshipType.IMPORTS,
                        node=child,
                        annotations={'path': full_path, 'alias': alias, 're_export': is_pub},
                    ))
            elif child.type == 'use_list':
                # use std::io::{Read, Write};
                # Find the scoped path before the use_list
                scope_path = ""
                for sibling in node.children:
                    if sibling == child:
                        break
                    if sibling.type == 'scoped_identifier':
                        scope_path = self._get_node_text(sibling) + "::"
                    elif sibling.type == 'identifier':
                        scope_path = self._get_node_text(sibling) + "::"
                    elif sibling.type == 'scoped_use_list':
                        pass  # handled below

                for list_child in child.children:
                    if list_child.type == 'identifier':
                        item_name = self._get_node_text(list_child)
                        full_path = f"{prefix}{scope_path}{item_name}"
                        self._current_imports[item_name] = full_path
                        self._relationships.append(self._make_relationship(
                            source=Path(self._current_file).stem,
                            target=full_path,
                            rel_type=RelationshipType.IMPORTS,
                            node=list_child,
                            annotations={'path': full_path, 're_export': is_pub},
                        ))
                    elif list_child.type == 'self':
                        # use std::io::{self, Read} — imports the module itself
                        module_name = scope_path.rstrip('::').rsplit('::', 1)[-1] if scope_path else ''
                        if module_name:
                            full_path = f"{prefix}{scope_path.rstrip('::')}"
                            self._current_imports[module_name] = full_path
                    elif list_child.type == 'use_as_clause':
                        self._extract_use_paths(list_child, prefix=f"{prefix}{scope_path}", is_pub=is_pub)
            elif child.type == 'scoped_use_list':
                # The tree structure: scoped_use_list has a path part and a use_list
                self._handle_scoped_use_list(child, prefix, is_pub)
            elif child.type == 'use_wildcard':
                # use std::io::*;
                # Find the preceding path
                path = prefix.rstrip('::') if prefix else ""
                for sibling in node.children:
                    if sibling == child:
                        break
                    if sibling.type in ('scoped_identifier', 'identifier'):
                        path = self._get_node_text(sibling)

                self._current_imports[f"{path}::*"] = f"{path}::*"
                self._relationships.append(self._make_relationship(
                    source=Path(self._current_file).stem,
                    target=f"{path}::*",
                    rel_type=RelationshipType.IMPORTS,
                    node=child,
                    annotations={
                        'path': f"{path}::*",
                        'glob': True,
                        're_export': is_pub,
                    },
                ))
            elif child.type == 'identifier' and child == node.children[-1]:
                # Simple: use Foo; (rare but valid)
                name = self._get_node_text(child)
                if name not in ('use', 'pub'):
                    full_path = f"{prefix}{name}" if prefix else name
                    self._current_imports[name] = full_path
                    self._relationships.append(self._make_relationship(
                        source=Path(self._current_file).stem,
                        target=full_path,
                        rel_type=RelationshipType.IMPORTS,
                        node=child,
                        annotations={'path': full_path, 're_export': is_pub},
                    ))

    def _handle_scoped_use_list(self, node: Node, prefix: str, is_pub: bool) -> None:
        """Handle scoped_use_list: foo::bar::{A, B}."""
        # scoped_use_list has path + :: + use_list
        path_parts = []
        use_list = None
        for child in node.children:
            if child.type == 'use_list':
                use_list = child
            elif child.type in ('identifier', 'scoped_identifier', 'self', 'crate', 'super'):
                path_parts.append(self._get_node_text(child))

        scope_path = "::".join(path_parts) + "::" if path_parts else ""
        full_prefix = f"{prefix}{scope_path}"

        if use_list:
            for list_child in use_list.children:
                if list_child.type == 'identifier':
                    item_name = self._get_node_text(list_child)
                    full_path = f"{full_prefix}{item_name}"
                    self._current_imports[item_name] = full_path
                    self._relationships.append(self._make_relationship(
                        source=Path(self._current_file).stem,
                        target=full_path,
                        rel_type=RelationshipType.IMPORTS,
                        node=list_child,
                        annotations={'path': full_path, 're_export': is_pub},
                    ))
                elif list_child.type == 'scoped_use_list':
                    self._handle_scoped_use_list(list_child, full_prefix, is_pub)
                elif list_child.type == 'self':
                    module_name = scope_path.rstrip('::').rsplit('::', 1)[-1] if scope_path else ''
                    if module_name:
                        full_path = f"{prefix}{scope_path.rstrip('::')}"
                        self._current_imports[module_name] = full_path
                elif list_child.type == 'use_wildcard':
                    glob_path = full_prefix.rstrip('::')
                    self._current_imports[f"{glob_path}::*"] = f"{glob_path}::*"
                    self._relationships.append(self._make_relationship(
                        source=Path(self._current_file).stem,
                        target=f"{glob_path}::*",
                        rel_type=RelationshipType.IMPORTS,
                        node=list_child,
                        annotations={'path': f"{glob_path}::*", 'glob': True, 're_export': is_pub},
                    ))

    def _handle_macro_definition(self, node: Node) -> None:
        """Handle macro_rules! definition — emit MACRO symbol."""
        name_node = self._find_child_by_type(node, 'identifier')
        if not name_node:
            return
        name = self._get_node_text(name_node)

        self._consume_pending_metadata()
        docstring = self._get_preceding_comment(node)

        sym = Symbol(
            name=name,
            symbol_type=SymbolType.MACRO,
            scope=Scope.GLOBAL,
            range=self._make_range(node),
            file_path=self._current_file,
            full_name=name,
            visibility="public",  # macro_rules! macros are typically pub
            docstring=docstring,
            source_text=self._get_source_text(node),
            metadata={'is_declarative': True},
        )
        self._symbols.append(sym)

    def _handle_union_item(self, node: Node) -> None:
        """Handle union definition — emit STRUCT symbol with is_union metadata."""
        name_node = self._find_child_by_type(node, 'type_identifier')
        if not name_node:
            return
        name = self._get_node_text(name_node)

        derives, attrs = self._consume_pending_metadata()
        docstring = self._get_preceding_comment(node)
        source_text = self._get_source_text(node)
        visibility = self._get_visibility(node)

        metadata: Dict[str, Any] = {'is_union': True}
        if derives:
            metadata['derives'] = derives

        sym = Symbol(
            name=name,
            symbol_type=SymbolType.STRUCT,
            scope=Scope.GLOBAL,
            range=self._make_range(node),
            file_path=self._current_file,
            full_name=name,
            visibility=visibility,
            docstring=docstring,
            source_text=source_text,
            signature=f"union {name}",
            metadata=metadata,
        )
        self._symbols.append(sym)
        self._file_type_names.add(name)

        # Extract fields
        field_list = self._find_child_by_type(node, 'field_declaration_list')
        if field_list:
            self._extract_struct_fields(name, field_list, node)

    def _handle_extern_crate(self, node: Node) -> None:
        """Handle extern crate declaration, potentially with alias."""
        # extern crate foo;
        # extern crate foo as bar;
        crate_name = None
        alias = None
        for child in node.children:
            if child.type == 'identifier':
                if crate_name is None:
                    crate_name = self._get_node_text(child)
                else:
                    alias = self._get_node_text(child)
            elif child.type == 'crate':
                continue

        if not crate_name:
            return

        effective_name = alias or crate_name
        self._extern_crate_aliases[effective_name] = crate_name

        self._relationships.append(self._make_relationship(
            source=Path(self._current_file).stem,
            target=crate_name,
            rel_type=RelationshipType.IMPORTS,
            node=node,
            annotations={
                'path': crate_name,
                'alias': alias,
                'cross_crate': True,
                'extern_crate': True,
            },
        ))

    # ========================================================================
    # Function signature helpers
    # ========================================================================

    def _has_self_parameter(self, node: Node) -> bool:
        """Check if function has a self parameter."""
        params = self._find_child_by_type(node, 'parameters')
        if not params:
            return False
        for child in params.children:
            if child.type in ('self_parameter', 'self'):
                return True
            # Also check for `self: ...` pattern
            if child.type == 'parameter' and self._get_node_text(child).strip().startswith('self'):
                return True
        return False

    def _extract_return_type(self, node: Node) -> Optional[str]:
        """Extract return type from a function node."""
        # Look for -> type pattern
        found_arrow = False
        for child in node.children:
            if self._get_node_text(child) == '->':
                found_arrow = True
            elif found_arrow and child.type not in ('block', 'where_clause', ';'):
                return self._get_node_text(child)
        return None

    def _extract_parameter_types(self, node: Node) -> List[str]:
        """Extract parameter types from a function's parameters node."""
        param_types = []
        params = self._find_child_by_type(node, 'parameters')
        if not params:
            return param_types

        for child in params.children:
            if child.type == 'parameter':
                # parameter has pattern + type
                for sub in reversed(child.children):
                    if sub.type in ('type_identifier', 'generic_type', 'reference_type',
                                    'scoped_type_identifier', 'primitive_type',
                                    'dynamic_type', 'abstract_type', 'tuple_type',
                                    'array_type', 'function_type'):
                        param_types.append(self._get_node_text(sub))
                        break
            elif child.type == 'self_parameter':
                param_types.append('self')

        return param_types

    def _build_function_signature(self, name: str, node: Node, ret_type: Optional[str]) -> str:
        """Build a human-readable function signature."""
        modifiers = self._extract_function_modifiers(node)
        parts = []
        if modifiers.get('is_async'):
            parts.append('async')
        if modifiers.get('is_unsafe'):
            parts.append('unsafe')
        parts.append('fn')
        parts.append(name)

        # Type parameters
        tp_node = self._find_child_by_type(node, 'type_parameters')
        if tp_node:
            parts[-1] += self._get_node_text(tp_node)

        # Parameters
        params = self._find_child_by_type(node, 'parameters')
        if params:
            parts.append(self._get_node_text(params))
        else:
            parts.append('()')

        if ret_type:
            parts.append(f"-> {ret_type}")

        return ' '.join(parts)

    # ========================================================================
    # Phase 2: Body Walking — Relationship Extraction
    # ========================================================================

    def _walk_body(self, node: Node, source_symbol: str) -> None:
        """Recursively walk a function/method body to extract calls and creates."""
        if not node:
            return

        ntype = node.type

        if ntype == 'call_expression':
            self._handle_call_expression(node, source_symbol)
        elif ntype == 'struct_expression':
            self._handle_struct_expression(node, source_symbol)
        elif ntype == 'macro_invocation':
            self._handle_macro_invocation(node, source_symbol)
        elif ntype == 'type_cast_expression':
            self._handle_type_cast(node, source_symbol)
        elif ntype == 'await_expression':
            # Walk inner expression; calls found will have context
            for child in node.children:
                self._walk_body(child, source_symbol)
            return
        elif ntype == 'closure_expression':
            # Walk closure body — calls attributed to enclosing function
            body = self._find_child_by_type(node, 'block')
            if body:
                self._walk_body(body, source_symbol)
            else:
                # Expression closure: |x| x + 1
                for child in node.children:
                    self._walk_body(child, source_symbol)
            return
        elif ntype == 'return_expression':
            for child in node.children:
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
            # Direct call: process(), new()
            target = self._get_node_text(func_node)
        elif func_node.type == 'field_expression':
            # Method call: obj.method() or Type::method()
            # field_expression → value . field_identifier
            field_id = self._find_child_by_type(func_node, 'field_identifier')
            if field_id:
                left = func_node.children[0] if func_node.children else None
                left_text = self._get_node_text(left) if left else ''
                right_text = self._get_node_text(field_id)
                target = f"{left_text}.{right_text}"
                call_annotations['is_method_call'] = True
        elif func_node.type == 'scoped_identifier':
            # Qualified call: Type::new(), module::func()
            target = self._get_node_text(func_node)
            # Resolve extern crate aliases
            parts = target.split('::', 1)
            if len(parts) == 2 and parts[0] in self._extern_crate_aliases:
                resolved_crate = self._extern_crate_aliases[parts[0]]
                call_annotations['resolved_crate'] = resolved_crate
        elif func_node.type == 'generic_function':
            # func::<T>(args)
            inner_name = self._find_child_by_type(func_node, 'identifier') or \
                         self._find_child_by_type(func_node, 'scoped_identifier') or \
                         self._find_child_by_type(func_node, 'field_expression')
            if inner_name:
                target = self._get_node_text(inner_name)
                call_annotations['is_turbofish'] = True

        if target:
            # Resolve Self to impl target
            if target.startswith('Self::') and self._current_impl_target:
                target = f"{self._current_impl_target}::{target[6:]}"
                call_annotations['via_self'] = True

            # Filter builtins
            base_name = target.rsplit('::', 1)[-1] if '::' in target else target
            if base_name.split('.', 1)[-1] if '.' in base_name else base_name not in RUST_BUILTIN_TYPES:
                self._relationships.append(self._make_relationship(
                    source=source_symbol,
                    target=target,
                    rel_type=RelationshipType.CALLS,
                    node=node,
                    annotations=call_annotations,
                ))

    def _handle_struct_expression(self, node: Node, source_symbol: str) -> None:
        """Handle struct expression (struct literal) to extract CREATES relationships."""
        # struct_expression: TypeName { field: value, ... }
        # First child is typically the type name
        type_node = node.children[0] if node.children else None
        if not type_node:
            return

        target = None
        annotations: Dict[str, Any] = {}

        if type_node.type == 'type_identifier':
            target = self._get_node_text(type_node)
        elif type_node.type == 'scoped_type_identifier':
            target = self._get_node_text(type_node)

        if target:
            # Resolve Self
            if target == 'Self' and self._current_impl_target:
                target = self._current_impl_target
                annotations['via_self'] = True

            if target not in RUST_BUILTIN_TYPES:
                self._relationships.append(self._make_relationship(
                    source=source_symbol,
                    target=target,
                    rel_type=RelationshipType.CREATES,
                    node=node,
                    annotations=annotations,
                ))

        # Walk field initializer values for nested struct expressions and calls
        field_init_list = self._find_child_by_type(node, 'field_initializer_list')
        if field_init_list:
            for child in field_init_list.children:
                if child.type == 'field_initializer':
                    # Walk the value expression (second meaningful child)
                    for sub in child.children:
                        if sub.type not in ('field_identifier', ':'):
                            self._walk_body(sub, source_symbol)
                elif child.type == 'shorthand_field_initializer':
                    pass  # Just a name, no nested expression
                elif child.type == 'base_field_initializer':
                    # ..other_struct
                    for sub in child.children:
                        self._walk_body(sub, source_symbol)

    def _handle_macro_invocation(self, node: Node, source_symbol: str) -> None:
        """Handle macro invocation to extract CALLS relationships."""
        # macro_invocation: identifier ! token_tree
        name_node = self._find_child_by_type(node, 'identifier')
        if not name_node:
            # Try scoped: path::macro!
            name_node = self._find_child_by_type(node, 'scoped_identifier')
        if not name_node:
            return

        macro_name = self._get_node_text(name_node)

        if macro_name and macro_name not in RUST_BUILTIN_MACROS:
            self._relationships.append(self._make_relationship(
                source=source_symbol,
                target=macro_name,
                rel_type=RelationshipType.CALLS,
                node=node,
                annotations={'is_macro': True},
            ))

    def _handle_type_cast(self, node: Node, source_symbol: str) -> None:
        """Handle type cast expression (expr as Type) to extract REFERENCES."""
        # type_cast_expression → expression as type
        for child in node.children:
            if child.type in ('type_identifier', 'generic_type', 'scoped_type_identifier'):
                target = self._get_node_text(child)
                core = self._extract_core_type_name(target)
                if core and core not in RUST_BUILTIN_TYPES:
                    self._relationships.append(self._make_relationship(
                        source=source_symbol,
                        target=core,
                        rel_type=RelationshipType.REFERENCES,
                        node=child,
                        annotations={'via_cast': True},
                    ))

    # ========================================================================
    # Signature Type Reference Extraction (dyn Trait, impl Trait, generic bounds)
    # ========================================================================

    def _extract_signature_type_references(self, source_symbol: str, func_node: Node) -> None:
        """Extract type references from function signatures (dyn/impl/bounds/where)."""
        # Walk parameters
        params = self._find_child_by_type(func_node, 'parameters')
        if params:
            self._extract_trait_references_recursive(params, source_symbol)

        # Walk return type
        found_arrow = False
        for child in func_node.children:
            if self._get_node_text(child) == '->':
                found_arrow = True
            elif found_arrow and child.type not in ('block', 'where_clause', ';'):
                self._extract_trait_references_recursive(child, source_symbol)
                break

        # Walk type_parameters for generic bounds
        tp_node = self._find_child_by_type(func_node, 'type_parameters')
        if tp_node:
            self._extract_generic_bound_references(tp_node, source_symbol)

        # Walk where_clause
        where_node = self._find_child_by_type(func_node, 'where_clause')
        if where_node:
            self._extract_where_clause_references(where_node, source_symbol)

    def _extract_trait_references_recursive(self, node: Node, source_symbol: str) -> None:
        """Walk a type node tree to find dyn/impl trait references."""
        if node.type == 'dynamic_type':
            # dyn Trait
            trait_node = self._find_child_by_type(node, 'type_identifier') or \
                         self._find_child_by_type(node, 'scoped_type_identifier')
            if trait_node:
                trait_text = self._get_node_text(trait_node)
                if trait_text not in RUST_BUILTIN_TYPES:
                    self._relationships.append(self._make_relationship(
                        source=source_symbol,
                        target=trait_text,
                        rel_type=RelationshipType.REFERENCES,
                        node=trait_node,
                        annotations={'dispatch': 'dynamic'},
                    ))
        elif node.type == 'abstract_type':
            # impl Trait
            trait_node = self._find_child_by_type(node, 'type_identifier') or \
                         self._find_child_by_type(node, 'scoped_type_identifier') or \
                         self._find_child_by_type(node, 'function_type')
            if trait_node:
                trait_text = self._get_node_text(trait_node)
                core = trait_text.split('(')[0] if '(' in trait_text else trait_text
                if core not in RUST_BUILTIN_TYPES:
                    self._relationships.append(self._make_relationship(
                        source=source_symbol,
                        target=core,
                        rel_type=RelationshipType.REFERENCES,
                        node=trait_node,
                        annotations={'dispatch': 'static'},
                    ))

        # Recurse into children
        for child in node.children:
            self._extract_trait_references_recursive(child, source_symbol)

    def _extract_generic_bound_references(self, tp_node: Node, source_symbol: str) -> None:
        """Extract trait references from generic type parameters with bounds."""
        for child in tp_node.children:
            if child.type in ('constrained_type_parameter', 'type_parameter'):
                type_param_node = self._find_child_by_type(child, 'type_identifier')
                type_param = self._get_node_text(type_param_node) if type_param_node else ''
                bounds = self._find_child_by_type(child, 'trait_bounds')
                if bounds:
                    self._extract_bounds_references(bounds, source_symbol, type_param, 'generic_bound')

    def _extract_where_clause_references(self, where_node: Node, source_symbol: str) -> None:
        """Extract trait references from where clauses."""
        for child in where_node.children:
            if child.type == 'where_predicate':
                type_param_node = self._find_child_by_type(child, 'type_identifier')
                type_param = self._get_node_text(type_param_node) if type_param_node else ''
                bounds = self._find_child_by_type(child, 'trait_bounds')
                if bounds:
                    self._extract_bounds_references(bounds, source_symbol, type_param, 'where_bound')

    def _extract_bounds_references(self, bounds_node: Node, source_symbol: str,
                                    type_param: str, dispatch: str) -> None:
        """Extract trait references from a trait_bounds node."""
        for child in bounds_node.children:
            if child.type == 'type_identifier':
                trait_name = self._get_node_text(child)
                if trait_name not in RUST_BUILTIN_TYPES:
                    self._relationships.append(self._make_relationship(
                        source=source_symbol,
                        target=trait_name,
                        rel_type=RelationshipType.REFERENCES,
                        node=child,
                        annotations={'dispatch': dispatch, 'type_param': type_param},
                    ))
            elif child.type == 'scoped_type_identifier':
                trait_name = self._get_node_text(child)
                if trait_name not in RUST_BUILTIN_TYPES:
                    self._relationships.append(self._make_relationship(
                        source=source_symbol,
                        target=trait_name,
                        rel_type=RelationshipType.REFERENCES,
                        node=child,
                        annotations={'dispatch': dispatch, 'type_param': type_param},
                    ))
            elif child.type == 'generic_type':
                # e.g., Iterator<Item = T> or Into<U>
                inner_type = self._find_child_by_type(child, 'type_identifier')
                if inner_type:
                    trait_name = self._get_node_text(inner_type)
                    if trait_name not in RUST_BUILTIN_TYPES:
                        self._relationships.append(self._make_relationship(
                            source=source_symbol,
                            target=trait_name,
                            rel_type=RelationshipType.REFERENCES,
                            node=child,
                            annotations={'dispatch': dispatch, 'type_param': type_param},
                        ))
            elif child.type == 'lifetime':
                pass  # Lifetimes in bounds are not relationship-bearing

    # ========================================================================
    # Phase 3: Cross-File Analysis
    # ========================================================================

    def parse_multiple_files(self, file_paths: List[str], max_workers: int = 4,
                              repo_path: Optional[str] = None) -> Dict[str, ParseResult]:
        """
        Parse multiple Rust files and resolve cross-file relationships.

        4-pass algorithm:
        0. Cargo workspace discovery (if repo_path provided)
        1. Parallel single-file parsing
        2. Build global registries
        3. Cross-file relationship enhancement (impl-to-struct, trait impls)
        """
        results: Dict[str, ParseResult] = {}
        total = len(file_paths)

        logger.info(f"Parsing {total} Rust files for cross-file analysis with {max_workers} workers")
        if this and getattr(this, 'module', None):
            this.module.invocation_thinking(
                f"I am on phase parsing\nStarting Rust multi-file parsing: {total} files\n"
                f"Reasoning: Collect struct/trait/method symbols & imports for cross-file resolution.\n"
                f"Next: Cargo workspace discovery, then parallel parse."
            )

        # === Pass 0: Cargo workspace discovery ===
        if repo_path:
            self._crate_map = self._build_crate_map(repo_path)
            if self._crate_map:
                logger.info(f"Discovered {len(self._crate_map)} crates in Cargo workspace")
                # Assign files to crates
                for fp in file_paths:
                    for crate_name, info in self._crate_map.items():
                        if fp.startswith(info.path):
                            self._file_to_crate[fp] = crate_name
                            break
            else:
                logger.info("No Cargo workspace found, using flat single-crate mode")

        # === Pass 1: Parallel single-file parsing ===
        results = self._parse_files_parallel(file_paths, max_workers)

        if this and getattr(this, 'module', None):
            succeeded = sum(1 for r in results.values() if r.symbols)
            pct = int((succeeded / total) * 100) if total else 100
            this.module.invocation_thinking(
                f"I am on phase parsing\nRust raw parsing complete: {succeeded}/{total} files ({pct}%)\n"
                f"Next: Build global type + method registries."
            )

        # === Pass 2: Build global registries ===
        self._global_type_registry.clear()
        self._global_function_registry.clear()
        self._global_method_registry.clear()
        self._global_trait_methods.clear()

        for file_path, result in results.items():
            if not result.symbols:
                continue
            for sym in result.symbols:
                if sym.symbol_type in (SymbolType.STRUCT, SymbolType.ENUM,
                                       SymbolType.TRAIT, SymbolType.TYPE_ALIAS):
                    self._global_type_registry[sym.name] = file_path
                elif sym.symbol_type == SymbolType.FUNCTION:
                    self._global_function_registry[sym.name] = file_path
                elif sym.symbol_type == SymbolType.METHOD and sym.parent_symbol:
                    self._global_method_registry.setdefault(
                        sym.parent_symbol, {}
                    )[sym.name] = file_path

                # Collect trait method sets
                if sym.symbol_type == SymbolType.TRAIT:
                    trait_methods = set()
                    for other in result.symbols:
                        if (other.symbol_type == SymbolType.METHOD and
                                other.parent_symbol == sym.name):
                            trait_methods.add(other.name)
                    if trait_methods:
                        self._global_trait_methods[sym.name] = trait_methods

        if this and getattr(this, 'module', None):
            this.module.invocation_thinking(
                f"I am on phase parsing\nGlobal registries: {len(self._global_type_registry)} types, "
                f"{len(self._global_function_registry)} functions, "
                f"{len(self._global_method_registry)} impl targets\n"
                f"Next: Cross-file relationship enhancement."
            )

        # === Pass 3: Cross-file relationship enhancement ===
        self._link_impl_methods_cross_file(results)
        self._resolve_cross_file_calls(results)

        if this and getattr(this, 'module', None):
            this.module.invocation_thinking(
                f"I am on phase parsing\nRust parsing complete: {len(results)}/{total} files\n"
                f"Cross-file analysis done."
            )

        return results

    def _parse_files_parallel(self, file_paths: List[str], max_workers: int) -> Dict[str, ParseResult]:
        """Parse multiple Rust files in parallel using ThreadPoolExecutor with batching."""
        results: Dict[str, ParseResult] = {}
        total = len(file_paths)
        batch_size = 500

        num_batches = (total + batch_size - 1) // batch_size
        if num_batches > 1:
            logger.info(f"Processing {total} Rust files in {num_batches} batches of up to {batch_size}")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            batch_files = file_paths[start_idx:end_idx]

            if num_batches > 1:
                logger.info(f"Rust batch {batch_idx + 1}/{num_batches}: files {start_idx + 1}-{end_idx}")

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_file = {}
                    for fp in batch_files:
                        try:
                            future = executor.submit(_parse_single_rust_file, fp)
                            future_to_file[future] = fp
                        except Exception as e:
                            logger.error(f"Failed to submit Rust file {fp}: {e}")
                            results[fp] = ParseResult(
                                file_path=fp, language="rust",
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
                                logger.info(f"Parsed {completed}/{total} Rust files...")
                        except concurrent.futures.TimeoutError:
                            logger.error(f"Timeout parsing Rust file (>60s): {file_path}")
                            results[file_path] = ParseResult(
                                file_path=file_path, language="rust",
                                symbols=[], relationships=[],
                                errors=["Timeout during parsing"]
                            )
                        except Exception as e:
                            logger.error(f"Failed to parse Rust file {file_path}: {e}")
                            results[file_path] = ParseResult(
                                file_path=file_path, language="rust",
                                symbols=[], relationships=[],
                                errors=[str(e)]
                            )
            except Exception as e:
                logger.error(f"Rust batch {batch_idx + 1} executor failed: {e}")
                for fp in batch_files:
                    if fp not in results:
                        results[fp] = ParseResult(
                            file_path=fp, language="rust",
                            symbols=[], relationships=[],
                            errors=[f"Batch failed: {str(e)}"]
                        )

        return results

    def _link_impl_methods_cross_file(self, results: Dict[str, ParseResult]) -> None:
        """Create DEFINES edges for impl methods whose target type is in a different file."""
        for file_path, result in results.items():
            for sym in result.symbols:
                if sym.symbol_type != SymbolType.METHOD or not sym.parent_symbol:
                    continue

                target_type = sym.parent_symbol
                type_file = self._global_type_registry.get(target_type)

                if type_file and type_file != file_path:
                    # Cross-file: method's parent type is defined in another file
                    result.relationships.append(Relationship(
                        source_symbol=target_type,
                        target_symbol=f"{target_type}.{sym.name}",
                        relationship_type=RelationshipType.DEFINES,
                        source_file=type_file,
                        target_file=file_path,
                        source_range=sym.range,
                        confidence=1.0,
                        annotations={
                            'member_type': 'method',
                            'cross_file': True,
                            'impl_kind': sym.metadata.get('impl_kind', 'inherent'),
                        },
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

                if '::' in target:
                    # Scoped call: Type::method() or module::func()
                    parts = target.rsplit('::', 1)
                    left, right = parts[0], parts[1]

                    # Check if left is a known type with methods
                    if left in self._global_method_registry:
                        method_file = self._global_method_registry[left].get(right)
                        if method_file:
                            rel.target_file = method_file
                    # Check if left is a known type (associated function)
                    elif left in self._global_type_registry:
                        rel.target_file = self._global_type_registry[left]
                elif '.' in target:
                    # Method call on instance: obj.method()
                    method_name = target.split('.', 1)[1]
                    # Try to find the method across all types
                    for type_name, methods in self._global_method_registry.items():
                        if method_name in methods:
                            rel.target_file = methods[method_name]
                            break
                else:
                    # Simple function call
                    func_file = self._global_function_registry.get(target)
                    if func_file:
                        rel.target_file = func_file

    # ========================================================================
    # Cargo Workspace Resolution
    # ========================================================================

    def _find_workspace_roots(self, repo_path: str) -> List[Path]:
        """Find all Cargo workspace roots in the repository."""
        roots: List[Path] = []
        repo = Path(repo_path)

        for cargo_toml in repo.rglob('Cargo.toml'):
            # Skip target directories and hidden directories
            parts = cargo_toml.relative_to(repo).parts
            if any(p.startswith('.') or p == 'target' for p in parts):
                continue

            data = _parse_cargo_toml(cargo_toml)
            if 'workspace' in data:
                roots.append(cargo_toml.parent)
            elif 'package' in data:
                # Single-crate project — check it's not inside an already-found workspace
                if not any(cargo_toml.parent.is_relative_to(r) for r in roots):
                    roots.append(cargo_toml.parent)

        return roots

    def _build_crate_map(self, repo_path: str) -> Dict[str, CrateInfo]:
        """Build crate_name → CrateInfo mapping from Cargo workspace."""
        crate_map: Dict[str, CrateInfo] = {}

        workspace_roots = self._find_workspace_roots(repo_path)
        if not workspace_roots:
            return crate_map

        for ws_root in workspace_roots:
            root_cargo = ws_root / 'Cargo.toml'
            if not root_cargo.exists():
                continue
            cargo_data = _parse_cargo_toml(root_cargo)

            # Check for workspace members
            members = cargo_data.get('workspace', {}).get('members', [])

            if not members and 'package' in cargo_data:
                # Single-crate project
                pkg_name = cargo_data.get('package', {}).get('name', '')
                if pkg_name:
                    normalized = pkg_name.replace('-', '_')
                    crate_map[normalized] = CrateInfo(
                        name=pkg_name,
                        normalized_name=normalized,
                        path=str(ws_root),
                        src_root=str(ws_root / 'src'),
                        is_proc_macro=_is_proc_macro(cargo_data),
                    )
                continue

            # Resolve member globs and parse each member's Cargo.toml
            for member_pattern in members:
                for member_dir in ws_root.glob(member_pattern):
                    if not member_dir.is_dir():
                        continue
                    member_cargo = member_dir / 'Cargo.toml'
                    if not member_cargo.exists():
                        continue
                    member_data = _parse_cargo_toml(member_cargo)
                    pkg_name = member_data.get('package', {}).get('name', '')
                    if not pkg_name:
                        continue
                    normalized = pkg_name.replace('-', '_')
                    crate_map[normalized] = CrateInfo(
                        name=pkg_name,
                        normalized_name=normalized,
                        path=str(member_dir),
                        src_root=str(member_dir / 'src'),
                        is_proc_macro=_is_proc_macro(member_data),
                    )

            # Also add the root crate if it has [package]
            root_pkg = cargo_data.get('package', {}).get('name', '')
            if root_pkg:
                normalized = root_pkg.replace('-', '_')
                if normalized not in crate_map:
                    crate_map[normalized] = CrateInfo(
                        name=root_pkg,
                        normalized_name=normalized,
                        path=str(ws_root),
                        src_root=str(ws_root / 'src'),
                        is_proc_macro=_is_proc_macro(cargo_data),
                    )

        return crate_map
