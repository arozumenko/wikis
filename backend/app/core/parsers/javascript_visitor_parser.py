"""JavaScript (and JSX) rich parser using Tree-sitter.

Phase 0-1 implementation:
 - Symbol extraction: module, classes, functions, methods, fields (this.prop), exported variables
 - Relationship extraction (unresolved targets initially): IMPORTS, EXPORTS, INHERITANCE, CREATES (new expressions), REFERENCES (JSX component usage), CALLS (basic identifier & member calls), COMPOSITION (constructor field initialized via new Type())
 - JSX component usage captured as REFERENCES relationships (tag -> component symbol)
 - Multi-file orchestration with two-pass enhancement: build global export index then resolve IMPORTS / JSX references to target files

Future phases (not yet implemented here): deeper call resolution, namespace import member resolution, CommonJS complex patterns, dedup/performance tuning.

KNOWN LIMITATIONS:
==================
1. PARAMETER AND RETURN TYPES NOT CAPTURED
   - Issue: JavaScript parser doesn't extract parameter_types or return_type metadata
   - Impact: Type-based navigation and transitive field access expansion don't work
   - Example that won't expand:
     
     const AppConfig = require('./AppConfig');
     const DatabaseConfig = require('./DatabaseConfig');
     
     /**
      * @param {AppConfig} config  ← JSDoc type annotation not captured
      * @returns {string}          ← Return type not captured
      */
     function connect(config) {
         const host = config.database.host;
         return host;
     }
   
   - Current behavior:
     * Function metadata has: parameter_types = []
     * Function metadata has: return_type = None
     * Expansion only includes: connect (function) → NetworkService (module)
     * Missing: AppConfig (parameter type) → DatabaseConfig (composition)
   
   - To fix:
     * Parse JSDoc comments (@param, @returns, @type tags)
     * Extract type information from JSDoc annotations
     * Store in symbol metadata: parameter_types=['AppConfig'], return_type='string'
     * Alternative: Support TypeScript type annotations if .ts files
   
   - Note: JavaScript DOES create composition edges at class level (✅)
     * AppConfig → DatabaseConfig composition edge exists
     * But can't use it without parameter type metadata
   
   - See: Java parser (java_visitor_parser.py) for JSDoc-like comment parsing
   - See: TypeScript parser for native type annotation support
   - Verified working in: C++ and Java (Oct 2025)

2. TODO: TypeScript support
   - JavaScript parser could be extended to handle TypeScript syntax
   - TypeScript has native type annotations (no JSDoc needed)
   - Would provide better type-based navigation
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from tree_sitter import Node
from tree_sitter_language_pack import get_language, get_parser

from .base_parser import (
    BaseParser, Symbol, Relationship, ParseResult, LanguageCapabilities,
    SymbolType, RelationshipType, Scope, Position, Range
)

logger = logging.getLogger(__name__)

# Optional thinking emitter
try:  # pragma: no cover
    from tools import this  # type: ignore
except Exception:  # pragma: no cover
    this = None  # type: ignore


# ----------------------------------------------------------------------------------
# Helper dataclasses for intermediate bookkeeping
# ----------------------------------------------------------------------------------

@dataclass
class ImportBinding:
    local: str
    source_path: str                 # raw module specifier (e.g. './utils', 'react')
    imported: Optional[str] = None   # original exported name (None for default)
    is_namespace: bool = False
    is_default: bool = False


@dataclass
class ExportEntry:
    name: str                        # exported name (or '<default>')
    internal_symbol: str             # underlying declared symbol name
    is_default: bool
    file_path: str
    symbol_type: Optional[SymbolType] = None


DEFAULT_EXPORT_SENTINEL = "<default>"

# ---------------------------------------------------------------------------
# Identifiers to ignore during body-reference scanning (JS keywords +
# well-known built-in globals).  We only emit REFERENCES edges for
# identifiers that match a *user-defined* symbol (import or same-file
# top-level declaration), so this set is a safety net for names that
# happen to collide with built-ins rather than an exhaustive list.
# ---------------------------------------------------------------------------
_JS_KEYWORDS = frozenset({
    # ECMAScript reserved words
    'break', 'case', 'catch', 'continue', 'debugger', 'default', 'delete',
    'do', 'else', 'export', 'extends', 'finally', 'for', 'function', 'if',
    'import', 'in', 'instanceof', 'new', 'return', 'switch', 'this',
    'throw', 'try', 'typeof', 'var', 'void', 'while', 'with',
    'class', 'const', 'enum', 'let', 'static', 'yield', 'await', 'async',
    'super', 'implements', 'interface', 'package', 'private', 'protected',
    'public',
    # Literals
    'true', 'false', 'null', 'undefined', 'NaN', 'Infinity',
    # Common built-in globals
    'arguments', 'eval', 'console', 'window', 'document', 'global',
    'globalThis', 'process', 'require', 'module', 'exports',
    'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval',
    'fetch', 'Response', 'Request', 'Headers', 'URL',
    'Array', 'Object', 'String', 'Number', 'Boolean', 'Symbol',
    'BigInt', 'Date', 'Math', 'JSON', 'RegExp', 'Map', 'Set',
    'WeakMap', 'WeakSet', 'Promise', 'Proxy', 'Reflect',
    'Error', 'TypeError', 'RangeError', 'SyntaxError', 'ReferenceError',
    'parseInt', 'parseFloat', 'isNaN', 'isFinite', 'encodeURIComponent',
    'decodeURIComponent', 'encodeURI', 'decodeURI',
    'Buffer', 'Uint8Array', 'Int8Array', 'ArrayBuffer', 'DataView',
})


def _parse_single_js_file(file_path: str) -> Tuple[str, ParseResult, Dict[str, ImportBinding], List[ExportEntry]]:
    """Worker helper for parallel parse of a single JS/JSX file."""
    parser = JavaScriptVisitorParser()
    result = parser.parse_file(file_path)
    return file_path, result, parser._import_bindings, parser._exports


class JavaScriptVisitorParser(BaseParser):
    """Visitor-style rich parser for JavaScript + JSX (phase 0-1).

    Implements multi-file parse with global export index enhancement.
    """

    def __init__(self):
        super().__init__("javascript")
        self.ts_language = None
        self.parser = None
        self._current_file: str = ""
        self._current_content: str = ""
        self._symbols: List[Symbol] = []
        self._relationships: List[Relationship] = []
        self._scope_stack: List[str] = []
        self._module_symbol_name: str = ""

        # Per-file bookkeeping
        self._import_bindings: Dict[str, ImportBinding] = {}
        self._exports: List[ExportEntry] = []
        self._class_fields: Dict[str, Set[str]] = {}
        self._jsx_usages: List[Tuple[str, Range, Optional[str]]] = []  # (raw_tag, range, enclosing_symbol)

        # Global (across multi-file pass) registries
        self._global_exports: Dict[str, List[ExportEntry]] = {}
        self._exports_by_file: Dict[str, List[ExportEntry]] = {}

        self._setup_tree_sitter()

    # ------------------------------------------------------------------
    # BaseParser interface implementations
    # ------------------------------------------------------------------
    def _define_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            language="javascript",
            supported_symbols={
                SymbolType.MODULE, SymbolType.CLASS, SymbolType.FUNCTION,
                SymbolType.METHOD, SymbolType.CONSTRUCTOR, SymbolType.FIELD,
                SymbolType.VARIABLE, SymbolType.CONSTANT
            },
            supported_relationships={
                RelationshipType.IMPORTS, RelationshipType.EXPORTS, RelationshipType.CALLS,
                RelationshipType.INHERITANCE, RelationshipType.REFERENCES, RelationshipType.COMPOSITION,
                RelationshipType.CREATES, RelationshipType.DEFINES
            },
            supports_ast_parsing=True,
            supports_type_inference=False,
            supports_cross_file_analysis=True,
            has_classes=True,
            has_interfaces=False,
            has_namespaces=False,
            has_generics=False,
            has_decorators=False,
            has_annotations=False,
            max_file_size_mb=8.0,
            typical_parse_time_ms=20.0
        )

    def _get_supported_extensions(self) -> Set[str]:
        return {'.js', '.mjs', '.cjs', '.jsx'}

    def _setup_tree_sitter(self) -> bool:
        try:
            self.ts_language = get_language('javascript')
            self.parser = get_parser('javascript')
            return True
        except Exception as e:
            logger.error(f"Failed to init tree-sitter javascript: {e}")
            return False

    # The BaseParser abstract methods `extract_symbols` / `extract_relationships`
    # are oriented around an AST node argument; this parser uses parse_file
    # directly, so we implement trivial wrappers.
    def extract_symbols(self, ast_node, file_path: str) -> List[Symbol]:  # type: ignore
        return self.parse_file(file_path).symbols

    def extract_relationships(self, ast_node, symbols: List[Symbol], file_path: str) -> List[Relationship]:  # type: ignore
        return self.parse_file(file_path).relationships

    # ------------------------------------------------------------------
    # Parsing entrypoints
    # ------------------------------------------------------------------
    def parse_file(self, file_path: Union[str, Path], content: Optional[str] = None) -> ParseResult:
        start = time.time()
        file_path = str(file_path)
        self._reset_file_state(file_path)

        try:
            if content is None:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            self._current_content = content

            if self.parser:
                tree = self.parser.parse(bytes(content, 'utf8'))
                root = tree.root_node
                self._create_module_symbol(root)
                self._visit_node(root)  # Pass 1: Extract symbols
                self._flush_jsx_usages()  # Flush JSX references
                self._extract_defines_relationships()  # Pass 2: Create DEFINES edges
                self._scan_all_body_references()  # Pass 3: Cross-reference scanning
            else:
                logger.warning("Tree-sitter parser unavailable for JavaScript; returning empty result")

            parse_time = time.time() - start
            return ParseResult(
                file_path=file_path,
                language="javascript",
                symbols=self._symbols,
                relationships=self._relationships,
                parse_time=parse_time,
                exports=[e.name for e in self._exports],
                imports=list({b.source_path for b in self._import_bindings.values()})
            )
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return ParseResult(
                file_path=file_path,
                language="javascript",
                symbols=[],
                relationships=[],
                errors=[str(e)]
            )

    def parse_multiple_files(self, file_paths: List[str], max_workers: int = 4) -> Dict[str, ParseResult]:
        """Two-pass multi-file parsing with global export enhancement.

        Falls back to sequential parse when max_workers <=1 or on ProcessPool failures,
        improving debuggability in constrained environments (e.g., pytest in ephemeral tmp dirs).
        """
        import concurrent.futures

        results: Dict[str, ParseResult] = {}
        per_file_imports: Dict[str, Dict[str, ImportBinding]] = {}
        per_file_exports: Dict[str, List[ExportEntry]] = {}

        def _collect(fp: str):
            fpr, res, imps, exps = _parse_single_js_file(fp)
            results[fpr] = res
            per_file_imports[fpr] = imps
            per_file_exports[fpr] = exps

        total = len(file_paths)
        logger.info(f"Parsing {total} JavaScript files for cross-file analysis with {max_workers} workers")
        if this and getattr(this, 'module', None):
            this.module.invocation_thinking(
                f"I am on phase parsing\nStarting JavaScript multi-file parsing: {total} files\nReasoning: Need AST + import/export maps to unify re-exports and resolve symbol flows across modules & JSX usage.\nNext: Parse in parallel with ~10% progress signals then build global export index."
            )
        
        # Batch size: process 500 files at a time to prevent memory issues
        batch_size = 500
        num_batches = (total + batch_size - 1) // batch_size
        
        if num_batches > 1:
            logger.info(f"Processing {total} JavaScript files in {num_batches} batch(es) of up to {batch_size} files each")
        
        start_time = time.time()
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            batch_files = file_paths[start_idx:end_idx]
            
            if num_batches > 1:
                logger.info(f"Processing batch {batch_idx + 1}/{num_batches}: files {start_idx + 1}-{end_idx} of {total}")
            
            try:
                # Use ThreadPoolExecutor for better stability (no SystemExit issues)
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all files in this batch
                    future_to_file = {}
                    for fp in batch_files:
                        try:
                            future = executor.submit(_parse_single_js_file, fp)
                            future_to_file[future] = fp
                        except Exception as e:
                            logger.error(f"Failed to submit file {fp}: {type(e).__name__}: {e}")
                            results[fp] = ParseResult(
                                file_path=fp,
                                language="javascript",
                                symbols=[],
                                relationships=[],
                                errors=[f"Submit failed: {str(e)}"]
                            )
                            per_file_imports[fp] = {}
                            per_file_exports[fp] = []
                    
                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            fp, result, import_bindings, exports = future.result(timeout=60)
                            results[fp] = result
                            per_file_imports[fp] = import_bindings
                            per_file_exports[fp] = exports
                            
                            # Progress reporting
                            if len(results) % max(1, int(total/10)) == 0 or len(results) == total:
                                elapsed = time.time() - start_time
                                logger.info(f"Parsed {len(results)}/{total} JavaScript files in {elapsed:.1f}s...")
                                if this and getattr(this, 'module', None):
                                    pct = int((len(results)/total)*100) if total else 100
                                    this.module.invocation_thinking(
                                        f"I am on phase parsing\nJavaScript raw parse progress: {len(results)}/{total} files ({pct}%)\nReasoning: Expanding module-level symbol + export surface for later re-export and import resolution.\nNext: Complete raw phase then aggregate exports globally."
                                    )
                        except concurrent.futures.TimeoutError:
                            logger.error(f"Timeout parsing JavaScript file (>60s): {file_path}")
                            results[file_path] = ParseResult(
                                file_path=file_path,
                                language="javascript",
                                symbols=[],
                                relationships=[],
                                errors=["Timeout during parsing"]
                            )
                            per_file_imports[file_path] = {}
                            per_file_exports[file_path] = []
                        except Exception as e:
                            logger.error(f"Failed to parse JavaScript file {file_path}: {type(e).__name__}: {e}", exc_info=True)
                            results[file_path] = ParseResult(
                                file_path=file_path,
                                language="javascript",
                                symbols=[],
                                relationships=[],
                                errors=[f"Parse error: {str(e)}"]
                            )
                            per_file_imports[file_path] = {}
                            per_file_exports[file_path] = []
            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} failed: {type(e).__name__}: {e}", exc_info=True)
                # Process remaining files in this batch sequentially as fallback
                for fp in batch_files:
                    if fp not in results:
                        try:
                            _collect(fp)
                        except Exception as inner_e:
                            logger.error(f"Sequential fallback also failed for {fp}: {inner_e}")
                            results[fp] = ParseResult(
                                file_path=fp,
                                language="javascript",
                                symbols=[],
                                relationships=[],
                                errors=[f"Batch and fallback failed: {str(inner_e)}"]
                            )
                            per_file_imports[fp] = {}
                            per_file_exports[fp] = []

        # Build global export index
        if this and getattr(this, 'module', None):
            this.module.invocation_thinking(
                "I am on phase parsing\nBuilding global JavaScript export index\nReasoning: Consolidate named/default/re-export chains to enable accurate cross-file call & reference mapping.\nNext: Summarize export coverage then resolve relationships."
            )
        global_exports: Dict[str, List[ExportEntry]] = {}
        for fp, entries in per_file_exports.items():
            for entry in entries:
                name_key = entry.name
                global_exports.setdefault(name_key, []).append(entry)
        if this and getattr(this, 'module', None):
            total_exports = sum(len(v) for v in global_exports.values())
            this.module.invocation_thinking(
                f"I am on phase parsing\nGlobal export index built: {len(global_exports)} keys / {total_exports} entries\nReasoning: Export surface unified; can now trace import->export->definition chains.\nNext: Resolve cross-file imports, calls, inheritance & JSX references."
            )

        # Build per-file class->methods index for deeper call resolution
        per_file_method_index: Dict[str, Dict[str, Set[str]]] = {}
        for fp2, res in results.items():
            cls_methods: Dict[str, Set[str]] = {}
            for sym in res.symbols:
                if sym.symbol_type == SymbolType.METHOD and sym.parent_symbol:
                    cls = sym.parent_symbol.split('.')[-1]
                    cls_methods.setdefault(cls, set()).add(sym.name)
            if cls_methods:
                per_file_method_index[fp2] = cls_methods

        # Build global symbol registry: maps symbol name → file path for
        # top-level CLASS / FUNCTION / CONSTANT symbols.  Used as a fallback
        # in _enhance_file_relationships when import bindings don't cover a
        # REFERENCES target (e.g. re-exports, barrel files, or same-package
        # symbols referenced without explicit import).
        global_symbol_registry: Dict[str, str] = {}
        for fp2, res in results.items():
            module_stem = Path(fp2).stem
            for sym in res.symbols:
                if sym.symbol_type in (SymbolType.CLASS, SymbolType.FUNCTION, SymbolType.CONSTANT):
                    expected_full = f"{module_stem}.{sym.name}"
                    if sym.full_name == expected_full:
                        # First-seen wins (mirrors TS behaviour)
                        if sym.name not in global_symbol_registry:
                            global_symbol_registry[sym.name] = fp2

        logger.info("Enhancing JavaScript relationships with cross-file resolution")
        if this and getattr(this, 'module', None):
            this.module.invocation_thinking("I am on phase parsing\nResolving cross-file exports, imports, calls & JSX references (JavaScript)\nReasoning: Enrich relationships with resolved targets for precise graph edges.\nNext: Emit final parsing completion summary.")
        # Enhance relationships (IMPORTS + JSX + INHERITANCE + CALLS resolution)
        for fp, result in results.items():
            self._enhance_file_relationships(fp, result, per_file_imports[fp], per_file_exports, global_exports, per_file_method_index, results, global_symbol_registry)

        logger.info(f"JavaScript parsing complete: {len(results)}/{total} files processed")
        if this and getattr(this, 'module', None):
            this.module.invocation_thinking(
                f"I am on phase parsing\nJavaScript parsing complete: {len(results)}/{total} files (100%)\nReasoning: All modules enriched with resolved relationships; downstream graph composition can proceed.\nNext: Higher-level aggregation and documentation phases consume these artifacts."
            )

        return results

    # ------------------------------------------------------------------
    # Internal state helpers
    # ------------------------------------------------------------------
    def _reset_file_state(self, file_path: str):
        self._current_file = file_path
        self._current_content = ""
        self._symbols = []
        self._relationships = []
        self._scope_stack = []
        self._module_symbol_name = Path(file_path).stem
        self._import_bindings = {}
        self._exports = []
        self._class_fields = {}
        self._jsx_usages = []
        # Stored (full_name, declaration_node) for post-traversal body scanning
        self._function_bodies: List[Tuple[str, Node]] = []

    # ------------------------------------------------------------------
    # Node visitation
    # ------------------------------------------------------------------
    def _visit_node(self, node: Node):
        if node is None:
            return
        t = node.type

        handler_map = {
            'import_statement': self._handle_import_statement,
            'import_declaration': self._handle_import_statement,  # defensive
            'export_statement': self._handle_export_statement,
            'function_declaration': self._handle_function_declaration,
            'class_declaration': self._handle_class_declaration,
            'method_definition': self._handle_method_definition,
            'field_definition': self._handle_field_definition,  # Phase 4 (JS-1)
            'lexical_declaration': self._handle_lexical_declaration,
            'variable_declaration': self._handle_variable_declaration,
            'call_expression': self._handle_call_expression,
            'new_expression': self._handle_new_expression,
            'jsx_opening_element': self._handle_jsx_element,
            'jsx_self_closing_element': self._handle_jsx_element,
        }

        handler = handler_map.get(t)
        if handler:
            handler(node)
        else:
            for child in getattr(node, 'children', []):
                self._visit_node(child)

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------
    def _byte_to_line_col(self, byte_offset: int) -> Tuple[int, int]:
        content = self._current_content
        lines_before = content[:byte_offset].count('\n')
        line_start = content.rfind('\n', 0, byte_offset) + 1
        col = byte_offset - line_start
        return lines_before + 1, col

    def _range(self, node: Node) -> Range:
        sl, sc = self._byte_to_line_col(node.start_byte)
        el, ec = self._byte_to_line_col(node.end_byte)
        return Range(start=Position(sl, sc), end=Position(el, ec))

    def _text(self, node: Node) -> str:
        try:
            return node.text.decode('utf-8')
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Symbol creation helpers
    # ------------------------------------------------------------------
    def _create_module_symbol(self, root: Node):
        """Track module scope for qualified names but don't emit a MODULE symbol.
        
        Modules are organizational, not architectural - they don't go in the graph.
        We only track them in scope_stack so children get qualified full_names like
        "module.ClassName" instead of just "ClassName".
        """
        # Add module to scope stack so children get qualified full_names
        self._scope_stack.append(self._module_symbol_name)

    def _enter_scope(self, name: str):
        self._scope_stack.append(name)

    def _exit_scope(self):
        if self._scope_stack:
            self._scope_stack.pop()

    def _current_scope_path(self) -> str:
        return '.'.join(self._scope_stack) if self._scope_stack else ''

    def _qualify(self, name: str) -> str:
        scope = self._current_scope_path()
        return f"{scope}.{name}" if scope else name

    # ------------------------------------------------------------------
    # JSDoc Parsing (Phase 4 JS-2)
    # ------------------------------------------------------------------
    # Regex patterns for JSDoc tags
    _JSDOC_PARAM_RE = __import__('re').compile(
        r'@param\s+\{([^}]+)\}\s+(\w+)'
    )
    _JSDOC_RETURNS_RE = __import__('re').compile(
        r'@returns?\s+\{([^}]+)\}'
    )
    _JSDOC_TYPE_RE = __import__('re').compile(
        r'@type\s+\{([^}]+)\}'
    )

    def _get_preceding_jsdoc(self, node: Node) -> Optional[str]:
        """Return the text of a JSDoc comment immediately preceding this node.
        
        In tree-sitter JS, comments are siblings of their target. We check
        the previous sibling for a /** ... */ block comment.
        
        For exported functions/classes, the JSDoc comment is a sibling of
        the export_statement wrapper, so we also check the parent's
        previous sibling.
        """
        # Direct previous sibling
        prev = node.prev_named_sibling
        if prev is not None and prev.type == 'comment':
            text = self._text(prev)
            if text.startswith('/**'):
                return text
        
        # Check parent's previous sibling (for export function / export class)
        parent = node.parent
        if parent is not None and parent.type == 'export_statement':
            prev_parent = parent.prev_named_sibling
            if prev_parent is not None and prev_parent.type == 'comment':
                text = self._text(prev_parent)
                if text.startswith('/**'):
                    return text
        
        return None

    def _parse_jsdoc(self, jsdoc: str) -> Dict[str, Any]:
        """Parse JSDoc comment text and extract @param, @returns, @type tags.
        
        Returns dict with possible keys:
            'params': dict of param_name -> type_string
            'returns': type_string
            'type': type_string
        """
        result: Dict[str, Any] = {}

        # @param {Type} name
        params = {}
        for m in self._JSDOC_PARAM_RE.finditer(jsdoc):
            params[m.group(2)] = m.group(1).strip()
        if params:
            result['params'] = params

        # @returns {Type}
        m = self._JSDOC_RETURNS_RE.search(jsdoc)
        if m:
            result['returns'] = m.group(1).strip()

        # @type {Type}
        m = self._JSDOC_TYPE_RE.search(jsdoc)
        if m:
            result['type'] = m.group(1).strip()

        return result

    def _emit_jsdoc_references(self, jsdoc_data: Dict[str, Any],
                               source_symbol: str, source_range: Range):
        """Emit REFERENCES for user-defined types mentioned in JSDoc tags.
        
        Filters out builtin types (string, number, boolean, etc.).
        """
        _BUILTIN_TYPES = {
            'string', 'number', 'boolean', 'object', 'undefined', 'null',
            'void', 'any', 'symbol', 'bigint', 'never', 'unknown',
            'String', 'Number', 'Boolean', 'Object', 'Function',
            'Array', 'Promise', 'Map', 'Set', 'Date', 'RegExp', 'Error',
            'Symbol', 'BigInt',
        }

        import re
        # Extract identifiers from all JSDoc type strings
        type_strings = []
        if 'params' in jsdoc_data:
            type_strings.extend(jsdoc_data['params'].values())
        if 'returns' in jsdoc_data:
            type_strings.append(jsdoc_data['returns'])
        if 'type' in jsdoc_data:
            type_strings.append(jsdoc_data['type'])

        # Extract identifiers from type strings like "Array<User>", "string|MyType"
        ident_re = re.compile(r'[A-Z]\w*')
        seen = set()
        for ts in type_strings:
            for match in ident_re.finditer(ts):
                name = match.group()
                if name not in _BUILTIN_TYPES and name not in seen:
                    seen.add(name)
                    self._relationships.append(Relationship(
                        source_symbol=source_symbol,
                        target_symbol=name,
                        relationship_type=RelationshipType.REFERENCES,
                        source_file=self._current_file,
                        source_range=source_range,
                        annotations={'reference_type': 'jsdoc'}
                    ))

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def _handle_import_statement(self, node: Node):
        # Patterns handled (tree-sitter javascript):
        # import Default from './mod'
        # import { A, B as C } from './mod'
        # import * as NS from './mod'
        # import './sideEffect'
        source_path = None
        import_clause = None
        for child in node.children:
            if child.type == 'string':
                source_path = self._text(child).strip('"\'')
            elif child.type == 'import_clause':
                import_clause = child

        if not source_path:
            return

        # Side-effect import
        if not import_clause:
            rel = Relationship(
                source_symbol=self._module_symbol_name,
                target_symbol=source_path,
                relationship_type=RelationshipType.IMPORTS,
                source_file=self._current_file
            )
            self._relationships.append(rel)
            return

        for ic_child in import_clause.children:
            t = ic_child.type
            if t == 'identifier':  # default import
                local = self._text(ic_child)
                self._import_bindings[local] = ImportBinding(local=local, source_path=source_path, imported=None, is_default=True)
                self._relationships.append(Relationship(
                    source_symbol=self._module_symbol_name,
                    target_symbol=local,
                    relationship_type=RelationshipType.IMPORTS,
                    source_file=self._current_file
                ))
            elif t == 'namespace_import':  # * as NS
                # structure: namespace_import -> '*' 'as' identifier
                ident = [c for c in ic_child.children if c.type == 'identifier']
                if ident:
                    local = self._text(ident[0])
                    self._import_bindings[local] = ImportBinding(local=local, source_path=source_path, imported='*', is_namespace=True)
                    self._relationships.append(Relationship(
                        source_symbol=self._module_symbol_name,
                        target_symbol=local,
                        relationship_type=RelationshipType.IMPORTS,
                        source_file=self._current_file
                    ))
            elif t == 'named_imports':  # { A, B as C }
                for spec in ic_child.children:
                    if spec.type == 'import_specifier':
                        imported = None
                        local = None
                        for sp_child in spec.children:
                            if sp_child.type == 'identifier':
                                if imported is None:
                                    imported = self._text(sp_child)
                                else:
                                    local = self._text(sp_child)
                        if imported and not local:
                            local = imported
                        if local:
                            self._import_bindings[local] = ImportBinding(local=local, source_path=source_path, imported=imported, is_default=False)
                            self._relationships.append(Relationship(
                                source_symbol=self._module_symbol_name,
                                target_symbol=local,
                                relationship_type=RelationshipType.IMPORTS,
                                source_file=self._current_file
                            ))

    def _handle_export_statement(self, node: Node):
        # Patterns: export function foo() {}, export class A {}, export { foo as bar }, export default ...
        # We'll detect declared symbols or specifiers.
        default_export = any(ch.type == 'default' for ch in node.children)
        # Direct function/class declaration child
        for child in node.children:
            if child.type in ('function_declaration', 'class_declaration'):
                name_node = next((c for c in child.children if c.type == 'identifier'), None)
                if name_node:
                    name = self._text(name_node)
                    export_name = DEFAULT_EXPORT_SENTINEL if default_export else name
                    self._exports.append(ExportEntry(name=export_name, internal_symbol=name, is_default=default_export, file_path=self._current_file))
        # Named exports: export { A, B as C }
        for ch in node.children:
            if ch.type == 'export_clause':
                for spec in ch.children:
                    if spec.type == 'export_specifier':
                        names = [c for c in spec.children if c.type == 'identifier']
                        if names:
                            original = self._text(names[0])
                            exported = self._text(names[-1]) if len(names) > 1 else original
                            self._exports.append(ExportEntry(name=exported, internal_symbol=original, is_default=False, file_path=self._current_file))
        # export default <identifier>
        if default_export and not any(c.type in ('function_declaration', 'class_declaration') for c in node.children):
            ident = next((c for c in node.children if c.type == 'identifier'), None)
            if ident:
                name = self._text(ident)
                self._exports.append(ExportEntry(name=DEFAULT_EXPORT_SENTINEL, internal_symbol=name, is_default=True, file_path=self._current_file))
        # Visit children (so class/function declarations get processed fully)
        for child in node.children:
            self._visit_node(child)

    def _handle_function_declaration(self, node: Node):
        name_node = next((c for c in node.children if c.type == 'identifier'), None)
        if not name_node:
            return
        name = self._text(name_node)
        
        # Phase 4 JS-2: Extract JSDoc metadata
        jsdoc_text = self._get_preceding_jsdoc(node)
        jsdoc_meta = {}
        if jsdoc_text:
            jsdoc_data = self._parse_jsdoc(jsdoc_text)
            if jsdoc_data.get('params'):
                jsdoc_meta['jsdoc_params'] = jsdoc_data['params']
            if jsdoc_data.get('returns'):
                jsdoc_meta['jsdoc_returns'] = jsdoc_data['returns']
        
        symbol = Symbol(
            name=name,
            symbol_type=SymbolType.FUNCTION,
            scope=Scope.FUNCTION,
            range=self._range(node),
            file_path=self._current_file,
            parent_symbol=self._current_scope_path() or self._module_symbol_name,
            full_name=self._qualify(name),
            source_text=self._text(node),
            metadata=jsdoc_meta if jsdoc_meta else {}
        )
        self._symbols.append(symbol)
        
        # Emit REFERENCES for user-defined types in JSDoc
        if jsdoc_text:
            jsdoc_data = self._parse_jsdoc(jsdoc_text)
            self._emit_jsdoc_references(jsdoc_data, symbol.full_name, self._range(node))
        
        # Store for post-traversal body reference scanning
        self._function_bodies.append((symbol.full_name, node))
        
        # Enter function scope before traversing body
        self._enter_scope(name)
        for child in node.children:
            self._visit_node(child)
        self._exit_scope()

    def _handle_class_declaration(self, node: Node):
        name_node = next((c for c in node.children if c.type == 'identifier'), None)
        if not name_node:
            return
        name = self._text(name_node)
        symbol = Symbol(
            name=name,
            symbol_type=SymbolType.CLASS,
            scope=Scope.CLASS,
            range=self._range(node),
            file_path=self._current_file,
            parent_symbol=self._current_scope_path() or self._module_symbol_name,
            full_name=self._qualify(name),
            source_text=self._text(node)
        )
        self._symbols.append(symbol)
        # Inheritance: look for class_heritage child
        heritage = next((c for c in node.children if c.type == 'class_heritage'), None)
        if heritage:
            ident = next((c for c in heritage.children if c.type == 'identifier'), None)
            if ident:
                base = self._text(ident)
                self._relationships.append(Relationship(
                    source_symbol=name,
                    target_symbol=base,
                    relationship_type=RelationshipType.INHERITANCE,
                    source_file=self._current_file
                ))
        else:
            # Fallback textual scan for 'extends Identifier'
            for idx, ch in enumerate(node.children):
                if self._text(ch) == 'extends' and idx + 1 < len(node.children):
                    next_node = node.children[idx + 1]
                    if next_node.type == 'identifier':
                        base = self._text(next_node)
                        self._relationships.append(Relationship(
                            source_symbol=name,
                            target_symbol=base,
                            relationship_type=RelationshipType.INHERITANCE,
                            source_file=self._current_file
                        ))
                        break
        self._enter_scope(name)
        for child in node.children:
            self._visit_node(child)
        self._exit_scope()

    def _handle_method_definition(self, node: Node):
        # method_definition -> property_identifier / identifier etc.
        name_node = next((c for c in node.children if c.type in ('property_identifier', 'identifier')), None)
        if not name_node:
            return
        name = self._text(name_node)
        full_name = self._qualify(name)
        
        # Check if method is static
        is_static = any(c.type == 'static' for c in node.children)
        
        # Phase 2: standardise constructor SymbolType
        sym_type = SymbolType.CONSTRUCTOR if name == 'constructor' else SymbolType.METHOD
        
        # Phase 4 JS-2: Extract JSDoc metadata
        jsdoc_text = self._get_preceding_jsdoc(node)
        jsdoc_meta = {}
        if jsdoc_text:
            jsdoc_data = self._parse_jsdoc(jsdoc_text)
            if jsdoc_data.get('params'):
                jsdoc_meta['jsdoc_params'] = jsdoc_data['params']
            if jsdoc_data.get('returns'):
                jsdoc_meta['jsdoc_returns'] = jsdoc_data['returns']
        
        symbol = Symbol(
            name=name,
            symbol_type=sym_type,
            scope=Scope.FUNCTION,
            range=self._range(node),
            file_path=self._current_file,
            parent_symbol=self._current_scope_path() or self._module_symbol_name,
            full_name=full_name,
            source_text=self._text(node),
            is_static=is_static,
            metadata=jsdoc_meta if jsdoc_meta else {}
        )
        self._symbols.append(symbol)
        
        # Emit REFERENCES for user-defined types in JSDoc
        if jsdoc_text:
            jsdoc_data = self._parse_jsdoc(jsdoc_text)
            self._emit_jsdoc_references(jsdoc_data, full_name, self._range(node))
        
        # Store for post-traversal body reference scanning
        self._function_bodies.append((full_name, node))
        
        self._enter_scope(name)
        for child in node.children:
            self._visit_node(child)
        self._exit_scope()

    def _handle_field_definition(self, node: Node):
        """Extract class FIELD from field_definition node (Phase 4 JS-1).
        
        Handles:
            x = 5;             → FIELD with default value
            static y = 10;     → static FIELD
            #secret = 'hidden'; → private FIELD
            name;              → FIELD with no initializer
        
        Tree-sitter AST for field_definition:
            field_definition
                [static]                          (optional modifier)
                property_identifier | private_property_identifier
                [= <expression>]                  (optional initializer)
        """
        if not self._scope_stack:
            return  # fields must be inside a class scope
        
        # Extract field name — property_identifier or private_property_identifier
        name_node = next(
            (c for c in node.children if c.type in ('property_identifier', 'private_property_identifier')),
            None
        )
        if not name_node:
            return
        
        name = self._text(name_node)
        is_private = name_node.type == 'private_property_identifier'
        is_static = any(c.type == 'static' for c in node.children)
        
        full_name = self._qualify(name)
        parent_symbol = self._current_scope_path()
        
        # Determine visibility
        if is_private:
            visibility = 'private'
        else:
            visibility = 'public'  # JS class fields default to public
        
        # Extract default value for metadata
        has_initializer = any(c.type == '=' for c in node.children)
        
        # Phase 4 JS-2: Extract JSDoc @type for fields
        jsdoc_text = self._get_preceding_jsdoc(node)
        jsdoc_meta: Dict[str, Any] = {
            'is_private': is_private,
            'has_initializer': has_initializer,
        }
        if jsdoc_text:
            jsdoc_data = self._parse_jsdoc(jsdoc_text)
            if jsdoc_data.get('type'):
                jsdoc_meta['jsdoc_type'] = jsdoc_data['type']
        
        symbol = Symbol(
            name=name,
            symbol_type=SymbolType.FIELD,
            scope=Scope.CLASS,
            range=self._range(node),
            file_path=self._current_file,
            parent_symbol=parent_symbol,
            full_name=full_name,
            source_text=self._text(node),
            is_static=is_static,
            visibility=visibility,
            metadata=jsdoc_meta
        )
        self._symbols.append(symbol)
        
        # Emit REFERENCES for user-defined types in JSDoc
        if jsdoc_text:
            jsdoc_data = self._parse_jsdoc(jsdoc_text)
            self._emit_jsdoc_references(jsdoc_data, full_name, self._range(node))
        
        # Track class fields for later COMPOSITION detection
        class_name = self._scope_stack[-1] if self._scope_stack else None
        if class_name:
            if class_name not in self._class_fields:
                self._class_fields[class_name] = set()
            self._class_fields[class_name].add(name)
        
        # Recurse into initializer for CALLS/CREATES extraction
        for child in node.children:
            if child.type not in ('property_identifier', 'private_property_identifier', 'static', '=', ';'):
                self._visit_node(child)

    def _handle_lexical_declaration(self, node: Node):
        # const / let declarations; mark exported if preceded by export parent
        is_export = node.parent and node.parent.type == 'export_statement'
        for child in node.children:
            if child.type == 'variable_declarator':
                ident = next((c for c in child.children if c.type == 'identifier'), None)
                if ident:
                    name = self._text(ident)
                    symbol = Symbol(
                        name=name,
                        symbol_type=SymbolType.CONSTANT,
                        scope=Scope.GLOBAL,
                        range=self._range(child),
                        file_path=self._current_file,
                        parent_symbol=self._module_symbol_name,
                        full_name=self._qualify(name),
                        source_text=self._text(child)
                    )
                    self._symbols.append(symbol)
                    if is_export:
                        self._exports.append(ExportEntry(name=name, internal_symbol=name, is_default=False, file_path=self._current_file, symbol_type=SymbolType.CONSTANT))
                else:
                    # Handle CommonJS destructured require:
                    #   const { X, Y } = require("./module")
                    # Treat destructured names as import bindings so that
                    # body scanning + enhancement can resolve cross-file refs.
                    self._try_register_require_bindings(child)
        # Continue traversal for initializers
        for c in node.children:
            self._visit_node(c)

    def _handle_variable_declaration(self, node: Node):
        # var declarations (older style)
        self._handle_lexical_declaration(node)

    def _handle_call_expression(self, node: Node):
        # Basic CALLS relationship (identifier() or obj.method())
        # We'll capture textual target; enhancement later may resolve
        callee = next((c for c in node.children if c.type in ('identifier', 'member_expression')), None)
        if callee:
            target = self._extract_callee_name(callee)
            if target:
                source_symbol = self._current_scope_path() or self._module_symbol_name
                self._relationships.append(Relationship(
                    source_symbol=source_symbol,
                    target_symbol=target,
                    relationship_type=RelationshipType.CALLS,
                    source_file=self._current_file
                ))
        # Traverse arguments
        for child in node.children:
            self._visit_node(child)

    def _handle_new_expression(self, node: Node):
        ident = next((c for c in node.children if c.type == 'identifier'), None)
        if ident:
            name = self._text(ident)
            source_symbol = self._current_scope_path() or self._module_symbol_name
            # CREATES (constructor call) - matches Python/Java/TypeScript parsers
            self._relationships.append(Relationship(
                source_symbol=source_symbol,
                target_symbol=name,
                relationship_type=RelationshipType.CREATES,
                source_file=self._current_file,
                confidence=0.95,  # High confidence for tree-sitter based detection
                annotations={'creation_type': 'new'}
            ))
            # Potential COMPOSITION if inside class constructor
            if self._scope_stack and self._scope_stack[-1] in ('constructor', '<init>'):
                self._relationships.append(Relationship(
                    source_symbol=source_symbol,
                    target_symbol=name,
                    relationship_type=RelationshipType.COMPOSITION,
                    source_file=self._current_file
                ))
        for c in node.children:
            self._visit_node(c)

    # Removed standalone extends/heritage handlers (handled in class declaration)

    def _handle_jsx_element(self, node: Node):
        # Opening or self-closing element; extract tag
        name_node = next((c for c in node.children if c.type in ('identifier', 'jsx_identifier', 'jsx_member_expression', 'member_expression')), None)
        if not name_node:
            return
        raw = self._text(name_node)
        # Filter intrinsic HTML (starts lowercase or contains '-')
        first_segment = raw.split('.')[0]
        if not first_segment or not first_segment[0].isupper() or '-' in first_segment:
            return
        enclosing = self._current_scope_path() or self._module_symbol_name
        self._jsx_usages.append((raw, self._range(name_node), enclosing))

    # ------------------------------------------------------------------
    # Helper extraction
    # ------------------------------------------------------------------
    def _extract_callee_name(self, node: Node) -> Optional[str]:
        # Recursive extraction to capture full member chains: obj.prop, this.method, super.method, Util.sm
        if node.type in ('identifier', 'this', 'super', 'property_identifier'):
            return self._text(node)
        if node.type == 'member_expression':
            if not node.children:
                return None
            # Left side can itself be a member_expression
            left = self._extract_callee_name(node.children[0])
            # Find the property identifier (could be property_identifier or identifier)
            prop = None
            for ch in node.children[1:]:
                if ch.type in ('property_identifier', 'identifier'):
                    prop = self._text(ch)
            if left and prop:
                return f"{left}.{prop}"
            return left or prop
        return None

    def _try_register_require_bindings(self, declarator_node: Node):
        """Detect ``const { X, Y } = require("./path")`` and register
        the destructured names as import bindings (CommonJS support).

        This enables body scanning + enhancement to create and resolve
        cross-file REFERENCES edges for CommonJS modules.
        """
        obj_pattern = next(
            (c for c in declarator_node.children if c.type == 'object_pattern'), None
        )
        call_expr = next(
            (c for c in declarator_node.children if c.type == 'call_expression'), None
        )
        if not (obj_pattern and call_expr):
            return

        # Verify the call is require("...")
        callee = next(
            (c for c in call_expr.children if c.type == 'identifier'), None
        )
        if not callee or self._text(callee) != 'require':
            return

        args_node = next(
            (c for c in call_expr.children if c.type == 'arguments'), None
        )
        if not args_node:
            return

        string_node = next(
            (c for c in args_node.children if c.type == 'string'), None
        )
        if not string_node:
            return

        source_path = self._text(string_node).strip('\'"')

        # Register each destructured name as an import binding
        for p_child in obj_pattern.children:
            if p_child.type in ('shorthand_property_identifier',
                                'shorthand_property_identifier_pattern'):
                local = self._text(p_child)
                self._import_bindings[local] = ImportBinding(
                    local=local,
                    source_path=source_path,
                    imported=local,
                    is_namespace=False,
                    is_default=False,
                )
            elif p_child.type == 'pair_pattern':
                # const { original: alias } = require(...)
                key_node = next(
                    (c for c in p_child.children
                     if c.type in ('property_identifier', 'identifier')),
                    None
                )
                value_node = next(
                    (c for c in p_child.children
                     if c.type == 'shorthand_property_identifier_pattern'
                        or (c.type == 'identifier' and c != key_node)),
                    None
                )
                if key_node and value_node:
                    original = self._text(key_node)
                    local = self._text(value_node)
                    self._import_bindings[local] = ImportBinding(
                        local=local,
                        source_path=source_path,
                        imported=original,
                        is_namespace=False,
                        is_default=False,
                    )

    # ------------------------------------------------------------------
    # Body reference scanning (cross-reference detection)
    # ------------------------------------------------------------------
    def _extract_param_names(self, params_node: Node) -> Set[str]:
        """Extract parameter names from a formal_parameters AST node."""
        names: Set[str] = set()
        for child in params_node.children:
            if child.type == 'identifier':
                names.add(self._text(child))
            elif child.type == 'assignment_pattern':
                # Default parameter: function(x = 5)
                ident = next((c for c in child.children if c.type == 'identifier'), None)
                if ident:
                    names.add(self._text(ident))
            elif child.type == 'rest_pattern':
                # Rest parameter: function(...args)
                ident = next((c for c in child.children if c.type == 'identifier'), None)
                if ident:
                    names.add(self._text(ident))
            elif child.type in ('object_pattern', 'array_pattern'):
                # Destructured parameters — extract nested identifiers
                names.update(self._extract_destructured_names(child))
        return names

    def _extract_destructured_names(self, node: Node) -> Set[str]:
        """Extract identifiers from destructuring patterns (object/array)."""
        names: Set[str] = set()
        stack = list(node.children)
        while stack:
            child = stack.pop()
            if child.type == 'identifier':
                names.add(self._text(child))
            elif child.type == 'shorthand_property_identifier_pattern':
                names.add(self._text(child))
            elif child.type in ('object_pattern', 'array_pattern',
                                'assignment_pattern', 'rest_pattern',
                                'pair_pattern'):
                stack.extend(child.children)
        return names

    def _extract_local_var_names(self, body_node: Node) -> Set[str]:
        """Extract locally-declared variable names from a statement_block.

        Only checks immediate statements (not nested blocks) — this is
        intentionally shallow to avoid over-filtering.
        """
        names: Set[str] = set()
        for stmt in body_node.children:
            if stmt.type in ('lexical_declaration', 'variable_declaration'):
                for child in stmt.children:
                    if child.type == 'variable_declarator':
                        ident = next((c for c in child.children if c.type == 'identifier'), None)
                        if ident:
                            names.add(self._text(ident))
        return names

    def _scan_all_body_references(self):
        """Post-traversal pass: scan function/method bodies for identifier
        references to known symbols (imported or same-file top-level).

        Creates REFERENCES edges similar to TypeScript's ``visit_identifier``
        mechanism, enabling smart expansion to discover constant/function/class
        usage across files and within the same file.
        """
        # Build the set of symbol names we consider "interesting" targets.
        # 1. Same-file top-level symbols (constants, functions, classes)
        # 2. Imported symbol names
        known_symbols: Set[str] = set()
        for sym in self._symbols:
            if sym.symbol_type in (SymbolType.CONSTANT, SymbolType.FUNCTION, SymbolType.CLASS):
                # Only truly top-level: full_name == "module.Name"
                expected_full = f"{self._module_symbol_name}.{sym.name}"
                if sym.full_name == expected_full:
                    known_symbols.add(sym.name)
        for local_name in self._import_bindings:
            known_symbols.add(local_name)

        if not known_symbols:
            return

        for full_name, decl_node in self._function_bodies:
            # Locate body (statement_block) and formal_parameters
            body_node = None
            params_node = None
            for child in decl_node.children:
                if child.type == 'statement_block':
                    body_node = child
                elif child.type == 'formal_parameters':
                    params_node = child

            if not body_node:
                continue

            # Build exclusion set for this function
            param_names = self._extract_param_names(params_node) if params_node else set()
            local_vars = self._extract_local_var_names(body_node)
            own_name = full_name.rsplit('.', 1)[-1]
            exclude = param_names | local_vars | _JS_KEYWORDS | {own_name}

            # Walk body descendants for identifier nodes
            seen_refs: Set[str] = set()
            stack = list(body_node.children)
            while stack:
                nd = stack.pop()
                if nd.type == 'identifier':
                    name = self._text(nd)
                    if name in known_symbols and name not in exclude and name not in seen_refs:
                        # Skip left-hand side of variable declarations
                        parent = nd.parent
                        if parent and parent.type == 'variable_declarator':
                            if parent.children and parent.children[0] == nd:
                                continue
                        seen_refs.add(name)
                        self._relationships.append(Relationship(
                            source_symbol=full_name,
                            target_symbol=name,
                            relationship_type=RelationshipType.REFERENCES,
                            source_file=self._current_file,
                            source_range=self._range(nd),
                        ))
                stack.extend(nd.children)

    def _flush_jsx_usages(self):
        seen = set()
        for raw, rng, enclosing in self._jsx_usages:
            key = (enclosing, raw)
            if key in seen:
                continue
            seen.add(key)
            self._relationships.append(Relationship(
                source_symbol=enclosing,
                target_symbol=raw,
                relationship_type=RelationshipType.REFERENCES,
                source_file=self._current_file,
                source_range=rng
            ))

    def _extract_defines_relationships(self):
        """Extract DEFINES relationships from classes to their members (Pass 2).
        
        Creates DEFINES edges for:
        - Class -> Method, Constructor, Field
        
        Note: MODULE → top-level DEFINES relationships are NOT created.
        Modules are organizational, not architectural. Parent-child containment
        is captured by parent_symbol metadata on each symbol.
        
        This follows the two-pass architecture:
        - Pass 1 (SymbolExtractor): Already done in _visit_node - creates symbols with parent_symbol
        - Pass 2 (RelationshipExtractor): This method - matches parent_symbol strings to create edges
        
        Reference: docs/RELATIONSHIP_EXTRACTOR_PATTERN.md
        """
        # Build index of symbols by parent for efficient lookup
        symbols_by_parent = {}
        for symbol in self._symbols:
            parent = symbol.parent_symbol
            if parent:
                if parent not in symbols_by_parent:
                    symbols_by_parent[parent] = []
                symbols_by_parent[parent].append(symbol)
        
        # For each class, create DEFINES edges to its members
        # (NOT from MODULE - modules are organizational, not architectural)
        for symbol in self._symbols:
            if symbol.symbol_type == SymbolType.CLASS:
                # Find members that belong to this class.
                # Members store parent_symbol as the qualified scope path (e.g. "test.Foo"),
                # so look up by both the short name and the fully-qualified name.
                class_name = symbol.name
                full_class_name = symbol.full_name   # e.g. "test.Foo"
                members = (
                    symbols_by_parent.get(class_name, []) +
                    symbols_by_parent.get(full_class_name, [])
                )
                
                for member in members:
                    # Create DEFINES for methods, constructors, and fields
                    if member.symbol_type in [SymbolType.METHOD, SymbolType.CONSTRUCTOR, SymbolType.FIELD]:
                        # Determine member_type for metadata
                        if member.symbol_type == SymbolType.FIELD:
                            member_type = 'field'
                        elif member.name == 'constructor':
                            member_type = 'constructor'
                        else:
                            member_type = 'method'
                        
                        # Get is_static from symbol field (set during Pass 1)
                        is_static = member.is_static
                        
                        # Create DEFINES relationship
                        relationship = Relationship(
                            source_symbol=class_name,
                            target_symbol=member.full_name,
                            relationship_type=RelationshipType.DEFINES,
                            source_file=self._current_file,
                            source_range=member.range,
                            annotations={'member_type': member_type, 'is_static': is_static}
                        )
                        self._relationships.append(relationship)

    # ------------------------------------------------------------------
    # Enhancement
    # ------------------------------------------------------------------
    def _enhance_file_relationships(self, file_path: str, result: ParseResult,
                                    imports: Dict[str, ImportBinding],
                                    per_file_exports: Dict[str, List[ExportEntry]],
                                    global_exports: Dict[str, List[ExportEntry]],
                                    per_file_method_index: Dict[str, Dict[str, Set[str]]],
                                    all_results: Dict[str, ParseResult],
                                    global_symbol_registry: Optional[Dict[str, str]] = None):
        # Map of import local name -> list of candidate export entries
        resolved_import_targets: Dict[str, List[ExportEntry]] = {}
        for local, binding in imports.items():
            target_file = self._resolve_import_path(file_path, binding.source_path, per_file_exports)
            if target_file and target_file in per_file_exports:
                if binding.is_namespace:
                    candidates = per_file_exports[target_file]
                elif binding.is_default:
                    candidates = [e for e in per_file_exports[target_file] if e.is_default]
                elif binding.imported:
                    candidates = [e for e in per_file_exports[target_file] if e.name == binding.imported]
                else:
                    candidates = [e for e in per_file_exports[target_file] if e.name == local]
                if candidates:
                    resolved_import_targets[local] = candidates
            else:
                name_key = binding.imported or local
                if name_key in global_exports and len(global_exports[name_key]) == 1:
                    resolved_import_targets[local] = global_exports[name_key]

        enhanced: List[Relationship] = []
        for rel in result.relationships:
            if rel.relationship_type == RelationshipType.IMPORTS:
                if rel.target_symbol in resolved_import_targets:
                    rel.target_file = resolved_import_targets[rel.target_symbol][0].file_path
                enhanced.append(rel)
            elif rel.relationship_type in (RelationshipType.REFERENCES, RelationshipType.CALLS, RelationshipType.INHERITANCE):
                target_root = rel.target_symbol.split('.')[0]
                candidates = resolved_import_targets.get(target_root)
                if candidates:
                    rel.target_file = candidates[0].file_path
                    if '.' in rel.target_symbol:
                        parts = rel.target_symbol.split('.')
                        if len(parts) == 2:
                            ns, member = parts
                            is_ns_binding = any(b.local == ns and b.is_namespace for b in imports.values())
                            export_match = [e for e in candidates if e.name == member]
                            if is_ns_binding or export_match:
                                rel.annotations['resolved_member'] = member
                                rel.annotations['namespace'] = ns
                                rel.annotations['resolution'] = 'namespace_member'
                                if rel.relationship_type == RelationshipType.CALLS:
                                    rel.annotations['call_kind'] = 'namespace_function'
                    else:
                        rel.annotations['resolution'] = 'import_binding'
                        cand = candidates[0]
                        rel.annotations['resolved_symbol'] = cand.internal_symbol or cand.name
                        if candidates[0].is_default:
                            rel.annotations['default_export'] = True
                        if rel.relationship_type == RelationshipType.CALLS:
                            rel.annotations['call_kind'] = 'imported_function'
                enhanced.append(rel)
            else:
                enhanced.append(rel)

        # Fallback: resolve unresolved REFERENCES via global symbol registry.
        # This covers symbols that are used but weren't resolved through
        # import bindings (e.g. barrel re-exports, side-loaded globals).
        if global_symbol_registry:
            for rel in enhanced:
                if (rel.relationship_type == RelationshipType.REFERENCES
                        and rel.target_file is None
                        and '.' not in rel.target_symbol):
                    registry_fp = global_symbol_registry.get(rel.target_symbol)
                    if registry_fp and registry_fp != file_path:
                        rel.target_file = registry_fp
                        rel.annotations['resolution'] = 'global_registry'

        # Deduplicate
        seen = set()
        unique: List[Relationship] = []
        for rel in enhanced:
            key = (rel.relationship_type, rel.source_symbol, rel.target_symbol, rel.target_file)
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        result.relationships = unique

        # Second pass for method calls
        method_index = per_file_method_index.get(file_path, {})
        inheritance_map: Dict[str, str] = {}
        for r in result.relationships:
            if r.relationship_type == RelationshipType.INHERITANCE and r.source_file == file_path:
                inheritance_map[r.source_symbol] = r.target_symbol
        for r in result.relationships:
            if r.relationship_type != RelationshipType.CALLS:
                continue
            if '.' not in r.target_symbol:
                continue
            root, member = r.target_symbol.split('.', 1)
            if root == 'this':
                cls = r.source_symbol.split('.')[0]
                if cls in method_index and member in method_index[cls]:
                    r.annotations.setdefault('resolution', 'instance_method')
                    r.annotations['call_kind'] = 'instance_method'
                    r.annotations['resolved_member'] = member
                    r.annotations['resolved_class'] = cls
                else:
                    base = inheritance_map.get(cls)
                    if base:
                        for fp2, _res in all_results.items():
                            if fp2 == file_path:
                                continue
                            base_methods = per_file_method_index.get(fp2, {})
                            if base in base_methods and member in base_methods[base]:
                                r.annotations.setdefault('resolution', 'inherited_method')
                                r.annotations['call_kind'] = 'inherited_method'
                                r.annotations['resolved_member'] = member
                                r.annotations['resolved_class'] = base
                                r.target_file = fp2
                                break
            elif root == 'super':
                cls = r.source_symbol.split('.')[0]
                base = inheritance_map.get(cls)
                if base:
                    for fp2, _res in all_results.items():
                        base_methods = per_file_method_index.get(fp2, {})
                        if base in base_methods and member in base_methods[base]:
                            r.annotations.setdefault('resolution', 'super_method')
                            r.annotations['call_kind'] = 'super_method'
                            r.annotations['resolved_member'] = member
                            r.annotations['resolved_class'] = base
                            r.target_file = fp2
                            break
            else:
                if root in method_index and member in method_index[root]:
                    r.annotations.setdefault('resolution', 'class_method')
                    r.annotations['call_kind'] = 'class_method'
                    r.annotations['resolved_member'] = member
                    r.annotations['resolved_class'] = root
                else:
                    for fp2, cls_methods in per_file_method_index.items():
                        if root in cls_methods and member in cls_methods[root]:
                            r.annotations.setdefault('resolution', 'class_method')
                            r.annotations['call_kind'] = 'class_method'
                            r.annotations['resolved_member'] = member
                            r.annotations['resolved_class'] = root
                            if r.target_file is None:
                                r.target_file = fp2
                            break

    def _resolve_import_path(self, from_file: str, source_path: str, per_file_exports: Dict[str, List[ExportEntry]]) -> Optional[str]:
        # Only resolve relative paths for now ('./', '../')
        if not (source_path.startswith('./') or source_path.startswith('../')):
            return None
        base_dir = Path(from_file).parent
        candidate = (base_dir / source_path)
        # We'll test a few candidate path strings (raw and resolved) to maximize matching
        candidate_variants = [candidate]
        try:
            resolved_variant = candidate.resolve()
            if resolved_variant != candidate:
                candidate_variants.append(resolved_variant)
        except Exception:
            pass
        # Try with extensions on each variant
        # Try with extensions
        for cand in candidate_variants:
            if cand.is_file():
                return str(cand)
            for ext in ('.js', '.jsx', '.mjs', '.cjs'):
                p = cand.with_suffix(ext)
                if p.is_file():
                    return str(p)
            if cand.is_dir():
                for ext in ('.js', '.jsx'):
                    p = cand / f"index{ext}"
                    if p.is_file():
                        return str(p)
        return None


# NOTE: Intentionally NOT registering in global parser registry; integration is done directly in graph builder.
