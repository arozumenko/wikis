"""
Enhanced Unified Graph Builder - Single System for Parsing, Chunking, and Graph Building

This module implements the unified architecture that eliminates double parsing by combining:
1. Rich parsers (Java, Python) for comprehensive analysis 
2. Basic tree-sitter parsers (14+ languages) for symbol-level parsing
3. Multi-tier code_graph building (comprehensive + basic relationships)
4. Symbol-level chunking (no character splitting)

Replaces both GraphAwareCodeSplitter and separate UnifiedGraphBuilder usage.
"""

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple, Optional, Any, Union

import networkx as nx
# Document generation
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from ..code_splitter import GraphAwareCodeSplitter
from ..parsers.base_parser import ParseResult
from ..parsers.cpp_enhanced_parser import CppEnhancedParser
from ..parsers.csharp_visitor_parser import CSharpVisitorParser
from ..parsers.go_visitor_parser import GoVisitorParser
from ..parsers.java_visitor_parser import JavaVisitorParser
from ..parsers.rust_visitor_parser import RustVisitorParser
from ..parsers.javascript_visitor_parser import JavaScriptVisitorParser
from ..parsers.python_parser import PythonParser
from ..parsers.typescript_enhanced_parser import TypeScriptEnhancedParser
from ..utils.resource_monitor import resource_monitor

logger = logging.getLogger(__name__)

# =============================================================================
# Feature Flags for Doc/Code Separation
# =============================================================================
# When enabled, documentation files are NOT added to the graph (code-only graph)
# Docs still go to vector store for semantic retrieval
# This reduces RAM usage by ~3x (no duplicate doc storage in graph+dense+BM25)
SEPARATE_DOC_INDEX = os.getenv("WIKIS_DOC_SEPARATE_INDEX", "0") == "1"

# Documentation symbol types — single source of truth in constants.py
from ..constants import (
    DOC_SYMBOL_TYPES, ARCHITECTURAL_SYMBOLS,
    DOCUMENTATION_EXTENSIONS as _DOC_EXTENSIONS_MAP,
    KNOWN_FILENAMES as _KNOWN_FILENAMES_MAP,
)

# Log feature flag state at import time
if SEPARATE_DOC_INDEX:
    logger.info("[FEATURE FLAG] WIKIS_DOC_SEPARATE_INDEX=1: Docs routed to vector store only (not graph)")
else:
    logger.debug("[FEATURE FLAG] WIKIS_DOC_SEPARATE_INDEX=0: Docs included in graph (legacy mode)")

# Optional thinking emitter (safe no-op if outside task context)
try:  # pragma: no cover
    from tools import this  # type: ignore
except Exception:  # pragma: no cover
    this = None  # type: ignore

# ── AST node-type → canonical type mapping (basic-tier languages) ──────
# Raw tree-sitter node types (e.g. ``function_declaration``) need to be
# normalised to the canonical vocabulary used by the structure planner
# (``function``, ``method``, ``class``, ``struct``, …).  This map covers
# Go, Rust, C#, Ruby, Kotlin and Scala – the six basic-tier languages.
_BASIC_AST_TYPE_MAP: Dict[str, str] = {
    # ---------- Go ----------
    'function_declaration': 'function',
    'method_declaration':   'method',
    'struct_item':          'struct',       # refined by _classify_go_type_declaration
    'interface_item':       'interface',    # refined by _classify_go_type_declaration
    'type_alias_item':      'type_alias',   # refined by _classify_go_type_declaration
    'type_declaration':     'struct',       # fallback when classifier isn't applied
    'import_spec':          'import',
    # ---------- Rust ----------
    'function_item':        'function',
    'impl_item':            'class',        # impl blocks act as class-like containers
    'struct_item':          'struct',       # also used by Go above
    'enum_item':            'enum',
    'trait_item':           'trait',
    'mod_item':             'module',
    'use_declaration':      'import',
    # ---------- C# ----------
    'class_declaration':    'class',
    'method_declaration':   'method',       # also used by Go above
    'constructor_declaration': 'constructor',
    'interface_declaration': 'interface',
    'enum_declaration':     'enum',
    'struct_declaration':   'struct',
    'property_declaration': 'constant',
    'using_directive':      'import',
    # ---------- Ruby ----------
    'method':               'method',
    'singleton_method':     'method',
    'class':                'class',
    'module':               'module',
    'constant_assignment':  'constant',
    # ---------- Kotlin ----------
    'function_declaration': 'function',     # shared with Go (first wins in dict)
    'class_declaration':    'class',        # shared with C#
    'object_declaration':   'class',
    'companion_object':     'class',
    'property_declaration': 'constant',     # shared with C#
    'import_header':        'import',
    # ---------- Scala ----------
    'function_definition':  'function',
    'class_definition':     'class',
    'object_definition':    'class',
    'trait_definition':     'trait',
    'val_definition':       'constant',
    'import_expression':    'import',
    # ---------- identity (already canonical) ----------
    'function':   'function',
    'method':     'method',
    'class':      'class',
    'struct':     'struct',
    'interface':  'interface',
    'enum':       'enum',
    'trait':      'trait',
    'constant':   'constant',
    'type_alias': 'type_alias',
    'constructor': 'constructor',
    'module':     'module',
    'import':     'import',
}

# Node types that represent import/using statements — filtered from the
# basic-tier graph to reduce noise (import nodes inflate symbol counts
# without contributing meaningful structure).
_IMPORT_TYPES: frozenset = frozenset({
    'import', 'import_spec', 'import_declaration', 'import_header',
    'import_expression', 'using_directive', 'use_declaration',
    'import_statement', 'import_from_statement',
})


@dataclass
class BasicSymbol:
    """Basic symbol extracted from tree-sitter parsing with relative path support"""
    name: str
    symbol_type: str  # AST node type
    start_line: int
    end_line: int
    file_path: str
    rel_path: str               # NEW: Relative path from repository root
    language: str
    source_text: str = ""
    imports: Set[str] = field(default_factory=set)
    calls: Set[str] = field(default_factory=set)
    parent_symbol: Optional[str] = None
    docstring: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None


@dataclass
class BasicRelationship:
    """Basic relationship extracted from tree-sitter parsing"""
    source_symbol: str
    target_symbol: str
    relationship_type: str  # 'imports', 'calls', 'references'
    source_file: str
    target_file: Optional[str] = None


@dataclass
class BasicParseResult:
    """Result of basic tree-sitter parsing"""
    file_path: str
    language: str
    symbols: List[BasicSymbol]
    relationships: List[BasicRelationship]
    parse_level: str = 'basic'
    imports: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class UnifiedAnalysis:
    """Unified analysis result containing both chunks and graphs"""
    documents: List[Document] = field(default_factory=list)           # Symbol-level chunks
    documents_iter: Optional[Iterable[Document]] = None               # Optional streaming iterator
    unified_graph: Optional[nx.MultiDiGraph] = None                   # Multi-tier code_graph with preserved relationships
    cross_language_relationships: List[Any] = field(default_factory=list)
    language_stats: Dict[str, Any] = field(default_factory=dict)
    rich_languages: Set[str] = field(default_factory=set)             # Languages with rich analysis
    basic_languages: Set[str] = field(default_factory=set)            # Languages with basic analysis


class EnhancedUnifiedGraphBuilder:
    """
    Single system for parsing, chunking, and comprehensive code_graph building.
    
    Eliminates double parsing by using appropriate parser tier for each language
    and generating both chunk documents and relationship graphs from single parse.
    """
    
    # Code Splitter language mappings (copied from original)
    SUPPORTED_LANGUAGES = {
        '.py': 'python',
        '.js': 'javascript', 
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.cs': 'csharp',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.c++': 'cpp',
        '.h': 'cpp',      # C++ headers
        '.hpp': 'cpp',    # C++ headers
        '.hh': 'cpp',     # C++ headers
        '.hxx': 'cpp',    # C++ headers
        '.c': 'c',
        '.rb': 'ruby',
        '.php': 'php',
        '.rs': 'rust',
        '.kt': 'kotlin',
        '.scala': 'scala'
    }
    
    # Documentation file extensions — imported from constants.py (single source of truth)
    DOCUMENTATION_EXTENSIONS = _DOC_EXTENSIONS_MAP

    # Known filenames for extensionless / special-name files
    KNOWN_FILENAMES = _KNOWN_FILENAMES_MAP
    
    def __init__(self, max_workers: int = 4, debug_mode: bool = False):
        # Tier 1: Rich parsers for comprehensive analysis
        self.rich_parsers = {
            'cpp': CppEnhancedParser(),
            'csharp': CSharpVisitorParser(),
            'go': GoVisitorParser(),
            'java': JavaVisitorParser(),
            'python': PythonParser(),
            'javascript': JavaScriptVisitorParser(),
            'typescript': TypeScriptEnhancedParser(),
            'rust': RustVisitorParser(),
        }
        
        # Configuration
        self.max_workers = max_workers
        self.debug_mode = debug_mode
        
        # Initialize code splitter for basic parsing (provides tree-sitter parsers)
        self.code_splitter = GraphAwareCodeSplitter()
        
        # Track which languages use which analysis level
        self.rich_languages = set(self.rich_parsers.keys())
        
        # Basic languages are those supported by code splitter but not in rich parsers
        # Based on actual code splitter symbol extraction implementation
        # Only include languages that have working target_nodes and _get_symbol_name implementation
        supported_basic_languages = [
            'ruby', 'kotlin', 'scala'
        ]
        
        self.basic_languages = set()
        for lang in supported_basic_languages:
            if lang in self.code_splitter.parsers and lang not in self.rich_languages:
                self.basic_languages.add(lang)
        
        # Note: C, C++, and PHP are in SUPPORTED_LANGUAGES for file detection
        # but do NOT have symbol extraction implementation in the code splitter yet
        # They would fall back to regex-based processing in the code splitter
        
        # File-level imports storage (for basic parsers, like code splitter)
        self.file_imports = {}  # file_path -> set of file-level imports
        
        # Store last analysis result for statistics
        self.last_analysis: Optional[UnifiedAnalysis] = None
        
        # ARCHITECTURAL FILTERING: Only store TOP-LEVEL symbols in vector store
        # These define the code structure and public API surface.
        # 
        # INCLUDED (top-level only):
        # - class, interface, struct, enum, trait - type definitions
        # - function - standalone functions (NOT methods inside classes)
        # - constant - module-level constants (important config/API values)
        # - macro - C/C++ preprocessor definitions
        # - documentation files - README, configs, markdown, etc.
        #
        # EXCLUDED (belong inside their parent symbols):
        # - method, constructor, destructor - part of their parent class
        # - variable - local variables are implementation details  
        # - field - class fields are part of their parent class
        # - module, namespace - organizational only, minimal content
        # Single source of truth: plugin_implementation/constants.py
        self.ARCHITECTURAL_SYMBOLS = ARCHITECTURAL_SYMBOLS
        
        logger.info(f"Enhanced Unified Graph Builder initialized:")
        logger.info(f"  Rich parsers: {list(self.rich_languages)}")
        logger.info(f"  Basic parsers: {list(self.basic_languages)}")
        logger.info(f"  Code splitter parsers available: {list(self.code_splitter.parsers.keys())}")
        logger.info(f"  Languages without symbol extraction: C, C++, PHP (use regex fallback)")
        logger.info(f"  Architectural symbols only: {sorted(self.ARCHITECTURAL_SYMBOLS)}")

    def _detect_repo_root(self, file_path: str) -> str:
        """
        Detect repository root by looking for common repository indicators.
        Generic approach that works for any project type.
        
        Args:
            file_path: Path to any file in the repository
            
        Returns:
            Detected repository root path
        """
        current_dir = os.path.dirname(os.path.abspath(file_path))
        
        # Generic repository indicators (no hardcoded project assumptions)
        repo_indicators = {
            # Version Control
            '.git', '.hg', '.svn',
            # Package Managers  
            'package.json', 'Cargo.toml', 'pom.xml', 'go.mod', 'setup.py', 'pyproject.toml',
            # Build Systems
            'CMakeLists.txt', 'Makefile', 'build.gradle', 'mix.exs', 'dune-project',
            # Common Project Files
            'README.md', 'LICENSE', '.gitignore', 'Gemfile', 'composer.json'
        }
        
        # Walk up directory tree looking for repository indicators
        while current_dir != os.path.dirname(current_dir):  # Until we reach filesystem root
            # Check if any repository indicator exists in current directory
            for indicator in repo_indicators:
                if os.path.exists(os.path.join(current_dir, indicator)):
                    return current_dir
            
            # Move up one directory level
            current_dir = os.path.dirname(current_dir)
        
        # Fallback: use directory containing the file if no indicators found
        return os.path.dirname(os.path.abspath(file_path))

    def _calculate_relative_path(self, full_path: str, repo_root: str) -> str:
        """
        Calculate relative path from repository root using standard OS functions.
        
        Args:
            full_path: Absolute file path 
            repo_root: Repository root path
            
        Returns:
            Relative path from repository root
        """
        try:
            return os.path.relpath(full_path, repo_root)
        except (ValueError, OSError):
            # Fallback: return just the filename if path calculation fails
            return os.path.basename(full_path)

    def _is_architectural_symbol(self, symbol_type) -> bool:
        """Check if a symbol type should be included in architectural-level processing."""
        # Handle both string and SymbolType enum
        if hasattr(symbol_type, 'value'):
            symbol_type = symbol_type.value
        elif hasattr(symbol_type, 'lower'):
            # Already a string
            pass
        else:
            # Convert to string as fallback
            symbol_type = str(symbol_type)
        
        return symbol_type.lower() in self.ARCHITECTURAL_SYMBOLS
    
    def _is_package_or_namespace_parent(self, parent_symbol: str, parse_results: Dict[str, Union['ParseResult', 'BasicParseResult']]) -> bool:
        """
        Multi-language check if parent symbol represents a package/namespace (file-level parent) vs a class/function (nested entity).
        
        Language patterns:
        - Java: Top-level classes have parent_symbol = "com.example.package" (package name)
        - Python: Top-level classes have parent_symbol = None (file level)
        - C#: Top-level classes have parent_symbol = "MyNamespace" or "Company.Product.Module"
        - TypeScript: Top-level classes have parent_symbol = "mymodule" or None
        - Go: Top-level functions/structs have parent_symbol = "package_name" or None
        - Rust: Top-level items have parent_symbol = "crate_name" or "module_name"
        - Kotlin: Top-level classes have parent_symbol = "com.example.package"
        - Scala: Top-level classes have parent_symbol = "com.example.package"
        
        Returns True if parent represents a package/namespace (symbol should be included in embeddings)
        Returns False if parent represents a class/function (symbol should be excluded as nested)
        """
        if not parent_symbol:
            return True  # No parent = top-level (Python, JavaScript, etc.)
        
        # Strategy 1: Check if any symbol in parse_results has this parent_symbol as their name
        # If we find a namespace/package/module symbol with this name, then it's a valid package parent
        for file_path, result in parse_results.items():
            for symbol in result.symbols:
                # Get symbol type string consistently
                if hasattr(result, 'parse_level') and result.parse_level == 'comprehensive':
                    symbol_type_str = symbol.symbol_type.value if hasattr(symbol.symbol_type, 'value') else str(symbol.symbol_type)
                else:
                    # Handle both string and SymbolType enum for basic/documentation symbols
                    if hasattr(symbol.symbol_type, 'value'):
                        symbol_type_str = symbol.symbol_type.value
                    else:
                        symbol_type_str = str(symbol.symbol_type)
                
                # If we find a namespace/module/package symbol with the same name as parent_symbol
                if (symbol.name == parent_symbol and 
                    symbol_type_str.lower() in {'namespace', 'module', 'package', 'crate'}):
                    return True
        
        # Strategy 2: Multi-language pattern-based detection for package/namespace naming conventions
        if '.' in parent_symbol:
            # Dotted names typically indicate packages/namespaces in many languages
            parts = parent_symbol.split('.')
            
            # Check if it looks like a package/namespace (lowercase, underscores, no obvious class patterns)
            looks_like_package = all(
                part.islower() or part.isdigit() or '_' in part or part.replace('_', '').islower()
                for part in parts
            )
            
            # Common package/namespace prefixes across languages
            namespace_prefixes = {
                # Java/Kotlin/Scala
                'com', 'org', 'net', 'io', 'java', 'javax', 'android', 'kotlin', 'scala',
                # C#
                'system', 'microsoft', 'windows', 'web', 'data', 'collections',
                # General organizational
                'app', 'api', 'core', 'common', 'shared', 'lib', 'framework', 'utils'
            }
            
            if (looks_like_package and parts[0].lower() in namespace_prefixes):
                return True
        
        # Strategy 3: Language-specific single-word namespace detection
        # C#: "MyNamespace", Go: "main", "fmt", Rust: "std", "tokio"
        single_word_namespaces = {
            # C# common namespaces
            'system', 'microsoft', 'windows', 'collections', 'generic', 'linq',
            # Go standard packages
            'main', 'fmt', 'http', 'json', 'time', 'context', 'sync', 'os', 'io',
            # Rust crates/modules
            'std', 'core', 'alloc', 'tokio', 'serde', 'clap', 'regex',
            # TypeScript/JavaScript modules
            'react', 'express', 'lodash', 'moment', 'axios'
        }
        
        if (parent_symbol.lower() in single_word_namespaces):
            return True
        
        # Strategy 4: Detect obvious class/struct/interface names (these indicate nested entities)
        # Multi-language class naming patterns
        class_indicators = {
            # Starts with uppercase (Java, C#, Kotlin, Scala style)
            'starts_uppercase': parent_symbol and parent_symbol[0].isupper(),
            # Contains "Class", "Interface", "Struct", "Enum" etc.
            'has_type_suffix': any(suffix in parent_symbol.lower() for suffix in 
                                 ['class', 'interface', 'struct', 'enum', 'trait', 'impl']),
            # CamelCase pattern (common in OOP languages)
            'is_camel_case': (parent_symbol and 
                            parent_symbol[0].isupper() and 
                            any(c.isupper() for c in parent_symbol[1:]) and
                            not '.' in parent_symbol and
                            '_' not in parent_symbol),
            # Generic type indicators (List<T>, Map<K,V>, etc.)
            'has_generics': '<' in parent_symbol and '>' in parent_symbol
        }
        
        # If multiple class indicators suggest this is a class/struct/interface name
        class_score = sum(class_indicators.values())
        if class_score >= 2:  # At least 2 indicators suggest it's a class
            return False  # Likely nested entity
        
        # Strategy 5: Language-specific special cases
        # Handle language-specific patterns
        
        # Go: package names are typically lowercase, simple words
        if (parent_symbol.islower() and 
            len(parent_symbol) <= 20 and  # Reasonable package name length
            '_' not in parent_symbol and
            not any(c.isupper() for c in parent_symbol)):
            return True  # Likely Go package
        
        # Rust: module names are typically lowercase with underscores
        if (parent_symbol.islower() and 
            ('_' in parent_symbol or parent_symbol.isalpha()) and
            len(parent_symbol) <= 30):
            return True  # Likely Rust module
        
        # Default: err on the side of inclusion for architectural embeddings
        # Better to include a few extra symbols than miss top-level classes
        return True
    
    def _initialize_basic_parsers(self):
        """
        This method is no longer needed since we use the code splitter directly.
        The code splitter handles tree-sitter parser initialization.
        """
        pass
    
    def analyze_repository(self, repo_path: str, 
                          include_patterns: Optional[List[str]] = None,
                          exclude_patterns: Optional[List[str]] = None,
                          stream_documents: bool = False) -> UnifiedAnalysis:
        """
        Single pass analysis: parse once, generate chunks + multi-tier code_graph
        
        Args:
            repo_path: Path to the repository root
            include_patterns: File patterns to include
            exclude_patterns: File patterns to exclude
            
        Returns:
            Unified analysis with documents and code_graph from single parsing
        """
        logger.info(f"Starting unified repository analysis: {repo_path}")
        
        # Store repo_path for relative path calculations
        self.repo_path = repo_path
        
        if this and getattr(this, 'module', None):
            this.module.invocation_thinking("I am on phase parsing\nDiscovering files by language\nReasoning: Establish language clusters to select rich vs basic parsing strategies.\nNext: Parse each language set with appropriate tier.")
        
        # Phase 1: Discover files by language
        with resource_monitor("graph_builder.discover_files", logger):
            files_by_language = self._discover_files_by_language(repo_path, include_patterns, exclude_patterns)
        
        # Phase 2: Single parse with appropriate parser tier
        parse_results = {}
        
        with resource_monitor("graph_builder.parse_by_language", logger):
            for language, file_paths in files_by_language.items():
                if language == 'unknown':
                    continue
                logger.info(f"Parsing {len(file_paths)} {language} files")
                if this and getattr(this, 'module', None):
                    tier = 'rich' if language in self.rich_parsers else ('basic' if language in self.basic_languages else 'unsupported')
                    this.module.invocation_thinking(
                        f"I am on phase parsing\nParsing {len(file_paths)} {language} files using {tier} parser\nReasoning: Extract symbols + relationships to feed unified multi-tier graph.\nNext: Aggregate language results then construct composite graph."
                    )
                
                if language == 'documentation':
                    # Process documentation files with text chunking
                    doc_results = self._parse_documentation_files(file_paths, repo_path)
                    parse_results.update(doc_results)
                elif language in self.rich_parsers:
                    # Use rich parser for comprehensive analysis
                    rich_results = self._parse_with_rich_parser(language, file_paths)
                    parse_results.update(rich_results)
                elif language in self.basic_languages:
                    # Use basic tree-sitter parser (code splitter style)
                    basic_results = self._parse_with_basic_parser(language, file_paths)
                    parse_results.update(basic_results)
                else:
                    # Language not supported for symbol-level parsing yet
                    # (e.g., C, C++, PHP - they have tree-sitter parsers but no symbol extraction)
                    logger.warning(f"Language {language} has tree-sitter parser but no symbol extraction implementation yet")
                    logger.info(f"Files will be skipped: {file_paths}")
                    # These files would fall back to regex-based processing in the original code splitter
        
        # Phase 3: Build multi-tier unified code_graph
        with resource_monitor("graph_builder.build_graph", logger):
            unified_graph = self._build_multi_tier_graph(parse_results)
        
        # Phase 3.5: Build node index for O(1) lookups (for relationship hints)
        with resource_monitor("graph_builder.build_node_index", logger):
            self._build_node_index(unified_graph)
        
        # Phase 4: Generate symbol-level chunk documents (NO character splitting)
        # By default, clear parse results after generating documents to save RAM
        documents_iter = None
        with resource_monitor("graph_builder.generate_documents", logger):
            if stream_documents:
                if os.getenv("WIKIS_KEEP_PARSE_RESULTS") != "1":
                    documents_iter = self._iter_symbol_chunks_with_cleanup(parse_results)
                else:
                    documents_iter = self._iter_symbol_chunks(parse_results)
                chunk_documents = []
            else:
                chunk_documents = self._generate_symbol_chunks(parse_results)
                # Clear parse results after document generation to save RAM
                # (streaming path clears via _iter_symbol_chunks_with_cleanup)
                if os.getenv("WIKIS_KEEP_PARSE_RESULTS") != "1":
                    try:
                        parse_results.clear()
                    except Exception:
                        pass
        
        # Phase 5: Detect cross-language relationships (from original UnifiedGraphBuilder)
        with resource_monitor("graph_builder.cross_language_rels", logger):
            cross_lang_rels = self._detect_cross_language_relationships(files_by_language, parse_results)
        
        # Phase 6: Generate statistics
        with resource_monitor("graph_builder.language_stats", logger):
            language_stats = self._generate_language_stats(parse_results)
        
        analysis = UnifiedAnalysis(
            documents=chunk_documents,
            documents_iter=documents_iter,
            unified_graph=unified_graph,
            cross_language_relationships=cross_lang_rels,
            language_stats=language_stats,
            rich_languages=self.rich_languages,
            basic_languages=self.basic_languages
        )
        
        # Store analysis for statistics/debugging only when explicitly requested
        # By default, don't keep reference to save RAM
        keep_analysis = os.getenv("WIKIS_KEEP_LAST_ANALYSIS") == "1"
        if keep_analysis:
            self.last_analysis = analysis
            self._last_analysis = analysis
        else:
            self.last_analysis = None
            self._last_analysis = None
        
        logger.info(f"Unified analysis complete:")
        if stream_documents:
            logger.info("  Documents generated: streaming")
        else:
            logger.info(f"  Documents generated: {len(chunk_documents)}")
        logger.info(f"  Graph nodes: {unified_graph.number_of_nodes() if unified_graph else 0}")
        logger.info(f"  Graph edges: {unified_graph.number_of_edges() if unified_graph else 0}")
        logger.info(f"  Cross-language relationships: {len(cross_lang_rels)}")
        if this and getattr(this, 'module', None):
            this.module.invocation_thinking(
                f"I am on phase parsing\nParsing + graph build complete: {len(chunk_documents)} docs, {unified_graph.number_of_nodes() if unified_graph else 0} nodes\nReasoning: Unified multi-language graph ready for indexing & documentation phases.\nNext: Higher layers will perform retrieval + summarization."
            )

        # Clear code splitter caches by default to save RAM
        # Set WIKIS_KEEP_CODE_SPLITTER_CACHE=1 to keep for debugging
        if os.getenv("WIKIS_KEEP_CODE_SPLITTER_CACHE") != "1":
            try:
                self.code_splitter.symbol_table.clear()
                self.code_splitter.file_imports.clear()
                if hasattr(self.code_splitter, "call_graph"):
                    self.code_splitter.call_graph.clear()
                if hasattr(self.code_splitter, "import_graph"):
                    self.code_splitter.import_graph.clear()
                if hasattr(self.code_splitter, "combined_graph"):
                    self.code_splitter.combined_graph.clear()
            except Exception:
                pass
        
        return analysis
    
    def _discover_files_by_language(self, repo_path: str,
                                   include_patterns: Optional[List[str]] = None,
                                   exclude_patterns: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """Discover and categorize files by programming language"""
        files_by_language = defaultdict(list)
        
        # Default exclude patterns — imported from constants (single source of truth).
        # NOTE: Does NOT use '**/.*' which would block .github, .gitlab, and
        # known dot-files like .editorconfig / .gitignore.
        if exclude_patterns is None:
            from ..constants import DEFAULT_EXCLUDE_PATTERNS
            exclude_patterns = list(DEFAULT_EXCLUDE_PATTERNS)
        
        repo_path = Path(repo_path)
        
        # Walk through all files in the repository (sorted for deterministic ordering)
        for file_path in sorted(repo_path.rglob('*')):
            if file_path.is_file():
                # Check exclusions
                if self._should_exclude_file(str(file_path), exclude_patterns):
                    continue
                
                # Check inclusions
                if include_patterns and not self._should_include_file(str(file_path), include_patterns):
                    continue
                
                # Categorize by language based on file extension
                file_extension = file_path.suffix.lower()
                language = self.SUPPORTED_LANGUAGES.get(file_extension)
                doc_type = self.DOCUMENTATION_EXTENSIONS.get(file_extension)
                
                # Fallback: check known filenames for extensionless / special files
                if not language and not doc_type:
                    doc_type = self.KNOWN_FILENAMES.get(file_path.name)
                
                if language:
                    files_by_language[language].append(str(file_path))
                elif doc_type:
                    files_by_language['documentation'].append(str(file_path))
                else:
                    files_by_language['unknown'].append(str(file_path))
        
        return dict(files_by_language)
    
    def _should_exclude_file(self, file_path: str, exclude_patterns: List[str]) -> bool:
        """Check if a file should be excluded based on patterns"""
        from fnmatch import fnmatch
        
        for pattern in exclude_patterns:
            if fnmatch(file_path, pattern):
                return True
        return False
    
    def _should_include_file(self, file_path: str, include_patterns: List[str]) -> bool:
        """Check if a file should be included based on patterns"""
        from fnmatch import fnmatch
        
        for pattern in include_patterns:
            if fnmatch(file_path, pattern):
                return True
        return False
    
    def _parse_with_rich_parser(self, language: str, file_paths: List[str]) -> Dict[str, ParseResult]:
        """
        Parse files using rich parser for comprehensive analysis.
        
        CRITICAL: Uses the magic 'parse_multiple_files' method which is the KEY to cross-file
        relationship detection for both Java and Python rich parsers. This method:
        - Builds complete symbol table across ALL files first
        - Resolves cross-file inheritance, implements, imports relationships  
        - Provides the main "magic" for comprehensive repository-wide analysis
        
        This is what distinguishes rich parsers from basic tree-sitter parsing!
        """
        parser = self.rich_parsers[language]
        
        if hasattr(parser, 'parse_multiple_files'):
            # 🎯 THE MAGIC: Multi-file parsing for cross-file relationship resolution
            # This is what enables comprehensive inheritance, implements, and import analysis
            # across the entire repository for Java and Python
            results = parser.parse_multiple_files(file_paths, max_workers=self.max_workers)
            
            # Clear parser's global state after multi-file parsing to free memory
            # Rich parsers accumulate symbol registries that can be large for big repos
            self._clear_rich_parser_state(parser)
            
            return results
        else:
            # Fallback to individual file parsing (should not happen for Java/Python)
            logger.warning(f"Rich parser for {language} missing parse_multiple_files method, falling back to individual parsing")
            results = {}
            for file_path in file_paths:
                try:
                    result = parser.parse_file(file_path)
                    results[file_path] = result
                except Exception as e:
                    logger.warning(f"Failed to parse {file_path} with rich parser: {e}")
            return results
    
    def _clear_rich_parser_state(self, parser) -> None:
        """Clear rich parser's accumulated state to free memory after multi-file parsing."""
        try:
            # Python parser global registries
            if hasattr(parser, '_global_class_locations'):
                parser._global_class_locations.clear()
            if hasattr(parser, '_global_function_locations'):
                parser._global_function_locations.clear()
            if hasattr(parser, '_global_symbol_registry'):
                parser._global_symbol_registry.clear()
            
            # Java parser registries
            if hasattr(parser, '_class_locations'):
                parser._class_locations.clear()
            if hasattr(parser, '_interface_locations'):
                parser._interface_locations.clear()
            if hasattr(parser, '_symbol_registry'):
                parser._symbol_registry.clear()
            
            # TypeScript/JavaScript parser registries  
            if hasattr(parser, '_type_locations'):
                parser._type_locations.clear()
            if hasattr(parser, '_function_locations'):
                parser._function_locations.clear()

            # Go parser registries
            if hasattr(parser, '_global_type_registry'):
                parser._global_type_registry.clear()
            if hasattr(parser, '_global_function_registry'):
                parser._global_function_registry.clear()
            if hasattr(parser, '_global_method_registry'):
                parser._global_method_registry.clear()
            if hasattr(parser, '_global_interface_methods'):
                parser._global_interface_methods.clear()
                
        except Exception as e:
            logger.debug(f"Failed to clear rich parser state: {e}")
    
    def _parse_with_basic_parser(self, language: str, file_paths: List[str]) -> Dict[str, BasicParseResult]:
        """
        Parse files using basic tree-sitter parser (code splitter style).
        Uses the exact same logic as the Code Splitter to ensure consistency.
        Now includes relative path calculation for all symbols.
        """
        results = {}
        
        # Detect repository root once for all files
        repo_root = ""
        if file_paths:
            repo_root = self._detect_repo_root(file_paths[0])
            logger.info(f"Detected repo root {repo_root}")
        
        for file_path in file_paths:
            try:
                symbols, relationships = self._parse_with_basic_parser_single_file(file_path, language, repo_root)
                rel_path = self._calculate_relative_path(file_path, repo_root)
                
                # Convert to BasicParseResult format by converting dict symbols to BasicSymbol objects
                basic_symbols = []
                basic_relationships = []
                
                for symbol_dict in symbols:
                    # Include ALL symbols (no filtering at parser level)
                    basic_symbol = BasicSymbol(
                        name=symbol_dict['name'],
                        symbol_type=symbol_dict['type'],
                        start_line=symbol_dict['start_line'],
                        end_line=symbol_dict['end_line'],
                        file_path=symbol_dict['file_path'],
                        rel_path=rel_path,  # Add relative path to all symbols
                        language=symbol_dict['language'],
                        source_text=symbol_dict['content'],
                        imports=set(symbol_dict.get('imports', [])),
                        calls=set(symbol_dict.get('calls', [])),
                        docstring=symbol_dict.get('docstring'),
                        parameters=symbol_dict.get('parameters', []),
                        return_type=symbol_dict.get('return_type')
                    )
                    basic_symbols.append(basic_symbol)
                
                for rel_dict in relationships:
                    basic_rel = BasicRelationship(
                        source_symbol=rel_dict['source'],
                        target_symbol=rel_dict['target'],
                        relationship_type=rel_dict['type'],
                        source_file=rel_dict['source_file'],
                        target_file=rel_dict.get('target_file')
                    )
                    basic_relationships.append(basic_rel)
                
                # Convert to BasicParseResult format
                result = BasicParseResult(
                    file_path=file_path,
                    language=language,
                    symbols=basic_symbols,
                    relationships=basic_relationships,
                    parse_level='basic',
                    imports=list(self.code_splitter.file_imports.get(file_path, []))
                )
                results[file_path] = result
                
            except Exception as e:
                logger.warning(f"Failed to parse {file_path} with basic parser: {e}")
        
        return results
    
    def _parse_with_basic_parser_single_file(self, file_path: str, language: str, repo_root: str = "") -> Tuple[List[Dict], List[Dict]]:
        """
        Parse single file using basic Tree-sitter parser for symbol extraction and basic code_graph building.
        Uses the exact same logic as the Code Splitter to ensure consistency.
        
        Args:
            file_path: Absolute path to the file to parse
            language: Programming language of the file
            repo_root: Repository root directory for calculating relative paths
        """
        if language not in self.code_splitter.SUPPORTED_LANGUAGES.values():
            logger.warning(f"Language {language} not supported by basic parser")
            return [], []
        
        if language not in self.code_splitter.parsers:
            logger.warning(f"No Tree-sitter parser available for {language}")
            return [], []
        
        try:
            # Clear the code splitter's state for this parsing session
            self.code_splitter.symbol_table.clear()
            self.code_splitter.file_imports.clear()
            
            # Calculate relative path from repository root
            rel_path = self._calculate_relative_path(file_path, repo_root) if repo_root else os.path.basename(file_path)
            logger.info(f"Detected file relative path - {rel_path} for {file_path}")
            documents = self.code_splitter._process_code_file(file_path, rel_path)
            
            symbols = []
            relationships = []
            
            # Convert documents to our symbol format
            for doc in documents:
                metadata = doc.metadata
                if metadata.get('chunk_type') == 'code' and 'symbol' in metadata:
                    symbol_dict = {
                        'name': metadata['symbol'],
                        'type': metadata.get('node_type', metadata.get('symbol_type', 'unknown')),
                        'start_line': metadata.get('start_line', 0),
                        'end_line': metadata.get('end_line', 0),
                        'file_path': rel_path,  # Use relative path instead of absolute path
                        'language': language,
                        'content': doc.page_content,
                        'imports': metadata.get('imports', []),
                        'calls': metadata.get('calls', []),
                        'parameters': metadata.get('parameters', []),
                        'return_type': metadata.get('return_type'),
                        'docstring': metadata.get('docstring')
                    }
                    symbols.append(symbol_dict)
                    
                    # Create relationships for imports and calls
                    for imported in metadata.get('imports', []):
                        relationships.append({
                            'source': metadata['symbol'],
                            'target': imported,
                            'type': 'imports',
                            'source_file': rel_path,  # Use relative path
                            'target_file': None  # Will be resolved later
                        })
                    
                    for called in metadata.get('calls', []):
                        relationships.append({
                            'source': metadata['symbol'],
                            'target': called,
                            'type': 'calls',
                            'source_file': rel_path,  # Use relative path
                            'target_file': None  # Will be resolved later
                        })
            
            # Add file-level imports as relationships
            if file_path in self.code_splitter.file_imports:
                file_imports = self.code_splitter.file_imports[file_path]
                for imported in file_imports:
                    relationships.append({
                        'source': f"file:{Path(rel_path).stem}",  # Use relative path
                        'target': imported,
                        'type': 'file_imports',
                        'source_file': rel_path,  # Use relative path
                        'target_file': None
                    })
            
            return symbols, relationships
            
        except Exception as e:
            logger.error(f"Error parsing {file_path} with basic parser: {e}")
            return [], []
    
    def _build_multi_tier_graph(self, parse_results: Dict[str, Union[ParseResult, BasicParseResult]]) -> nx.MultiDiGraph:
        """Build unified code_graph with both comprehensive and basic relationship levels.
        
        When WIKIS_DOC_SEPARATE_INDEX=1, documentation files are excluded from the graph
        to reduce RAM usage. Docs still go to vector store via _generate_symbol_chunks.
        """
        # Use MultiDiGraph to preserve multiple relationship types between same nodes
        graph = nx.MultiDiGraph()
        graph.graph['analysis_type'] = 'multi_tier'
        
        # Filter out documentation results when SEPARATE_DOC_INDEX is enabled
        # This keeps graph code-only for vector store, reducing RAM by ~3x.
        # However, doc nodes are ALWAYS added to the NX graph (third pass below)
        # so they enter the unified DB for clustering and wiki structure.
        if SEPARATE_DOC_INDEX:
            code_results = {
                path: result for path, result in parse_results.items()
                if not (hasattr(result, 'parse_level') and result.parse_level == 'documentation')
            }
            # Keep doc results available for the graph third-pass (unified DB needs them)
            doc_results_separated = {
                path: result for path, result in parse_results.items()
                if hasattr(result, 'parse_level') and result.parse_level == 'documentation'
            }
            logger.info(
                f"[SEPARATE_DOC_INDEX] {len(doc_results_separated)} doc files excluded from "
                f"vector store but will be added to graph for unified DB"
            )
            parse_results_for_graph = code_results
        else:
            doc_results_separated = {}
            parse_results_for_graph = parse_results
        
        # Separate rich, basic, and documentation parse results
        rich_results = {path: result for path, result in parse_results_for_graph.items() 
                       if not hasattr(result, 'parse_level') or result.parse_level not in ('basic', 'documentation')}
        basic_results = {path: result for path, result in parse_results_for_graph.items() 
                        if hasattr(result, 'parse_level') and result.parse_level == 'basic'}
        # Merge: docs from parse_results_for_graph (SEPARATE_DOC_INDEX=0) + separated docs
        doc_results = {path: result for path, result in parse_results_for_graph.items()
                      if hasattr(result, 'parse_level') and result.parse_level == 'documentation'}
        doc_results.update(doc_results_separated)
        
        logger.info(f"Building multi-tier code_graph: {len(rich_results)} rich, {len(basic_results)} basic, {len(doc_results)} doc")
        
        # Add rich language graphs (comprehensive relationships)
        # Group by language first to avoid processing the same language multiple times
        rich_languages_found = {}
        for file_path, result in rich_results.items():
            language = result.language
            if language not in rich_languages_found:
                rich_languages_found[language] = []
            rich_languages_found[language].append((file_path, result))
        
        # --- Progress instrumentation (percentage by language groups) ---
        # Count total language groups to process (rich + basic present)
        rich_group_count = len(rich_languages_found)
        basic_group_count = sum(1 for language in self.basic_languages 
                                if any(res.language == language for res in basic_results.values()))
        total_groups = rich_group_count + basic_group_count if (rich_group_count + basic_group_count) > 0 else 1
        completed_groups = 0
        if this and getattr(this, 'module', None):
            this.module.invocation_thinking(
                f"I am on phase parsing\nGraph build: initializing unified multi-language graph ({total_groups} language groups)"
            )

        # Process each rich language once
        for language, file_results in rich_languages_found.items():
            lang_results = {fp: res for fp, res in file_results}
            file_count = len(lang_results)
            if lang_results:
                if this and getattr(this, 'module', None):
                    pct = int((completed_groups/total_groups)*100) if total_groups else 100
                    this.module.invocation_thinking(
                        f"I am on phase parsing\nGraph build progress: {pct}% (processing {language}, {file_count} files)"
                    )
                lang_graph = self._build_comprehensive_language_graph(lang_results, language)
                self._merge_graph(graph, lang_graph)
                completed_groups += 1
                if this and getattr(this, 'module', None):
                    pct = int((completed_groups/total_groups)*100) if total_groups else 100
                    this.module.invocation_thinking(
                        f"I am on phase parsing\nGraph build progress: {pct}% (completed {language})"
                    )

        # Add basic language graphs (code splitter style relationships)
        for language in self.basic_languages:
            lang_basic_results = {fp: res for fp, res in basic_results.items() if res.language == language}
            if lang_basic_results:
                file_count = len(lang_basic_results)
                if this and getattr(this, 'module', None):
                    pct = int((completed_groups/total_groups)*100) if total_groups else 100
                    this.module.invocation_thinking(
                        f"I am on phase parsing\nGraph build progress: {pct}% (processing {language}, {file_count} files)"
                    )
                basic_graph = self._build_basic_language_graph(lang_basic_results, language)
                self._merge_graph(graph, basic_graph)
                completed_groups += 1
                if this and getattr(this, 'module', None):
                    pct = int((completed_groups/total_groups)*100) if total_groups else 100
                    this.module.invocation_thinking(
                        f"I am on phase parsing\nGraph build progress: {pct}% (completed {language})"
                    )

        # Add documentation nodes to the graph (README, markdown, configs, etc.)
        # Without this, doc files are parsed but never added to the graph,
        # so they don't make it into the unified DB, clustering, or expansion.
        if doc_results:
            doc_node_count = 0
            for file_path, result in doc_results.items():
                file_name = Path(file_path).stem
                doc_lang = result.language  # e.g. 'markdown', 'yaml', 'text'
                for symbol in result.symbols:
                    node_id = f"{doc_lang}::{file_name}::{symbol.name}"
                    # Handle duplicates
                    base_node_id = node_id
                    counter = 1
                    while node_id in graph.nodes:
                        node_id = f"{base_node_id}#{counter}"
                        counter += 1

                    symbol_type_str = symbol.symbol_type
                    if hasattr(symbol_type_str, 'value'):
                        symbol_type_str = symbol_type_str.value
                    symbol_type_str = str(symbol_type_str).lower()

                    rel_path = getattr(symbol, 'rel_path', '') or file_path

                    node_attrs = {
                        'name': symbol.name,
                        'type': symbol_type_str,
                        'symbol_name': symbol.name,
                        'symbol_type': symbol_type_str,
                        'file_path': file_path,
                        'rel_path': rel_path,
                        'file_name': file_name,
                        'language': doc_lang,
                        'start_line': symbol.start_line,
                        'end_line': symbol.end_line,
                        'analysis_level': 'documentation',
                        'source_text': symbol.source_text or '',
                        'docstring': symbol.docstring or '',
                        'parameters': symbol.parameters if hasattr(symbol, 'parameters') else [],
                        'return_type': symbol.return_type or '' if hasattr(symbol, 'return_type') else '',
                    }
                    graph.add_node(node_id, **node_attrs)
                    doc_node_count += 1

            logger.info(
                "Added %d documentation nodes from %d files to graph",
                doc_node_count, len(doc_results),
            )

        # Final completion message (force 100%)
        if this and getattr(this, 'module', None):
            this.module.invocation_thinking(
                "I am on phase parsing\nGraph build progress: 100% (all language groups merged)"
            )
        # Summary with counts
        if this and getattr(this, 'module', None):
            try:
                node_count = graph.number_of_nodes()
                edge_count = graph.number_of_edges()
                this.module.invocation_thinking(
                    f"I am on phase parsing\nGraph build summary: {node_count} nodes, {edge_count} relationships added"
                )
            except Exception:
                pass

        # Build graph lookup indexes for O(1) resolution (safe if missing fields)
        try:
            self._build_graph_indexes(graph)
        except Exception as exc:
            logger.warning(f"Failed to build graph lookup indexes: {exc}")
        
        return graph

    @staticmethod
    def _simple_symbol_name(name: str) -> str:
        if not name:
            return ""
        return name.split('.')[-1].split('::')[-1]

    @staticmethod
    def _get_type_priority() -> Dict[str, int]:
        return {
            # Top-level architectural types (highest priority - these define system structure)
            'class': 10,
            'interface': 10,
            'trait': 10,             # Rust, Scala
            'protocol': 10,          # Swift
            'enum': 9,
            'struct': 9,             # C, C++, Rust, Go
            'record': 9,             # Java 14+, C#
            'data_class': 9,         # Kotlin
            'object': 9,             # Kotlin/Scala singleton objects
            'module': 8,             # Python, TypeScript modules
            'namespace': 8,          # C++, C#
            # Top-level constructs (medium-high priority)
            'function': 7,           # Top-level functions
            'constant': 6,           # Module-level constants
            'type_alias': 6,         # Type definitions
            'annotation': 6,         # Java annotations
            'decorator': 6,          # Python decorators
            'macro': 6,              # Rust macros, C preprocessor
            # Implementation details (low priority - accessed via parent)
            'method': 3,
            'constructor': 2,        # Explicitly lower than class (same name as class)
            'field': 2,
            'property': 2,
            'parameter': 1,
            'variable': 1,
            'local_variable': 1,
            'argument': 1,
            # Unknown/fallback
            'unknown': 0
        }

    def _build_graph_indexes(self, graph: nx.MultiDiGraph) -> None:
        """
        Build lookup indexes on the graph for O(1) resolution in content expansion.
        Indexes are additive and safe for mixed comprehensive/basic nodes.
        """
        if not graph:
            return

        type_priority = self._get_type_priority()

        graph._node_index = {}
        graph._simple_name_index = {}
        graph._full_name_index = {}
        graph._name_index = defaultdict(list)
        graph._suffix_index = defaultdict(list)
        graph._decl_impl_index = defaultdict(list)
        graph._constant_def_index = defaultdict(list)

        def _symbol_type_of(node_id: str) -> str:
            node_data = graph.nodes.get(node_id, {})
            symbol_type_raw = node_data.get('symbol_type') or node_data.get('type') or 'unknown'
            return symbol_type_raw.value.lower() if hasattr(symbol_type_raw, 'value') else str(symbol_type_raw).lower()

        def _maybe_set(index: Dict, key, node_id: str) -> None:
            if key in index:
                existing_node_id = index[key]
                existing_type = _symbol_type_of(existing_node_id)
                new_type = _symbol_type_of(node_id)
                if type_priority.get(new_type, 0) > type_priority.get(existing_type, 0):
                    index[key] = node_id
            else:
                index[key] = node_id

        for node_id, node_data in graph.nodes(data=True):
            symbol_name = node_data.get('symbol_name') or node_data.get('name') or ''
            file_path = node_data.get('file_path', '')
            language = (node_data.get('language') or '').lower()

            if symbol_name and file_path and language:
                _maybe_set(graph._node_index, (symbol_name, file_path, language), node_id)

                simple_name = self._simple_symbol_name(symbol_name)
                if simple_name:
                    _maybe_set(graph._simple_name_index, (simple_name, file_path, language), node_id)

                # Full name index (prefer parser-provided full_name)
                full_name = node_data.get('full_name')
                if not full_name:
                    symbol_obj = node_data.get('symbol')
                    full_name = getattr(symbol_obj, 'full_name', None) if symbol_obj else None
                if full_name:
                    _maybe_set(graph._full_name_index, full_name, node_id)

                graph._name_index[symbol_name].append(node_id)

                # Suffix index from node_id (after language::file::)
                parts = node_id.split('::', 2)
                suffix = parts[2] if len(parts) == 3 else node_id
                graph._suffix_index[suffix].append(node_id)

            # Constant definition index (best-effort, refined via edges below)
            symbol_type = _symbol_type_of(node_id)
            if symbol_name and symbol_type == 'constant':
                graph._constant_def_index[symbol_name].append(node_id)

        # Build decl/impl index from edges (implementation -> declaration)
        for source, target, edge_data in graph.edges(data=True):
            rel_type = edge_data.get('relationship_type') or edge_data.get('type') or ''
            if str(rel_type).lower() != 'defines_body':
                continue

            graph._decl_impl_index[target].append(source)

            # If this links constants, include in constant definition index
            target_type = _symbol_type_of(target)
            source_type = _symbol_type_of(source)
            if target_type == 'constant' or source_type == 'constant':
                source_name = graph.nodes.get(source, {}).get('symbol_name') or graph.nodes.get(source, {}).get('name')
                if source_name:
                    graph._constant_def_index[source_name].append(source)

        # Sort list-based indexes by type priority (highest first)
        def _priority(node_id: str) -> int:
            return type_priority.get(_symbol_type_of(node_id), 0)

        for key, nodes in graph._name_index.items():
            nodes.sort(key=_priority, reverse=True)

        for key, nodes in graph._suffix_index.items():
            nodes.sort(key=_priority, reverse=True)
    
    def _build_comprehensive_language_graph(self, parse_results: Dict[str, ParseResult], language: str) -> nx.MultiDiGraph:
        """Build comprehensive code_graph for rich parsers with sequential processing"""
        
        start_time = time.time()
        logger.info(f"Building comprehensive {language} code_graph for {len(parse_results)} files...")
        
        # Analyze potential symbol loss before processing
        symbol_analysis = self._analyze_symbol_loss(parse_results, language)
        
        # Use MultiDiGraph to preserve all relationship types
        graph = nx.MultiDiGraph()
        graph.graph['language'] = language
        graph.graph['analysis_level'] = 'comprehensive'
        
        # Use synchronous graph building (no event loop needed)
        symbol_registry = self._build_graph_sync(graph, parse_results, language, symbol_analysis)
        
        build_time = time.time() - start_time
        logger.info(f"Comprehensive {language} code_graph built in {build_time:.2f}s: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        return graph
    
    def _analyze_symbol_loss(self, parse_results: Dict[str, ParseResult], language: str = "python") -> Dict[str, Any]:
        """Analyze where symbols are being lost during processing"""
        analysis = {
            'total_expected': 0,
            'valid_symbols': 0,  # Only count symbols that will actually be added
            'files_analyzed': 0,
            'symbols_per_file': {},
            'duplicate_definitions': {},  # Only legitimate duplicates (same symbol defined multiple times in same file)
            'empty_symbols': 0,
            'symbols_with_issues': []
        }
        
        # Track symbol definitions (not references/usage)
        symbol_definitions = {}  # symbol_name -> [(file_path, symbol_type)]
        processed_symbols = set()  # Track globally unique node IDs
        
        for file_path, result in parse_results.items():
            file_name = Path(file_path).stem
            symbol_count = len(result.symbols)
            valid_count = 0
            
            analysis['total_expected'] += symbol_count
            analysis['files_analyzed'] += 1
            analysis['symbols_per_file'][file_name] = symbol_count
            
            for symbol in result.symbols:
                if not symbol.name or symbol.name.strip() == "":
                    analysis['empty_symbols'] += 1
                    analysis['symbols_with_issues'].append(f"Empty symbol name in {file_name}")
                    continue
                
                # Create the same node_id as in _process_file_symbols
                node_id = f"{language}::{file_name}::{symbol.name}"
                
                # Only count each unique node_id once
                if node_id not in processed_symbols:
                    valid_count += 1
                    processed_symbols.add(node_id)
                    
                    # Track symbol definitions for legitimate duplicate detection
                    # Use full node_id to properly distinguish symbols from different files
                    symbol_key = f"{file_name}::{symbol.name}"  # Include file context
                    if symbol_key not in symbol_definitions:
                        symbol_definitions[symbol_key] = []
                    symbol_definitions[symbol_key].append((file_path, symbol.symbol_type.value if hasattr(symbol.symbol_type, 'value') else str(symbol.symbol_type)))
            
            analysis['symbols_per_file'][f"{file_name}_valid"] = valid_count
        
        analysis['valid_symbols'] = len(processed_symbols)
        
        # Identify legitimate duplicates (same symbol defined multiple times in the SAME file)
        for symbol_key, definitions in symbol_definitions.items():
            if len(definitions) > 1:
                # Group by symbol type
                by_type = {}
                for file_path, symbol_type in definitions:
                    if symbol_type not in by_type:
                        by_type[symbol_type] = []
                    by_type[symbol_type].append(Path(file_path).stem)
                
                # Only report as duplicate if same type defined multiple times in same context
                for symbol_type, files in by_type.items():
                    if len(files) > 1:
                        # Extract the symbol name from the key
                        symbol_name = symbol_key.split('::')[-1]
                        analysis['duplicate_definitions'][f"{symbol_name}({symbol_type})"] = files
        
        logger.info(f"SYMBOL LOSS ANALYSIS:")
        logger.info(f"  Total expected symbols: {analysis['total_expected']}")
        logger.info(f"  Valid symbols (will be added): {analysis['valid_symbols']}")
        logger.info(f"  Files analyzed: {analysis['files_analyzed']}")
        logger.info(f"  Empty symbol names: {analysis['empty_symbols']}")
        logger.info(f"  Legitimate duplicate definitions: {len(analysis['duplicate_definitions'])}")
        
        if analysis['duplicate_definitions']:
            logger.warning(f"Legitimate duplicate definitions found (same symbol multiple times in same file): {list(analysis['duplicate_definitions'].items())[:5]}")
        
        return analysis

    def _build_graph_sync(self, graph: nx.MultiDiGraph, parse_results: Dict[str, ParseResult], 
                               language: str, symbol_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Synchronous code_graph building with parallel relationship processing"""
        # Enhanced symbol registry for better cross-file resolution
        symbol_registry = {
            'by_name': {},           # symbol_name -> [list of node_ids]
            'by_qualified_name': {}, # file.symbol_name -> node_id  
            'by_full_path': {}       # file_path::symbol_name -> node_id
        }
        
        # Phase 1: Add all symbols as nodes (batched)
        logger.info(f"Phase 1: Adding symbols as nodes...")
        self._add_symbols_batch(graph, parse_results, language, symbol_registry, symbol_analysis)
        
        # Phase 2: Add relationships as edges (bulk operation with threading)
        logger.info(f"Phase 2: Adding relationships as edges...")
        self._add_relationships_bulk(graph, parse_results, language, symbol_registry)
        
        return symbol_registry
    
    def _add_symbols_batch(self, graph: nx.MultiDiGraph, parse_results: Dict[str, ParseResult], 
                                language: str, symbol_registry: Dict[str, Any], symbol_analysis: Dict[str, Any]):
        """Add symbols as nodes synchronously"""
        file_items = list(parse_results.items())
        
        # Use stored repo_path if available, otherwise detect from files
        repo_root = ""
        if hasattr(self, 'repo_path') and self.repo_path:
            repo_root = self.repo_path
        elif file_items:
            repo_root = self._detect_repo_root(file_items[0][0])
        
        # Use valid symbols count from analysis
        total_symbols_expected = symbol_analysis['valid_symbols']
        
        logger.info(f"Expected to add {total_symbols_expected} symbols to code_graph")
        
        # Process all files synchronously (fast enough for node creation)
        for i, (file_path, result) in enumerate(file_items):
            self._process_file_symbols(graph, file_path, result, language, symbol_registry, repo_root)
            
            # Progress logging every 500 files
            if (i + 1) % 500 == 0:
                symbols_added = graph.number_of_nodes()
                logger.info(f"Processed {i + 1}/{len(file_items)} files (symbols)")
                logger.info(f"Symbols in code_graph so far: {symbols_added}/{total_symbols_expected}")
        
        final_symbols = graph.number_of_nodes()
        logger.info(f"SYMBOL SUMMARY: Expected {total_symbols_expected}, Added {final_symbols}, Loss: {total_symbols_expected - final_symbols}")
    
    def _get_local_qualified_name(self, symbol: Any, file_name: str) -> str:
        """
        Get local qualified name for a symbol to use in node ID.
        
        For nested symbols, includes parent context to disambiguate.
        Example: AppConfig.__init__ vs DatabaseConfig.__init__
        
        Uses full_name minus the module prefix to get the local qualified path.
        All parsers now use '.' separator in full_name (standardized).
        
        Node ID uniqueness strategy:
        - file_name is already in the node_id: language::file_name::local_name
        - So if full_name = "file_name.X.Y", we want local_name = "X.Y" (strip file_name prefix)
        - MODULE/NAMESPACE symbols are skipped at graph level, so no collision handling needed
        
        Args:
            symbol: Symbol object with name, full_name, parent_symbol
            file_name: File stem (module name)
            
        Returns:
            Local qualified name for node ID (e.g., "AppConfig.__init__", "AppConfig.database")
        """
        full_name = getattr(symbol, 'full_name', None)
        
        if full_name:
            parts = full_name.split('.')
            
            # Strip module prefix if present
            # If full_name starts with file_name (module), remove it to avoid redundancy
            # e.g., "User.User.name" with file=User -> "User.name" (class.field)
            # e.g., "test.DatabaseConfig" with file=test -> "DatabaseConfig" (class)
            if len(parts) >= 2 and parts[0] == file_name:
                local_parts = parts[1:]
            else:
                local_parts = parts
            
            if local_parts:
                return '.'.join(local_parts)
        
        # Fallback to simple name
        return symbol.name
    
    def _process_file_symbols(self, graph: nx.MultiDiGraph, file_path: str, result: ParseResult,
                                   language: str, symbol_registry: Dict[str, Any], repo_root: str = ""):
        """Process symbols from a single file with relative path support"""
        file_name = Path(file_path).stem
        rel_path = self._calculate_relative_path(file_path, repo_root) if repo_root else file_path
        
        for symbol in result.symbols:
            # Skip symbols with empty names
            if not symbol.name or symbol.name.strip() == "":
                continue
            
            # Safety check: Skip MODULE/NAMESPACE symbols if any parser still emits them
            # (Parsers should no longer emit these after standardization cleanup)
            symbol_type_str = symbol.symbol_type.value if hasattr(symbol.symbol_type, 'value') else str(symbol.symbol_type)
            if symbol_type_str.lower() in ('module', 'namespace'):
                continue
            
            # Build qualified local name for node ID
            # For nested symbols, include parent context: Parent.symbol
            # This disambiguates e.g., AppConfig.__init__ vs DatabaseConfig.__init__
            local_qualified_name = self._get_local_qualified_name(symbol, file_name)
            node_id = f"{language}::{file_name}::{local_qualified_name}"
            
            # Only skip if this EXACT node already exists (same file + same symbol)
            # Do NOT skip symbols with same name from different files - those are legitimate!
            if graph.has_node(node_id):
                existing_file = graph.nodes[node_id].get('file_path', '')
                if existing_file == file_path:
                    # True duplicate from same file, skip
                    continue
                else:
                    # Different file (e.g., declaration vs implementation)
                    # Create unique node_id by appending file hash
                    file_hash = hash(file_path) & 0xFFFF
                    node_id = f"{node_id}_{file_hash:04x}"
            
            # Extract line numbers from Range object for flat access
            _range = getattr(symbol, 'range', None)
            _start_line = _range.start.line if _range else 0
            _end_line = _range.end.line if _range else 0

            # Add symbol to graph
            graph.add_node(node_id,
                         symbol=symbol,
                         file_path=file_path,
                         rel_path=rel_path,          # Add relative path
                         file_name=file_name,
                         language=language,
                         symbol_type=symbol.symbol_type.value if hasattr(symbol.symbol_type, 'value') else str(symbol.symbol_type),
                         symbol_name=symbol.name,
                         start_line=_start_line,
                         end_line=_end_line,
                         parent_symbol=getattr(symbol, 'parent_symbol', None),
                         analysis_level='comprehensive')
            
            # Register symbol in enhanced registry for better cross-file resolution
            # 1. By simple name (list of all instances)
            if symbol.name not in symbol_registry['by_name']:
                symbol_registry['by_name'][symbol.name] = []
            symbol_registry['by_name'][symbol.name].append(node_id)
            
            # 2. By local qualified name (Parent.symbol) - for relationship resolution
            symbol_registry['by_qualified_name'][local_qualified_name] = node_id
            
            # 3. By full qualified name (file.Parent.symbol) - for cross-file resolution
            full_name = getattr(symbol, 'full_name', None)
            if full_name:
                symbol_registry['by_qualified_name'][full_name] = node_id
            
            # 4. By full path (file_path::symbol)
            full_path_name = f"{file_path}::{symbol.name}"
            symbol_registry['by_full_path'][full_path_name] = node_id
    
    def _build_node_lookup_cache(self, graph: nx.MultiDiGraph) -> Dict[str, bool]:
        """
        Build a fast O(1) lookup cache for all nodes in the graph.
        This eliminates expensive graph.has_node() calls during relationship resolution.
        Returns a set-like dict for O(1) membership testing.
        """
        return {node: True for node in graph.nodes()}
    
    def _add_relationships_bulk(self, graph: nx.MultiDiGraph, parse_results: Dict[str, ParseResult],
                               language: str, symbol_registry: Dict[str, Any]):
        """Add relationships as edges using bulk NetworkX operation (sequential preparation, bulk addition)"""
        
        total_relationships_expected = sum(len(result.relationships) for result in parse_results.values())
        logger.info(f"Expected to add {total_relationships_expected} relationships to code_graph")
        
        start_time = time.time()
        
        # Build fast lookup cache of all existing nodes (O(n) once instead of O(n) per relationship)
        logger.info("Building node lookup cache for fast resolution...")
        cache_start = time.time()
        node_cache = self._build_node_lookup_cache(graph)
        cache_time = time.time() - cache_start
        logger.info(f"Node cache built with {len(node_cache)} nodes in {cache_time:.2f}s")
        
        # Collect all edges to add in bulk (sequential to avoid thread-safety issues)
        edges_to_add = []
        relationships_attempted = 0
        relationships_skipped = 0
        
        total_files = len(parse_results)
        processed_files = 0
        
        # Process files sequentially (NetworkX and dicts are not thread-safe)
        for file_path, result in parse_results.items():
            file_name = Path(file_path).stem
            processed_files += 1
            
            for relationship in result.relationships:
                relationships_attempted += 1
                source_symbol = relationship.source_symbol
                target_symbol = relationship.target_symbol
                rel_type = relationship.relationship_type.value
                
                # Safety check: Skip DEFINES from MODULE/NAMESPACE/PACKAGE sources
                # (Parsers should no longer emit these after standardization cleanup)
                if rel_type == 'defines':
                    annotations = getattr(relationship, 'annotations', {}) or {}
                    container_type = annotations.get('container_type', '').lower()
                    if container_type in ('module', 'namespace', 'package'):
                        relationships_skipped += 1
                        continue
                
                # Synchronous source node resolution with cache
                source_node = self._resolve_source_node_sync(
                    source_symbol, language, file_name, file_path, graph, symbol_registry, parse_results, node_cache
                )
                
                # Synchronous target node resolution with cache
                # For cross-file relationships (like DEFINES_BODY), use target_file if available
                target_file_hint = None
                if hasattr(relationship, 'target_file') and relationship.target_file:
                    target_file_hint = Path(relationship.target_file).stem
                
                target_node = self._resolve_target_node_sync(
                    target_symbol, language, file_name, symbol_registry, graph, parse_results, node_cache, target_file_hint
                )
                
                # Validate nodes
                if not source_node:
                    if self.debug_mode:
                        print(f"❌ Failed to resolve source node: {source_symbol} in {file_name}")
                    logger.debug(f"Source resolution failed: {source_symbol} in {file_name}")
                    relationships_skipped += 1
                    continue
                    
                if not target_node:
                    # Target node is None means it's an external reference (builtin/stdlib/3rd party)
                    # _resolve_target_node_sync returns None for unresolved symbols
                    if self.debug_mode:
                        print(f"⏭️ Skipping unresolved target: {target_symbol} in {file_name}")
                    logger.debug(f"Skipping unresolved target: {target_symbol} (external reference)")
                    relationships_skipped += 1
                    continue
                
                # Prepare edge data (node creation will happen during bulk addition)
                target_file_name = target_node.split('::')[1] if '::' in target_node else file_name
                
                # Preserve relationship annotations from parser
                rel_annotations = {}
                if hasattr(relationship, 'annotations') and relationship.annotations:
                    rel_annotations = relationship.annotations.copy()
                
                # Collect edge for bulk addition
                edge_data = {
                    'relationship_type': rel_type,
                    'source_file': file_name,
                    'target_file': target_file_name,
                    'analysis_level': 'comprehensive',
                    'annotations': rel_annotations,
                    # Store metadata for deferred node creation
                    '_source_symbol': source_symbol,
                    '_target_symbol': target_symbol,
                    '_source_node': source_node,
                    '_target_node': target_node,
                    '_file_path': file_path,
                    '_file_name': file_name,
                    '_language': language,
                }
                
                edges_to_add.append((source_node, target_node, edge_data))
                
                if self.debug_mode:
                    anno_str = f" {rel_annotations}" if rel_annotations else ""
                    print(f"✅ Prepared edge {source_node} --{rel_type}-->{anno_str} {target_node}")
            
            # Progress logging every 500 files
            if processed_files % 500 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Prepared relationships from {processed_files}/{total_files} files "
                          f"({len(edges_to_add)} edges ready) in {elapsed:.1f}s")
        
        preparation_time = time.time() - start_time
        logger.info(f"Preparation complete: {len(edges_to_add)} edges in {preparation_time:.2f}s")
        
        # Bulk add all edges at once (massive speedup)
        logger.info(f"Adding {len(edges_to_add)} edges to code_graph in bulk...")
        bulk_start = time.time()
        
        # NetworkX add_edges_from for MultiDiGraph
        # Create missing nodes first, then add edges
        for source, target, edge_data in edges_to_add:
            # Extract metadata (use pop for internal markers, preserve source_context for hints)
            source_symbol = edge_data.pop('_source_symbol', None)
            target_symbol = edge_data.pop('_target_symbol', None)
            source_node = edge_data.pop('_source_node', source)
            target_node = edge_data.pop('_target_node', target)
            file_path = edge_data.pop('_file_path', '')
            file_name = edge_data.pop('_file_name', '')
            language_val = edge_data.pop('_language', language)
            rel_type = edge_data.get('relationship_type', 'unknown')
            
            # Preserve source_context on edge for transitive hints (e.g., "via self.repo field")
            # This enables hints like "→ uses UserRepo (via repo field in UserService)"
            if source_symbol:
                edge_data['source_context'] = source_symbol
            if target_symbol:
                edge_data['target_context'] = target_symbol
            
            # Ensure source node exists
            if not graph.has_node(source_node):
                graph.add_node(source_node,
                             symbol_name=source_symbol.split('.')[-1] if source_symbol else source_node.split('::')[-1],
                             file_path=file_path,
                             file_name=file_name,
                             language=language_val,
                             symbol_type="inferred",
                             analysis_level='comprehensive')
            
            # Ensure target node exists
            if not graph.has_node(target_node):
                target_parts = target_node.split('::')
                target_name = target_parts[-1] if target_parts else target_symbol
                target_file_name = target_parts[1] if len(target_parts) > 1 else file_name
                
                # Infer target type based on relationship
                if rel_type == 'inheritance':
                    target_type = "class"
                elif rel_type == 'calls':
                    has_scope = '.' in (target_symbol or '') or '::' in (target_symbol or '')
                    target_type = "function" if not has_scope else "method"
                elif rel_type == 'imports':
                    has_scope = '.' in (target_symbol or '') or '::' in (target_symbol or '')
                    target_type = "module" if not has_scope else "class"
                else:
                    target_type = "unknown"
                
                graph.add_node(target_node,
                             symbol_name=target_name,
                             file_path=file_path,
                             file_name=target_file_name,
                             language=language_val,
                             symbol_type=target_type,
                             analysis_level='comprehensive')
            
            # Add the edge
            graph.add_edge(source, target, **edge_data)
        
        bulk_time = time.time() - bulk_start
        total_time = time.time() - start_time
        
        final_edges = graph.number_of_edges()
        relationships_added = len(edges_to_add)
        
        logger.info(f"RELATIONSHIP SUMMARY:")
        logger.info(f"  - Expected: {total_relationships_expected}")
        logger.info(f"  - Attempted: {relationships_attempted}")
        logger.info(f"  - Added: {relationships_added}")
        logger.info(f"  - Skipped: {relationships_skipped}")
        logger.info(f"  - Final edges in code_graph: {final_edges}")
        logger.info(f"  - Preparation time: {preparation_time:.2f}s")
        logger.info(f"  - Bulk addition time: {bulk_time:.2f}s")
        logger.info(f"  - Total time: {total_time:.2f}s")
        
        if relationships_added != final_edges:
            logger.warning(f"⚠️ Edge count mismatch: Added {relationships_added} but code_graph has {final_edges}")
            logger.warning(f"   This may indicate duplicate edges or MultiDiGraph consolidation")
    
    def _resolve_source_node_sync(self, source_symbol: str, language: str, file_name: str, 
                                   file_path: str, graph: nx.MultiDiGraph, symbol_registry: Dict[str, Any],
                                   parse_results: Dict[str, ParseResult], node_cache: Dict[str, bool]) -> str:
        """Synchronous version of source node resolution with O(1) cache lookup"""
        
        # Strategy 1: Direct match in current file (check this FIRST before file-level)
        direct_source = f"{language}::{file_name}::{source_symbol}"
        if direct_source in node_cache:
            return direct_source
        
        # NOTE: C++ qualified name workaround REMOVED after parser standardization
        # C++ parser now uses '.' separator like all other parsers (see PARSER_GRAPH_STANDARDIZATION.md)
        
        # Strategy 2: File-level imports (source = module name)
        # Only use this if NOT found as actual symbol (to handle Java where class name == file name)
        if source_symbol == file_name:
            source_node = f"{language}::{file_name}::__file__"
            if source_node not in node_cache:
                graph.add_node(source_node,
                             symbol_name="__file__",
                             file_path=file_path,
                             file_name=file_name,
                             language=language,
                             symbol_type="module",
                             analysis_level='comprehensive')
                node_cache[source_node] = True  # Update cache
            return source_node
        
        # Strategy 3: Handle dotted names (class.method, module.function, class.field)
        # For qualified names like "app.AppConfig.database", try QUALIFIED names first
        # Priority: qualified partial > single part (avoids matching parent class for child field)
        if '.' in source_symbol:
            parts = source_symbol.split('.')
            
            # First pass: try QUALIFIED names (more specific = better match)
            # For 'app.AppConfig.database', try:
            # 1. 'AppConfig.database' (qualified with parent - most specific)
            # 2. 'app.AppConfig.database' (full name)
            for i in range(len(parts) - 1, -1, -1):  # Reverse order: from end to start
                partial_name = '.'.join(parts[i:])
                candidate = f"{language}::{file_name}::{partial_name}"
                if candidate in node_cache:
                    return candidate
            
            # Second pass: try single parts (fallback - less specific)
            # Only if no qualified match found
            for i in range(len(parts) - 1, -1, -1):
                single_part = parts[i]
                candidate = f"{language}::{file_name}::{single_part}"
                if candidate in node_cache:
                    return candidate
        
        # Strategy 4: Registry lookup by name
        if source_symbol in symbol_registry['by_name']:
            candidates = symbol_registry['by_name'][source_symbol]
            
            # Prefer candidates from the current file
            for candidate in candidates:
                if file_name in candidate and candidate in node_cache:
                    return candidate
            
            # Fallback to any candidate
            for candidate in candidates:
                if candidate in node_cache:
                    return candidate
        
        # Strategy 5: Qualified name lookup
        if source_symbol in symbol_registry['by_qualified_name']:
            candidate = symbol_registry['by_qualified_name'][source_symbol]
            if candidate in node_cache:
                return candidate
        
        # Strategy 6: REMOVED - Cross-file search eliminated (too slow, replaced by registry)
        
        # Strategy 7: Create fallback node
        # NOTE: C++ qualified name workaround REMOVED after parser standardization
        # C++ parser now uses '.' separator like all other parsers (see PARSER_GRAPH_STANDARDIZATION.md)
        fallback_node = f"{language}::{file_name}::{source_symbol}"
        return fallback_node
    
    def _resolve_target_node_sync(self, target_symbol: str, language: str, file_name: str,
                                  symbol_registry: Dict[str, Any], graph: nx.MultiDiGraph,
                                  parse_results: Dict[str, ParseResult], node_cache: Dict[str, bool],
                                  target_file_hint: Optional[str] = None) -> Optional[str]:
        """Synchronous version of target node resolution with O(1) cache lookup
        
        Args:
            target_file_hint: Optional file name stem to prefer for cross-file relationships
        """
        
        # Strategy 0: If target_file_hint provided, try that file first (for cross-file relationships)
        if target_file_hint and target_file_hint != file_name:
            hinted_target = f"{language}::{target_file_hint}::{target_symbol}"
            if hinted_target in node_cache:
                return hinted_target
        
        # Strategy 1: Direct match in same file
        direct_target = f"{language}::{file_name}::{target_symbol}"
        if direct_target in node_cache:
            return direct_target
        
        # Strategy 2: Handle self references
        if target_symbol.startswith('self.'):
            attribute_name = target_symbol[5:]
            self_target = f"{language}::{file_name}::{attribute_name}"
            if self_target in node_cache:
                return self_target
        
        # Strategy 3: Handle qualified names for imports and containment
        # For qualified names like "test.AppConfig.__init__", try the MOST SPECIFIC (rightmost) part first
        if '.' in target_symbol:
            parts = target_symbol.split('.')
            
            # First handle simple 2-part module.class case
            if len(parts) == 2:
                module_name, class_name = parts
                qualified_target = f"{language}::{module_name}::{class_name}"
                if qualified_target in node_cache:
                    return qualified_target
                
                unqualified_target = f"{language}::{file_name}::{class_name}"
                if unqualified_target in node_cache:
                    return unqualified_target
            
            # For longer qualified names, try QUALIFIED names first (more specific = better match)
            # For 'test.AppConfig.__init__', try:
            # 1. 'AppConfig.__init__' (qualified with parent - most specific)
            # 2. 'test.AppConfig.__init__' (full name)
            # 3. '__init__' alone (fallback - less specific)
            # Priority: qualified partial > single part (avoids matching parent class for child method)
            for i in range(len(parts) - 1, -1, -1):  # Reverse order: from end to start
                # First try the qualified name from this part onwards (more specific)
                partial_name = '.'.join(parts[i:])
                candidate = f"{language}::{file_name}::{partial_name}"
                if candidate in node_cache:
                    return candidate
            
            # Fallback: try single parts (less specific - only if no qualified match found)
            for i in range(len(parts) - 1, -1, -1):
                single_part = parts[i]
                candidate = f"{language}::{file_name}::{single_part}"
                if candidate in node_cache:
                    return candidate
        
        # Strategy 4: Registry lookup by exact name
        if target_symbol in symbol_registry['by_name']:
            candidates = symbol_registry['by_name'][target_symbol]
            definition_types = {'class', 'function', 'method', 'interface', 'enum', 'macro', 'constant', 'struct'}
            
            # First pass: look for definitions
            for candidate in candidates:
                if candidate in node_cache:
                    node_data = graph.nodes[candidate]
                    symbol_type_raw = node_data.get('symbol_type', '')
                    if hasattr(symbol_type_raw, 'value'):
                        symbol_type = symbol_type_raw.value.lower()
                    else:
                        symbol_type = str(symbol_type_raw).lower()
                    if symbol_type in definition_types:
                        return candidate
            
            # Second pass: any valid candidate
            for candidate in candidates:
                if candidate in node_cache:
                    return candidate
        
        # Strategy 5: Registry lookup by qualified name
        if target_symbol in symbol_registry['by_qualified_name']:
            candidate = symbol_registry['by_qualified_name'][target_symbol]
            if candidate in node_cache:
                return candidate
        
        # Strategy 6: REMOVED - Cross-file search eliminated (too slow, replaced by registry)
        # The symbol_registry already contains cross-file symbols from _add_symbols_batch
        
        # Strategy 7: Flexible matching for common patterns
        if '.' in target_symbol:
            method_name = target_symbol.split('.')[-1]
            
            method_candidate = f"{language}::{file_name}::{method_name}"
            if method_candidate in node_cache:
                return method_candidate
            
            if method_name in symbol_registry['by_name']:
                candidates = symbol_registry['by_name'][method_name]
                for candidate in candidates:
                    if candidate in node_cache:
                        node_data = graph.nodes[candidate]
                        symbol_type_raw = node_data.get('symbol_type', '')
                        if hasattr(symbol_type_raw, 'value'):
                            symbol_type = symbol_type_raw.value.lower()
                        else:
                            symbol_type = str(symbol_type_raw).lower()
                        if symbol_type in {'method', 'function'}:
                            return candidate
        
        # Strategy 8: UNRESOLVED SYMBOL = EXTERNAL REFERENCE
        # If we reach here, the symbol could NOT be resolved to any node defined in the project.
        # This means it's an external reference (builtin, stdlib, 3rd party library).
        # 
        # Key insight: Symbols defined in user code ARE in the symbol_registry and node_cache.
        # If Strategies 1-7 all failed, the symbol is NOT user code - skip it.
        #
        # This is language-agnostic: works for Python, Java, TypeScript, C++, etc.
        # No need for explicit builtin lists - the resolution logic handles it.
        if self.debug_mode:
            print(f"⏭️ Skipping unresolved symbol: {target_symbol} (not defined in project)")
        logger.debug(f"Skipping unresolved symbol: {target_symbol} (external reference)")
        return None
    
    def _merge_graph(self, target_graph: nx.MultiDiGraph, source_graph):
        """Merge source code_graph into target code_graph.

        Accepts either a MultiDiGraph or a DiGraph as source. If the source is a
        simple DiGraph we synthesize edge keys to preserve multiple edges when
        merging into the MultiDiGraph target.
        """
        if not source_graph:
            return

        # Add / merge nodes
        for node_id, node_attrs in source_graph.nodes(data=True):
            if node_id not in target_graph:
                target_graph.add_node(node_id, **node_attrs)
            else:
                existing_attrs = target_graph.nodes[node_id]
                merged_attrs = {**existing_attrs, **node_attrs}
                target_graph.add_node(node_id, **merged_attrs)

        # Add edges. Support both MultiDiGraph and DiGraph
        if isinstance(source_graph, nx.MultiDiGraph):
            for u, v, key, edge_attrs in source_graph.edges(data=True, keys=True):
                target_graph.add_edge(u, v, key=key, **edge_attrs)
        else:
            # DiGraph: no keys; create deterministic synthetic keys to avoid overwriting
            edge_index = 0
            for u, v, edge_attrs in source_graph.edges(data=True):
                synthetic_key = f"d:{edge_index}"
                target_graph.add_edge(u, v, key=synthetic_key, **edge_attrs)
                edge_index += 1
    
    def _build_basic_language_graph(self, parse_results: Dict[str, BasicParseResult], language: str) -> nx.MultiDiGraph:
        """Build basic code_graph for tree-sitter parsers.

        Use MultiDiGraph so merging logic is consistent with comprehensive graphs
        and multiple relationship edges between the same pair of nodes are preserved.
        """
        graph = nx.MultiDiGraph()
        graph.graph['language'] = language
        graph.graph['analysis_level'] = 'basic'
        
        # Simple code_graph building for basic parsers
        for file_path, result in parse_results.items():
            file_name = Path(file_path).stem
            
            # Add symbols as nodes
            for symbol in result.symbols:
                node_id = f"{language}::{file_name}::{symbol.name}"
                
                # Handle duplicates
                base_node_id = node_id
                counter = 1
                while node_id in graph.nodes:
                    node_id = f"{base_node_id}#{counter}"
                    counter += 1
                
                # Normalise raw AST node type to canonical category
                symbol_type_str = symbol.symbol_type.value if hasattr(symbol.symbol_type, 'value') else str(symbol.symbol_type)
                canonical_type = _BASIC_AST_TYPE_MAP.get(symbol_type_str, symbol_type_str)

                # Skip import/using nodes — they inflate total_symbols without
                # contributing meaningful structure.
                if canonical_type == 'import' or symbol_type_str in _IMPORT_TYPES:
                    continue

                node_attrs = {
                    'name': symbol.name,
                    'type': canonical_type,
                    'symbol_name': symbol.name,                # canonical key for FTS5
                    'symbol_type': canonical_type,             # canonical key for FTS5
                    'file_path': file_path,
                    'rel_path': getattr(symbol, 'rel_path', '') or file_path,  # canonical key
                    'file_name': file_name,
                    'language': language,
                    'start_line': symbol.start_line,
                    'end_line': symbol.end_line,
                    'analysis_level': 'basic',
                    'source_text': symbol.source_text,
                    'docstring': symbol.docstring or '',
                    'parameters': symbol.parameters,
                    'return_type': symbol.return_type or ''
                }
                
                graph.add_node(node_id, **node_attrs)
            
            # Add relationships as edges
            for relationship in result.relationships:
                # Simple resolution for basic relationships
                source_id = f"{language}::{file_name}::{relationship.source_symbol}"
                target_id = f"{language}::{file_name}::{relationship.target_symbol}"
                
                if source_id in graph.nodes and target_id in graph.nodes:
                    edge_attrs = {
                        'type': relationship.relationship_type,
                        'source_file': relationship.source_file,
                        'target_file': relationship.target_file,
                        'analysis_level': 'basic',
                        'language': language
                    }
                    # Add without specifying key; MultiDiGraph will create one
                    graph.add_edge(source_id, target_id, **edge_attrs)
        
        return graph
    
    def _is_java_standard_library_reference(self, target_symbol: str, target_node: str) -> bool:
        """Check if a target symbol/node references Java standard library."""
        if not target_symbol:
            return False
            
        # Check for fully qualified Java standard library packages
        java_std_packages = {
            'java.lang',
            'java.util',
            'java.io',
            'java.nio',
            'java.time',
            'java.math',
            'java.net',
            'java.text',
            'java.util.regex',
            'java.util.concurrent',
            'java.util.function',
            'java.util.stream',
            'java.security',
            'java.awt',
            'javax.swing',
            'javax.annotation'
        }
        
        # Check if it's a fully qualified Java standard library type
        if '.' in target_symbol:
            for package in java_std_packages:
                if target_symbol.startswith(package + '.'):
                    return True
        
        # Check for common built-in methods on standard library types
        builtin_methods = {
            'isEmpty', 'trim', 'length', 'contains', 'add', 'remove', 'get', 'set',
            'toString', 'equals', 'hashCode', 'getClass', 'format', 'valueOf',
            'randomUUID', 'now', 'hash', 'compile', 'matches', 'matcher',
            'orElseThrow', 'isPresent', 'of', 'empty'
        }
        
        # Check if the target symbol is a method call on a standard library type
        if '.' in target_symbol:
            method_name = target_symbol.split('.')[-1]
            if method_name in builtin_methods:
                return True
        
        return False

    def _extract_cpp_metadata(self, symbol: Any) -> Dict[str, Any]:
        """Extract C++ specific metadata from symbol comments for document metadata."""
        if not hasattr(symbol, 'comments'):
            return {}
        
        metadata = {}
        for comment in symbol.comments:
            if comment.startswith('namespaces='):
                metadata['namespaces'] = comment.split('=', 1)[1]
            elif comment.startswith('template_params='):
                metadata['template_params'] = comment.split('=', 1)[1]
                metadata['is_template'] = True
            elif comment.startswith('template_template_params='):
                metadata['template_template_params'] = comment.split('=', 1)[1]
            elif comment == 'final_class':
                metadata['is_final'] = True
            elif comment == 'scoped_enum':
                metadata['is_scoped_enum'] = True
            elif comment.startswith('const_kind='):
                metadata['const_kind'] = comment.split('=', 1)[1]
            elif comment == 'lambda':
                metadata['is_lambda'] = True
        
        return metadata
    
    def _generate_symbol_chunks(self, parse_results: Dict[str, Union[ParseResult, BasicParseResult]]) -> List[Document]:
        """
        Generate symbol-level chunk documents from parse results for vector store.
        
        When SEPARATE_DOC_INDEX=1: ONLY documentation files go to vector store (code is in graph only).
        When SEPARATE_DOC_INDEX=0: All architectural symbols go to vector store (legacy mode).
        
        Sequential processing for thread safety.
        """
        total_files = len(parse_results)
        
        # Count doc files vs code files
        doc_files_count = 0
        code_files_count = 0
        for file_path, result in parse_results.items():
            if getattr(result, 'parse_level', 'comprehensive') == 'documentation':
                doc_files_count += 1
            else:
                code_files_count += 1
        
        if SEPARATE_DOC_INDEX:
            # SEPARATION MODE: Only docs go to vector store, code stays in graph only
            logger.info(
                f"[SEPARATE_DOC_INDEX] Vector store: {doc_files_count} doc files ONLY "
                f"(skipping {code_files_count} code files - code is in graph only)"
            )
        else:
            logger.info(f"Generating symbol chunks from {total_files} parsed files...")
        
        start_time = time.time()
        
        documents = []
        total_symbols_processed = 0
        architectural_symbols_kept = 0
        doc_chunks_generated = 0
        code_files_skipped = 0
        processed_files = 0
        
        # Process files sequentially
        for file_path, result in parse_results.items():
            processed_files += 1
            is_doc_file = getattr(result, 'parse_level', 'comprehensive') == 'documentation'
            
            # SEPARATE_DOC_INDEX: Skip ALL code files - they go to graph only
            if SEPARATE_DOC_INDEX and not is_doc_file:
                code_files_skipped += 1
                continue
            
            try:
                file_name = Path(file_path).name
                file_symbol_count = len(result.symbols)
                
                # Log files with large symbol counts that might cause issues
                if file_symbol_count > 1000:
                    logger.warning(f"Processing file with {file_symbol_count} symbols: {file_path}")
                
                # Create document for each symbol
                for symbol in result.symbols:
                    total_symbols_processed += 1
                    
                    # Calculate relative path if symbol doesn't have it
                    symbol_rel_path = getattr(symbol, 'rel_path', None)
                    if not symbol_rel_path and hasattr(self, 'repo_path'):
                        symbol_rel_path = self._calculate_relative_path(file_path, self.repo_path)
                    if not symbol_rel_path:
                        symbol_rel_path = file_path  # Fallback to absolute if calculation fails
                    
                    # Determine if this is rich or basic parsing
                    analysis_level = getattr(result, 'parse_level', 'comprehensive')
                    
                    # ARCHITECTURAL FILTERING: Only include architectural symbols in vector store
                    symbol_type_str = None
                    if analysis_level == 'documentation':
                        symbol_type_str = symbol.symbol_type
                    elif analysis_level == 'basic':
                        symbol_type_str = symbol.symbol_type
                    else:
                        # Rich symbol
                        symbol_type_str = symbol.symbol_type.value if hasattr(symbol.symbol_type, 'value') else str(symbol.symbol_type)
                    
                    # Skip non-architectural symbols for vector store
                    if not self._is_architectural_symbol(symbol_type_str):
                        continue
                    
                    # CRITICAL: Only include TOP-LEVEL entities in architectural embeddings
                    if analysis_level == 'comprehensive':
                        parent_symbol = getattr(symbol, 'parent_symbol', None)
                        if parent_symbol:
                            if not self._is_package_or_namespace_parent(parent_symbol, parse_results):
                                continue
                    
                    # Track doc chunks (code files are skipped above when SEPARATE_DOC_INDEX=1)
                    if is_doc_file:
                        doc_chunks_generated += 1
                    
                    # Build metadata and create document
                    if analysis_level == 'documentation':
                        content = symbol.source_text or ''
                        symbol_type = symbol.symbol_type
                        metadata = {
                            'chunk_type': 'text',
                            'symbol_name': symbol.name,
                            'symbol_type': symbol_type,
                            'file_path': file_path,
                            'rel_path': symbol_rel_path,  # Use calculated relative path
                            'source': symbol_rel_path,
                            'file_name': file_name,
                            'language': symbol.language,
                            'start_line': symbol.start_line,
                            'end_line': symbol.end_line,
                            'analysis_level': 'documentation',
                            'docstring': symbol.docstring or '',
                            'summary': symbol.docstring or '',
                            'content_type': symbol.return_type or 'text',
                            'section_id': getattr(symbol, 'section_id', None),
                            'section_order': getattr(symbol, 'section_order', None)
                        }
                    elif analysis_level == 'basic':
                        content = symbol.source_text or ''
                        symbol_type = symbol.symbol_type
                        metadata = {
                            'chunk_type': 'symbol',
                            'symbol_name': symbol.name,
                            'symbol_type': symbol_type,
                            'file_path': file_path,
                            'rel_path': symbol_rel_path,  # Use calculated relative path
                            'source': symbol_rel_path,
                            'file_name': file_name,
                            'language': symbol.language,
                            'start_line': symbol.start_line,
                            'end_line': symbol.end_line,
                            'analysis_level': 'basic',
                            'docstring': symbol.docstring or '',
                            'parameters': symbol.parameters,
                            'return_type': symbol.return_type or ''
                        }
                    else:
                        # Rich symbol
                        content = getattr(symbol, 'source_text', '') or ''
                        symbol_type = symbol.symbol_type.value if hasattr(symbol.symbol_type, 'value') else str(symbol.symbol_type)
                        
                        # Extract C++ specific metadata if language is C++
                        cpp_meta = {}
                        if result.language == 'cpp':
                            cpp_meta = self._extract_cpp_metadata(symbol)
                        
                        # Get symbol metadata (generic_params, is_exported, decorators, etc.)
                        symbol_metadata = getattr(symbol, 'metadata', {}) or {}
                        
                        metadata = {
                            'chunk_type': 'symbol',
                            'symbol_name': symbol.name,
                            'symbol_type': symbol_type,
                            'file_path': file_path,
                            'rel_path': symbol_rel_path,  # Use calculated relative path
                            'source': symbol_rel_path,
                            'file_name': file_name,
                            'language': result.language,
                            'start_line': symbol.range.start.line if symbol.range else 0,
                            'end_line': symbol.range.end.line if symbol.range else 0,
                            'analysis_level': 'comprehensive',
                            'docstring': getattr(symbol, 'docstring', '') or '',
                            'parameters': getattr(symbol, 'parameters', []),
                            'return_type': getattr(symbol, 'return_type', '') or '',
                            'access_modifier': getattr(symbol, 'access_modifier', ''),
                            'is_abstract': getattr(symbol, 'is_abstract', False),
                            'is_static': getattr(symbol, 'is_static', False),
                            'namespaces': cpp_meta.get('namespaces', ''),
                            'template_params': cpp_meta.get('template_params', ''),
                            'is_template': cpp_meta.get('is_template', False),
                            'is_final': cpp_meta.get('is_final', False),
                            'is_scoped_enum': cpp_meta.get('is_scoped_enum', False),
                            'is_lambda': cpp_meta.get('is_lambda', False),
                            'const_kind': cpp_meta.get('const_kind', ''),
                            # Include symbol metadata (generic_params, decorators, is_exported, etc.)
                            'annotations': symbol_metadata,
                        }
                    
                    # Create document
                    doc = Document(
                        page_content=content,
                        metadata=metadata
                    )
                    documents.append(doc)
                    architectural_symbols_kept += 1
            
            except Exception as e:
                logger.error(f"Failed to process symbols from file {file_path}: {type(e).__name__}: {e}", exc_info=True)
                logger.error(f"Skipping {file_symbol_count} symbols from this file")
            
            # Progress logging every 500 files
            if processed_files % 500 == 0 or processed_files == total_files:
                elapsed = time.time() - start_time
                logger.info(f"Processing symbols: {processed_files}/{total_files} files, "
                          f"{architectural_symbols_kept} architectural symbols created from "
                          f"{total_symbols_processed} total in {elapsed:.1f}s")
        
        generation_time = time.time() - start_time
        
        # Final summary
        if SEPARATE_DOC_INDEX:
            logger.info(
                f"[SEPARATE_DOC_INDEX] Vector store: {len(documents)} doc chunks "
                f"from {doc_files_count} doc files (skipped {code_files_skipped} code files) "
                f"in {generation_time:.2f}s"
            )
        else:
            logger.info(f"Symbol chunk generation complete: {len(documents)} documents created "
                       f"from {architectural_symbols_kept} architectural symbols "
                       f"(filtered from {total_symbols_processed} total symbols across {total_files} files) "
                       f"in {generation_time:.2f}s")
        
        return documents

    def _iter_symbol_chunks(self, parse_results: Dict[str, Union[ParseResult, BasicParseResult]]):
        """Yield symbol-level chunk documents to enable streaming indexing."""
        total_files = len(parse_results)
        
        # Count doc files vs code files
        doc_files_count = 0
        code_files_count = 0
        for file_path, result in parse_results.items():
            if getattr(result, 'parse_level', 'comprehensive') == 'documentation':
                doc_files_count += 1
            else:
                code_files_count += 1
        
        if SEPARATE_DOC_INDEX:
            # SEPARATION MODE: Only docs go to vector store, code stays in graph only
            logger.info(
                f"[SEPARATE_DOC_INDEX] Vector store: {doc_files_count} doc files ONLY "
                f"(skipping {code_files_count} code files - code is in graph only)"
            )
        else:
            logger.info(f"Generating symbol chunks from {total_files} parsed files (streaming)...")

        start_time = time.time()

        total_symbols_processed = 0
        architectural_symbols_kept = 0
        doc_chunks_generated = 0
        code_files_skipped = 0
        processed_files = 0

        for file_path, result in parse_results.items():
            processed_files += 1
            is_doc_file = getattr(result, 'parse_level', 'comprehensive') == 'documentation'

            # SEPARATE_DOC_INDEX: Skip ALL code files - they go to graph only
            if SEPARATE_DOC_INDEX and not is_doc_file:
                code_files_skipped += 1
                continue

            try:
                file_name = Path(file_path).name
                file_symbol_count = len(result.symbols)

                if file_symbol_count > 1000:
                    logger.warning(f"Processing file with {file_symbol_count} symbols: {file_path}")

                for symbol in result.symbols:
                    total_symbols_processed += 1

                    symbol_rel_path = getattr(symbol, 'rel_path', None)
                    if not symbol_rel_path and hasattr(self, 'repo_path'):
                        symbol_rel_path = self._calculate_relative_path(file_path, self.repo_path)
                    if not symbol_rel_path:
                        symbol_rel_path = file_path

                    analysis_level = getattr(result, 'parse_level', 'comprehensive')

                    if analysis_level == 'documentation':
                        symbol_type_str = symbol.symbol_type
                    elif analysis_level == 'basic':
                        symbol_type_str = symbol.symbol_type
                    else:
                        symbol_type_str = symbol.symbol_type.value if hasattr(symbol.symbol_type, 'value') else str(symbol.symbol_type)

                    if not self._is_architectural_symbol(symbol_type_str):
                        continue

                    if analysis_level == 'comprehensive':
                        parent_symbol = getattr(symbol, 'parent_symbol', None)
                        if parent_symbol:
                            if not self._is_package_or_namespace_parent(parent_symbol, parse_results):
                                continue

                    # Track doc chunks (code files are skipped above when SEPARATE_DOC_INDEX=1)
                    if is_doc_file:
                        doc_chunks_generated += 1

                    if analysis_level == 'documentation':
                        content = symbol.source_text or ''
                        symbol_type = symbol.symbol_type
                        metadata = {
                            'chunk_type': 'text',
                            'symbol_name': symbol.name,
                            'symbol_type': symbol_type,
                            'file_path': file_path,
                            'rel_path': symbol_rel_path,
                            'source': symbol_rel_path,
                            'file_name': file_name,
                            'language': symbol.language,
                            'start_line': symbol.start_line,
                            'end_line': symbol.end_line,
                            'analysis_level': 'documentation',
                            'docstring': symbol.docstring or '',
                            'summary': symbol.docstring or '',
                            'content_type': symbol.return_type or 'text',
                            'section_id': getattr(symbol, 'section_id', None),
                            'section_order': getattr(symbol, 'section_order', None)
                        }
                    elif analysis_level == 'basic':
                        content = symbol.source_text or ''
                        symbol_type = symbol.symbol_type
                        metadata = {
                            'chunk_type': 'symbol',
                            'symbol_name': symbol.name,
                            'symbol_type': symbol_type,
                            'file_path': file_path,
                            'rel_path': symbol_rel_path,
                            'source': symbol_rel_path,
                            'file_name': file_name,
                            'language': symbol.language,
                            'start_line': symbol.start_line,
                            'end_line': symbol.end_line,
                            'analysis_level': 'basic',
                            'docstring': symbol.docstring or '',
                            'parameters': symbol.parameters,
                            'return_type': symbol.return_type or ''
                        }
                    else:
                        content = getattr(symbol, 'source_text', '') or ''
                        symbol_type = symbol.symbol_type.value if hasattr(symbol.symbol_type, 'value') else str(symbol.symbol_type)

                        cpp_meta = {}
                        if result.language == 'cpp':
                            cpp_meta = self._extract_cpp_metadata(symbol)

                        symbol_metadata = getattr(symbol, 'metadata', {}) or {}

                        metadata = {
                            'chunk_type': 'symbol',
                            'symbol_name': symbol.name,
                            'symbol_type': symbol_type,
                            'file_path': file_path,
                            'rel_path': symbol_rel_path,
                            'source': symbol_rel_path,
                            'file_name': file_name,
                            'language': result.language,
                            'start_line': symbol.range.start.line if symbol.range else 0,
                            'end_line': symbol.range.end.line if symbol.range else 0,
                            'analysis_level': 'comprehensive',
                            'docstring': getattr(symbol, 'docstring', '') or '',
                            'parameters': getattr(symbol, 'parameters', []),
                            'return_type': getattr(symbol, 'return_type', '') or '',
                            'access_modifier': getattr(symbol, 'access_modifier', ''),
                            'is_abstract': getattr(symbol, 'is_abstract', False),
                            'is_static': getattr(symbol, 'is_static', False),
                            'namespaces': cpp_meta.get('namespaces', ''),
                            'template_params': cpp_meta.get('template_params', ''),
                            'is_template': cpp_meta.get('is_template', False),
                            'is_final': cpp_meta.get('is_final', False),
                            'is_scoped_enum': cpp_meta.get('is_scoped_enum', False),
                            'is_lambda': cpp_meta.get('is_lambda', False),
                            'const_kind': cpp_meta.get('const_kind', ''),
                            'annotations': symbol_metadata,
                        }

                    doc = Document(page_content=content, metadata=metadata)
                    architectural_symbols_kept += 1
                    yield doc

            except Exception as e:
                logger.warning(f"Failed to process symbols for {file_path}: {e}")

        elapsed = time.time() - start_time
        
        # Final summary
        if SEPARATE_DOC_INDEX:
            logger.info(
                f"[SEPARATE_DOC_INDEX] Vector store: {architectural_symbols_kept} doc chunks "
                f"from {doc_files_count} doc files (skipped {code_files_skipped} code files) "
                f"in {elapsed:.2f}s (streaming)"
            )
        else:
            logger.info(
                f"Generated {architectural_symbols_kept} architectural symbol docs "
                f"from {processed_files} files in {elapsed:.2f}s (streaming)."
            )

    def _iter_symbol_chunks_with_cleanup(self, parse_results: Dict[str, Union[ParseResult, BasicParseResult]]):
        """Stream symbol chunks and clear parse results after consumption."""
        try:
            for doc in self._iter_symbol_chunks(parse_results):
                yield doc
        finally:
            try:
                parse_results.clear()
            except Exception:
                pass

    
    def _detect_cross_language_relationships(self, files_by_language: Dict[str, List[str]], 
                                           parse_results: Dict[str, Union[ParseResult, BasicParseResult]]) -> List[Any]:
        """Detect cross-language relationships (simplified for now)"""
        cross_lang_rels = []
        
        # For now, return empty list - this would require more sophisticated analysis
        # to detect things like Java calling Python scripts, etc.
        
        return cross_lang_rels
    
    def _generate_language_stats(self, parse_results: Dict[str, Union[ParseResult, BasicParseResult]]) -> Dict[str, Any]:
        """Generate statistics by language"""
        stats = {}
        
        # Group results by language
        by_language = defaultdict(list)
        for file_path, result in parse_results.items():
            language = result.language
            by_language[language].append(result)
        
        # Generate stats for each language
        for language, results in by_language.items():
            total_symbols = sum(len(result.symbols) for result in results)
            total_relationships = sum(len(result.relationships) for result in results)
            
            # Determine analysis level
            analysis_levels = set()
            for result in results:
                if hasattr(result, 'parse_level'):
                    analysis_levels.add(result.parse_level)
                else:
                    analysis_levels.add('comprehensive')
            
            stats[language] = {
                'files_count': len(results),
                'symbols_count': total_symbols,
                'relationships_count': total_relationships,
                'analysis_levels': list(analysis_levels)
            }
        
        return stats
    
    def _parse_documentation_files(self, file_paths: List[str], repo_path: str) -> Dict[str, 'BasicParseResult']:
        """
        Process documentation files (markdown, text, etc.) with text chunking
        instead of symbol extraction.
        """
        results = {}
        
        for file_path in file_paths:
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Determine file type (extension first, then known filename)
                file_extension = Path(file_path).suffix.lower()
                doc_type = self.DOCUMENTATION_EXTENSIONS.get(file_extension)
                if not doc_type:
                    doc_type = self.KNOWN_FILENAMES.get(Path(file_path).name, 'text')
                
                # Simple text chunking for documentation
                chunks = self._chunk_text_content(content, file_path, doc_type)
                
                # Create symbols for each text chunk
                symbols = []
                rel_path = os.path.relpath(file_path, repo_path)
                for i, chunk in enumerate(chunks):
                    # Embed file path in chunk content so file-type keywords
                    # (e.g. "yaml", "gradle", "makefile") appear in the vector
                    # embedding and are discoverable via semantic search.
                    path_header = f"[File: {rel_path}]\n"
                    enriched_content = path_header + chunk['content']
                    
                    symbol = BasicSymbol(
                        name=chunk.get('summary', f"{Path(file_path).stem}_chunk_{i}"),
                        symbol_type=chunk.get('section_type', 'text_chunk'),
                        start_line=chunk.get('start_line', 0),
                        end_line=chunk.get('end_line', 0),
                        file_path=file_path,
                        rel_path=rel_path,
                        language=doc_type,
                        source_text=enriched_content,
                        docstring=chunk.get('summary', ''),
                        parameters=[],
                        return_type=doc_type
                    )
                    # Add section ID and order for reconstruction
                    if 'section_id' in chunk:
                        symbol.section_id = chunk['section_id']
                    if 'section_order' in chunk:
                        symbol.section_order = chunk['section_order']
                    symbols.append(symbol)
                
                # Create parse result
                result = BasicParseResult(
                    file_path=file_path,
                    language=doc_type,
                    symbols=symbols,
                    relationships=[],  # No relationships for text chunks
                    parse_level='documentation',
                    imports=[]
                )
                results[file_path] = result
                
                logger.debug(f"Processed documentation file {file_path}: {len(symbols)} text chunks")
                
            except Exception as e:
                logger.warning(f"Failed to parse documentation file {file_path}: {e}")
        
        return results
    
    def _chunk_text_content(self, content: str, file_path: str, doc_type: str) -> List[Dict[str, Any]]:
        """
        Chunk text content into manageable pieces for documentation files.
        Only split if content is larger than reasonable size (15-17k characters).
        """
        chunks = []
        
        # Check if document is reasonably sized (13-17k characters)
        content_size = len(content)
        max_reasonable_size = 17000  # 17k characters
        min_split_size = 13000       # 13k characters - only split if larger than this
        
        logger.debug(f"Document size: {content_size} characters, file: {file_path}")
        
        # If document is reasonably sized, keep it as a single chunk
        if content_size <= max_reasonable_size:
            logger.info(f"Document {file_path} ({content_size} chars) fits in context - keeping as single chunk")
            return [{
                'content': content,
                'summary': Path(file_path).stem,
                'start_line': 1,
                'end_line': len(content.split('\n')),
                'headers': {},
                'section_type': f'{doc_type}_document',
                'section_id': 0,
                'section_order': 0
            }]
        
        # Only split if document is larger than minimum split threshold
        if content_size > min_split_size:
            logger.info(f"Document {file_path} ({content_size} chars) is large - attempting to split")
            
            # For markdown files, try to split by headers
            if doc_type == 'markdown':
                chunks = self._chunk_markdown_content(content, file_path)
            else:
                # For other text files, use simple line-based chunking
                chunks = self._chunk_generic_text(content, file_path)
        else:
            logger.info(f"Document {file_path} ({content_size} chars) below split threshold - keeping as single chunk")
            chunks = [{
                'content': content,
                'summary': Path(file_path).stem,
                'start_line': 1,
                'end_line': len(content.split('\n')),
                'headers': {},
                'section_type': f'{doc_type}_document',
                'section_id': 0,
                'section_order': 0
            }]
        
        return chunks
    
    def _chunk_markdown_content(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Split markdown content using LangChain's MarkdownHeaderTextSplitter
        keeping complete sections without character-based splitting.
        Each chunk gets a sequential ID for reconstruction in original order.
        """
        chunks = []
        
        try:
            # Define headers to split on (h1, h2, h3, h4)
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"), 
                ("###", "Header 3"),
                ("####", "Header 4"),
            ]
            
            # Create markdown splitter
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                strip_headers=False,  # Keep headers in content
                return_each_line=True
            )
            
            # Split the markdown content
            md_header_splits = markdown_splitter.split_text(content)
            
            # Process each markdown section - keep complete sections without further splitting
            for i, split in enumerate(md_header_splits):
                section_content = split.page_content
                section_metadata = split.metadata
                
                # Extract section name from metadata or content
                section_name = self._extract_section_name(section_metadata, section_content, i)
                
                # Create chunk with sequential ID for reconstruction
                chunks.append({
                    'content': section_content,
                    'summary': section_name,
                    'start_line': self._estimate_line_number(content, section_content),
                    'end_line': self._estimate_line_number(content, section_content) + section_content.count('\n'),
                    'headers': section_metadata,
                    'section_type': 'markdown_section',
                    'section_id': i,  # Sequential ID for reconstruction
                    'section_order': i  # Explicit order field for sorting during reconstruction
                })
                    
        except Exception as e:
            logger.warning(f"Failed to split markdown with LangChain splitter: {e}, falling back to simple splitting")
            # Fallback to simple chunking
            chunks = self._chunk_generic_text(content, file_path, max_lines=50)
        
        # If no chunks found, create one chunk with entire content
        if not chunks:
            chunks.append({
                'content': content,
                'summary': Path(file_path).stem,
                'start_line': 1,
                'end_line': len(content.split('\n')),
                'headers': {},
                'section_type': 'markdown_document',
                'section_id': 0,
                'section_order': 0
            })
        
        return chunks
    
    def _extract_section_name(self, metadata: Dict[str, str], content: str, index: int) -> str:
        """Extract a meaningful section name from metadata or content."""
        # Try to get section name from metadata (headers)
        if metadata:
            # Get the deepest header level
            for header_level in ["Header 4", "Header 3", "Header 2", "Header 1"]:
                if header_level in metadata:
                    return metadata[header_level].strip()
        
        # Fallback: extract from first line of content
        first_line = content.split('\n')[0].strip()
        if first_line.startswith('#'):
            return first_line.lstrip('#').strip()
        
        # Final fallback
        return f"Section {index + 1}"
    
    def _estimate_line_number(self, full_content: str, chunk_content: str) -> int:
        """Estimate the line number where a chunk starts in the full content."""
        try:
            chunk_start = full_content.find(chunk_content)
            if chunk_start == -1:
                return 1
            
            # Count newlines before the chunk
            lines_before = full_content[:chunk_start].count('\n')
            return lines_before + 1
        except Exception:
            return 1
    
    def _chunk_generic_text(self, content: str, file_path: str, max_lines: int = 30) -> List[Dict[str, Any]]:
        """
        Chunk generic text content into manageable pieces using line-based splitting.
        
        Args:
            content: The text content to chunk
            file_path: Path to the source file
            max_lines: Maximum number of lines per chunk
            
        Returns:
            List of chunk dictionaries with content, summary, and line info
        """
        chunks = []
        lines = content.split('\n')
        
        # Simple line-based chunking
        for i in range(0, len(lines), max_lines):
            chunk_lines = lines[i:i + max_lines]
            chunk_content = '\n'.join(chunk_lines)
            
            # Skip empty chunks
            if not chunk_content.strip():
                continue
            
            # Extract first meaningful line as summary
            summary_line = ""
            for line in chunk_lines:
                if line.strip():
                    summary_line = line.strip()[:50]  # First 50 chars
                    if len(line.strip()) > 50:
                        summary_line += "..."
                    break
            
            if not summary_line:
                summary_line = f"{Path(file_path).stem} chunk {len(chunks) + 1}"
            
            chunks.append({
                'content': chunk_content,
                'summary': summary_line,
                'start_line': i + 1,
                'end_line': min(i + max_lines, len(lines)),
                'section_type': 'text_chunk'
            })
        
        # If no chunks created, create one with entire content
        if not chunks:
            chunks.append({
                'content': content,
                'summary': Path(file_path).stem,
                'start_line': 1,
                'end_line': len(lines),
                'section_type': 'text_document'
            })
        
        return chunks
    
    def _build_node_index(self, graph: nx.MultiDiGraph) -> None:
        """
        Build O(1) node index: {(symbol_name, file_path, language): node_id}
        
        Handles edge cases:
        - Prefer architectural symbols over implementation details
        - Constructor named same as class: prefer the class symbol
        - Multiple symbols with same name: prefer by architectural significance
        
        Priority order (architectural significance):
        - High: class, interface, trait, protocol, enum, struct, record, data_class, 
                object (Kotlin/Scala singleton), module (top-level architectural)
        - Medium: function (top-level), constant (module-level), type_alias, annotation, decorator
        - Low: method, constructor, field, parameter, local_variable (implementation details)
        """
        graph._node_index = {}
        
        # Priority levels for symbol types (higher = more architecturally significant)
        type_priority = self._get_type_priority()
        
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            symbol_name = node_data.get('symbol_name', '')
            file_path = node_data.get('file_path', '')
            language = node_data.get('language', '')
            symbol_type = node_data.get('symbol_type', 'unknown')
            
            if symbol_name and file_path:
                key = (symbol_name, file_path, language)
                
                # If key exists, check priority (prefer architectural symbols)
                if key in graph._node_index:
                    existing_node_id = graph._node_index[key]
                    existing_type = graph.nodes[existing_node_id].get('symbol_type', 'unknown')
                    
                    # Replace if new symbol has higher architectural priority
                    new_priority = type_priority.get(symbol_type, 0)
                    existing_priority = type_priority.get(existing_type, 0)
                    
                    if new_priority > existing_priority:
                        graph._node_index[key] = node_id
                else:
                    graph._node_index[key] = node_id
        
        logger.debug(f"Built node index with {len(graph._node_index)} entries")

