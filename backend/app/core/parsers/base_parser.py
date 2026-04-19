"""
Base Parser Interface for Wiki Toolkit's Modular Parser Architecture

This module defines the plugin_implementation interfaces and base classes for language-specific
AST parsers that extract symbols, relationships, and context from source code.

Key Features:
- Pure AST parsing (no regex fallback)
- Standardized relationship extraction
- Hierarchical context support
- Language capability matrix
- Unified scoping model
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """Comprehensive relationship types for code analysis"""
    # Basic relationships
    IMPORTS = "imports"                    # import statements
    CALLS = "calls"                       # function/method calls
    
    # Object-oriented relationships
    INHERITANCE = "inheritance"           # class inheritance (extends/implements)
    IMPLEMENTATION = "implementation"     # interface implementation
    COMPOSITION = "composition"          # has-a relationships (strong ownership)
    AGGREGATION = "aggregation"          # has-a relationships (weak ownership)
    
    # Structural relationships
    DEFINES = "defines"                  # contains definition (class defines method)
    CONTAINS = "contains"                # structural containment
    DECORATES = "decorates"              # decorator relationships
    ANNOTATES = "annotates"              # type annotations
    
    # Data flow relationships
    ASSIGNS = "assigns"                  # variable assignments
    RETURNS = "returns"                  # return value relationships
    PARAMETER = "parameter"              # parameter relationships
    
    # Module relationships
    EXPORTS = "exports"                  # module exports
    REFERENCES = "references"            # general symbol references
    SPECIALIZES = "specializes"          # template specialization -> primary template
    INSTANTIATES = "instantiates"        # code site instantiates a template with concrete args
    DECLARES = "declares"               # declaration of a symbol (e.g., method prototype)
    DEFINES_BODY = "defines_body"        # out-of-class or separate body definition linking to declaration
    OVERRIDES = "overrides"             # derived method overrides virtual/pure virtual base
    HIDES = "hides"                     # derived method hides non-virtual base method (name shadow)
    # Phase C: type usage & object creation semantics
    USES_TYPE = "uses_type"              # a symbol (function/method) uses a type (param / return / template arg)
    CREATES = "creates"                  # object construction (stack / heap / array / placement new)
    READS = "reads"                      # function/method reads a field / external state
    WRITES = "writes"                    # function/method writes/mutates a field / external state
    CAPTURES = "captures"                # lambda captures variable / this
    ALIAS_OF = "alias_of"               # type alias / typedef ultimate target mapping


class SymbolType(Enum):
    """Types of symbols that can be extracted from code"""
    FUNCTION = "function"
    METHOD = "method"
    CONSTRUCTOR = "constructor"  # C++/Java constructors
    CLASS = "class"
    INTERFACE = "interface"
    VARIABLE = "variable"
    CONSTANT = "constant"
    PROPERTY = "property"
    FIELD = "field"
    PARAMETER = "parameter"
    MODULE = "module"
    NAMESPACE = "namespace"
    ENUM = "enum"
    STRUCT = "struct"
    TRAIT = "trait"
    TYPE_ALIAS = "type_alias"
    ANNOTATION = "annotation"
    DECORATOR = "decorator"
    MACRO = "macro"


class Scope(Enum):
    """Unified scoping model across languages"""
    GLOBAL = "global"                    # Global/module scope
    CLASS = "class"                      # Class scope
    FUNCTION = "function"                # Function/method scope (unified)
    BLOCK = "block"                      # Block scope (if/for/try/etc)
    LOCAL = "local"                      # Local variable scope


@dataclass
class Position:
    """Source code position information"""
    line: int
    column: int
    
    def __str__(self) -> str:
        return f"{self.line}:{self.column}"


@dataclass
class Range:
    """Source code range information"""
    start: Position
    end: Position
    
    def __str__(self) -> str:
        return f"{self.start}-{self.end}"
    
    def contains_line(self, line: int) -> bool:
        """Check if range contains the given line"""
        return self.start.line <= line <= self.end.line


@dataclass
class Symbol:
    """Extracted symbol information"""
    name: str
    symbol_type: SymbolType
    scope: Scope
    range: Range
    file_path: str
    
    # Hierarchical context
    parent_symbol: Optional[str] = None    # Parent class/function name
    full_name: Optional[str] = None        # Fully qualified name
    
    # Metadata
    visibility: Optional[str] = None       # public/private/protected
    is_static: bool = False
    is_abstract: bool = False
    is_async: bool = False
    
    # Documentation
    docstring: Optional[str] = None
    comments: List[str] = field(default_factory=list)
    
    # Type information
    return_type: Optional[str] = None
    parameter_types: List[str] = field(default_factory=list)
    
    # Additional context
    source_text: Optional[str] = None      # Full source code of the symbol
    signature: Optional[str] = None        # Method/function signature
    metadata: Dict[str, Any] = field(default_factory=dict)  # Flexible metadata storage
    
    def get_qualified_name(self) -> str:
        """Get the fully qualified name of the symbol"""
        if self.full_name:
            return self.full_name
        
        if self.parent_symbol:
            return f"{self.parent_symbol}.{self.name}"
        
        return self.name
    
    def is_in_scope(self, target_scope: Scope) -> bool:
        """Check if symbol is in the specified scope"""
        return self.scope == target_scope


@dataclass
class Relationship:
    """Extracted relationship between symbols"""
    source_symbol: str                     # Source symbol name
    target_symbol: str                     # Target symbol name
    relationship_type: RelationshipType
    
    # Context information
    source_file: str
    target_file: Optional[str] = None      # May be in different file
    source_range: Optional[Range] = None   # Where relationship occurs in source
    
    # Relationship metadata
    confidence: float = 1.0                # Confidence in relationship (0.0-1.0)
    weight: float = 1.0                    # Relationship strength/importance
    is_direct: bool = True                 # Direct vs inferred relationship
    
    # Additional context
    context: Optional[str] = None          # Surrounding code context
    annotations: Dict[str, Any] = field(default_factory=dict)
    
    def get_key(self) -> str:
        """Get unique key for this relationship (includes source location to distinguish multiple relationships of same type)"""
        location_key = f"{self.source_range.start.line}:{self.source_range.start.column}"
        return f"{self.source_symbol}->{self.target_symbol}:{self.relationship_type.value}@{location_key}"


@dataclass
class ParseResult:
    """Result of parsing a source file"""
    file_path: str
    language: str
    
    # Extracted elements
    symbols: List[Symbol]
    relationships: List[Relationship]
    
    # File metadata
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    
    # Context information
    module_docstring: Optional[str] = None
    file_hash: Optional[str] = None        # For caching
    parse_time: Optional[float] = None     # Performance tracking
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def get_symbols_by_type(self, symbol_type: SymbolType) -> List[Symbol]:
        """Get all symbols of a specific type"""
        return [s for s in self.symbols if s.symbol_type == symbol_type]
    
    def get_symbols_in_scope(self, scope: Scope) -> List[Symbol]:
        """Get all symbols in a specific scope"""
        return [s for s in self.symbols if s.scope == scope]
    
    def get_relationships_by_type(self, rel_type: RelationshipType) -> List[Relationship]:
        """Get all relationships of a specific type"""
        return [r for r in self.relationships if r.relationship_type == rel_type]


@dataclass
class LanguageCapabilities:
    """Capabilities matrix for a specific language parser"""
    language: str
    
    # Supported symbol types
    supported_symbols: Set[SymbolType]
    
    # Supported relationship types
    supported_relationships: Set[RelationshipType]
    
    # Parser features
    supports_ast_parsing: bool = True
    supports_type_inference: bool = False
    supports_cross_file_analysis: bool = True
    supports_incremental_parsing: bool = False
    
    # Language-specific features
    has_classes: bool = True
    has_interfaces: bool = False
    has_namespaces: bool = False
    has_generics: bool = False
    has_decorators: bool = False
    has_annotations: bool = False
    
    # Performance characteristics
    max_file_size_mb: float = 10.0         # Maximum recommended file size
    typical_parse_time_ms: float = 100.0   # Typical parse time per MB
    
    def can_parse(self, symbol_type: SymbolType) -> bool:
        """Check if parser can extract this symbol type"""
        return symbol_type in self.supported_symbols
    
    def can_extract_relationship(self, rel_type: RelationshipType) -> bool:
        """Check if parser can extract this relationship type"""
        return rel_type in self.supported_relationships


class BaseParser(ABC):
    """
    Abstract base class for language-specific parsers
    
    All language parsers must inherit from this class and implement
    the abstract methods. This ensures consistent interface and
    behavior across all supported languages.
    """
    
    def __init__(self, language: str):
        """
        Initialize parser for specific language
        
        Args:
            language: Name of the programming language
        """
        self.language = language
        self.capabilities = self._define_capabilities()
        self.logger = logging.getLogger(f"{__name__}.{language}")
    
    @abstractmethod
    def _define_capabilities(self) -> LanguageCapabilities:
        """
        Define the capabilities of this parser
        
        Returns:
            LanguageCapabilities object describing what this parser can do
        """
        pass
    
    @abstractmethod
    def parse_file(self, file_path: Union[str, Path], content: Optional[str] = None) -> ParseResult:
        """
        Parse a source file and extract symbols and relationships
        
        Args:
            file_path: Path to the source file
            content: Optional file content (if not provided, will read from file)
            
        Returns:
            ParseResult containing extracted symbols and relationships
        """
        pass
    
    @abstractmethod
    def extract_symbols(self, ast_node: Any, file_path: str) -> List[Symbol]:
        """
        Extract symbols from AST node
        
        Args:
            ast_node: Root AST node from tree-sitter or similar
            file_path: Path to the source file
            
        Returns:
            List of extracted symbols
        """
        pass
    
    @abstractmethod
    def extract_relationships(self, ast_node: Any, symbols: List[Symbol], file_path: str) -> List[Relationship]:
        """
        Extract relationships between symbols from AST
        
        Args:
            ast_node: Root AST node
            symbols: Previously extracted symbols
            file_path: Path to the source file
            
        Returns:
            List of extracted relationships
        """
        pass
    
    def get_context_for_symbol(self, symbol: Symbol, content: str, expansion_level: int = 2) -> str:
        """
        Get hierarchical context for a symbol based on expansion level
        
        Args:
            symbol: The symbol to get context for
            content: Full file content
            expansion_level: 1=symbol only, 2=+enclosing, 3=+dependencies
            
        Returns:
            Context string with appropriate level of detail
        """
        try:
            lines = content.split('\n')
            
            if expansion_level == 1:
                # Just the symbol itself
                return self._extract_symbol_content(symbol, lines)
            
            elif expansion_level == 2:
                # Symbol + enclosing class/module
                return self._extract_enclosing_context(symbol, lines)
            
            elif expansion_level == 3:
                # Symbol + enclosing + related symbols
                return self._extract_full_context(symbol, lines)
            
            else:
                return self._extract_symbol_content(symbol, lines)
                
        except Exception as e:
            self.logger.error(f"Failed to extract context for {symbol.name}: {e}")
            return symbol.source_text or symbol.name
    
    def _extract_symbol_content(self, symbol: Symbol, lines: List[str]) -> str:
        """Extract just the symbol's source code"""
        if symbol.source_text:
            return symbol.source_text
        
        try:
            start_line = max(0, symbol.range.start.line - 1)
            end_line = min(len(lines), symbol.range.end.line)
            return '\n'.join(lines[start_line:end_line])
        except Exception:
            return symbol.name
    
    def _extract_enclosing_context(self, symbol: Symbol, lines: List[str]) -> str:
        """Extract symbol + enclosing class/function context"""
        # Default implementation - can be overridden by language-specific parsers
        try:
            # Expand range to include more context
            start_line = max(0, symbol.range.start.line - 5)
            end_line = min(len(lines), symbol.range.end.line + 2)
            return '\n'.join(lines[start_line:end_line])
        except Exception:
            return self._extract_symbol_content(symbol, lines)
    
    def _extract_full_context(self, symbol: Symbol, lines: List[str]) -> str:
        """Extract symbol + enclosing + related symbols context"""
        # Default implementation - can be overridden by language-specific parsers
        try:
            # Expand range significantly
            start_line = max(0, symbol.range.start.line - 10)
            end_line = min(len(lines), symbol.range.end.line + 5)
            return '\n'.join(lines[start_line:end_line])
        except Exception:
            return self._extract_enclosing_context(symbol, lines)
    
    def validate_result(self, result: ParseResult) -> ParseResult:
        """
        Validate and clean up parse result
        
        Args:
            result: Raw parse result
            
        Returns:
            Validated and cleaned parse result
        """
        # Remove duplicate symbols
        unique_symbols = []
        seen_symbols = set()
        
        for symbol in result.symbols:
            symbol_key = f"{symbol.name}:{symbol.symbol_type.value}:{symbol.range}"
            if symbol_key not in seen_symbols:
                unique_symbols.append(symbol)
                seen_symbols.add(symbol_key)
        
        result.symbols = unique_symbols
        
        # Remove duplicate relationships
        unique_relationships = []
        seen_relationships = set()
        
        for relationship in result.relationships:
            rel_key = relationship.get_key()
            if rel_key not in seen_relationships:
                unique_relationships.append(relationship)
                seen_relationships.add(rel_key)
        
        result.relationships = unique_relationships
        
        # Validate relationship targets exist
        # Build set of both names and full names for validation
        symbol_names = {s.name for s in result.symbols}
        symbol_full_names = {s.full_name for s in result.symbols if s.full_name}
        all_symbol_identifiers = symbol_names | symbol_full_names
        
        # Also accept file-level sources (e.g. explicit template instantiation at TU scope).
        # These use the file stem as source_symbol because there's no enclosing symbol.
        file_stem = Path(result.file_path).stem if result.file_path else None
        
        validated_relationships = []
        
        for relationship in result.relationships:
            source_ok = (relationship.source_symbol in all_symbol_identifiers or
                         relationship.source_symbol == file_stem)
            target_ok = relationship.target_symbol in all_symbol_identifiers
            if source_ok or target_ok:
                validated_relationships.append(relationship)
            else:
                result.warnings.append(f"Orphaned relationship: {relationship.get_key()}")
        
        result.relationships = validated_relationships
        
        return result
    
    def get_capabilities(self) -> LanguageCapabilities:
        """Get parser capabilities"""
        return self.capabilities
    
    def supports_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if this parser can handle the given file
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if parser can handle this file
        """
        # Default implementation checks file extension
        # Can be overridden by language-specific parsers
        path = Path(file_path)
        return path.suffix.lower() in self._get_supported_extensions()
    
    @abstractmethod
    def _get_supported_extensions(self) -> Set[str]:
        """
        Get set of supported file extensions
        
        Returns:
            Set of file extensions (including the dot, e.g., {'.py', '.pyx'})
        """
        pass


class ParserRegistry:
    """
    Registry for managing language-specific parsers
    
    Provides a central point for parser discovery, instantiation,
    and capability querying across all supported languages.
    """
    
    def __init__(self):
        """Initialize empty parser registry"""
        self._parsers: Dict[str, BaseParser] = {}
        self._extension_map: Dict[str, str] = {}  # extension -> language
    
    def register_parser(self, parser: BaseParser) -> None:
        """
        Register a language parser
        
        Args:
            parser: Parser instance to register
        """
        language = parser.language.lower()
        self._parsers[language] = parser
        
        # Update extension mapping
        for ext in parser._get_supported_extensions():
            self._extension_map[ext.lower()] = language
        
        logger.info(f"Registered parser for {language}")
    
    def get_parser(self, language: str) -> Optional[BaseParser]:
        """
        Get parser for specific language
        
        Args:
            language: Language name
            
        Returns:
            Parser instance or None if not found
        """
        return self._parsers.get(language.lower())
    
    def get_parser_for_file(self, file_path: Union[str, Path]) -> Optional[BaseParser]:
        """
        Get appropriate parser for a file based on extension
        
        Args:
            file_path: Path to the file
            
        Returns:
            Parser instance or None if no suitable parser found
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        language = self._extension_map.get(extension)
        if language:
            return self._parsers.get(language)
        
        return None
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return list(self._parsers.keys())
    
    def get_supported_extensions(self) -> Set[str]:
        """Get set of all supported file extensions"""
        return set(self._extension_map.keys())
    
    def get_language_capabilities(self, language: str) -> Optional[LanguageCapabilities]:
        """
        Get capabilities for a specific language
        
        Args:
            language: Language name
            
        Returns:
            LanguageCapabilities or None if language not supported
        """
        parser = self.get_parser(language)
        return parser.get_capabilities() if parser else None


# Global parser registry instance
parser_registry = ParserRegistry()
