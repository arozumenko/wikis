"""
Enhanced TypeScript Parser implementing comprehensive TypeScript/TSX analysis.

Based on the C++ Enhanced Parser architecture (cpp_enhanced_parser.py) with TypeScript-specific adaptations.

Key Features:
- Complete TypeScript type system analysis (interfaces, types, generics, union/intersection types)
- Cross-file module resolution (import/export tracking with module dependency graph)
- TSX/React component detection (class & functional components, props extraction)
- Decorator pattern extraction (Angular, NestJS, custom decorators)
- Multi-file batch parsing with 3-pass relationship resolution
- Parallel parsing with ThreadPoolExecutor (like C++ parser)

Architecture:
- Tree-sitter based AST parsing (typescript & tsx grammars via language pack)
- Visitor pattern for symbol extraction (SymbolExtractor + RelationshipExtractor)
- Global registries for cross-file resolution (classes, interfaces, functions, types, modules)
- Three-pass parsing: Pass 1 (parse all files) → Pass 2 (build registries) → Pass 3 (enhance relationships)

Symbol Types Extracted:
- Classes (with abstract, generic, decorator metadata)
- Interfaces (with extends, generic parameters)
- Type Aliases (union, intersection, mapped types)
- Functions (arrow functions, function declarations, methods)
- Enums (numeric and string enums)
- Variables/Constants (const, let, var with type inference)
- Namespaces/Modules (ambient and regular)

Relationships Detected:
- INHERITANCE (extends for classes)
- IMPLEMENTATION (implements for classes/interfaces)
- IMPORTS/EXPORTS (module dependencies)
- CALLS (function/method invocations)
- REFERENCES (type usage, variable references)
- CREATES (object instantiation with new)
- INSTANTIATES (generic type instantiation)
- COMPOSITION/AGGREGATION (class fields, optional properties)

TypeScript-Specific Handling:
- Generic constraints and defaults (T extends Base = DefaultType)
- Optional properties and parameters (field?: string)
- Union and intersection types (A | B, A & B)
- Type guards and assertions
- Decorators with arguments (@Component({...}))
- JSX/TSX elements and React patterns
- Ambient declarations (declare module, declare global)
- Module augmentation and declaration merging
"""

import concurrent.futures
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from tree_sitter_language_pack import get_language, get_parser

from .base_parser import (
    BaseParser,
    LanguageCapabilities,
    ParseResult,
    Position,
    Range,
    Relationship,
    RelationshipType,
    Scope,
    Symbol,
    SymbolType,
)

logger = logging.getLogger(__name__)


# Global symbol registries for cross-file resolution
_global_class_locations: Dict[str, str] = {}  # class_name -> file_path
_global_interface_locations: Dict[str, str] = {}  # interface_name -> file_path
_global_function_locations: Dict[str, str] = {}  # function_name -> file_path
_global_type_locations: Dict[str, str] = {}  # type_alias_name -> file_path
_global_symbol_registry: Dict[str, str] = {}  # symbol_name -> full_path
_global_module_exports: Dict[str, Set[str]] = {}  # file_path -> exported_symbols


def _parse_single_typescript_file(file_path: str) -> Tuple[str, ParseResult]:
    """
    Worker function for parallel TypeScript file parsing.
    
    Creates a fresh parser instance per thread (tree-sitter parsers are not thread-safe).
    This is the same pattern as C++ parser's _parse_single_cpp_file.
    
    Args:
        file_path: Path to TypeScript/TSX file
        
    Returns:
        Tuple of (file_path, ParseResult)
    """
    parser = TypeScriptEnhancedParser()
    result = parser.parse_file(file_path)
    return file_path, result


class TypeScriptEnhancedParser(BaseParser):
    """
    Enhanced TypeScript Parser implementing clean modular parser architecture.
    
    Based on C++ Enhanced Parser with TypeScript-specific adaptations:
    - Comprehensive symbol extraction (classes, interfaces, types, functions, enums)
    - Full relationship detection (inheritance, implementation, imports, calls, references)
    - Generic type parameter handling (constraints, defaults, variance)
    - Decorator extraction and metadata (Angular, NestJS patterns)
    - TSX/React component detection (props, lifecycle, hooks)
    - Cross-file resolution via global registries (module dependencies)
    - Parallel parsing with batching (ThreadPoolExecutor)
    
    Supported Extensions:
    - .ts (TypeScript)
    - .tsx (TypeScript + JSX)
    - .d.ts (TypeScript declarations)
    - .mts (TypeScript module)
    - .cts (TypeScript CommonJS)
    """
    
    def __init__(self):
        super().__init__("typescript")
        self.tsx_parser = None  # Separate parser for TSX files
        self.ts_language = None
        self.tsx_language = None
        self._current_file = None
        self._current_content = None
        self._current_scope_stack: List[str] = []
        self._field_relationships = []  # Store field-level relationships for merging
        self._setup_successful = self._setup_tree_sitter()
    
    def _setup_tree_sitter(self) -> bool:
        """
        Setup TypeScript and TSX parsers using tree-sitter language pack.
        
        Uses get_language() and get_parser() which handle all tree-sitter setup automatically.
        This is the exact same pattern as C++ parser's _setup_tree_sitter.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            # TypeScript parser for .ts files
            self.ts_language = get_language('typescript')
            self.parser = get_parser('typescript')
            
            # TSX parser for .tsx files (TypeScript + JSX)
            self.tsx_language = get_language('tsx')
            self.tsx_parser = get_parser('tsx')
            
            logger.info("TypeScript parsers initialized successfully (typescript + tsx)")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize TypeScript parsers: {e}")
            self.parser = None
            self.tsx_parser = None
            return False
    
    def _define_capabilities(self) -> LanguageCapabilities:
        """
        Define TypeScript language capabilities.
        
        Maps TypeScript constructs to SymbolType and RelationshipType enums.
        Similar to C++ capabilities but adapted for TypeScript type system.
        
        Returns:
            LanguageCapabilities with supported symbols and relationships
        """
        return LanguageCapabilities(
            language="typescript",
            supported_symbols={
                # Core OOP symbols (like C++ classes/structs)
                SymbolType.CLASS,           # class declarations
                SymbolType.INTERFACE,       # interface declarations (TS-specific)
                SymbolType.TYPE_ALIAS,      # type aliases (was STRUCT — fixed to match base_parser enum)
                
                # Functions (like C++)
                SymbolType.FUNCTION,        # function declarations, arrow functions
                SymbolType.METHOD,          # class methods
                SymbolType.CONSTRUCTOR,     # class constructors
                
                # Data members (like C++)
                SymbolType.FIELD,           # class properties/fields
                SymbolType.PARAMETER,       # function/method parameters
                SymbolType.VARIABLE,        # variables (let, const, var)
                SymbolType.CONSTANT,        # constants, enum members
                
                # Organizational (like C++ namespaces)
                SymbolType.MODULE           # namespaces, modules, exports
            },
            supported_relationships={
                # OOP relationships (like C++)
                RelationshipType.INHERITANCE,       # extends for classes
                RelationshipType.IMPLEMENTATION,    # implements for classes
                RelationshipType.COMPOSITION,       # class fields (required)
                RelationshipType.AGGREGATION,       # optional fields, interface properties
                
                # Code relationships (like C++)
                RelationshipType.CALLS,             # function/method calls
                RelationshipType.REFERENCES,        # variable/type usage
                RelationshipType.CREATES,           # object instantiation (new)
                RelationshipType.INSTANTIATES,      # generic instantiation
                
                # Module relationships (TS-specific, similar to C++ includes)
                RelationshipType.IMPORTS,           # import statements
                # RelationshipType.EXPORTS          # export statements (if we add EXPORTS type)
            },
            supports_ast_parsing=True,
            has_generics=True  # TypeScript generics (like C++ templates)
        )
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        TypeScript supports multiple extensions for different module systems.
        
        Returns:
            List of file extensions (.ts, .tsx, .d.ts, .mts, .cts)
        """
        return ['.ts', '.tsx', '.d.ts', '.mts', '.cts']
    
    def _get_supported_extensions(self) -> Set[str]:
        """
        Get set of supported file extensions (required by BaseParser).
        
        Returns:
            Set of file extensions including the dot
        """
        return {'.ts', '.tsx', '.d.ts', '.mts', '.cts'}
    
    def can_parse(self, file_path: str) -> bool:
        """
        Check if this parser can handle the file.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file extension is supported and parser is initialized
        """
        if not self._setup_successful:
            return False
        
        ext = Path(file_path).suffix.lower()
        return ext in self.get_supported_extensions()
    
    def parse_batch(self, file_paths: List[Union[str, Path]]) -> List[ParseResult]:
        """
        Parse multiple TypeScript files in batch.
        
        Simple sequential parsing for compatibility.
        For large batches, use parse_multiple_files instead (parallel + cross-file resolution).
        
        Args:
            file_paths: List of file paths to parse
            
        Returns:
            List of ParseResult objects, one for each file
        """
        results = []
        for file_path in file_paths:
            try:
                result = self.parse_file(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                # Add error result
                results.append(ParseResult(
                    file_path=str(file_path),
                    language="typescript",
                    symbols=[],
                    relationships=[],
                    parse_time=0.0,
                    errors=[f"Parse error: {e}"]
                ))
        return results
    
    def parse_file(self, file_path: Union[str, Path], content: Optional[str] = None) -> ParseResult:
        """
        Parse a single TypeScript/TSX file and extract symbols and relationships.
        
        Args:
            file_path: Path to TypeScript/TSX file
            content: Optional file content (if None, reads from file_path)
            
        Returns:
            ParseResult with symbols, relationships, and metadata
        """
        start_time = time.time()
        file_path = str(file_path)
        self._current_file = file_path
        
        try:
            # Read content if not provided
            if content is None:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            
            self._current_content = content
            
            # Calculate file hash
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Select parser based on file extension (TSX vs TS)
            parser_to_use = self.tsx_parser if file_path.endswith('.tsx') else self.parser
            
            # Parse with tree-sitter
            if parser_to_use is None:
                return ParseResult(
                    file_path=file_path,
                    language="typescript",
                    symbols=[],
                    relationships=[],
                    file_hash=file_hash,
                    parse_time=time.time() - start_time,
                    errors=["Tree-sitter parser not available"]
                )
            
            tree = parser_to_use.parse(bytes(content, 'utf8'))
            if tree.root_node is None:
                return ParseResult(
                    file_path=file_path,
                    language="typescript",
                    symbols=[],
                    relationships=[],
                    file_hash=file_hash,
                    parse_time=time.time() - start_time,
                    errors=["Failed to parse TypeScript file"]
                )
            
            # Extract symbols and relationships
            symbols = self.extract_symbols(tree.root_node, file_path)
            relationships = self.extract_relationships(tree.root_node, symbols, file_path)
            
            result = ParseResult(
                file_path=file_path,
                language="typescript",
                symbols=symbols,
                relationships=relationships,
                file_hash=file_hash,
                parse_time=time.time() - start_time
            )
            
            return self.validate_result(result)
            
        except Exception as e:
            logger.error(f"Failed to parse TypeScript file {file_path}: {e}")
            return ParseResult(
                file_path=file_path,
                language="typescript",
                symbols=[],
                relationships=[],
                parse_time=time.time() - start_time,
                errors=[f"Parse error: {e}"]
            )
    
    def extract_symbols(self, ast_node: Any, file_path: str) -> List[Symbol]:
        """
        Extract symbols from TypeScript tree-sitter AST.
        
        Uses visitor pattern like C++ parser with SymbolExtractor class.
        
        Args:
            ast_node: Root tree-sitter AST node
            file_path: Path to source file
            
        Returns:
            List of Symbol objects
        """
        symbols = []
        self._current_scope_stack = []
        
        class SymbolExtractor:
            """Extracts symbols from TypeScript AST using visitor pattern (C++ style)"""
            
            def __init__(self, parser):
                self.parser = parser
                self.symbols = []
                self.relationships = []  # Field-level relationships (composition/aggregation)
                self.scope_stack = []    # Current namespace/module/class scope (dot-separated)
                self.module_stack = []   # Track module boundaries
                self.current_generic_params = None  # Like C++ template_params
                self.current_decorators = []  # TypeScript decorators
                self.tsx_mode = file_path.endswith('.tsx')  # Track if processing TSX
            
            def visit(self, node):
                """Main visitor dispatch (C++ pattern)"""
                # Nodes that handle recursion manually (don't auto-recurse)
                no_recurse_types = {
                    'class_declaration', 'abstract_class_declaration', 'interface_declaration',
                    'function_declaration', 'arrow_function', 'method_definition',
                    'namespace_declaration', 'module_declaration', 'internal_module',
                    'enum_declaration', 'type_alias_declaration'
                }
                
                # Dispatch to specific visitor
                method_name = f'visit_{node.type}'
                handled = False
                if hasattr(self, method_name):
                    getattr(self, method_name)(node)
                    handled = True
                
                # Skip recursion for manually-handled nodes
                if handled and node.type in no_recurse_types:
                    return
                
                # Visit children
                for child in node.children:
                    self.visit(child)
            
            def visit_program(self, node):
                """Root node - like C++ translation_unit.
                
                Track module in scope for qualified names but don't emit a MODULE symbol
                (modules are organizational, not architectural).
                """
                from pathlib import Path
                module_name = Path(file_path).stem
                
                # Track module in scope for children (needed for qualified full_name)
                self.scope_stack.append(module_name)
            
            def visit_class_declaration(self, node):
                """Extract CLASS symbol - similar to C++ class_specifier"""
                # Get class name
                name_node = self.parser._find_child_by_type(node, 'type_identifier')
                if not name_node:
                    return
                
                name = self.parser._get_node_text(name_node)
                parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}.{name}" if parent_symbol else name
                
                # Check for abstract class modifier
                is_abstract = self._has_modifier(node, 'abstract')
                
                # Check for export modifier
                is_exported = self._has_modifier(node, 'export')
                
                # Extract generic parameters (like C++ templates)
                generic_params = self._extract_generic_parameters(node)
                
                # Extract heritage (extends/implements)
                extends_clause = self.parser._find_child_by_type(node, 'class_heritage')
                extends_types = []
                implements_types = []
                
                if extends_clause:
                    # Parse extends and implements clauses
                    extends_types, implements_types = self._parse_class_heritage(extends_clause)
                
                # Create symbol
                symbol = Symbol(
                    name=name,
                    symbol_type=SymbolType.CLASS,
                    scope=Scope.CLASS if self.scope_stack else Scope.GLOBAL,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    source_text=self.parser._get_node_text(node),
                    is_abstract=is_abstract  # Unified API field
                )
                
                # Add metadata for language-specific features
                metadata = {}
                if is_exported:
                    metadata['is_exported'] = True
                if generic_params:
                    metadata['is_generic'] = True
                    metadata['generic_params'] = generic_params
                if self.current_decorators:
                    metadata['decorators'] = self.current_decorators.copy()
                    self.current_decorators.clear()
                
                # Check if React component (TSX-specific)
                if self.tsx_mode and self._is_react_component(extends_types):
                    metadata['is_react_component'] = True
                    metadata['component_type'] = 'class'
                
                if metadata:
                    symbol.metadata = metadata
                
                self.symbols.append(symbol)
                
                # Enter scope and visit children (C++ pattern)
                self.scope_stack.append(name)
                
                body = self.parser._find_child_by_type(node, 'class_body')
                if body:
                    for child in body.children:
                        self.visit(child)
                
                self.scope_stack.pop()
                
                # Extract INHERITANCE relationships
                for extends_type in extends_types:
                    relationship = Relationship(
                        source_symbol=full_name,
                        target_symbol=extends_type,
                        relationship_type=RelationshipType.INHERITANCE,
                        source_file=file_path,
                        source_range=self.parser._node_to_range(node)
                    )
                    self.relationships.append(relationship)
                
                # Extract IMPLEMENTATION relationships
                for impl_type in implements_types:
                    relationship = Relationship(
                        source_symbol=full_name,
                        target_symbol=impl_type,
                        relationship_type=RelationshipType.IMPLEMENTATION,
                        source_file=file_path,
                        source_range=self.parser._node_to_range(node)
                    )
                    self.relationships.append(relationship)
            
            def visit_abstract_class_declaration(self, node):
                """Extract ABSTRACT CLASS symbol - delegates to visit_class_declaration"""
                # Tree-sitter uses a separate node type for abstract classes
                # Just delegate to the regular class handler, which already checks for abstract modifier
                # Debug: Uncomment to trace execution
                # print(f"DEBUG: visit_abstract_class_declaration called")
                self.visit_class_declaration(node)
            
            def visit_interface_declaration(self, node):
                """Extract INTERFACE symbol - TypeScript-specific"""
                # Get interface name
                name_node = self.parser._find_child_by_type(node, 'type_identifier')
                if not name_node:
                    return
                
                name = self.parser._get_node_text(name_node)
                parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}.{name}" if parent_symbol else name
                
                # Check for export modifier
                is_exported = self._has_modifier(node, 'export')
                
                # Extract generic parameters
                generic_params = self._extract_generic_parameters(node)
                
                # Extract extends clause (interfaces can extend multiple)
                extends_types = self._extract_interface_extends(node)
                
                # Create symbol (use INTERFACE type)
                symbol = Symbol(
                    name=name,
                    symbol_type=SymbolType.INTERFACE,
                    scope=Scope.GLOBAL if not parent_symbol else Scope.CLASS,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    source_text=self.parser._get_node_text(node)
                )
                
                # Metadata for language-specific features
                metadata = {}
                if is_exported:
                    metadata['is_exported'] = True
                if generic_params:
                    metadata['is_generic'] = True
                    metadata['generic_params'] = generic_params
                if self.current_decorators:
                    metadata['decorators'] = self.current_decorators.copy()
                    self.current_decorators.clear()
                
                if metadata:
                    symbol.metadata = metadata
                
                self.symbols.append(symbol)
                
                # Enter scope for members
                self.scope_stack.append(name)
                
                body = self.parser._find_child_by_type(node, 'interface_body')
                if body:
                    for child in body.children:
                        self.visit(child)
                
                self.scope_stack.pop()
                
                # Extract INHERITANCE relationships (interface extends)
                for extends_type in extends_types:
                    relationship = Relationship(
                        source_symbol=full_name,
                        target_symbol=extends_type,
                        relationship_type=RelationshipType.INHERITANCE,
                        source_file=file_path,
                        source_range=self.parser._node_to_range(node)
                    )
                    self.relationships.append(relationship)
            
            def visit_type_alias_declaration(self, node):
                """Extract TYPE ALIAS - TypeScript-specific (like C++ typedef/using)"""
                name_node = self.parser._find_child_by_type(node, 'type_identifier')
                if not name_node:
                    return
                
                name = self.parser._get_node_text(name_node)
                parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}.{name}" if parent_symbol else name
                
                # Check for export modifier
                is_exported = self._has_modifier(node, 'export')
                
                # Extract the aliased type
                type_node = self.parser._find_child_by_type(node, ['union_type', 'intersection_type', 
                                                                    'object_type', 'type_identifier'])
                aliased_type = self.parser._get_node_text(type_node) if type_node else None
                
                # Extract generic parameters
                generic_params = self._extract_generic_parameters(node)
                
                # Create symbol — use TYPE_ALIAS (not STRUCT)
                symbol = Symbol(
                    name=name,
                    symbol_type=SymbolType.TYPE_ALIAS,  # Fixed: was STRUCT, now matches base_parser enum & C++ parser
                    scope=Scope.GLOBAL if not parent_symbol else Scope.CLASS,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    return_type=aliased_type,
                    metadata={
                        'is_type_alias': True,
                        'aliased_type': aliased_type,
                        'is_exported': is_exported,
                        'is_generic': bool(generic_params),
                        'generic_params': generic_params if generic_params else None
                    }
                )
                self.symbols.append(symbol)
            
            def visit_enum_declaration(self, node):
                """Extract ENUM symbol"""
                # Get enum name
                name_node = self.parser._find_child_by_type(node, 'identifier')
                if not name_node:
                    return
                
                name = self.parser._get_node_text(name_node)
                parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}.{name}" if parent_symbol else name
                
                # Check for export and const modifiers
                is_exported = self._has_modifier(node, 'export')
                is_const = self._has_modifier(node, 'const')
                
                # Create enum symbol
                symbol = Symbol(
                    name=name,
                    symbol_type=SymbolType.ENUM,  # Use ENUM type properly
                    scope=Scope.CLASS if self.scope_stack else Scope.GLOBAL,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    source_text=self.parser._get_node_text(node)
                )
                
                # Metadata for language-specific features
                metadata = {
                    'is_enum': True,
                    'is_const_enum': is_const
                }
                if is_exported:
                    metadata['is_exported'] = True
                
                symbol.metadata = metadata
                self.symbols.append(symbol)
                
                # Extract enum members as CONSTANT children
                self.scope_stack.append(name)
                
                enum_body = self.parser._find_child_by_type(node, 'enum_body')
                if enum_body:
                    for child in enum_body.children:
                        if child.type == 'property_identifier':
                            self._extract_enum_member(child, full_name)
                
                self.scope_stack.pop()
            
            def visit_function_declaration(self, node):
                """Extract FUNCTION symbol"""
                # Get function name
                name_node = self.parser._find_child_by_type(node, 'identifier')
                if not name_node:
                    return
                
                name = self.parser._get_node_text(name_node)
                parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}.{name}" if parent_symbol else name
                
                # Check for export and async modifiers
                is_exported = self._has_modifier(node, 'export')
                is_async = self._has_modifier(node, 'async')
                
                # Extract generic parameters
                generic_params = self._extract_generic_parameters(node)
                
                # Extract parameters
                parameters = self._extract_function_parameters(node)
                
                # Extract return type
                return_type = self._extract_return_type(node)
                
                # Create function symbol
                symbol = Symbol(
                    name=name,
                    symbol_type=SymbolType.FUNCTION,
                    scope=Scope.FUNCTION,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    return_type=return_type,
                    parameter_types=[p.get('type') for p in parameters if p.get('type')],
                    source_text=self.parser._get_node_text(node),
                    is_async=is_async  # Unified API field
                )
                
                # Metadata for language-specific features
                metadata = {}
                if is_exported:
                    metadata['is_exported'] = True
                if generic_params:
                    metadata['is_generic'] = True
                    metadata['generic_params'] = generic_params
                if self.current_decorators:
                    metadata['decorators'] = self.current_decorators.copy()
                    self.current_decorators.clear()
                
                # Check if React functional component (TSX)
                if self.tsx_mode and self._is_react_functional_component(node):
                    metadata['is_react_component'] = True
                    metadata['component_type'] = 'functional'
                
                if metadata:
                    symbol.metadata = metadata
                
                self.symbols.append(symbol)
                
                # Add PARAMETER symbols
                for param in parameters:
                    param_symbol = Symbol(
                        name=param['name'],
                        symbol_type=SymbolType.PARAMETER,
                        scope=Scope.FUNCTION,
                        range=param.get('range', symbol.range),
                        file_path=file_path,
                        parent_symbol=full_name,
                        full_name=f"{full_name}::{param['name']}",
                        return_type=param.get('type')
                    )
                    self.symbols.append(param_symbol)
            
            def visit_method_definition(self, node):
                """Extract METHOD symbol from class"""
                # Get method name
                name_node = self.parser._find_child_by_type(node, ['property_identifier', 'identifier'])
                if not name_node:
                    return
                
                name = self.parser._get_node_text(name_node)
                
                # Skip if this is a constructor (handled separately)
                if name == 'constructor':
                    self._extract_constructor(node)
                    return
                
                parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}.{name}" if parent_symbol else name
                
                # Check modifiers
                is_static = self._has_modifier(node, 'static')
                is_async = self._has_modifier(node, 'async')
                is_abstract = self._has_modifier(node, 'abstract')
                access_modifier = self._get_access_modifier(node)  # public/private/protected
                
                # Extract generic parameters
                generic_params = self._extract_generic_parameters(node)
                
                # Extract parameters
                parameters = self._extract_function_parameters(node)
                
                # Extract return type
                return_type = self._extract_return_type(node)
                
                # Create method symbol
                symbol = Symbol(
                    name=name,
                    symbol_type=SymbolType.METHOD,
                    scope=Scope.FUNCTION,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    return_type=return_type,
                    parameter_types=[p.get('type') for p in parameters if p.get('type')],
                    source_text=self.parser._get_node_text(node),
                    is_static=is_static,       # Unified API field
                    is_async=is_async,         # Unified API field
                    is_abstract=is_abstract,   # Unified API field
                    visibility=access_modifier # Unified API field
                )
                
                # Metadata for language-specific features
                metadata = {}
                if generic_params:
                    metadata['is_generic'] = True
                    metadata['generic_params'] = generic_params
                if self.current_decorators:
                    metadata['decorators'] = self.current_decorators.copy()
                    self.current_decorators.clear()
                
                if metadata:
                    symbol.metadata = metadata
                
                self.symbols.append(symbol)
                
                # Add PARAMETER symbols
                for param in parameters:
                    param_symbol = Symbol(
                        name=param['name'],
                        symbol_type=SymbolType.PARAMETER,
                        scope=Scope.FUNCTION,
                        range=param.get('range', symbol.range),
                        file_path=file_path,
                        parent_symbol=full_name,
                        full_name=f"{full_name}::{param['name']}",
                        return_type=param.get('type')
                    )
                    self.symbols.append(param_symbol)
            
            def visit_public_field_definition(self, node):
                """Extract class field/property"""
                self._extract_class_field(node)
            
            def visit_method_signature(self, node):
                """Extract METHOD symbol from interface (TypeScript-specific)"""
                # Get method name
                name_node = self.parser._find_child_by_type(node, 'property_identifier')
                if not name_node:
                    return
                
                name = self.parser._get_node_text(name_node)
                parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}.{name}" if parent_symbol else name
                
                # Extract parameters
                parameters = self._extract_function_parameters(node)
                
                # Extract return type
                return_type = self._extract_return_type(node)
                
                # Create method symbol (interface methods are always abstract)
                symbol = Symbol(
                    name=name,
                    symbol_type=SymbolType.METHOD,
                    scope=Scope.FUNCTION,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    return_type=return_type,
                    parameter_types=[p.get('type') for p in parameters if p.get('type')],
                    source_text=self.parser._get_node_text(node),
                    is_abstract=True,  # Interface methods are always abstract
                    is_static=False
                )
                
                self.symbols.append(symbol)
            
            def visit_property_signature(self, node):
                """Extract PROPERTY/FIELD symbol from interface (TypeScript-specific)"""
                # Get property name
                name_node = self.parser._find_child_by_type(node, 'property_identifier')
                if not name_node:
                    return
                
                name = self.parser._get_node_text(name_node)
                parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}.{name}" if parent_symbol else name
                
                # Check for optional marker (?)
                is_optional = any(child.type == '?' for child in node.children)
                
                # Check for readonly modifier
                is_readonly = self._has_modifier(node, 'readonly')
                
                # Extract type annotation
                type_annotation = self._extract_type_annotation(node)
                
                # Create property symbol
                symbol = Symbol(
                    name=name,
                    symbol_type=SymbolType.PROPERTY,
                    scope=Scope.CLASS,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    return_type=type_annotation,
                    source_text=self.parser._get_node_text(node),
                    is_static=False
                )
                
                # Add TypeScript-specific metadata
                metadata = {}
                if is_optional:
                    metadata['is_optional'] = True
                if is_readonly:
                    metadata['is_readonly'] = True
                
                if metadata:
                    symbol.metadata = metadata
                
                self.symbols.append(symbol)
                
                # Create COMPOSITION/AGGREGATION relationship for property type
                # This ensures interface properties with user-defined types create relationships
                if type_annotation:
                    self._extract_field_type_relationship(type_annotation, full_name, name, node)
            
            def visit_abstract_method_signature(self, node):
                """Extract abstract METHOD symbol from abstract class (TypeScript-specific)"""
                # Get method name
                name_node = self.parser._find_child_by_type(node, 'property_identifier')
                if not name_node:
                    return
                
                name = self.parser._get_node_text(name_node)
                parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}.{name}" if parent_symbol else name
                
                # Extract parameters
                parameters = self._extract_function_parameters(node)
                
                # Extract return type
                return_type = self._extract_return_type(node)
                
                # Create method symbol (abstract methods are always abstract)
                symbol = Symbol(
                    name=name,
                    symbol_type=SymbolType.METHOD,
                    scope=Scope.FUNCTION,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    return_type=return_type,
                    parameter_types=[p.get('type') for p in parameters if p.get('type')],
                    source_text=self.parser._get_node_text(node),
                    is_abstract=True,  # Abstract method signature
                    is_static=False
                )
                
                self.symbols.append(symbol)
            
            def visit_lexical_declaration(self, node):
                """Extract variables (const, let, var)"""
                # Check if this is at top-level (not inside a function)
                if not self.scope_stack or self.scope_stack[0] == Path(file_path).stem:
                    # Top-level variable
                    self._extract_variable_declaration(node)
            
            def visit_variable_declaration(self, node):
                """Extract variable from variable_declarator"""
                self._extract_variable_from_declarator(node)
            
            def visit_import_statement(self, node):
                """Extract IMPORTS relationships"""
                # Get source module
                source_node = self.parser._find_child_by_type(node, 'string')
                if not source_node:
                    return
                
                source_module = self.parser._get_node_text(source_node).strip('"').strip("'")
                
                # Get imported symbols
                import_clause = self.parser._find_child_by_type(node, 'import_clause')
                if import_clause:
                    imported_symbols = self._extract_imported_symbols(import_clause)
                    
                    # Create IMPORTS relationships for each imported symbol
                    for imported_name in imported_symbols:
                        relationship = Relationship(
                            source_symbol=Path(file_path).stem,  # File-level import
                            target_symbol=f"{source_module}::{imported_name}",
                            relationship_type=RelationshipType.IMPORTS,
                            source_file=file_path,
                            source_range=self.parser._node_to_range(node),
                            annotations={
                                'import_type': 'named',
                                'source_module': source_module,
                                'imported_name': imported_name
                            }
                        )
                        self.relationships.append(relationship)
            
            def visit_decorator(self, node):
                """Extract decorator information (Angular/NestJS pattern)"""
                # Get decorator name (could be call_expression or identifier)
                decorator_name = None
                call_expr = self.parser._find_child_by_type(node, 'call_expression')
                
                if call_expr:
                    identifier = self.parser._find_child_by_type(call_expr, 'identifier')
                    if identifier:
                        decorator_name = self.parser._get_node_text(identifier)
                else:
                    identifier = self.parser._find_child_by_type(node, 'identifier')
                    if identifier:
                        decorator_name = self.parser._get_node_text(identifier)
                
                if decorator_name:
                    # Store for next symbol
                    self.current_decorators.append({
                        'name': decorator_name,
                        'range': self.parser._node_to_range(node)
                    })
            
            # Helper methods
            
            def _has_modifier(self, node, modifier_name: str) -> bool:
                """Check if node has a specific modifier (export, abstract, async, etc.)"""
                # Check direct children for modifier
                for child in node.children:
                    if child.type == modifier_name:
                        return True
                # Check parent for export
                if modifier_name == 'export' and node.parent:
                    for child in node.parent.children:
                        if child.type == 'export':
                            return True
                return False
            
            def _get_access_modifier(self, node) -> Optional[str]:
                """Get access modifier (public, private, protected)"""
                # Try direct children first
                for child in node.children:
                    if child.type in ['public', 'private', 'protected']:
                        return child.type
                    # TypeScript may use accessibility_modifier wrapper
                    if child.type == 'accessibility_modifier':
                        return self.parser._get_node_text(child)
                return None
            
            def _extract_generic_parameters(self, node) -> List[str]:
                """Extract generic type parameters (like C++ templates)"""
                params = []
                type_params = self.parser._find_child_by_type(node, 'type_parameters')
                if type_params:
                    for child in type_params.children:
                        if child.type == 'type_parameter':
                            name_node = self.parser._find_child_by_type(child, 'type_identifier')
                            if name_node:
                                params.append(self.parser._get_node_text(name_node))
                return params
            
            def _parse_class_heritage(self, heritage_node):
                """Parse class heritage clause (extends + implements)"""
                extends_types = []
                implements_types = []
                
                if not heritage_node:
                    return extends_types, implements_types
                
                for child in heritage_node.children:
                    if child.type == 'extends_clause':
                        # Get extended class - can be identifier or type_identifier
                        type_node = self.parser._find_child_by_type(child, ['identifier', 'type_identifier', 'generic_type', 'member_expression'])
                        if type_node:
                            extends_types.append(self.parser._get_node_text(type_node))
                    
                    elif child.type == 'implements_clause':
                        # Get implemented interfaces (can be multiple)
                        # Check all children for type identifiers, including generic types
                        for subchild in child.children:
                            if subchild.type in ['type_identifier', 'identifier', 'generic_type']:
                                impl_type = self.parser._get_node_text(subchild)
                                if impl_type not in implements_types and impl_type != 'implements':
                                    implements_types.append(impl_type)
                
                return extends_types, implements_types
            
            def _extract_interface_extends(self, node) -> List[str]:
                """Extract interface extends clause
                
                Handles both node types:
                - extends_type_clause: TypeScript's tree-sitter uses this for interface extends
                - extends_clause: Some versions may use this
                """
                extends_types = []
                
                # Try extends_type_clause first (TypeScript's actual AST node type)
                extends_clause = self.parser._find_child_by_type(node, 'extends_type_clause')
                if not extends_clause:
                    # Fallback to extends_clause
                    extends_clause = self.parser._find_child_by_type(node, 'extends_clause')
                
                if extends_clause:
                    # Can extend multiple interfaces
                    for child in extends_clause.children:
                        if child.type in ['type_identifier', 'generic_type']:
                            extends_types.append(self.parser._get_node_text(child))
                
                return extends_types
            
            def _extract_function_parameters(self, node) -> List[Dict]:
                """Extract function parameters with types"""
                params = []
                formal_params = self.parser._find_child_by_type(node, 'formal_parameters')
                
                if formal_params:
                    for child in formal_params.children:
                        if child.type in ['required_parameter', 'optional_parameter']:
                            param_info = self._extract_parameter_info(child)
                            if param_info:
                                params.append(param_info)
                
                return params
            
            def _extract_parameter_info(self, param_node) -> Optional[Dict]:
                """Extract parameter name and type"""
                # Get parameter name
                identifier = self.parser._find_child_by_type(param_node, 'identifier')
                if not identifier:
                    return None
                
                param_name = self.parser._get_node_text(identifier)
                
                # Get parameter type annotation
                type_annotation = self.parser._find_child_by_type(param_node, 'type_annotation')
                param_type = None
                
                if type_annotation:
                    # Type annotation contains the actual type
                    for child in type_annotation.children:
                        if child.type != ':':
                            param_type = self.parser._get_node_text(child)
                            break
                
                # Check if optional
                is_optional = param_node.type == 'optional_parameter'
                
                return {
                    'name': param_name,
                    'type': param_type,
                    'is_optional': is_optional,
                    'range': self.parser._node_to_range(param_node)
                }
            
            def _extract_return_type(self, node) -> Optional[str]:
                """Extract return type from function/method"""
                type_annotation = self.parser._find_child_by_type(node, 'type_annotation')
                if type_annotation:
                    # Skip the ':' and get the actual type
                    for child in type_annotation.children:
                        if child.type != ':':
                            return self.parser._get_node_text(child)
                return None
            
            def _extract_type_annotation(self, node) -> Optional[str]:
                """Extract type annotation from property/field (alias for _extract_return_type)"""
                return self._extract_return_type(node)
            
            def _extract_react_fc_prop_types(self, type_node) -> List[Dict]:
                """
                Extract prop types from React.FC<PropsType> or React.FunctionComponent<PropsType>.
                
                Handles patterns like:
                - React.FC<{users: User[]}>
                - React.FunctionComponent<{name: string, age: number}>
                - FC<PropsInterface>
                
                Returns list of parameter dicts with 'name' and 'type' keys.
                """
                prop_types = []
                
                # Check if this is a generic_type (e.g., React.FC<...>)
                if type_node.type != 'generic_type':
                    return prop_types
                
                # Get the base type name (should be FC, FunctionComponent, etc.)
                type_identifier = None
                for child in type_node.children:
                    if child.type in ['type_identifier', 'member_expression', 'nested_type_identifier']:
                        type_text = self.parser._get_node_text(child)
                        # Check if it's a React component type
                        if 'FC' in type_text or 'FunctionComponent' in type_text:
                            type_identifier = type_text
                            break
                
                if not type_identifier:
                    return prop_types
                
                # Find type_arguments node containing the props type
                type_arguments = self.parser._find_child_by_type(type_node, 'type_arguments')
                if not type_arguments:
                    return prop_types
                
                # Get the first type argument (the props type)
                for child in type_arguments.children:
                    if child.type in ['object_type', 'type_identifier', 'type_reference']:
                        if child.type == 'object_type':
                            # Inline props: React.FC<{users: User[], count: number}>
                            prop_types.extend(self._extract_object_type_properties(child))
                        else:
                            # Named props type: React.FC<UserListProps>
                            props_type_name = self.parser._get_node_text(child)
                            # Return a single parameter representing the props object
                            prop_types.append({
                                'name': 'props',
                                'type': props_type_name,
                                'is_optional': False
                            })
                        break
                
                return prop_types
            
            def _extract_object_type_properties(self, object_type_node) -> List[Dict]:
                """
                Extract properties from an object type like {users: User[], count: number}.
                Returns list of dicts with 'name' and 'type' keys.
                """
                properties = []
                
                for child in object_type_node.children:
                    if child.type == 'property_signature':
                        # Get property name
                        prop_name_node = self.parser._find_child_by_type(child, 'property_identifier')
                        if not prop_name_node:
                            continue
                        
                        prop_name = self.parser._get_node_text(prop_name_node)
                        
                        # Get property type
                        type_annotation = self.parser._find_child_by_type(child, 'type_annotation')
                        prop_type = None
                        if type_annotation:
                            for type_child in type_annotation.children:
                                if type_child.type != ':':
                                    prop_type = self.parser._get_node_text(type_child)
                                    break
                        
                        # Check if optional (has '?' in signature)
                        is_optional = '?' in self.parser._get_node_text(child)
                        
                        if prop_type:
                            properties.append({
                                'name': prop_name,
                                'type': prop_type,
                                'is_optional': is_optional
                            })
                
                return properties
            
            def _extract_constructor(self, node):
                """Extract CONSTRUCTOR symbol"""
                parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}.constructor" if parent_symbol else "constructor"
                
                # Extract parameters
                parameters = self._extract_function_parameters(node)
                
                # Create constructor symbol
                symbol = Symbol(
                    name='constructor',
                    symbol_type=SymbolType.CONSTRUCTOR,
                    scope=Scope.FUNCTION,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    parameter_types=[p.get('type') for p in parameters if p.get('type')],
                    source_text=self.parser._get_node_text(node)
                )
                self.symbols.append(symbol)
                
                # Add PARAMETER symbols
                for param in parameters:
                    param_symbol = Symbol(
                        name=param['name'],
                        symbol_type=SymbolType.PARAMETER,
                        scope=Scope.FUNCTION,
                        range=param.get('range', symbol.range),
                        file_path=file_path,
                        parent_symbol=full_name,
                        full_name=f"{full_name}::{param['name']}",
                        return_type=param.get('type')
                    )
                    self.symbols.append(param_symbol)
            
            def _extract_class_field(self, node):
                """Extract class field/property"""
                if not self.scope_stack:
                    return
                
                # Get field name
                name_node = self.parser._find_child_by_type(node, ['property_identifier', 'identifier'])
                if not name_node:
                    return
                
                field_name = self.parser._get_node_text(name_node)
                parent_symbol = '.'.join(self.scope_stack)
                full_name = f"{parent_symbol}.{field_name}"
                
                # Get access modifier
                access_modifier = self._get_access_modifier(node)
                
                # Check if static, readonly, or optional
                is_static = self._has_modifier(node, 'static')
                is_readonly = self._has_modifier(node, 'readonly')
                is_optional = any(child.type == '?' for child in node.children)
                
                # Get field type
                type_annotation = self.parser._find_child_by_type(node, 'type_annotation')
                field_type = None
                if type_annotation:
                    for child in type_annotation.children:
                        if child.type != ':':
                            field_type = self.parser._get_node_text(child)
                            break
                
                # Create field symbol with unified API fields
                symbol = Symbol(
                    name=field_name,
                    symbol_type=SymbolType.FIELD,
                    scope=Scope.CLASS,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    return_type=field_type,
                    is_static=is_static,           # Unified API field
                    visibility=access_modifier     # Unified API field
                )
                
                # Metadata for TypeScript-specific features
                metadata = {}
                if is_readonly:
                    metadata['is_readonly'] = True
                if is_optional:
                    metadata['is_optional'] = True
                if self.current_decorators:
                    metadata['decorators'] = self.current_decorators.copy()
                    self.current_decorators.clear()
                
                if metadata:
                    symbol.metadata = metadata
                
                self.symbols.append(symbol)
                
                # Create COMPOSITION/AGGREGATION relationship for field type
                if field_type:
                    self._extract_field_type_relationship(field_type, full_name, field_name, node)
            
            def _extract_variable_declaration(self, node):
                """Extract top-level variable declaration"""
                # Get kind (const, let, var)
                kind = None
                for child in node.children:
                    if child.type in ['const', 'let', 'var']:
                        kind = child.type
                        break
                
                # Get variable declarator
                declarator = self.parser._find_child_by_type(node, 'variable_declarator')
                if declarator:
                    self._extract_variable_from_declarator(declarator, kind)
            
            def _extract_variable_from_declarator(self, node, kind='const'):
                """Extract variable from variable_declarator"""
                # Get variable name
                identifier = self.parser._find_child_by_type(node, 'identifier')
                if not identifier:
                    return
                
                var_name = self.parser._get_node_text(identifier)
                parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}.{var_name}" if parent_symbol else var_name
                
                # Check if this is an arrow function assignment (React functional component pattern)
                arrow_func = self.parser._find_child_by_type(node, 'arrow_function')
                if arrow_func:
                    # Extract as a function instead of variable
                    self._extract_arrow_function_as_symbol(node, arrow_func, var_name, full_name, parent_symbol)
                    return
                
                # Get type annotation
                type_annotation = self.parser._find_child_by_type(node, 'type_annotation')
                var_type = None
                if type_annotation:
                    for child in type_annotation.children:
                        if child.type != ':':
                            var_type = self.parser._get_node_text(child)
                            break
                
                # Check if exported
                is_exported = node.parent and self._has_modifier(node.parent.parent, 'export')
                
                # Create variable/constant symbol
                symbol_type = SymbolType.CONSTANT if kind == 'const' else SymbolType.VARIABLE
                
                symbol = Symbol(
                    name=var_name,
                    symbol_type=symbol_type,
                    scope=Scope.GLOBAL if not parent_symbol else Scope.CLASS,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    return_type=var_type,
                    source_text=self.parser._get_node_text(node),  # Include source text for constants
                    metadata={
                        'kind': kind,
                        'is_exported': is_exported
                    }
                )
                self.symbols.append(symbol)
            
            def _extract_arrow_function_as_symbol(self, declarator_node, arrow_func_node, name, full_name, parent_symbol):
                """Extract arrow function assigned to const/let as a FUNCTION symbol"""
                # Get type annotation (may include React.FC or similar)
                type_annotation = self.parser._find_child_by_type(declarator_node, 'type_annotation')
                return_type = None
                react_prop_types = []  # For React.FC<PropsType>
                
                if type_annotation:
                    for child in type_annotation.children:
                        if child.type != ':':
                            return_type = self.parser._get_node_text(child)
                            # Extract prop types from React.FC<{prop: Type}> syntax
                            react_prop_types = self._extract_react_fc_prop_types(child)
                            break
                
                # Extract parameters from arrow function
                parameters = self._extract_function_parameters(arrow_func_node)
                
                # If no parameters extracted but we have React.FC prop types, use those
                # React.FC<{users: User[]}> means the component takes {users: User[]} as props
                if not parameters and react_prop_types:
                    parameters = react_prop_types
                
                # Check if exported
                is_exported = declarator_node.parent and self._has_modifier(declarator_node.parent.parent, 'export')
                
                # Check if this is async
                is_async = self._has_modifier(arrow_func_node, 'async')
                
                # Create function symbol
                symbol = Symbol(
                    name=name,
                    symbol_type=SymbolType.FUNCTION,
                    scope=Scope.FUNCTION,
                    range=self.parser._node_to_range(declarator_node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    return_type=return_type,
                    parameter_types=[p.get('type') for p in parameters if p.get('type')],
                    source_text=self.parser._get_node_text(declarator_node),
                    is_async=is_async
                )
                
                # Add metadata
                metadata = {}
                if is_exported:
                    metadata['is_exported'] = True
                
                # Check if React functional component (TSX-specific)
                if self.tsx_mode and self._is_react_functional_component(arrow_func_node):
                    metadata['is_react_component'] = True
                    metadata['component_type'] = 'functional'
                
                if metadata:
                    symbol.metadata = metadata
                
                self.symbols.append(symbol)
            
            def _extract_enum_member(self, node, parent_full_name):
                """Extract enum member as CONSTANT"""
                member_name = self.parser._get_node_text(node)
                full_name = f"{parent_full_name}.{member_name}"
                
                symbol = Symbol(
                    name=member_name,
                    symbol_type=SymbolType.CONSTANT,
                    scope=Scope.CLASS,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_full_name,
                    full_name=full_name,
                    metadata={'is_enum_member': True}
                )
                self.symbols.append(symbol)
            
            def _extract_imported_symbols(self, import_clause) -> List[str]:
                """Extract list of imported symbol names"""
                symbols = []
                
                # Handle named imports: { Foo, Bar }
                named_imports = self.parser._find_child_by_type(import_clause, 'named_imports')
                if named_imports:
                    for child in named_imports.children:
                        if child.type == 'import_specifier':
                            identifier = self.parser._find_child_by_type(child, 'identifier')
                            if identifier:
                                symbols.append(self.parser._get_node_text(identifier))
                
                # Handle namespace import: * as Foo
                namespace_import = self.parser._find_child_by_type(import_clause, 'namespace_import')
                if namespace_import:
                    identifier = self.parser._find_child_by_type(namespace_import, 'identifier')
                    if identifier:
                        symbols.append(self.parser._get_node_text(identifier))
                
                # Handle default import: Foo
                identifier = self.parser._find_child_by_type(import_clause, 'identifier')
                if identifier and identifier.parent == import_clause:
                    symbols.append(self.parser._get_node_text(identifier))
                
                return symbols
            
            def _extract_field_type_relationship(self, field_type, field_symbol, field_name, node):
                """Extract COMPOSITION/AGGREGATION relationship for class field
                
                Args:
                    field_type: The field's type annotation
                    field_symbol: The field's full qualified name (e.g., 'UserService.currentUser')
                    field_name: The field's simple name (e.g., 'currentUser')
                    node: The AST node
                """
                # Determine if optional (aggregation) or required (composition)
                is_optional = '?' in field_type or 'undefined' in field_type or 'null' in field_type
                
                # Extract base type (remove optional markers, array markers, etc.)
                base_type = field_type.replace('?', '').replace('[]', '').replace('undefined', '').replace('null', '').strip()
                
                # Check if user-defined type (not primitive)
                primitives = {'string', 'number', 'boolean', 'any', 'void', 'never', 'unknown'}
                if base_type.lower() not in primitives:
                    rel_type = RelationshipType.AGGREGATION if is_optional else RelationshipType.COMPOSITION
                    
                    relationship = Relationship(
                        source_symbol=field_symbol,
                        target_symbol=base_type,
                        relationship_type=rel_type,
                        source_file=file_path,
                        source_range=self.parser._node_to_range(node),
                        annotations={
                            'field_name': field_name,
                            'field_type': field_type
                        }
                    )
                    self.relationships.append(relationship)
            
            def _is_react_component(self, extends_types: List[str]) -> bool:
                """Check if class extends React.Component"""
                for base in extends_types:
                    if 'Component' in base or 'PureComponent' in base:
                        return True
                return False
            
            def _is_react_functional_component(self, node) -> bool:
                """Check if function returns JSX (React functional component)"""
                # Look for jsx_element or jsx_fragment in return statement
                for child in node.children:
                    if self._contains_jsx(child):
                        return True
                return False
            
            def _contains_jsx(self, node) -> bool:
                """Recursively check if node contains JSX"""
                if node.type in ['jsx_element', 'jsx_fragment', 'jsx_self_closing_element']:
                    return True
                for child in node.children:
                    if self._contains_jsx(child):
                        return True
                return False
        
        # Create extractor and visit AST
        extractor = SymbolExtractor(self)
        extractor.visit(ast_node)
        
        # Store field-level relationships for later merging
        self._field_relationships = extractor.relationships
        
        return extractor.symbols
    
    def extract_relationships(self, ast_node: Any, symbols: List[Symbol], file_path: str) -> List[Relationship]:
        """
        Extract relationships from TypeScript tree-sitter AST.
        
        Uses visitor pattern like C++ parser with RelationshipExtractor class.
        
        Args:
            ast_node: Root tree-sitter AST node
            symbols: List of symbols from extract_symbols
            file_path: Path to source file
            
        Returns:
            List of Relationship objects
        """
        if not self._setup_successful:
            return []
        
        # Build symbol lookup index for fast access (like C++ parser)
        symbol_by_name = {}
        symbol_by_full_name = {}
        for symbol in symbols:
            # Store lists to handle overloads (multiple symbols with same name)
            if symbol.name not in symbol_by_name:
                symbol_by_name[symbol.name] = []
            symbol_by_name[symbol.name].append(symbol)
            
            if symbol.full_name:
                if symbol.full_name not in symbol_by_full_name:
                    symbol_by_full_name[symbol.full_name] = []
                symbol_by_full_name[symbol.full_name].append(symbol)
        
        class RelationshipExtractor:
            """Extracts relationships from TypeScript AST using visitor pattern (C++ style)"""
            
            def __init__(self, parser, symbols, symbol_by_name, symbol_by_full_name, file_path):
                self.parser = parser
                self.symbols = symbols
                self.symbol_by_name = symbol_by_name
                self.symbol_by_full_name = symbol_by_full_name
                self.file_path = file_path
                self.relationships = []
                self.scope_stack = []  # Track current scope (class names, dot-separated)
                self.current_symbol = None  # Track current function/method being analyzed
                
                # TypeScript keywords to exclude from REFERENCES
                self.ts_keywords = {
                    'abstract', 'any', 'as', 'async', 'await', 'boolean', 'break', 'case',
                    'catch', 'class', 'const', 'constructor', 'continue', 'debugger', 'declare',
                    'default', 'delete', 'do', 'else', 'enum', 'export', 'extends', 'false',
                    'finally', 'for', 'from', 'function', 'get', 'if', 'implements', 'import',
                    'in', 'instanceof', 'interface', 'is', 'keyof', 'let', 'module', 'namespace',
                    'never', 'new', 'null', 'number', 'of', 'package', 'private', 'protected',
                    'public', 'readonly', 'require', 'return', 'set', 'static', 'string',
                    'super', 'switch', 'symbol', 'this', 'throw', 'true', 'try', 'type',
                    'typeof', 'undefined', 'unique', 'unknown', 'var', 'void', 'while', 'with',
                    'yield'
                }
            
            def visit(self, node):
                """Main visitor dispatch (C++ pattern)"""
                method_name = f'visit_{node.type}'
                has_custom_visitor = hasattr(self, method_name)
                
                if has_custom_visitor:
                    getattr(self, method_name)(node)
                
                # Visit children for most nodes
                # Some visitors handle recursion manually
                for child in node.children:
                    self.visit(child)
            
            def visit_class_declaration(self, node):
                """Track class scope for relationship context"""
                # Get class name
                name_node = self.parser._find_child_by_type(node, 'type_identifier')
                if name_node:
                    name = self.parser._get_node_text(name_node)
                    self.scope_stack.append(name)
                    
                    # Visit children (inheritance handled by SymbolExtractor)
                    for child in node.children:
                        self.visit(child)
                    
                    self.scope_stack.pop()
            
            def visit_abstract_class_declaration(self, node):
                """Track abstract class scope - delegates to visit_class_declaration"""
                # Tree-sitter uses separate node type for abstract classes
                self.visit_class_declaration(node)
            
            def visit_interface_declaration(self, node):
                """Track interface scope"""
                name_node = self.parser._find_child_by_type(node, 'type_identifier')
                if name_node:
                    name = self.parser._get_node_text(name_node)
                    self.scope_stack.append(name)
                    
                    for child in node.children:
                        self.visit(child)
                    
                    self.scope_stack.pop()
            
            def visit_function_declaration(self, node):
                """Track current function for CALLS and REFERENCES"""
                name_node = self.parser._find_child_by_type(node, 'identifier')
                if not name_node:
                    return
                
                name = self.parser._get_node_text(name_node)
                parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}.{name}" if parent_symbol else name
                
                # Find the symbol for this function
                symbol = self._find_symbol(full_name, name)
                
                # Extract REFERENCES from parameters and return type
                if symbol:
                    # Extract parameter type references
                    formal_params = self.parser._find_child_by_type(node, 'formal_parameters')
                    if formal_params:
                        self._extract_parameter_references(formal_params, symbol)
                    
                    # Extract return type references
                    self._extract_return_type_references(node, symbol)
                
                # Set current symbol for relationship tracking
                old_symbol = self.current_symbol
                self.current_symbol = symbol
                
                # Visit function body to extract CALLS and REFERENCES
                body = self.parser._find_child_by_type(node, 'statement_block')
                if body:
                    for child in body.children:
                        self.visit(child)
                
                # Restore previous symbol
                self.current_symbol = old_symbol
            
            def visit_method_definition(self, node):
                """Track current method for CALLS and REFERENCES"""
                name_node = self.parser._find_child_by_type(node, ['property_identifier', 'identifier'])
                if not name_node:
                    return
                
                name = self.parser._get_node_text(name_node)
                parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}.{name}" if parent_symbol else name
                
                # Find the symbol for this method
                symbol = self._find_symbol(full_name, name)
                
                # Extract REFERENCES from parameters and return type
                if symbol:
                    # Extract parameter type references
                    formal_params = self.parser._find_child_by_type(node, 'formal_parameters')
                    if formal_params:
                        self._extract_parameter_references(formal_params, symbol)
                    
                    # Extract return type references
                    self._extract_return_type_references(node, symbol)
                
                # Set current symbol
                old_symbol = self.current_symbol
                self.current_symbol = symbol
                
                # Visit method body
                body = self.parser._find_child_by_type(node, 'statement_block')
                if body:
                    for child in body.children:
                        self.visit(child)
                
                # Restore previous symbol
                self.current_symbol = old_symbol
            
            def visit_lexical_declaration(self, node):
                """Track arrow functions assigned to const/let for CALLS and REFERENCES"""
                # Find variable declarator with arrow function
                declarator = self.parser._find_child_by_type(node, 'variable_declarator')
                if not declarator:
                    return
                
                # Check if this is an arrow function assignment
                arrow_func = self.parser._find_child_by_type(declarator, 'arrow_function')
                if not arrow_func:
                    return
                
                # Get function name from identifier
                identifier = self.parser._find_child_by_type(declarator, 'identifier')
                if not identifier:
                    return
                
                name = self.parser._get_node_text(identifier)
                parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}.{name}" if parent_symbol else name
                
                # Find the symbol for this function
                symbol = self._find_symbol(full_name, name)
                
                # Extract REFERENCES from parameters and return type
                if symbol:
                    # Extract parameter type references
                    formal_params = self.parser._find_child_by_type(arrow_func, 'formal_parameters')
                    if formal_params:
                        self._extract_parameter_references(formal_params, symbol)
                    
                    # Extract return type references from declarator (type annotation on const)
                    self._extract_return_type_references(declarator, symbol)
                
                # Set current symbol for relationship tracking
                old_symbol = self.current_symbol
                self.current_symbol = symbol
                
                # Visit arrow function body to extract CALLS, REFERENCES, and JSX
                body = self.parser._find_child_by_type(arrow_func, 'statement_block')
                if body:
                    for child in body.children:
                        self.visit(child)
                else:
                    # Arrow function with expression body (no braces)
                    # Visit all children of arrow_func to find expression
                    for child in arrow_func.children:
                        if child.type not in ['formal_parameters', '=>', 'async']:
                            self.visit(child)
                
                # Restore previous symbol
                self.current_symbol = old_symbol
            
            def visit_call_expression(self, node):
                """Extract CALLS relationships from function calls"""
                if not self.current_symbol:
                    return
                
                # Extract function being called
                function_node = node.children[0] if node.children else None
                if not function_node:
                    return
                
                called_name = None
                
                if function_node.type == 'identifier':
                    # Simple function call: foo()
                    called_name = self.parser._get_node_text(function_node)
                
                elif function_node.type == 'member_expression':
                    # Method call: obj.method()
                    property = self.parser._find_child_by_type(function_node, 'property_identifier')
                    if property:
                        method_name = self.parser._get_node_text(property)
                        # Try to resolve to Class.method
                        called_name = method_name  # Simplified for now
                
                if called_name and called_name not in self.ts_keywords:
                    relationship = Relationship(
                        source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                        target_symbol=called_name,
                        relationship_type=RelationshipType.CALLS,
                        source_file=self.file_path,
                        source_range=self.parser._node_to_range(node)
                    )
                    self.relationships.append(relationship)
            
            def visit_new_expression(self, node):
                """Extract CREATES relationship for object instantiation
                
                Handles:
                    - new ClassName()
                    - new namespace.ClassName()
                    - new GenericClass<Type>()
                """
                if not self.current_symbol:
                    return
                
                # Get the type being instantiated
                type_node = None
                type_name = None
                
                for child in node.children:
                    if child.type == 'identifier':
                        # Simple: new User()
                        type_name = self.parser._get_node_text(child)
                        type_node = child
                        break
                    elif child.type == 'member_expression':
                        # Qualified: new MyModule.MyClass()
                        property = self.parser._find_child_by_type(child, 'property_identifier')
                        if property:
                            type_name = self.parser._get_node_text(property)
                            type_node = property
                        break
                    elif child.type == 'generic_type':
                        # Generic: new Map<string, User>()
                        base_type = self.parser._find_child_by_type(child, 'type_identifier')
                        if base_type:
                            type_name = self.parser._get_node_text(base_type)
                            type_node = base_type
                        
                        # Extract generic argument references
                        type_args = self.parser._find_child_by_type(child, 'type_arguments')
                        if type_args:
                            for arg_child in type_args.children:
                                if arg_child.type not in ['<', '>', ',']:
                                    self._extract_type_references_recursive(
                                        arg_child, 
                                        self.current_symbol, 
                                        'constructor_generic_arg'
                                    )
                        break
                
                if type_name and type_name not in self.ts_keywords:
                    relationship = Relationship(
                        source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                        target_symbol=type_name,
                        relationship_type=RelationshipType.CREATES,
                        source_file=self.file_path,
                        source_range=self.parser._node_to_range(type_node if type_node else node),
                        annotations={'creation_type': 'new'}
                    )
                    self.relationships.append(relationship)
            
            def visit_type_alias_declaration(self, node):
                """Extract REFERENCES and ALIAS_OF from type alias declarations.
                
                Handles:
                    type Foo = Bar;                    → ALIAS_OF(Foo→Bar) + REFERENCES(Foo→Bar)
                    type Union = A | B;                → REFERENCES(Union→A), REFERENCES(Union→B)
                    type Inter = A & B;                → REFERENCES(Inter→A), REFERENCES(Inter→B)
                    type Generic = Map<string, Widget>; → REFERENCES(Generic→Widget)
                """
                name_node = self.parser._find_child_by_type(node, 'type_identifier')
                if not name_node:
                    return
                
                name = self.parser._get_node_text(name_node)
                parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}.{name}" if parent_symbol else name
                
                # Look up the Symbol created by SymbolExtractor
                symbol = self._find_symbol(full_name, name)
                if not symbol:
                    return
                
                # Find the type value node (skip 'type' keyword, name, '=', generic params)
                type_value = None
                past_equals = False
                for child in node.children:
                    if child.type == '=':
                        past_equals = True
                        continue
                    if past_equals and child.type not in [';', 'type_parameters']:
                        type_value = child
                        break
                
                if not type_value:
                    return
                
                # Emit REFERENCES for all constituent types
                self._extract_type_references_recursive(type_value, symbol, 'type_alias')
                
                # Emit ALIAS_OF for simple aliases (type X = Y)
                if type_value.type == 'type_identifier':
                    target_name = self.parser._get_node_text(type_value)
                    if target_name and target_name not in self.ts_keywords:
                        alias_rel = Relationship(
                            source_symbol=symbol.full_name or symbol.name,
                            target_symbol=target_name,
                            relationship_type=RelationshipType.ALIAS_OF,
                            source_file=self.file_path,
                            source_range=self.parser._node_to_range(node),
                            annotations={'alias_name': name, 'aliased_type': target_name}
                        )
                        self.relationships.append(alias_rel)
                elif type_value.type == 'generic_type':
                    # For generic aliases like type X = Map<string, Y>, ALIAS_OF → base type
                    # Skip builtin generics (Array, Map, etc.) — only emit for user-defined
                    builtin_generics = {'Array', 'Promise', 'Map', 'Set', 'Record', 'Partial',
                                        'Required', 'Readonly', 'Pick', 'Omit'}
                    base = self.parser._find_child_by_type(type_value, 'type_identifier')
                    if base:
                        base_name = self.parser._get_node_text(base)
                        if base_name and base_name not in self.ts_keywords and base_name not in builtin_generics:
                            alias_rel = Relationship(
                                source_symbol=symbol.full_name or symbol.name,
                                target_symbol=base_name,
                                relationship_type=RelationshipType.ALIAS_OF,
                                source_file=self.file_path,
                                source_range=self.parser._node_to_range(node),
                                annotations={'alias_name': name, 'aliased_type': self.parser._get_node_text(type_value)}
                            )
                            self.relationships.append(alias_rel)
            
            def visit_type_identifier(self, node):
                """Extract REFERENCES relationships from type usage"""
                if not self.current_symbol:
                    return
                
                type_name = self.parser._get_node_text(node)
                
                # Skip keywords and primitives
                if type_name in self.ts_keywords:
                    return
                
                # Create REFERENCES relationship
                relationship = Relationship(
                    source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                    target_symbol=type_name,
                    relationship_type=RelationshipType.REFERENCES,
                    source_file=self.file_path,
                    source_range=self.parser._node_to_range(node),
                    annotations={'reference_type': 'type_usage'}
                )
                self.relationships.append(relationship)
            
            def visit_identifier(self, node):
                """Extract REFERENCES relationships from identifier usage"""
                if not self.current_symbol:
                    return
                
                name = self.parser._get_node_text(node)
                
                # Skip keywords
                if name in self.ts_keywords:
                    return
                
                # Skip if this is the current function name itself
                if name == self.current_symbol.name:
                    return
                
                # Check if this references a known symbol
                target_symbol = self._find_symbol(name, name)
                
                # Only create reference if we found a symbol or it looks like a type
                if target_symbol or (name and name[0].isupper()):
                    relationship = Relationship(
                        source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                        target_symbol=name,
                        relationship_type=RelationshipType.REFERENCES,
                        source_file=self.file_path,
                        source_range=self.parser._node_to_range(node)
                    )
                    self.relationships.append(relationship)
            
            def visit_jsx_element(self, node):
                """Extract REFERENCES for custom React components in JSX"""
                if not self.current_symbol:
                    return
                
                # Get opening element
                opening = self.parser._find_child_by_type(node, 'jsx_opening_element')
                if opening:
                    self._extract_jsx_component_reference(opening)
                
                # Visit children
                for child in node.children:
                    self.visit(child)
            
            def visit_jsx_self_closing_element(self, node):
                """Extract REFERENCES for self-closing React components"""
                if not self.current_symbol:
                    return
                
                self._extract_jsx_component_reference(node)
            
            def _extract_jsx_component_reference(self, jsx_node):
                """Extract component reference from JSX element"""
                # Find component identifier
                identifier = self.parser._find_child_by_type(jsx_node, 'identifier')
                if not identifier:
                    return
                
                component_name = self.parser._get_node_text(identifier)
                
                # Skip built-in HTML tags (lowercase)
                if component_name and component_name[0].islower():
                    return
                
                # Custom component (starts with uppercase) - create REFERENCES
                if component_name:
                    relationship = Relationship(
                        source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                        target_symbol=component_name,
                        relationship_type=RelationshipType.REFERENCES,
                        source_file=self.file_path,
                        source_range=self.parser._node_to_range(jsx_node),
                        annotations={'reference_type': 'jsx_component'}
                    )
                    self.relationships.append(relationship)
            
            def _extract_parameter_references(self, formal_params_node, function_symbol):
                """Extract REFERENCES from function parameter types
                
                Handles:
                    - Simple types: function foo(user: User)
                    - Generic types: function foo(items: Array<User>)
                    - Union types: function foo(id: string | number)
                    - Optional: function foo(user?: User)
                """
                if not function_symbol or not formal_params_node:
                    return
                
                for child in formal_params_node.children:
                    if child.type in ['required_parameter', 'optional_parameter']:
                        # Find type annotation
                        type_annotation = self.parser._find_child_by_type(child, 'type_annotation')
                        if type_annotation:
                            self._extract_type_references(type_annotation, function_symbol, 'parameter_type')
            
            def _extract_return_type_references(self, function_node, function_symbol):
                """Extract REFERENCES from function return type
                
                Handles:
                    - Simple: function foo(): User
                    - Generic: function foo(): Promise<User>
                    - Union: function foo(): User | null
                """
                if not function_symbol:
                    return
                
                # Find type annotation (comes after parameters)
                type_annotation = self.parser._find_child_by_type(function_node, 'type_annotation')
                if type_annotation:
                    self._extract_type_references(type_annotation, function_symbol, 'return_type')
            
            def _extract_type_references(self, type_node, source_symbol, reference_type):
                """Extract REFERENCES from type nodes recursively
                
                Handles all TypeScript type constructs:
                    - type_identifier: User, Customer
                    - generic_type: Array<User>, Map<string, User>
                    - union_type: User | Admin
                    - intersection_type: User & Timestamped
                    - array_type: User[]
                """
                if not type_node:
                    return
                
                # Find actual type (skip 'type_annotation' wrapper)
                type_value = None
                for child in type_node.children:
                    if child.type != ':':
                        type_value = child
                        break
                
                if not type_value:
                    return
                
                self._extract_type_references_recursive(type_value, source_symbol, reference_type)
            
            def _extract_type_references_recursive(self, type_node, source_symbol, reference_type):
                """Recursively extract type references"""
                if not type_node:
                    return
                
                if type_node.type == 'type_identifier':
                    # Simple type reference
                    type_name = self.parser._get_node_text(type_node)
                    if type_name and type_name not in self.ts_keywords:
                        relationship = Relationship(
                            source_symbol=source_symbol.full_name or source_symbol.name,
                            target_symbol=type_name,
                            relationship_type=RelationshipType.REFERENCES,
                            source_file=self.file_path,
                            source_range=self.parser._node_to_range(type_node),
                            annotations={'reference_type': reference_type}
                        )
                        self.relationships.append(relationship)
                
                elif type_node.type == 'generic_type':
                    # Generic type: Array<User>, Promise<Response>
                    # Extract base type
                    base = self.parser._find_child_by_type(type_node, 'type_identifier')
                    if base:
                        base_name = self.parser._get_node_text(base)
                        # Skip built-in generics
                        builtin_generics = {'Array', 'Promise', 'Map', 'Set', 'Record', 'Partial', 
                                          'Required', 'Readonly', 'Pick', 'Omit'}
                        if base_name not in builtin_generics:
                            relationship = Relationship(
                                source_symbol=source_symbol.full_name or source_symbol.name,
                                target_symbol=base_name,
                                relationship_type=RelationshipType.REFERENCES,
                                source_file=self.file_path,
                                source_range=self.parser._node_to_range(base),
                                annotations={'reference_type': f'{reference_type}_generic'}
                            )
                            self.relationships.append(relationship)
                    
                    # Extract type arguments
                    type_args = self.parser._find_child_by_type(type_node, 'type_arguments')
                    if type_args:
                        for child in type_args.children:
                            if child.type not in ['<', '>', ',']:
                                self._extract_type_references_recursive(child, source_symbol, f'{reference_type}_generic_arg')
                
                elif type_node.type in ['union_type', 'intersection_type']:
                    # Union/Intersection: User | Admin, User & Timestamped
                    for child in type_node.children:
                        if child.type not in ['|', '&']:
                            self._extract_type_references_recursive(child, source_symbol, reference_type)
                
                elif type_node.type == 'array_type':
                    # Array type: User[]
                    element_type = type_node.children[0] if type_node.children else None
                    if element_type:
                        self._extract_type_references_recursive(element_type, source_symbol, f'{reference_type}_array')
                
                elif type_node.type == 'parenthesized_type':
                    # Parenthesized: (User | Admin)
                    for child in type_node.children:
                        if child.type not in ['(', ')']:
                            self._extract_type_references_recursive(child, source_symbol, reference_type)
            
            # Helper methods
            
            def _find_symbol(self, full_name, name):
                """Find symbol by full name or name"""
                # Try full name first
                candidates = self.symbol_by_full_name.get(full_name)
                if candidates:
                    return candidates[0]
                
                # Try simple name
                candidates = self.symbol_by_name.get(name)
                if candidates:
                    return candidates[0]
                
                return None
        
        # Create extractor and visit AST
        extractor = RelationshipExtractor(self, symbols, symbol_by_name, symbol_by_full_name, file_path)
        extractor.visit(ast_node)
        
        # Extract DEFINES relationships (Pass 2: RelationshipExtractor)
        defines_relationships = self._extract_defines_relationships(symbols, file_path)
        extractor.relationships.extend(defines_relationships)
        
        # Merge field-level relationships from symbol extraction
        all_relationships = extractor.relationships[:]
        if hasattr(self, '_field_relationships'):
            all_relationships.extend(self._field_relationships)
            self._field_relationships = []  # Clear for next parse
        
        return all_relationships
    
    def _extract_defines_relationships(self, symbols: List[Symbol], file_path: str) -> List[Relationship]:
        """
        Extract DEFINES relationships from modules/classes/interfaces to their members (Pass 2).
        
        Creates DEFINES edges for:
        - Module → Top-level Class, Interface, Function, Enum, Constant
        - Class → Method, Field, Constructor
        - Interface → Method signature, Property
        - Abstract Class → Abstract Method
        
        Note: MODULE → top-level DEFINES relationships are NOT created.
        Modules are organizational, not architectural. Parent-child containment
        is captured by parent_symbol metadata on each symbol.
        
        This follows the two-pass architecture documented in RELATIONSHIP_EXTRACTOR_PATTERN.md:
        - Pass 1 (SymbolExtractor): Already done - creates symbols with parent_symbol
        - Pass 2 (RelationshipExtractor): This method - matches parent_symbol strings to create edges
        
        Args:
            symbols: List of symbols extracted in Pass 1
            file_path: Path to source file
            
        Returns:
            List of DEFINES relationships
        """
        relationships = []
        
        # Build index of symbols by parent for efficient lookup
        symbols_by_parent = {}
        for symbol in symbols:
            parent = symbol.parent_symbol
            if parent:
                if parent not in symbols_by_parent:
                    symbols_by_parent[parent] = []
                symbols_by_parent[parent].append(symbol)
        
        # For each class/interface, create DEFINES edges to its members
        # (NOT from MODULE - modules are organizational, not architectural)
        for symbol in symbols:
            # Only classes and interfaces can DEFINE members
            if symbol.symbol_type not in [SymbolType.CLASS, SymbolType.INTERFACE]:
                continue
            
            # Find members that belong to this container
            # Note: Members have parent_symbol set to the full name (e.g., "module.Class")
            # so we need to look up by full_name, not just name
            container_full_name = symbol.full_name if symbol.full_name else symbol.name
            members = symbols_by_parent.get(container_full_name, [])
            
            for member in members:
                # Only create DEFINES for methods, constructors, and fields
                # (NOT nested classes, NOT top-level functions)
                if member.symbol_type not in [SymbolType.METHOD, SymbolType.CONSTRUCTOR, 
                                              SymbolType.FIELD, SymbolType.PROPERTY]:
                    continue
                
                # Determine member_type for annotations
                member_type_map = {
                    SymbolType.METHOD: 'method',
                    SymbolType.CONSTRUCTOR: 'constructor',
                    SymbolType.FIELD: 'field',
                    SymbolType.PROPERTY: 'property'
                }
                member_type = member_type_map.get(member.symbol_type, 'unknown')
                
                # Build annotations from symbol metadata
                annotations = {
                    'member_type': member_type,
                    'is_static': member.is_static if hasattr(member, 'is_static') else False,
                }
                
                # Add TypeScript-specific annotations
                if hasattr(member, 'visibility') and member.visibility:
                    annotations['visibility'] = member.visibility
                
                if hasattr(member, 'is_abstract') and member.is_abstract:
                    annotations['is_abstract'] = True
                
                # Check for readonly in direct field or metadata
                if hasattr(member, 'is_readonly') and member.is_readonly:
                    annotations['is_readonly'] = True
                elif hasattr(member, 'metadata') and member.metadata and member.metadata.get('is_readonly'):
                    annotations['is_readonly'] = True
                
                # Check for optional modifier in metadata
                if hasattr(member, 'metadata') and member.metadata:
                    if member.metadata.get('is_optional'):
                        annotations['is_optional'] = True
                
                # Create DEFINES relationship
                relationship = Relationship(
                    source_symbol=container_full_name,
                    target_symbol=member.full_name,
                    relationship_type=RelationshipType.DEFINES,
                    source_file=file_path,
                    source_range=member.range,
                    annotations=annotations
                )
                relationships.append(relationship)
        
        return relationships
    
    # Helper methods (adapted from C++ parser)
    
    def _node_to_range(self, node) -> Range:
        """Convert tree-sitter node to Range"""
        return Range(
            start=Position(line=node.start_point[0] + 1, column=node.start_point[1]),
            end=Position(line=node.end_point[0] + 1, column=node.end_point[1])
        )
    
    def _get_node_text(self, node) -> str:
        """Extract text from tree-sitter node"""
        if node is None:
            return ""
        return self._current_content[node.start_byte:node.end_byte]
    
    def _find_child_by_type(self, node, child_type: Union[str, List[str]]):
        """Find first child with given type(s)"""
        if isinstance(child_type, str):
            child_type = [child_type]
        for child in node.children:
            if child.type in child_type:
                return child
        return None
    
    def _get_children_by_type(self, node, child_type: str):
        """Get all children with given type"""
        return [child for child in node.children if child.type == child_type]
    
    def parse_multiple_files(self, file_paths: List[str], max_workers: int = 4) -> Dict[str, ParseResult]:
        """
        Parse multiple files with cross-file resolution (3-pass approach).
        
        This is the KEY method for comprehensive repository-wide analysis.
        Follows the exact same pattern as C++ parser's parse_multiple_files.
        
        Pass 1: Parse all files in parallel → symbols + local relationships
        Pass 2: Build global symbol registries → class/interface/function/type/module locations  
        Pass 3: Enhance relationships → resolve imports, cross-file type references
        
        Args:
            file_paths: List of TypeScript/TSX file paths
            max_workers: Number of parallel workers for parsing
            
        Returns:
            Dict mapping file_path -> ParseResult with enhanced cross-file relationships
        """
        # Global symbol registries for cross-file resolution
        global _global_class_locations, _global_interface_locations
        global _global_function_locations, _global_type_locations
        global _global_symbol_registry, _global_module_exports
        
        _global_class_locations = {}
        _global_interface_locations = {}
        _global_function_locations = {}
        _global_type_locations = {}
        _global_symbol_registry = {}
        _global_module_exports = {}
        
        total = len(file_paths)
        logger.info(f"Parsing {total} TypeScript files for cross-file analysis with {max_workers} workers")
        
        # PASS 1: Parse all files in parallel
        try:
            results = self._parse_files_parallel(file_paths, max_workers)
            successful_parses = sum(1 for r in results.values() if r.symbols)
            logger.info(f"Pass 1 complete: {successful_parses}/{len(results)} files parsed successfully")
        except Exception as e:
            logger.error(f"Pass 1 (parallel parsing) failed: {type(e).__name__}: {e}", exc_info=True)
            raise
        
        # PASS 2: Build global symbol registries
        logger.info(f"Pass 2: Building global TypeScript symbol registries from {successful_parses} files")
        try:
            processed = 0
            for file_path, result in results.items():
                if result.symbols:  # Only process files that parsed successfully
                    self._extract_global_symbols(file_path, result)
                    processed += 1
                    if processed % 500 == 0:  # Log progress every 500 files
                        logger.info(f"Pass 2: Processed {processed}/{successful_parses} files")
            logger.info(f"Pass 2 complete: {len(_global_symbol_registry)} symbols registered")
        except Exception as e:
            logger.error(f"Pass 2 (symbol registry) failed: {type(e).__name__}: {e}", exc_info=True)
            # Continue anyway - cross-file resolution is best-effort
        
        # PASS 3: Enhance relationships with cross-file information
        logger.info(f"Pass 3: Enhancing relationships with cross-file resolution for {successful_parses} files")
        try:
            processed = 0
            for file_path, result in results.items():
                if result.symbols:  # Only process files that parsed successfully
                    self._enhance_cross_file_relationships(file_path, result)
                    processed += 1
                    if processed % 500 == 0:  # Log progress every 500 files
                        logger.info(f"Pass 3: Processed {processed}/{successful_parses} files")
            logger.info(f"Pass 3 complete: Cross-file relationships enhanced")
        except Exception as e:
            logger.error(f"Pass 3 (relationship enhancement) failed: {type(e).__name__}: {e}", exc_info=True)
            # Continue anyway - we still have basic relationships
        
        logger.info(f"TypeScript parsing complete: {len(results)}/{total} files processed, {successful_parses} successful")
        return results
    
    def _parse_files_parallel(self, file_paths: List[str], max_workers: int) -> Dict[str, ParseResult]:
        """
        Parse files in parallel using ThreadPoolExecutor with batching.
        
        Same pattern as C++ parser: ThreadPoolExecutor for stability, batching for memory efficiency.
        
        Args:
            file_paths: List of file paths to parse
            max_workers: Number of parallel workers
            
        Returns:
            Dict mapping file_path -> ParseResult
        """
        results = {}
        total = len(file_paths)
        
        # Batch size: process 500 files at a time to prevent memory issues
        batch_size = 500
        num_batches = (total + batch_size - 1) // batch_size
        
        if num_batches > 1:
            logger.info(f"Processing {total} files in {num_batches} batch(es) of up to {batch_size} files each")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total)
            batch_files = file_paths[start_idx:end_idx]
            
            if num_batches > 1:
                logger.info(f"Processing batch {batch_idx + 1}/{num_batches}: files {start_idx + 1}-{end_idx} of {total}")
            
            try:
                # Use ThreadPoolExecutor for better stability and lower memory overhead
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all files in this batch
                    future_to_path = {}
                    for fp in batch_files:
                        try:
                            future = executor.submit(_parse_single_typescript_file, fp)
                            future_to_path[future] = fp
                        except Exception as e:
                            logger.error(f"Failed to submit file {fp}: {type(e).__name__}: {e}")
                            results[fp] = ParseResult(
                                file_path=fp,
                                language="typescript",
                                symbols=[],
                                relationships=[],
                                errors=[f"Submit failed: {str(e)}"]
                            )
                    
                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(future_to_path):
                        file_path = future_to_path[future]
                        try:
                            parsed_file_path, result = future.result(timeout=60)  # 60 second timeout per file
                            results[parsed_file_path] = result
                        except concurrent.futures.TimeoutError:
                            logger.error(f"Timeout parsing file (>60s): {file_path}")
                            results[file_path] = ParseResult(
                                file_path=file_path,
                                language="typescript",
                                symbols=[],
                                relationships=[],
                                errors=["Timeout during parsing"]
                            )
                        except Exception as e:
                            logger.error(f"Failed to parse file {file_path}: {type(e).__name__}: {e}", exc_info=True)
                            results[file_path] = ParseResult(
                                file_path=file_path,
                                language="typescript",
                                symbols=[],
                                relationships=[],
                                errors=[f"{type(e).__name__}: {str(e)}"]
                            )
            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} executor failed: {type(e).__name__}: {e}", exc_info=True)
                # Mark remaining files in batch as failed
                for fp in batch_files:
                    if fp not in results:
                        results[fp] = ParseResult(
                            file_path=fp,
                            language="typescript",
                            symbols=[],
                            relationships=[],
                            errors=[f"Batch processing failed: {str(e)}"]
                        )
        
        successful = sum(1 for r in results.values() if r.symbols)
        logger.info(f"Parallel parsing complete: {successful}/{total} files parsed successfully")
        return results
    
    def _extract_global_symbols(self, file_path: str, result: ParseResult):
        """
        Extract symbols for global cross-file resolution.
        
        Stores symbol locations in global registries for Pass 3 enhancement.
        Same pattern as C++ parser but with TypeScript-specific symbol types.
        
        Args:
            file_path: Source file path
            result: ParseResult with symbols
        """
        global _global_class_locations, _global_interface_locations
        global _global_function_locations, _global_type_locations
        global _global_symbol_registry, _global_module_exports
        
        file_name = Path(file_path).stem
        
        # Store class, interface, function, type, and module locations globally
        for symbol in result.symbols:
            if symbol.symbol_type == SymbolType.CLASS:
                # Store class location
                _global_class_locations[symbol.name] = file_path
                if symbol.full_name:
                    _global_class_locations[symbol.full_name] = file_path
                    _global_symbol_registry[symbol.full_name] = file_path
                _global_symbol_registry[symbol.name] = file_path
                
            elif symbol.symbol_type == SymbolType.INTERFACE:
                # Store interface location (TS-specific)
                _global_interface_locations[symbol.name] = file_path
                if symbol.full_name:
                    _global_interface_locations[symbol.full_name] = file_path
                    _global_symbol_registry[symbol.full_name] = file_path
                _global_symbol_registry[symbol.name] = file_path
                
            elif symbol.symbol_type in [SymbolType.FUNCTION, SymbolType.METHOD]:
                # Store function/method location
                _global_function_locations[symbol.name] = file_path
                if symbol.full_name:
                    _global_function_locations[symbol.full_name] = file_path
                    _global_symbol_registry[symbol.full_name] = file_path
                _global_symbol_registry[symbol.name] = file_path
                
            elif symbol.symbol_type == SymbolType.TYPE_ALIAS:
                # Store type alias location
                _global_type_locations[symbol.name] = file_path
                if symbol.full_name:
                    _global_type_locations[symbol.full_name] = file_path
                    _global_symbol_registry[symbol.full_name] = file_path
                _global_symbol_registry[symbol.name] = file_path
            
            elif symbol.symbol_type == SymbolType.CONSTANT:
                # Store constant/enum member location
                if symbol.full_name:
                    _global_symbol_registry[symbol.full_name] = file_path
                _global_symbol_registry[symbol.name] = file_path
    
    def _enhance_cross_file_relationships(self, file_path: str, result: ParseResult):
        """
        Enhance relationships with cross-file target information.
        
        Uses global registries to resolve target files for relationships.
        Same pattern as C++ parser.
        
        Args:
            file_path: Source file path
            result: ParseResult to enhance (modified in place)
        """
        enhanced_relationships = []
        
        for rel in result.relationships:
            enhanced_rel = self._enhance_relationship(rel, file_path)
            enhanced_relationships.append(enhanced_rel)
        
        # Replace with enhanced relationships
        result.relationships.clear()
        result.relationships.extend(enhanced_relationships)
    
    def _enhance_relationship(self, rel: Relationship, source_file: str) -> Relationship:
        """
        Enhance a single relationship with cross-file target information.
        
        Args:
            rel: Relationship to enhance
            source_file: Source file path
            
        Returns:
            Enhanced Relationship with target_file set if cross-file
        """
        global _global_class_locations, _global_interface_locations
        global _global_function_locations, _global_type_locations, _global_symbol_registry
        
        # For key relationship types, try to resolve target to actual file
        if rel.relationship_type in [RelationshipType.INHERITANCE, RelationshipType.IMPLEMENTATION,
                                     RelationshipType.CALLS, RelationshipType.REFERENCES, 
                                     RelationshipType.CREATES, RelationshipType.IMPORTS]:
            target_symbol = rel.target_symbol
            
            # Try to find the target file
            target_file = None
            
            # Check if target is a known class
            if target_symbol in _global_class_locations:
                target_file = _global_class_locations[target_symbol]
            # Check if target is a known interface
            elif target_symbol in _global_interface_locations:
                target_file = _global_interface_locations[target_symbol]
            # Check if target is a known function
            elif target_symbol in _global_function_locations:
                target_file = _global_function_locations[target_symbol]
            # Check if target is a known type alias
            elif target_symbol in _global_type_locations:
                target_file = _global_type_locations[target_symbol]
            # Check the general registry
            elif target_symbol in _global_symbol_registry:
                target_file = _global_symbol_registry[target_symbol]
            
            # If we found a cross-file target, update the relationship
            if target_file and target_file != rel.source_file:
                return Relationship(
                    source_symbol=rel.source_symbol,
                    target_symbol=rel.target_symbol,
                    relationship_type=rel.relationship_type,
                    source_file=rel.source_file,
                    target_file=target_file,
                    source_range=rel.source_range,
                    confidence=rel.confidence,
                    weight=rel.weight,
                    is_direct=rel.is_direct,
                    context=rel.context,
                    annotations=rel.annotations
                )
        
        # Return original relationship if no enhancement
        return rel
