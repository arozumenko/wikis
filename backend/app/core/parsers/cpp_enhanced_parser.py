"""
Enhanced C++ Parser implementing clean modular parser architecture

This parser provides comprehensive C++ analysis with support for:
- Complete symbol extraction (classes, structs, functions, methods, templates, macros)
- Advanced relationship detection (inheritance, calls, references, specializes)
- Template handling (declarations and instantiations)
- Macro handling (constant and function-like)
- Cross-file resolution
- Granular symbol extraction (parameters, constructors, fields)

Based on Python parser architecture but adapted for C++ tree-sitter nodes.
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any, Tuple
import concurrent.futures

from .base_parser import (
    BaseParser, LanguageCapabilities, ParseResult, Symbol, Relationship,
    SymbolType, RelationshipType, Scope, Position, Range
)

logger = logging.getLogger(__name__)

# Tree-sitter imports
try:
    from tree_sitter_language_pack import get_language, get_parser  # type: ignore
except Exception:
    get_language = None  # type: ignore
    get_parser = None  # type: ignore


def _parse_single_cpp_file(file_path: str) -> Tuple[str, ParseResult]:
    """
    Parse a single C++ file - used by parallel workers.
    
    Creates a fresh parser instance for each file because:
    - Tree-sitter parsers are NOT thread-safe
    - Parse trees cannot be shared across threads
    - Each worker thread needs its own parser instance
    """
    try:
        parser = CppEnhancedParser()
        result = parser.parse_file(file_path)
        return file_path, result
    except Exception as e:
        logger.warning(f"Failed to parse {file_path}: {e}")
        return file_path, ParseResult(
            file_path=file_path,
            language="cpp",
            symbols=[],
            relationships=[],
            errors=[str(e)]
        )


def _collect_qualified_parts_from_node(node) -> list:
    """Recursively collect all name parts from a qualified_identifier node.
    
    Tree-sitter nests qualified_identifier for deeply qualified names:
        outer::inner::Widget →
          qualified_identifier
            namespace_identifier('outer')
            qualified_identifier
              namespace_identifier('inner')
              type_identifier('Widget')
    
    A flat iteration only picks up the first part ('outer').
    This helper walks the nested structure to return ['outer', 'inner', 'Widget'].
    
    Args:
        node: A tree-sitter qualified_identifier node.
        
    Returns:
        List of name parts in order (e.g. ['outer', 'inner', 'Widget']).
    """
    parts = []
    
    def _collect(n):
        for child in n.children:
            if child.type in ('namespace_identifier', 'identifier', 'type_identifier'):
                parts.append(child.text.decode())
            elif child.type == 'operator_name':
                parts.append(child.text.decode())
            elif child.type == 'template_type':
                # Template type like Array<T> — include full text with template args
                parts.append(child.text.decode())
            elif child.type == 'template_function':
                # Template function in qualified name: std::make_unique<T>
                name_node = None
                for tc in child.children:
                    if tc.type == 'identifier':
                        name_node = tc
                        break
                if name_node:
                    parts.append(name_node.text.decode())
            elif child.type == 'qualified_identifier':
                _collect(child)  # recurse
    
    _collect(node)
    return parts


class CppEnhancedParser(BaseParser):
    """
    Clean C++ parser based on Python parser architecture
    
    Key features:
    - Granular symbol extraction (parameters, constructors, fields)
    - Template support (declarations and instantiations)
    - Macro support (constant and function-like)
    - Comprehensive relationship tracking
    - Cross-file resolution (3-pass approach)
    """
    
    def __init__(self):
        super().__init__("cpp")
        self.parser = None
        self.ts_language = None
        self._current_file = ""
        self._current_content = ""
        self._current_scope_stack: List[str] = []
        self._current_class_scope_stack: List[str] = []  # Track only classes/structs, not namespaces
        self._setup_successful = self._setup_tree_sitter()
        
        # Global registries for cross-file resolution
        self._global_class_locations: Dict[str, str] = {}
        self._global_function_locations: Dict[str, str] = {}
        self._global_symbol_registry: Dict[str, str] = {}
    
    def _define_capabilities(self) -> LanguageCapabilities:
        """Define C++ parser capabilities"""
        return LanguageCapabilities(
            language="cpp",
            supported_symbols={
                SymbolType.MODULE,
                SymbolType.CLASS,
                SymbolType.STRUCT,
                SymbolType.FUNCTION,
                SymbolType.METHOD,
                SymbolType.CONSTRUCTOR,
                SymbolType.FIELD,
                SymbolType.PARAMETER,
                SymbolType.VARIABLE,
                SymbolType.CONSTANT,
                SymbolType.MACRO,
                SymbolType.ENUM,
                SymbolType.TYPE_ALIAS,
                SymbolType.NAMESPACE
            },
            supported_relationships={
                RelationshipType.INHERITANCE,
                RelationshipType.COMPOSITION,
                RelationshipType.AGGREGATION,
                RelationshipType.CALLS,
                RelationshipType.CREATES,
                RelationshipType.REFERENCES,
                RelationshipType.DEFINES,
                RelationshipType.DEFINES_BODY,
                RelationshipType.IMPORTS,
                RelationshipType.SPECIALIZES,
                RelationshipType.OVERRIDES,
                RelationshipType.INSTANTIATES,
                RelationshipType.ALIAS_OF
            },
            supports_ast_parsing=True,
            supports_type_inference=True,
            supports_cross_file_analysis=True,
            has_classes=True,
            has_interfaces=False,
            has_namespaces=True,
            has_generics=True,  # Templates
            has_decorators=False,
            has_annotations=False,
            max_file_size_mb=20.0,
            typical_parse_time_ms=80.0
        )
    
    def _get_supported_extensions(self) -> Set[str]:
        """Get supported C++ file extensions"""
        return {'.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx', '.c', '.hh'}
    
    def _normalize_full_name(self, full_name: str) -> str:
        """
        Normalize full_name to use '.' separator instead of '::'.
        
        This standardizes C++ qualified names to match other language parsers
        (Python, Java, TypeScript, JavaScript) which all use '.' separator.
        
        Example: 'AppConfig::database' -> 'AppConfig.database'
        
        See PARSER_GRAPH_STANDARDIZATION.md for rationale.
        """
        if full_name:
            return full_name.replace('::', '.')
        return full_name
    
    def _normalize_parse_result(self, symbols: List[Symbol], relationships: List[Relationship]) -> None:
        """
        Normalize all qualified names in symbols and relationships to use '.' separator.
        
        This is a post-processing step that standardizes C++ output to match other parsers.
        Modifies symbols and relationships in-place.
        """
        # Normalize symbols
        for symbol in symbols:
            if symbol.full_name:
                symbol.full_name = self._normalize_full_name(symbol.full_name)
            if symbol.parent_symbol:
                symbol.parent_symbol = self._normalize_full_name(symbol.parent_symbol)
        
        # Normalize relationships
        for rel in relationships:
            if rel.source_symbol:
                rel.source_symbol = self._normalize_full_name(rel.source_symbol)
            if rel.target_symbol:
                rel.target_symbol = self._normalize_full_name(rel.target_symbol)
    
    def _setup_tree_sitter(self) -> bool:
        """Setup tree-sitter for C++"""
        if get_language is None or get_parser is None:
            logger.warning("tree_sitter_language_pack not available")
            return False
        try:
            self.ts_language = get_language('cpp')
            self.parser = get_parser('cpp')
            return True
        except Exception as e:
            logger.error(f"Failed to initialize tree-sitter C++ parser: {e}")
            return False
    
    def parse_batch(self, file_paths: List[Union[str, Path]]) -> List[ParseResult]:
        """
        Parse multiple C++ files in batch
        
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
                    language="cpp",
                    symbols=[],
                    relationships=[],
                    parse_time=0.0,
                    errors=[f"Parse error: {e}"]
                ))
        return results
    
    def parse_file(self, file_path: Union[str, Path], content: Optional[str] = None) -> ParseResult:
        """
        Parse a C++ file and extract symbols and relationships
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
            
            # Parse with tree-sitter
            if self.parser is None:
                return ParseResult(
                    file_path=file_path,
                    language="cpp",
                    symbols=[],
                    relationships=[],
                    file_hash=file_hash,
                    parse_time=time.time() - start_time,
                    errors=["Tree-sitter parser not available"]
                )
            
            tree = self.parser.parse(bytes(content, 'utf8'))
            if tree.root_node is None:
                return ParseResult(
                    file_path=file_path,
                    language="cpp",
                    symbols=[],
                    relationships=[],
                    file_hash=file_hash,
                    parse_time=time.time() - start_time,
                    errors=["Failed to parse C++ file"]
                )
            
            # Extract symbols and relationships
            symbols = self.extract_symbols(tree.root_node, file_path)
            relationships = self.extract_relationships(tree.root_node, symbols, file_path)
            
            # Normalize all full_name fields to use '.' separator instead of '::'
            # This standardizes C++ output to match other language parsers
            self._normalize_parse_result(symbols, relationships)
            
            result = ParseResult(
                file_path=file_path,
                language="cpp",
                symbols=symbols,
                relationships=relationships,
                file_hash=file_hash,
                parse_time=time.time() - start_time
            )
            
            return self.validate_result(result)
            
        except Exception as e:
            logger.error(f"Failed to parse C++ file {file_path}: {e}")
            return ParseResult(
                file_path=file_path,
                language="cpp",
                symbols=[],
                relationships=[],
                parse_time=time.time() - start_time,
                errors=[f"Parse error: {e}"]
            )
    
    def extract_symbols(self, ast_node: Any, file_path: str) -> List[Symbol]:
        """Extract symbols from C++ tree-sitter AST"""
        symbols = []
        self._current_scope_stack = []
        
        class SymbolExtractor:
            def __init__(self, parser):
                self.parser = parser
                self.symbols = []
                self.relationships = []  # For COMPOSITION/AGGREGATION/OVERRIDES relationships
                self.scope_stack = []
                self.class_scope_stack = []  # Track only classes/structs, not namespaces
                self.current_template_params = None
                self.class_bases = {}  # Phase 5: map class full_name -> [base class names]
            
            def visit(self, node):
                """Main visitor dispatch"""
                # Nodes that should NOT recurse after handling
                # class/struct: don't recurse after extracting (children are manually visited in _extract_class_or_struct)
                # function: don't recurse after extracting (parameters are handled in visit_function_definition)
                # namespace: don't recurse after extracting (children are manually visited in visit_namespace_definition)
                # enum: don't recurse after extracting (enumerators are manually visited in visit_enum_specifier)
                # template_declaration: children visited inside visit_template_declaration with current_template_params set
                # NOTE: field_declaration DOES recurse because it can contain nested classes
                no_recurse_types = {'class_specifier', 'struct_specifier', 'function_definition', 'namespace_definition', 'enum_specifier', 'template_declaration'}
                
                # Dispatch based on node type
                method_name = f'visit_{node.type}'
                handled = False
                if hasattr(self, method_name):
                    getattr(self, method_name)(node)
                    handled = True
                
                # Skip recursion for specific node types to prevent duplication
                if handled and node.type in no_recurse_types:
                    return
                
                # Visit children
                for child in node.children:
                    self.visit(child)
            
            def visit_translation_unit(self, node):
                """Root node - create MODULE symbol
                
                Standardized pattern (matches TypeScript, Python):
                1. Push module_name to scope_stack (for qualified names)
                2. All top-level classes/structs get parent_symbol = module_name
                3. Don't emit MODULE symbol (organizational, not architectural)
                
                This ensures User.cpp with class User creates:
                - CLASS: full_name = "User.User" (after normalization)
                """
                module_name = Path(file_path).stem
                
                # Push module to scope_stack so top-level classes get proper parent
                # (but don't emit MODULE symbol - modules are organizational, not architectural)
                self.scope_stack.append(module_name)
            
            def visit_namespace_definition(self, node):
                """Track namespace scope for qualified names.
                
                Namespaces are organizational, not architectural - we don't emit
                NAMESPACE symbols. We only track them in scope_stack for qualified
                full_names like "namespace.ClassName".
                """
                # Get namespace name
                name_node = self.parser._find_child_by_type(node, 'namespace_identifier')
                if not name_node:
                    return
                
                name = self.parser._get_node_text(name_node)
                
                # Push namespace onto scope stack (but don't emit symbol)
                self.scope_stack.append(name)
                
                # Visit declaration list (namespace body)
                declaration_list = self.parser._find_child_by_type(node, 'declaration_list')
                if declaration_list:
                    for child in declaration_list.children:
                        if child.type not in ['{', '}']:
                            self.visit(child)
                
                # Pop namespace from scope
                self.scope_stack.pop()
            
            def visit_class_specifier(self, node):
                """Extract CLASS symbol"""
                self._extract_class_or_struct(node, SymbolType.CLASS)
            
            def visit_struct_specifier(self, node):
                """Extract STRUCT symbol"""
                self._extract_class_or_struct(node, SymbolType.STRUCT)
            
            def visit_enum_specifier(self, node):
                """Extract ENUM symbol (both enum and enum class, including unnamed enums)"""
                # Get enum name (may be None for unnamed enums)
                name_node = self.parser._find_child_by_type(node, 'type_identifier')
                
                # Check if it's enum class (scoped enum)
                is_scoped = any(child.type == 'class' for child in node.children)
                
                if name_node:
                    name = self.parser._get_node_text(name_node)
                    if not name:
                        # Empty name from tree-sitter (e.g., 'enum : int { ... }')
                        name = None
                
                if not name_node or not name:
                    # Unnamed enum: generate synthetic name from first enumerator(s)
                    # e.g., enum { max_packed_args = 15 } -> "<anonymous_enum:max_packed_args>"
                    enumerator_list = self.parser._find_child_by_type(node, 'enumerator_list')
                    first_enumerator_name = None
                    if enumerator_list:
                        for child in enumerator_list.children:
                            if child.type == 'enumerator':
                                id_node = self.parser._find_child_by_type(child, 'identifier')
                                if id_node:
                                    first_enumerator_name = self.parser._get_node_text(id_node)
                                    break
                    
                    if first_enumerator_name:
                        name = f"<anonymous_enum:{first_enumerator_name}>"
                    else:
                        # No enumerators at all, skip
                        return
                
                parent_symbol = '::'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}::{name}" if parent_symbol else name
                
                # Create enum symbol with proper ENUM type
                symbol = Symbol(
                    name=name,
                    symbol_type=SymbolType.ENUM,
                    scope=Scope.CLASS,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    source_text=self.parser._get_node_text(node),
                    metadata={'is_enum': True, 'is_scoped_enum': is_scoped,
                              'is_anonymous': not bool(name_node and self.parser._get_node_text(name_node))}
                )
                self.symbols.append(symbol)
                
                # Extract enumerators as CONSTANT children
                self.scope_stack.append(name)
                enumerator_list = self.parser._find_child_by_type(node, 'enumerator_list')
                if enumerator_list:
                    for child in enumerator_list.children:
                        if child.type == 'enumerator':
                            self._extract_enumerator(child, full_name)
                self.scope_stack.pop()
            
            def visit_alias_declaration(self, node):
                """Extract type alias (using statement)"""
                # Get alias name (left side of =)
                name_node = None
                for child in node.children:
                    if child.type == 'type_identifier':
                        name_node = child
                        break
                
                if not name_node:
                    return
                
                name = self.parser._get_node_text(name_node)
                parent_symbol = '::'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}::{name}" if parent_symbol else name
                
                # Get aliased type (right side of =)
                aliased_type = None
                for child in node.children:
                    if child.type == 'type_descriptor':
                        aliased_type = self.parser._get_node_text(child).strip()
                        break
                
                # Create type alias symbol with source_text for vectorstore/graph content
                symbol = Symbol(
                    name=name,
                    symbol_type=SymbolType.TYPE_ALIAS,
                    scope=Scope.GLOBAL,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    return_type=aliased_type,
                    source_text=self.parser._get_node_text(node),
                    metadata={'is_type_alias': True, 'aliased_type': aliased_type}
                )
                self.symbols.append(symbol)
                
                # Emit relationships for the aliased type
                if aliased_type:
                    # Extract user-defined types referenced by the alias target
                    udt_types = self._extract_user_defined_types_from_field(aliased_type)
                    for base_type, is_ptr_or_ref in udt_types:
                        # ALIAS_OF: this type alias refers to the target type
                        relationship = Relationship(
                            source_symbol=full_name,
                            target_symbol=base_type,
                            relationship_type=RelationshipType.ALIAS_OF,
                            source_file=file_path,
                            source_range=self.parser._node_to_range(node),
                            annotations={
                                'alias_name': name,
                                'aliased_type': aliased_type
                            }
                        )
                        self.relationships.append(relationship)
            
            def visit_type_definition(self, node):
                """Extract typedef"""
                # Get typedef name (last type_identifier or from function_declarator for function pointer typedefs)
                name_node = None
                for child in reversed(node.children):
                    if child.type == 'type_identifier':
                        name_node = child
                        break
                    elif child.type == 'function_declarator':
                        # typedef void (*callback_t)(int, int) — name is inside parenthesized_declarator
                        paren = self.parser._find_child_by_type(child, 'parenthesized_declarator')
                        if paren:
                            ptr_decl = self.parser._find_child_by_type(paren, 'pointer_declarator')
                            if ptr_decl:
                                name_node = self.parser._find_child_by_type(ptr_decl, 'type_identifier')
                        if name_node:
                            break
                
                if not name_node:
                    return
                
                name = self.parser._get_node_text(name_node)
                parent_symbol = '::'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}::{name}" if parent_symbol else name
                
                # Get base type (everything before the name)
                base_type = None
                for child in node.children:
                    if child.type in ['qualified_identifier', 'primitive_type', 'struct_specifier',
                                      'enum_specifier', 'type_identifier', 'template_type',
                                      'sized_type_specifier', 'placeholder_type_specifier']:
                        base_type = self.parser._get_node_text(child).strip()
                        break
                
                # Create typedef symbol with source_text for vectorstore/graph content
                symbol = Symbol(
                    name=name,
                    symbol_type=SymbolType.TYPE_ALIAS,
                    scope=Scope.GLOBAL,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    return_type=base_type,
                    source_text=self.parser._get_node_text(node),
                    metadata={'is_typedef': True, 'base_type': base_type}
                )
                self.symbols.append(symbol)
                
                # Emit relationships for the base type
                if base_type:
                    udt_types = self._extract_user_defined_types_from_field(base_type)
                    for udt_name, is_ptr_or_ref in udt_types:
                        relationship = Relationship(
                            source_symbol=full_name,
                            target_symbol=udt_name,
                            relationship_type=RelationshipType.ALIAS_OF,
                            source_file=file_path,
                            source_range=self.parser._node_to_range(node),
                            annotations={
                                'alias_name': name,
                                'base_type': base_type
                            }
                        )
                        self.relationships.append(relationship)
            
            def visit_template_declaration(self, node):
                """Handle template declarations"""
                # Extract template parameters
                params = []
                param_list = self.parser._find_child_by_type(node, 'template_parameter_list')
                if param_list:
                    for child in param_list.children:
                        if child.type in ['type_parameter_declaration', 'parameter_declaration', 
                                         'optional_type_parameter_declaration', 'variadic_type_parameter_declaration']:
                            param_name = self._extract_template_param_name(child)
                            if param_name:
                                params.append(param_name)
                
                # Store template params for use by wrapped declaration
                old_params = self.current_template_params
                self.current_template_params = params
                
                # Find and visit the wrapped declaration (class, function, alias, typedef, etc.)
                # All children are visited here with current_template_params set so
                # that _extract_user_defined_types_from_field can filter T/K/V etc.
                for child in node.children:
                    if child.type in ['class_specifier', 'struct_specifier', 'function_definition',
                                      'alias_declaration', 'type_definition']:
                        self.visit(child)
                    elif child.type == 'declaration':
                        # Template method declaration (inside class body)
                        self._handle_template_method_declaration(child)
                
                # Restore previous template params
                self.current_template_params = old_params
            
            def _handle_template_method_declaration(self, decl_node):
                """Handle template method declarations inside class bodies."""
                # Find function_declarator in the declaration
                func_declarator = self.parser._find_child_by_type(decl_node, 'function_declarator')
                if not func_declarator:
                    return
                
                # Extract method name (handles operators too)
                name = self._extract_function_name(func_declarator)
                if not name:
                    return
                
                parent_symbol = '::'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}::{name}" if parent_symbol else name
                
                # Extract return type
                return_type = None
                for child in decl_node.children:
                    if child.type in ['primitive_type', 'type_identifier',
                                      'qualified_identifier', 'template_type',
                                      'sized_type_specifier',
                                      'placeholder_type_specifier']:
                        return_type = self.parser._get_node_text(child)
                        break
                
                # Extract parameters
                parameters = self._extract_function_parameters(func_declarator)
                
                # Create method symbol
                # Check scope to determine if this is truly a method (inside class/struct)
                # or a free function (at namespace/global scope)
                is_method = bool(self.class_scope_stack)
                symbol = Symbol(
                    name=name,
                    symbol_type=SymbolType.METHOD if is_method else SymbolType.FUNCTION,
                    scope=Scope.FUNCTION,
                    range=self.parser._node_to_range(decl_node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    return_type=return_type,
                    parameter_types=[p.get('type') for p in parameters if p.get('type')]
                )
                
                # Add template metadata
                if self.current_template_params:
                    symbol.metadata = {
                        'is_template': True,
                        'template_params': self.current_template_params
                    }
                
                self.symbols.append(symbol)
                
                # Extract function type relationships (parameters and return type)
                self._extract_function_type_relationships(
                    function_symbol=full_name,
                    return_type=return_type,
                    parameter_types=[p.get('type') for p in parameters if p.get('type')],
                    node=decl_node
                )
                
                # Add PARAMETER symbols
                for param in parameters:
                    param_symbol = Symbol(
                        name=param['name'],
                        symbol_type=SymbolType.PARAMETER,
                        scope=Scope.FUNCTION,
                        range=param.get('range', symbol.range),
                        file_path=file_path,
                        parent_symbol=full_name,
                        return_type=param.get('type')
                    )
                    self.symbols.append(param_symbol)
            
            def _extract_class_or_struct(self, node, symbol_type):
                """Common logic for class and struct extraction"""
                # Get class/struct name
                name = self._extract_class_name(node)
                if not name:
                    return
                
                parent_symbol = '::'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}::{name}" if parent_symbol else name
                
                # Check if this is a forward declaration (class Foo; without body)
                has_body = self.parser._find_child_by_type(node, 'field_declaration_list') is not None
                
                # Extract base classes (only for full definitions)
                bases = self._extract_base_classes(node) if has_body else []
                
                # Create symbol
                symbol = Symbol(
                    name=name,
                    symbol_type=symbol_type,
                    scope=Scope.CLASS if self.scope_stack else Scope.GLOBAL,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    source_text=self.parser._get_node_text(node)  # Include declaration text even for forward decls
                )
                
                # Mark forward declarations in metadata
                # NOTE: Keep source_text - it contains the declaration line (e.g., "class Foo;")
                # which is needed for the vector store and later augmentation with definitions
                metadata = {}
                if not has_body:
                    metadata['is_forward_declaration'] = True
                    metadata['declaration_kind'] = 'forward'
                    # Don't clear source_text - it has the declaration line
                
                # Add template metadata if in template context
                if self.current_template_params:
                    metadata['is_template'] = True
                    metadata['template_params'] = self.current_template_params
                
                if metadata:
                    symbol.metadata = metadata
                
                self.symbols.append(symbol)
                
                # Only enter scope and visit children for full definitions (not forward declarations)
                if has_body:
                    self.scope_stack.append(name)
                    self.class_scope_stack.append(name)  # Track class/struct scope separately
                    # Phase 5: store base classes for override resolution
                    if bases:
                        self.class_bases[full_name] = bases
                    for child in node.children:
                        self.visit(child)
                    self.class_scope_stack.pop()
                    self.scope_stack.pop()
            
            def _extract_class_name(self, node):
                """Extract class/struct name from node"""
                for child in node.children:
                    if child.type == 'type_identifier':
                        return self.parser._get_node_text(child)
                return None
            
            def _extract_base_classes(self, node):
                """Extract base class names"""
                bases = []
                base_clause = self.parser._find_child_by_type(node, 'base_class_clause')
                if base_clause:
                    for child in base_clause.children:
                        if child.type in ['type_identifier', 'qualified_identifier']:
                            bases.append(self.parser._get_node_text(child))
                        elif child.type == 'template_type':
                            # Extract base template name
                            base_name = self._extract_base_template_name(child)
                            if base_name:
                                bases.append(base_name)
                return bases
            
            def _extract_base_template_name(self, template_node):
                """Extract base name from template type"""
                for child in template_node.children:
                    if child.type == 'type_identifier':
                        return self.parser._get_node_text(child)
                return None
            
            def _extract_enumerator(self, enumerator_node, parent_full_name):
                """Extract enumerator (enum value) as CONSTANT"""
                # Get enumerator name
                name_node = self.parser._find_child_by_type(enumerator_node, 'identifier')
                if not name_node:
                    return
                
                name = self.parser._get_node_text(name_node)
                full_name = f"{parent_full_name}::{name}"
                
                # Create constant symbol for the enumerator
                symbol = Symbol(
                    name=name,
                    symbol_type=SymbolType.CONSTANT,
                    scope=Scope.CLASS,
                    range=self.parser._node_to_range(enumerator_node),
                    file_path=file_path,
                    parent_symbol=parent_full_name,
                    full_name=full_name,
                    source_text=self.parser._get_node_text(enumerator_node)
                )
                self.symbols.append(symbol)
            
            def visit_declaration(self, node):
                """Extract function declarations and const declarations"""
                # Check if this is a function declaration by looking for function_declarator
                declarator = self.parser._find_child_by_type(node, 'function_declarator')
                if declarator:
                    self._extract_function_declaration(node, declarator)
                    return
                
                # Check if this is a const or extern declaration
                # Handle: const int X = 1; extern const Color darkGrey; extern int Y;
                has_const = False
                has_extern = False
                for child in node.children:
                    if child.type == 'type_qualifier':
                        qualifier_text = self.parser._get_node_text(child).lower()
                        if 'const' in qualifier_text:
                            has_const = True
                    elif child.type == 'storage_class_specifier':
                        specifier_text = self.parser._get_node_text(child).lower()
                        if 'extern' in specifier_text:
                            has_extern = True
                
                # Extract as constant if it has const qualifier or is extern declaration
                if has_const or has_extern:
                    self._extract_const_declaration(node)
                    return
            
            def _extract_function_declaration(self, node, declarator):
                """Extract function declaration as a symbol"""
                # Get function name
                qualified_parts = self._extract_qualified_name_parts(declarator)
                if qualified_parts and len(qualified_parts) > 1:
                    # Qualified name
                    name = qualified_parts[-1]
                    parent_symbol = '::'.join(qualified_parts[:-1])
                    full_name = '::'.join(qualified_parts)
                    symbol_type = SymbolType.METHOD
                else:
                    # Simple name
                    name = self._extract_function_name(declarator)
                    if not name:
                        return
                    
                    parent_symbol = '::'.join(self.scope_stack) if self.scope_stack else None
                    full_name = f"{parent_symbol}::{name}" if parent_symbol else name
                    # Free functions in namespaces are FUNCTION, methods in classes are METHOD
                    symbol_type = SymbolType.METHOD if self.class_scope_stack else SymbolType.FUNCTION
                
                # Extract return type
                return_type = self._extract_return_type(node)
                
                # Extract parameters
                parameters = self._extract_function_parameters(declarator)
                
                # Create function symbol
                symbol = Symbol(
                    name=name,
                    symbol_type=symbol_type,
                    scope=Scope.FUNCTION,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    return_type=return_type,
                    parameter_types=[p.get('type') for p in parameters if p.get('type')],
                    source_text=self.parser._get_node_text(node)  # Include declaration text
                )
                
                # Add metadata
                metadata = {}
                if self.current_template_params:
                    metadata['is_template'] = True
                    metadata['template_params'] = self.current_template_params
                
                # Mark as declaration (no body)
                metadata['has_body'] = False
                
                if metadata:
                    symbol.metadata = metadata
                
                self.symbols.append(symbol)
            
            def _extract_const_declaration(self, node):
                """Extract const declaration as CONSTANT symbol (both definitions and forward declarations)"""
                # Get the type
                const_type = None
                for child in node.children:
                    if child.type in ['primitive_type', 'type_identifier',
                                      'qualified_identifier', 'template_type',
                                      'sized_type_specifier',
                                      'placeholder_type_specifier']:
                        const_type = self.parser._get_node_text(child)
                        break
                
                # Extract all init_declarators (can have multiple: const int a=1, b=2;)
                found_declarators = False
                for child in node.children:
                    if child.type == 'init_declarator':
                        found_declarators = True
                        # Get the identifier
                        identifier_node = self.parser._find_child_by_type(child, 'identifier')
                        if not identifier_node:
                            continue
                        
                        name = self.parser._get_node_text(identifier_node)
                        parent_symbol = '::'.join(self.scope_stack) if self.scope_stack else None
                        full_name = f"{parent_symbol}::{name}" if parent_symbol else name
                        
                        # Create constant symbol
                        symbol = Symbol(
                            name=name,
                            symbol_type=SymbolType.CONSTANT,
                            scope=Scope.GLOBAL if not parent_symbol else Scope.CLASS,
                            range=self.parser._node_to_range(child),
                            file_path=file_path,
                            parent_symbol=parent_symbol,
                            full_name=full_name,
                            return_type=const_type,
                            source_text=self.parser._get_node_text(node)  # Include full declaration line
                        )
                        self.symbols.append(symbol)
                
                # Handle forward declarations without initializers (extern const int MAX_SIZE;)
                if not found_declarators:
                    for child in node.children:
                        if child.type == 'identifier':
                            name = self.parser._get_node_text(child)
                            parent_symbol = '::'.join(self.scope_stack) if self.scope_stack else None
                            full_name = f"{parent_symbol}::{name}" if parent_symbol else name
                            
                            # Create constant symbol
                            symbol = Symbol(
                                name=name,
                                symbol_type=SymbolType.CONSTANT,
                                scope=Scope.GLOBAL if not parent_symbol else Scope.CLASS,
                                range=self.parser._node_to_range(child),
                                file_path=file_path,
                                parent_symbol=parent_symbol,
                                full_name=full_name,
                                return_type=const_type,
                                source_text=self.parser._get_node_text(node)  # Include full declaration line
                            )
                            self.symbols.append(symbol)
            
            def visit_function_definition(self, node):
                """Extract FUNCTION, METHOD, or CONSTRUCTOR symbol"""
                # Check if this is actually a macro-wrapped class (e.g., class EXPORT MyClass)
                # Tree-sitter misparses these as function_definition
                class_node = self.parser._find_child_by_type(node, 'class_specifier')
                struct_node = self.parser._find_child_by_type(node, 'struct_specifier')
                
                if class_node:
                    # This is a macro-wrapped class - extract it properly
                    self.visit_class_specifier(class_node)
                    return
                elif struct_node:
                    # This is a macro-wrapped struct
                    self.visit_struct_specifier(struct_node)
                    return
                
                # Get function name (might be wrapped in reference/pointer declarator)
                declarator = self.parser._find_child_by_type(node, 'function_declarator')
                if not declarator:
                    # Check inside reference_declarator or pointer_declarator
                    for child in node.children:
                        if child.type in ['reference_declarator', 'pointer_declarator']:
                            declarator = self.parser._find_child_by_type(child, 'function_declarator')
                            if declarator:
                                break
                
                if not declarator:
                    return
                
                # Check for qualified name (Class::method) in .cpp files
                qualified_parts = self._extract_qualified_name_parts(declarator)
                if qualified_parts and len(qualified_parts) > 1:
                    # Qualified name: Point::distance or math::Point::distance or math::max
                    # Note: qualified_parts are relative to current scope, need to make absolute
                    name = qualified_parts[-1]
                    
                    # Make the qualified parts absolute by prepending scope_stack
                    if self.scope_stack:
                        absolute_parts = self.scope_stack + qualified_parts
                    else:
                        absolute_parts = qualified_parts
                    
                    # Parent is everything except the last part
                    parent_symbol = '::'.join(absolute_parts[:-1])
                    full_name = '::'.join(absolute_parts)
                    
                    # Determine symbol type
                    parent_class = qualified_parts[-2] if len(qualified_parts) >= 2 else None
                    # Strip template parameters from parent_class for comparison (e.g., Array<T> -> Array)
                    parent_class_base = parent_class.split('<')[0] if parent_class else None
                    if name == parent_class_base:
                        # Constructor: function name matches parent class name (ignoring template params)
                        symbol_type = SymbolType.CONSTRUCTOR
                    else:
                        # Check if this is a class/struct method or a free function in namespace
                        # Heuristic: If the immediate parent (last qualifier before name) starts with uppercase,
                        # it's likely a class/struct → METHOD. Otherwise, it's likely a namespace → FUNCTION.
                        immediate_parent = qualified_parts[-2] if len(qualified_parts) >= 2 else None
                        if immediate_parent and immediate_parent[0].isupper():
                            # Looks like a class/struct (PascalCase) → METHOD
                            symbol_type = SymbolType.METHOD
                        else:
                            # Looks like a namespace (lowercase) or top-level → FUNCTION
                            symbol_type = SymbolType.FUNCTION
                else:
                    # Non-qualified name: use scope_stack
                    name = self._extract_function_name(declarator)
                    if not name:
                        return
                    
                    parent_symbol = '::'.join(self.scope_stack) if self.scope_stack else None
                    full_name = f"{parent_symbol}::{name}" if parent_symbol else name
                    
                    # Determine symbol type
                    if self.class_scope_stack and parent_symbol and name == self.class_scope_stack[-1]:
                        # Constructor: function name matches parent *class* name
                        # Must check class_scope_stack (not scope_stack which includes module/namespace)
                        symbol_type = SymbolType.CONSTRUCTOR
                    elif self.class_scope_stack:
                        # Inside class/struct: METHOD
                        symbol_type = SymbolType.METHOD
                    else:
                        # Free function (possibly in namespace): FUNCTION
                        symbol_type = SymbolType.FUNCTION
                
                # Extract return type
                return_type = self._extract_return_type(node)
                
                # Extract parameters
                parameters = self._extract_function_parameters(declarator)
                
                # Check if this is a const method
                is_const = self._is_const_method(declarator)
                
                # Phase 5: Check for override specifier
                is_override = False
                for child in declarator.children:
                    if child.type == 'virtual_specifier' and self.parser._get_node_text(child) == 'override':
                        is_override = True
                        break
                
                # Create function symbol
                symbol = Symbol(
                    name=name,
                    symbol_type=symbol_type,
                    scope=Scope.FUNCTION,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    return_type=return_type,
                    parameter_types=[p.get('type') for p in parameters if p.get('type')],
                    source_text=self.parser._get_node_text(node)
                )
                
                # Add metadata
                metadata = {}
                if self.current_template_params:
                    metadata['is_template'] = True
                    metadata['template_params'] = self.current_template_params
                
                # Add const method flag
                if is_const:
                    metadata['is_const_method'] = True
                
                # Phase 5: override metadata
                if is_override:
                    metadata['is_override'] = True
                
                # Mark as having a body (this is a definition, not just a declaration)
                metadata['has_body'] = True
                metadata['has_qualified_name'] = bool(qualified_parts and len(qualified_parts) > 1)
                
                if metadata:
                    symbol.metadata = metadata
                
                self.symbols.append(symbol)
                
                # Phase 5: Emit OVERRIDES relationship for override methods
                if is_override and parent_symbol:
                    base_classes = self.class_bases.get(parent_symbol, [])
                    if base_classes:
                        base_class = base_classes[0]
                        overrides_rel = Relationship(
                            source_symbol=full_name,
                            target_symbol=f"{base_class}::{name}",
                            relationship_type=RelationshipType.OVERRIDES,
                            source_file=file_path,
                            source_range=self.parser._node_to_range(node),
                            annotations={'override_keyword': 'override'}
                        )
                        self.relationships.append(overrides_rel)
                
                # Extract function type relationships (parameters and return type)
                self._extract_function_type_relationships(
                    function_symbol=full_name,
                    return_type=return_type,
                    parameter_types=[p.get('type') for p in parameters if p.get('type')],
                    node=node
                )
                
                # Add PARAMETER symbols
                for param in parameters:
                    param_symbol = Symbol(
                        name=param['name'],
                        symbol_type=SymbolType.PARAMETER,
                        scope=Scope.FUNCTION,
                        range=param.get('range', symbol.range),
                        file_path=file_path,
                        parent_symbol=full_name,
                        return_type=param.get('type')
                    )
                    self.symbols.append(param_symbol)
            
            def _extract_qualified_name_parts(self, declarator_node):
                """Extract all parts from a qualified name (e.g., ['math', 'Point', 'distance'])"""
                for child in declarator_node.children:
                    if child.type == 'qualified_identifier':
                        # Parse the qualified_identifier node structure
                        parts = []
                        self._collect_qualified_parts(child, parts)
                        return parts if parts else None
                return None
            
            def _collect_qualified_parts(self, node, parts):
                """Recursively collect parts from qualified_identifier"""
                for child in node.children:
                    if child.type == 'namespace_identifier':
                        # This is actually a class or namespace name
                        parts.append(self.parser._get_node_text(child))
                    elif child.type == 'template_type':
                        # Template type like Array<T> - include full text with template args
                        parts.append(self.parser._get_node_text(child))
                    elif child.type == 'identifier':
                        # Final method/function name
                        parts.append(self.parser._get_node_text(child))
                    elif child.type == 'operator_name':
                        # Operator overload (e.g., operator+, operator+=)
                        parts.append(self.parser._get_node_text(child))
                    elif child.type == 'qualified_identifier':
                        # Nested qualified identifier (for deeply nested names)
                        self._collect_qualified_parts(child, parts)
            
            def _extract_function_name(self, declarator_node):
                """Extract function name from function_declarator"""
                for child in declarator_node.children:
                    if child.type == 'identifier':
                        return self.parser._get_node_text(child)
                    elif child.type == 'operator_name':
                        # Operator overload: operator+, operator+=, etc.
                        return self.parser._get_node_text(child)
                    elif child.type == 'qualified_identifier':
                        # For qualified names like ClassName::methodName
                        parts = self.parser._get_node_text(child).split('::')
                        return parts[-1] if parts else None
                    elif child.type == 'field_identifier':
                        return self.parser._get_node_text(child)
                return None
            
            def _extract_return_type(self, function_node):
                """Extract return type from function definition"""
                for child in function_node.children:
                    if child.type in ['primitive_type', 'type_identifier',
                                      'qualified_identifier', 'template_type',
                                      'sized_type_specifier',
                                      'placeholder_type_specifier']:
                        return self.parser._get_node_text(child)
                return None
            
            def _is_const_method(self, declarator_node):
                """Check if function_declarator has const qualifier (for const methods)"""
                if not declarator_node or declarator_node.type != 'function_declarator':
                    return False
                
                # Check for type_qualifier child with 'const'
                for child in declarator_node.children:
                    if child.type == 'type_qualifier':
                        if 'const' in self.parser._get_node_text(child):
                            return True
                return False
            
            def _extract_function_parameters(self, declarator_node):
                """Extract function parameters"""
                params = []
                param_list = self.parser._find_child_by_type(declarator_node, 'parameter_list')
                if param_list:
                    for child in param_list.children:
                        if child.type == 'parameter_declaration':
                            param_info = self._extract_parameter_info(child)
                            if param_info:
                                params.append(param_info)
                return params
            
            def _extract_parameter_info(self, param_node):
                """Extract parameter name and type with full qualifiers (const, &, &&, *)"""
                param_type = None
                param_name = None
                is_const_type = False  # const before type: const int
                pointer_qualifier = ''  # '', '*', '**', etc.
                const_after_pointer = False  # const after *: int* const
                ref_qualifier = ''  # '', '&', or '&&'
                
                for child in param_node.children:
                    if child.type == 'type_qualifier':
                        # Const qualifier before type: const int
                        if 'const' in self.parser._get_node_text(child):
                            is_const_type = True
                    elif child.type in ['primitive_type', 'type_identifier',
                                        'qualified_identifier', 'template_type',
                                        'sized_type_specifier',
                                        'placeholder_type_specifier']:
                        param_type = self.parser._get_node_text(child)
                    elif child.type == 'identifier':
                        param_name = self.parser._get_node_text(child)
                    elif child.type == 'pointer_declarator':
                        # Pointer declarator: int* p, const int* p, int* const p, etc.
                        # Extract full pointer qualifier from text, removing the identifier
                        
                        declarator_text = self.parser._get_node_text(child)
                        
                        # Extract parameter name first
                        for subchild in child.children:
                            if subchild.type == 'identifier':
                                param_name = self.parser._get_node_text(subchild)
                                break
                            elif subchild.type == 'pointer_declarator':
                                # Recursively find identifier in nested pointer
                                def find_identifier(node):
                                    for c in node.children:
                                        if c.type == 'identifier':
                                            return self.parser._get_node_text(c)
                                        elif c.type == 'pointer_declarator':
                                            result = find_identifier(c)
                                            if result:
                                                return result
                                    return None
                                param_name = find_identifier(subchild)
                                if param_name:
                                    break
                        
                        # Extract pointer qualifier: everything between type and identifier
                        # Remove whitespace and identifier from declarator text
                        if param_name:
                            # Remove the identifier and any trailing content
                            ptr_text = declarator_text
                            if param_name in ptr_text:
                                # Split on the identifier and take everything before it
                                ptr_text = ptr_text[:ptr_text.index(param_name)]
                            # Clean up: remove extra whitespace, keep * and const
                            import re
                            # Match patterns like: *, * const, **, * const*, etc.
                            # Keep all * and const keywords, normalize spacing
                            ptr_text = ptr_text.strip()
                            # Normalize: * const should have space, ** should not
                            ptr_text = re.sub(r'\s+', ' ', ptr_text)  # Normalize whitespace
                            ptr_text = re.sub(r'\*\s+const', '* const', ptr_text)  # * const with space
                            ptr_text = re.sub(r'\*\s+\*', '**', ptr_text)  # ** without space (but keep * const* format)
                            # Handle * const* pattern
                            ptr_text = re.sub(r'\*\s+const\s+\*', '* const*', ptr_text)
                            
                            pointer_qualifier = ptr_text
                    elif child.type == 'reference_declarator':
                        # Reference declarator: int& r, int&& rr
                        ref_text = self.parser._get_node_text(child)
                        if '&&' in ref_text:
                            ref_qualifier = '&&'
                        elif '&' in ref_text:
                            ref_qualifier = '&'
                        # Extract name from reference declarator
                        for subchild in child.children:
                            if subchild.type == 'identifier':
                                param_name = self.parser._get_node_text(subchild)
                
                if param_name and param_type:
                    # Build full type signature: const int*, int* const, const int* const, etc.
                    full_type = param_type
                    
                    # Add const before type
                    if is_const_type:
                        full_type = f"const {full_type}"
                    
                    # Add pointer qualifier
                    if pointer_qualifier:
                        full_type = f"{full_type}{pointer_qualifier}"
                    
                    # Add const after pointer
                    if const_after_pointer:
                        full_type = f"{full_type} const"
                    
                    # Add reference qualifier
                    if ref_qualifier:
                        full_type = f"{full_type}{ref_qualifier}"
                    
                    return {
                        'name': param_name,
                        'type': full_type,  # Now includes const, pointer, and ref qualifiers
                        'range': self.parser._node_to_range(param_node)
                    }
                return None
            
            def visit_field_declaration(self, node):
                """Extract FIELD or METHOD symbols from class/struct"""
                if not self.scope_stack:
                    return  # Skip global variables for now
                
                # Check if this is actually a method declaration (has function_declarator)
                # It might be direct child or wrapped in reference_declarator/pointer_declarator
                func_declarator = self.parser._find_child_by_type(node, 'function_declarator')
                if not func_declarator:
                    # Check inside reference_declarator or pointer_declarator
                    for child in node.children:
                        if child.type in ['reference_declarator', 'pointer_declarator']:
                            func_declarator = self.parser._find_child_by_type(child, 'function_declarator')
                            if func_declarator:
                                break
                
                if func_declarator:
                    # This is a method declaration - extract it as a METHOD
                    self._extract_method_declaration(node, func_declarator)
                    return
                
                # Extract base field type (without pointer/reference markers)
                # Handles: primitive_type (int), type_identifier (Point),
                # qualified_identifier (std::vector<T>), template_type (optional<T>),
                # sized_type_specifier (unsigned char, long long),
                # enum_specifier (enum Color), struct_specifier (struct Inner {...}),
                # placeholder_type_specifier (auto, decltype(auto))
                base_field_type = None
                for child in node.children:
                    if child.type in ['primitive_type', 'type_identifier',
                                      'qualified_identifier', 'template_type',
                                      'sized_type_specifier',
                                      'enum_specifier', 'struct_specifier',
                                      'placeholder_type_specifier']:
                        base_field_type = self.parser._get_node_text(child)
                        break
                
                # Extract field declarators and field identifiers (may have multiple fields declared at once)
                for child in node.children:
                    field_name = None
                    field_type = base_field_type  # Start with base type
                    
                    if child.type == 'field_declarator':
                        field_name = self._extract_field_name(child)
                        # For pointer/reference fields, extract complete type including * or &
                        # Check if field_declarator contains pointer_declarator or reference_declarator
                        pointer_declarator = self.parser._find_child_by_type(child, 'pointer_declarator')
                        reference_declarator = self.parser._find_child_by_type(child, 'reference_declarator')
                        if pointer_declarator:
                            # Count the * symbols in pointer_declarator
                            pointer_text = self.parser._get_node_text(pointer_declarator)
                            star_count = pointer_text.count('*')
                            field_type = base_field_type + ('*' * star_count) if base_field_type else None
                        elif reference_declarator:
                            field_type = base_field_type + '&' if base_field_type else None
                    elif child.type == 'pointer_declarator':
                        # Direct pointer declarator (e.g., Point* ptr;)
                        field_name = self._extract_field_name_from_declarator(child)
                        pointer_text = self.parser._get_node_text(child)
                        star_count = pointer_text.count('*')
                        field_type = base_field_type + ('*' * star_count) if base_field_type else None
                    elif child.type == 'reference_declarator':
                        # Direct reference declarator (e.g., Point& ref;)
                        field_name = self._extract_field_name_from_declarator(child)
                        field_type = base_field_type + '&' if base_field_type else None
                    elif child.type == 'array_declarator':
                        # Array fields: int values[3]; Point corners[4]; char buf[256];
                        # The field_identifier is nested inside array_declarator(s)
                        # For multi-dim arrays, array_declarator nests: arr[4][4] ->
                        #   array_declarator(array_declarator(field_identifier, [, 4, ]), [, 4, ])
                        inner = child
                        while inner.type == 'array_declarator':
                            nested = self.parser._find_child_by_type(inner, 'array_declarator')
                            if nested:
                                inner = nested
                            else:
                                break
                        field_id = self.parser._find_child_by_type(inner, 'field_identifier')
                        if field_id:
                            field_name = self.parser._get_node_text(field_id)
                        # field_type stays as base_field_type (the element type)
                    elif child.type == 'field_identifier':
                        # Handle simple declarations like: float x, y;
                        # Also handles bitfield declarations (field_identifier + bitfield_clause)
                        field_name = self.parser._get_node_text(child)
                    
                    if field_name:
                        parent_symbol = '::'.join(self.scope_stack)
                        full_name = f"{parent_symbol}::{field_name}"
                        
                        symbol = Symbol(
                            name=field_name,
                            symbol_type=SymbolType.FIELD,
                            scope=Scope.CLASS,
                            range=self.parser._node_to_range(child),
                            file_path=file_path,
                            parent_symbol=parent_symbol,
                            full_name=full_name,
                            return_type=field_type
                        )
                        self.symbols.append(symbol)
                        
                        # Create COMPOSITION/AGGREGATION relationship for user-defined types
                        if field_type:
                            self._extract_field_type_relationship(field_type, full_name, field_name, node)
            
            def _extract_method_declaration(self, field_decl_node, func_declarator):
                """Extract METHOD symbol from method declaration in class/struct"""
                # Get method name
                name = self._extract_function_name(func_declarator)
                if not name:
                    return
                
                parent_symbol = '::'.join(self.scope_stack)
                full_name = f"{parent_symbol}::{name}"
                
                # Determine if this is a constructor
                if name == self.scope_stack[-1]:
                    symbol_type = SymbolType.CONSTRUCTOR
                else:
                    symbol_type = SymbolType.METHOD
                
                # Extract return type
                return_type = self._extract_return_type(field_decl_node)
                
                # Extract parameters
                parameters = self._extract_function_parameters(func_declarator)
                
                # Check if this is a const method
                is_const = self._is_const_method(func_declarator)
                
                # Phase 5: Check for override specifier
                is_override = False
                for child in func_declarator.children:
                    if child.type == 'virtual_specifier' and self.parser._get_node_text(child) == 'override':
                        is_override = True
                        break
                
                # Create METHOD symbol
                symbol = Symbol(
                    name=name,
                    symbol_type=symbol_type,
                    scope=Scope.FUNCTION,
                    range=self.parser._node_to_range(field_decl_node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    return_type=return_type,
                    parameter_types=[p.get('type') for p in parameters if p.get('type')],
                    source_text=self.parser._get_node_text(field_decl_node)
                )
                
                # Add metadata for const methods and template info
                metadata = {}
                if is_const:
                    metadata['is_const_method'] = True
                if is_override:
                    metadata['is_override'] = True
                
                # Add template info if in template context
                if self.current_template_params:
                    metadata['is_template'] = True
                    metadata['template_params'] = self.current_template_params
                    template_info = f"TEMPLATE<{', '.join(self.current_template_params)}>"
                    if symbol.comments:
                        symbol.comments.append(template_info)
                    else:
                        symbol.comments = [template_info]
                
                # Store metadata on symbol
                if metadata:
                    symbol.metadata = metadata
                
                self.symbols.append(symbol)
                
                # Phase 5: Emit OVERRIDES relationship for override methods
                if is_override and parent_symbol:
                    # Find base class from class_bases mapping
                    base_classes = self.class_bases.get(parent_symbol, [])
                    if base_classes:
                        base_class = base_classes[0]  # Use first base class
                        overrides_rel = Relationship(
                            source_symbol=full_name,
                            target_symbol=f"{base_class}::{name}",
                            relationship_type=RelationshipType.OVERRIDES,
                            source_file=file_path,
                            source_range=self.parser._node_to_range(field_decl_node),
                            annotations={'override_keyword': 'override'}
                        )
                        self.relationships.append(overrides_rel)
                
                # Add PARAMETER symbols
                for param in parameters:
                    param_symbol = Symbol(
                        name=param['name'],
                        symbol_type=SymbolType.PARAMETER,
                        scope=Scope.FUNCTION,
                        range=param.get('range', self.parser._node_to_range(field_decl_node)),
                        file_path=file_path,
                        parent_symbol=full_name,
                        full_name=f"{full_name}::{param['name']}",
                        return_type=param.get('type')
                    )
                    self.symbols.append(param_symbol)
            
            def visit_preproc_def(self, node):
                """Extract MACRO symbols from simple #define macros.
                
                Both simple (#define FOO 42) and value-less (#define GUARD_H)
                macros are emitted as SymbolType.MACRO so the graph builder
                stores them with symbol_type='macro' — matching
                ARCHITECTURAL_SYMBOLS and enabling macro-aware expansion.
                """
                # Get macro name
                name_node = self.parser._find_child_by_type(node, 'identifier')
                if not name_node:
                    return
                
                name = self.parser._get_node_text(name_node)
                
                # Get macro value if present
                value = None
                value_node = self.parser._find_child_by_type(node, 'preproc_arg')
                if value_node:
                    value = self.parser._get_node_text(value_node)
                
                # Create MACRO symbol for #define
                parent_symbol = '::'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}::{name}" if parent_symbol else name
                
                symbol = Symbol(
                    name=name,
                    symbol_type=SymbolType.MACRO,
                    scope=Scope.GLOBAL,  # Macros are typically global
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    source_text=self.parser._get_node_text(node),
                    metadata={'macro': True, 'value': value}
                )
                self.symbols.append(symbol)
            
            def visit_preproc_function_def(self, node):
                """Extract MACRO symbols from function-like #define macros.
                
                Function-like macros (#define FOO(x) ...) are emitted as
                SymbolType.MACRO (not FUNCTION) so the graph stores them
                with symbol_type='macro'.  The 'function_macro' flag in
                metadata distinguishes them from simple value macros.
                """
                # Get macro name
                name_node = self.parser._find_child_by_type(node, 'identifier')
                if not name_node:
                    return
                
                name = self.parser._get_node_text(name_node)
                
                # Extract macro parameters
                params = []
                param_list = self.parser._find_child_by_type(node, 'preproc_params')
                if param_list:
                    for child in param_list.children:
                        if child.type == 'identifier':
                            params.append(self.parser._get_node_text(child))
                
                # Get macro body
                body = None
                body_node = self.parser._find_child_by_type(node, 'preproc_arg')
                if body_node:
                    body = self.parser._get_node_text(body_node)
                
                # Create MACRO symbol for function-like macros
                parent_symbol = '::'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}::{name}" if parent_symbol else name
                
                symbol = Symbol(
                    name=name,
                    symbol_type=SymbolType.MACRO,
                    scope=Scope.GLOBAL,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    parameter_types=[p for p in params],  # Store param names
                    source_text=self.parser._get_node_text(node),
                    metadata={'macro': True, 'function_macro': True, 'body': body}
                )
                self.symbols.append(symbol)
                
                # Create PARAMETER symbols for macro parameters
                for i, param in enumerate(params):
                    param_symbol = Symbol(
                        name=param,
                        symbol_type=SymbolType.PARAMETER,
                        scope=Scope.FUNCTION,
                        range=self.parser._node_to_range(param_list),  # Approximate
                        file_path=file_path,
                        parent_symbol=full_name,
                        full_name=f"{full_name}::{param}",
                        metadata={'macro_param': True, 'position': i}
                    )
                    self.symbols.append(param_symbol)
            
            def visit_preproc_include(self, node):
                """Extract #include directives as symbols for tracking dependencies
                
                Handles:
                    - System includes: #include <iostream>
                    - Local includes: #include "myheader.h"
                
                Creates symbols that can be used for IMPORTS relationships.
                """
                # file_path is available from the closure (extract_symbols parameter)
                
                # Extract include path
                include_path = None
                is_system = False
                
                # Check for system include: <...>
                system_lib = self.parser._find_child_by_type(node, 'system_lib_string')
                if system_lib:
                    # Get text and strip < >
                    include_path = self.parser._get_node_text(system_lib).strip('<>')
                    is_system = True
                else:
                    # Check for local include: "..."
                    string_lit = self.parser._find_child_by_type(node, 'string_literal')
                    if string_lit:
                        string_content = self.parser._find_child_by_type(string_lit, 'string_content')
                        if string_content:
                            include_path = self.parser._get_node_text(string_content)
                            is_system = False
                
                if include_path:
                    # Create a symbol for this include
                    # We use a special naming convention: include::<path>
                    parent_symbol = '::'.join(self.scope_stack) if self.scope_stack else None
                    include_name = f"include::{include_path}"
                    full_name = f"{parent_symbol}::{include_name}" if parent_symbol else include_name
                    
                    symbol = Symbol(
                        name=include_name,
                        symbol_type=SymbolType.VARIABLE,  # Use VARIABLE as closest match for includes
                        scope=Scope.GLOBAL,
                        range=self.parser._node_to_range(node),
                        file_path=file_path,
                        parent_symbol=parent_symbol,
                        full_name=full_name,
                        source_text=self.parser._get_node_text(node),
                        metadata={
                            'is_include': True,
                            'is_system': is_system,
                            'include_path': include_path,
                            'include_type': 'system' if is_system else 'local'
                        }
                    )
                    self.symbols.append(symbol)
            
            def _extract_field_name(self, field_declarator_node):
                """Extract field name from field_declarator
                
                Handles:
                    - Simple: int x;  (field_identifier is direct child)
                    - Pointer: int* ptr;  (field_identifier is inside pointer_declarator)
                    - Reference: int& ref;  (field_identifier is inside reference_declarator)
                """
                # First try direct child
                field_id = self.parser._find_child_by_type(field_declarator_node, 'field_identifier')
                if field_id:
                    return self.parser._get_node_text(field_id)
                
                # Try inside pointer_declarator or reference_declarator
                for child in field_declarator_node.children:
                    if child.type in ['pointer_declarator', 'reference_declarator']:
                        field_id = self.parser._find_child_by_type(child, 'field_identifier')
                        if field_id:
                            return self.parser._get_node_text(field_id)
                
                return None
            
            def _extract_field_name_from_declarator(self, declarator_node):
                """Extract field name from pointer_declarator or reference_declarator
                
                These nodes can appear as direct children of field_declaration.
                Handles nested pointer_declarators for multi-level pointers (e.g., Point** pptr).
                """
                field_id = self.parser._find_child_by_type(declarator_node, 'field_identifier')
                if field_id:
                    return self.parser._get_node_text(field_id)
                # Recurse into nested pointer_declarator / reference_declarator
                for child in declarator_node.children:
                    if child.type in ['pointer_declarator', 'reference_declarator']:
                        result = self._extract_field_name_from_declarator(child)
                        if result:
                            return result
                return None
            
            def _extract_field_type_relationship(self, field_type, field_symbol, field_name, node):
                """Extract COMPOSITION/AGGREGATION relationship for field with user-defined type
                
                Args:
                    field_type: The type of the field (e.g., 'Point', 'Point*', 'vector<Point>')
                    field_symbol: The field's full qualified name (e.g., 'Container::obj')
                    field_name: The field's simple name (e.g., 'obj')
                    node: The AST node for source range
                """
                if not field_type:
                    return
                
                # Check if this is a user-defined type
                base_types = self._extract_user_defined_types_from_field(field_type)
                
                for base_type, is_pointer_or_ref in base_types:
                    # Determine relationship type
                    rel_type = RelationshipType.AGGREGATION if is_pointer_or_ref else RelationshipType.COMPOSITION
                    
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
            
            def _extract_function_type_relationships(self, function_symbol, return_type, parameter_types, node):
                """Extract USES relationships for function parameters and return type
                
                Args:
                    function_symbol: The function's full name (e.g., 'validateUser')
                    return_type: The return type string (e.g., 'User', 'std::shared_ptr<User>')
                    parameter_types: List of parameter type strings
                    node: The AST node for source range
                """
                # Extract user-defined types from return type
                if return_type:
                    return_types = self._extract_user_defined_types_from_field(return_type)
                    for base_type, _ in return_types:
                        relationship = Relationship(
                            source_symbol=function_symbol,
                            target_symbol=base_type,
                            relationship_type=RelationshipType.REFERENCES,  # Function REFERENCES the return type
                            source_file=file_path,
                            source_range=self.parser._node_to_range(node),
                            annotations={
                                'usage': 'return_type',
                                'return_type': return_type
                            }
                        )
                        self.relationships.append(relationship)
                
                # Extract user-defined types from parameters
                for param_type in parameter_types:
                    if param_type:
                        param_types_extracted = self._extract_user_defined_types_from_field(param_type)
                        for base_type, _ in param_types_extracted:
                            relationship = Relationship(
                                source_symbol=function_symbol,
                                target_symbol=base_type,
                                relationship_type=RelationshipType.REFERENCES,  # Function REFERENCES the parameter type
                                source_file=file_path,
                                source_range=self.parser._node_to_range(node),
                                annotations={
                                    'usage': 'parameter_type',
                                    'parameter_type': param_type
                                }
                            )
                            self.relationships.append(relationship)
            
            def _extract_user_defined_types_from_field(self, field_type, _depth=0):
                """Extract user-defined types from field type string
                
                Returns list of (type_name, is_pointer_or_ref) tuples.
                Recursively decomposes nested template arguments with
                bracket-aware splitting.  Template parameters from the
                current ``self.current_template_params`` are filtered out.
                
                Examples:
                    'Point' -> [('Point', False)]
                    'Point*' -> [('Point', True)]
                    'Point&' -> [('Point', True)]
                    'vector<Point>' -> [('vector', False), ('Point', False)]
                    'map<string, Point>' -> [('map', False), ('Point', False)]
                    'map<string, pair<int, MyType>>' -> [('map', False), ('pair', False), ('MyType', True)]
                """
                # Guard against pathological recursion
                if _depth > 10:
                    return []
                
                PRIMITIVE_TYPES = {
                    'int', 'float', 'double', 'char', 'bool', 'void',
                    'short', 'long', 'unsigned', 'signed',
                    'unsigned char', 'unsigned int', 'unsigned short', 'unsigned long',
                    'unsigned long long', 'signed char', 'signed int',
                    'long long', 'long int', 'long double', 'short int',
                    'int8_t', 'int16_t', 'int32_t', 'int64_t',
                    'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
                    'size_t', 'ptrdiff_t', 'ssize_t',
                    'string', 'wstring',  # std::string treated as standard
                    'auto', 'decltype',
                    'char16_t', 'char32_t', 'wchar_t',
                }
                
                # Collect current template parameters to filter (e.g., T, U)
                template_params = set(getattr(self, 'current_template_params', None) or [])
                
                results = []
                
                # Check if pointer or reference
                is_pointer_or_ref = '*' in field_type or '&' in field_type
                
                # Extract base type (remove const, volatile, *, &, spaces)
                base_type = field_type.replace('const', '').replace('volatile', '').replace('*', '').replace('&', '').strip()
                
                # Strip C-style elaborated type specifiers: "enum Color" -> "Color",
                # "struct Foo" -> "Foo", "class Bar" -> "Bar", "union Baz" -> "Baz"
                for prefix in ('enum ', 'struct ', 'class ', 'union '):
                    if base_type.startswith(prefix):
                        base_type = base_type[len(prefix):].strip()
                        break
                
                # Remove namespace qualifiers for primitive check (keep full name for UDT)
                # e.g., "std::vector<Point>" — the namespace-stripped base is "vector<Point>"
                base_for_check = base_type.split('::')[-1] if '::' in base_type and '<' not in base_type else base_type
                
                # Handle template types: extract base and arguments
                if '<' in base_type and '>' in base_type:
                    # Extract base template name
                    template_base = base_type[:base_type.index('<')].strip()
                    if template_base and template_base not in PRIMITIVE_TYPES and template_base not in template_params:
                        results.append((template_base, False))  # Container itself
                    
                    # Extract template arguments
                    start = base_type.index('<')
                    end = base_type.rindex('>')
                    args_str = base_type[start+1:end]
                    
                    # Use bracket-aware splitting (handles nested templates)
                    args = self._split_template_args(args_str)
                    for arg in args:
                        # Clean argument
                        clean_arg = arg.replace('const', '').replace('*', '').replace('&', '').strip()
                        if not clean_arg or clean_arg in PRIMITIVE_TYPES or clean_arg in template_params:
                            continue
                        # Recurse into nested templates
                        if '<' in clean_arg and '>' in clean_arg:
                            nested = self._extract_user_defined_types_from_field(clean_arg, _depth + 1)
                            results.extend(nested)
                        else:
                            # Simple type argument — aggregation relationship
                            results.append((clean_arg, True))
                else:
                    # Simple type (no templates)
                    if base_type and base_type not in PRIMITIVE_TYPES and base_type not in template_params:
                        results.append((base_type, is_pointer_or_ref))
                
                return results
            
            @staticmethod
            def _split_template_args(args_str):
                """Split template arguments by comma, respecting nested angle brackets.
                
                Standard ``str.split(',')`` fails on nested templates like
                ``pair<int, MyType>`` (splits into ``pair<int`` and ``MyType>``).
                This method tracks ``<>`` nesting depth so commas inside inner
                templates are not treated as delimiters.
                
                Args:
                    args_str: The string between the outermost ``<`` and ``>``.
                
                Returns:
                    List of stripped argument strings.
                """
                result = []
                depth = 0
                current = []
                for ch in args_str:
                    if ch == '<':
                        depth += 1
                        current.append(ch)
                    elif ch == '>':
                        depth -= 1
                        current.append(ch)
                    elif ch == ',' and depth == 0:
                        result.append(''.join(current).strip())
                        current = []
                    else:
                        current.append(ch)
                # Last segment
                tail = ''.join(current).strip()
                if tail:
                    result.append(tail)
                return result
            
            def _extract_template_param_name(self, param_node):
                """Extract template parameter name"""
                for child in param_node.children:
                    if child.type in ['type_identifier', 'identifier']:
                        return self.parser._get_node_text(child)
                return None
        
        extractor = SymbolExtractor(self)
        extractor.visit(ast_node)
        
        # Store field-level relationships for later merging
        self._field_relationships = extractor.relationships
        
        return extractor.symbols
    
    class RelationshipExtractor:
        """Extracts relationships from C++ AST using visitor pattern"""
        
        def __init__(self, parser, symbols, symbol_by_name, symbol_by_full_name, file_path):
            self.parser = parser
            self.symbols = symbols
            self.symbol_by_name = symbol_by_name
            self.symbol_by_full_name = symbol_by_full_name
            self.file_path = file_path
            self.relationships = []
            self.scope_stack = []  # Track current scope (class names)
            self.current_symbol = None  # Track current function/method being analyzed
            self.current_template_params = []  # Track template parameters
            
            # Keywords to exclude from REFERENCES
            self.cpp_keywords = {
                'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
                'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
                'inline', 'int', 'long', 'register', 'return', 'short', 'signed',
                'sizeof', 'static', 'struct', 'switch', 'typedef', 'union', 'unsigned',
                'void', 'volatile', 'while', 'class', 'namespace', 'template', 'typename',
                'this', 'public', 'private', 'protected', 'virtual', 'const', 'constexpr',
                'nullptr', 'true', 'false', 'operator', 'new', 'delete', 'throw', 'try',
                'catch', 'bool', 'explicit', 'friend', 'mutable', 'using', 'static_cast',
                'dynamic_cast', 'reinterpret_cast', 'const_cast'
            }
        
        def _find_symbol_by_signature(self, full_name, name, parameter_types, is_const=False):
            """Find symbol matching both name, parameter signature, and const qualifier (for overloads)"""
            # Get candidate symbols - returns list or None
            candidates = self.symbol_by_full_name.get(full_name) or self.symbol_by_name.get(name)
            
            if not candidates:
                return None
            
            # If only one candidate, return it (common case - no overloading)
            if len(candidates) == 1:
                return candidates[0]
            
            # Multiple candidates - match by parameter types AND const qualifier
            for candidate in candidates:
                param_match = (hasattr(candidate, 'parameter_types') and 
                             candidate.parameter_types == parameter_types)
                const_match = (candidate.metadata and 
                             candidate.metadata.get('is_const_method', False) == is_const)
                
                if param_match and const_match:
                    return candidate
            
            # Fallback: Try matching by parameter types only (for non-const methods)
            for candidate in candidates:
                if hasattr(candidate, 'parameter_types') and candidate.parameter_types == parameter_types:
                    # Check if this is a non-const method when we're looking for non-const
                    if not is_const and not (candidate.metadata and candidate.metadata.get('is_const_method')):
                        return candidate
            
            # If no exact match, return first candidate as fallback
            # This handles cases where parameter_types extraction might differ slightly
            return candidates[0]
        
        def _get_single_symbol(self, full_name, name):
            """Get single symbol by name (for non-overloaded cases like classes, fields)"""
            candidates = self.symbol_by_full_name.get(full_name) or self.symbol_by_name.get(name)
            return candidates[0] if candidates else None
        
        def _extract_parameter_types_from_declarator(self, declarator):
            """Extract parameter types from function declarator for signature matching"""
            parameter_types = []
            
            # Find parameter_list node in declarator
            param_list = self.parser._find_child_by_type(declarator, 'parameter_list')
            if not param_list:
                return parameter_types
            
            # Extract each parameter declaration
            for child in param_list.children:
                if child.type == 'parameter_declaration':
                    # Extract type from parameter declaration
                    param_type = self._extract_type_from_param(child)
                    if param_type:
                        parameter_types.append(param_type)
            
            return parameter_types
        
        def _extract_type_from_param(self, param_node):
            """Extract full type string from parameter_declaration node including const, &, &&, *"""
            param_type = None
            is_const_type = False  # const before type: const int
            pointer_qualifier = ''  # '', '*', '**', etc.
            const_after_pointer = False  # const after *: int* const
            ref_qualifier = ''  # '', '&', or '&&'
            
            for child in param_node.children:
                if child.type == 'type_qualifier':
                    # Const qualifier before type: const int
                    if 'const' in self.parser._get_node_text(child):
                        is_const_type = True
                elif child.type in ['primitive_type', 'type_identifier',
                                    'qualified_identifier', 'template_type',
                                    'sized_type_specifier',
                                    'placeholder_type_specifier']:
                    param_type = self.parser._get_node_text(child)
                elif child.type == 'pointer_declarator':
                    # Extract full pointer text to preserve const placement
                    full_text = self.parser._get_node_text(child)
                    
                    # Remove parameter name to get just the pointer qualifiers
                    # Strategy: extract everything except the last identifier
                    parts = []
                    
                    def extract_pointer_parts(node, collected_parts):
                        """Recursively extract pointer qualifiers"""
                        for subchild in node.children:
                            if subchild.type == '*':
                                collected_parts.append('*')
                            elif subchild.type == 'type_qualifier':
                                if 'const' in self.parser._get_node_text(subchild):
                                    collected_parts.append(' const')
                            elif subchild.type == 'pointer_declarator':
                                # Nested pointer - recurse
                                extract_pointer_parts(subchild, collected_parts)
                            # Skip identifiers (parameter names)
                    
                    extract_pointer_parts(child, parts)
                    pointer_qualifier = ''.join(parts)
                    
                elif child.type == 'reference_declarator':
                    # Reference declarator: int& r, int&& rr
                    ref_text = self.parser._get_node_text(child)
                    if '&&' in ref_text:
                        ref_qualifier = '&&'
                    elif '&' in ref_text:
                        ref_qualifier = '&'
                    # Also get base type if not already captured
                    if not param_type:
                        base_type = self.parser._find_child_by_type(child, 'type_identifier')
                        if base_type:
                            param_type = self.parser._get_node_text(base_type)
            
            if param_type:
                # Build full type signature: const int*, int* const, const int* const, etc.
                full_type = param_type
                
                # Add const before type
                if is_const_type:
                    full_type = f"const {full_type}"
                
                # Add pointer qualifier (includes const placement)
                if pointer_qualifier:
                    full_type = f"{full_type}{pointer_qualifier}"
                
                # Add reference qualifier
                if ref_qualifier:
                    full_type = f"{full_type}{ref_qualifier}"
                
                return full_type
            
            return None
        
        def _is_const_method(self, declarator_node):
            """Check if function_declarator has const qualifier (for const methods)"""
            if not declarator_node or declarator_node.type != 'function_declarator':
                return False
            
            # Check for type_qualifier child with 'const'
            for child in declarator_node.children:
                if child.type == 'type_qualifier':
                    if 'const' in self.parser._get_node_text(child):
                        return True
            return False
        
        def visit(self, node):
            """Main visitor dispatch"""
            method_name = f'visit_{node.type}'
            has_custom_visitor = hasattr(self, method_name)
            
            if has_custom_visitor:
                getattr(self, method_name)(node)
            
            # Check for field access (field_expression) - but skip if it's a method call
            # Method calls are handled in visit_call_expression
            if node.type == 'field_expression' and node.parent and node.parent.type != 'call_expression':
                self._extract_field_access(node)
            
            # Visit children (unless custom visitor handles them)
            # For nodes with scope management or special handling, children are visited within the visitor  
            # Don't recurse into: class/struct (scope managed), namespace (scope managed), 
            # declaration (fully handled), function_definition (body visited manually)
            if not has_custom_visitor or node.type not in ['class_specifier', 'struct_specifier', 'namespace_definition', 'declaration', 'function_definition', 'translation_unit', 'template_instantiation', 'template_declaration']:
                for child in node.children:
                    self.visit(child)
        
        def visit_translation_unit(self, node):
            """Root node - push module name to scope_stack for standardized qualified names.
            
            Matches the symbol extractor which also pushes module_name to scope_stack.
            This ensures class full_name matches parent_symbol of members.
            """
            module_name = Path(self.file_path).stem
            self.scope_stack.append(module_name)
            
            # Visit children
            for child in node.children:
                self.visit(child)
            
            # Pop module (we don't need to, but for consistency)
            self.scope_stack.pop()
        
        def visit_namespace_definition(self, node):
            """Track namespace scope for relationship context.
            
            Namespaces are organizational, not architectural - we don't create
            NAMESPACE→child DEFINES relationships. Parent-child containment is
            captured by parent_symbol metadata on each symbol.
            """
            name_node = self.parser._find_child_by_type(node, 'namespace_identifier')
            if name_node:
                name = self.parser._get_node_text(name_node)
                self.scope_stack.append(name)
                
                # Visit children (no DEFINES relationships from namespace)
                for child in node.children:
                    self.visit(child)
                
                self.scope_stack.pop()
        
        def visit_class_specifier(self, node):
            """Track class scope for relationship context"""
            name = self._extract_class_name(node)
            if name:
                self.scope_stack.append(name)
                
                # Extract INHERITANCE relationships
                self._extract_inheritance(node, name)
                
                # Visit children first to parse all members (methods, fields)
                for child in node.children:
                    self.visit(child)
                
                # Extract DEFINES relationships AFTER members are parsed
                # (class defines its member methods and fields)
                self._extract_defines_relationships(name)
                
                self.scope_stack.pop()
        
        def visit_struct_specifier(self, node):
            """Track struct scope (same as class)"""
            self.visit_class_specifier(node)
        
        def visit_field_declaration(self, node):
            """Extract REFERENCES from member variable type declarations.
            
            Handles:
            - type_identifier: Point member; → REFERENCES to Point
            - qualified_identifier: fmt::memory_buffer buf; → REFERENCES to fmt::memory_buffer
            - qualified_identifier with template: std::vector<Point> v; → REFERENCES to Point (template arg)
            """
            if not self.scope_stack:
                return  # Skip global variables
            
            parent_symbol_name = '::'.join(self.scope_stack)
            
            # Collect active template params to skip (e.g. T, K, V)
            template_params = set(self.current_template_params or [])
            
            # Extract the type and emit REFERENCES
            for child in node.children:
                if child.type == 'type_identifier':
                    # Plain user type: Point member;
                    type_name = self.parser._get_node_text(child)
                    if type_name and type_name not in self.cpp_keywords and type_name not in template_params:
                        class ParentSymbol:
                            def __init__(self, full_name):
                                self.full_name = full_name
                                self.name = full_name.split('::')[-1] if '::' in full_name else full_name
                        relationship = Relationship(
                            source_symbol=parent_symbol_name,
                            target_symbol=type_name,
                            relationship_type=RelationshipType.REFERENCES,
                            source_file=self.file_path,
                            source_range=self.parser._node_to_range(child),
                            annotations={'reference_type': 'field_type'}
                        )
                        self.relationships.append(relationship)
                    break
                elif child.type == 'qualified_identifier':
                    # Qualified type: ns::Type or std::vector<Point>
                    template_node = self.parser._find_child_by_type(child, 'template_type')
                    if template_node:
                        # Qualified template: std::vector<Point>
                        # Extract the qualified template name
                        parts = []
                        for subchild in child.children:
                            if subchild.type == 'namespace_identifier':
                                parts.append(self.parser._get_node_text(subchild))
                            elif subchild.type == 'template_type':
                                name_node = self.parser._find_child_by_type(subchild, 'type_identifier')
                                if name_node:
                                    parts.append(self.parser._get_node_text(name_node))
                        type_name = '::'.join(parts) if parts else None
                        
                        # Emit REFERENCES for the qualified template type itself (non-std)
                        std_types = {'std::string', 'std::wstring', 'std::vector', 'std::list', 'std::map',
                                     'std::set', 'std::unordered_map', 'std::unordered_set', 'std::shared_ptr',
                                     'std::unique_ptr', 'std::weak_ptr', 'std::optional', 'std::variant',
                                     'std::tuple', 'std::pair', 'std::function', 'std::array'}
                        if type_name and type_name not in std_types:
                            relationship = Relationship(
                                source_symbol=parent_symbol_name,
                                target_symbol=type_name,
                                relationship_type=RelationshipType.REFERENCES,
                                source_file=self.file_path,
                                source_range=self.parser._node_to_range(child),
                                annotations={'reference_type': 'field_type'}
                            )
                            self.relationships.append(relationship)
                        
                        # Extract template argument types as REFERENCES
                        class ParentSymbol:
                            def __init__(self, full_name):
                                self.full_name = full_name
                                self.name = full_name.split('::')[-1] if '::' in full_name else full_name
                        parent_sym = ParentSymbol(parent_symbol_name)
                        self._extract_template_argument_references(template_node, parent_sym)
                    else:
                        # Plain qualified type: fmt::memory_buffer buf;
                        parts = _collect_qualified_parts_from_node(child)
                        type_name = '::'.join(parts)
                        if type_name and type_name not in self.cpp_keywords:
                            relationship = Relationship(
                                source_symbol=parent_symbol_name,
                                target_symbol=type_name,
                                relationship_type=RelationshipType.REFERENCES,
                                source_file=self.file_path,
                                source_range=self.parser._node_to_range(child),
                                annotations={'reference_type': 'field_type'}
                            )
                            self.relationships.append(relationship)
                    break
                elif child.type == 'template_type':
                    # Unqualified template type: optional<Point>, vector<int>
                    class ParentSymbol:
                        def __init__(self, full_name):
                            self.full_name = full_name
                            self.name = full_name.split('::')[-1] if '::' in full_name else full_name
                    parent_sym = ParentSymbol(parent_symbol_name)
                    self._extract_template_argument_references(child, parent_sym)
                    break
            
            # Continue visiting children for nested declarations
            for child in node.children:
                self.visit(child)
        
        def visit_function_definition(self, node):
            """Track current function for CALLS and REFERENCES"""
            # Find function_declarator (may be nested in pointer_declarator, reference_declarator, etc.)
            declarator = self.parser._find_child_by_type(node, 'function_declarator')
            if not declarator:
                # Check if it's nested inside a pointer/reference declarator
                for child in node.children:
                    if child.type in ['pointer_declarator', 'reference_declarator']:
                        declarator = self.parser._find_child_by_type(child, 'function_declarator')
                        if declarator:
                            break
            
            if not declarator:
                return
            
            # Extract function name - check for qualified name first (e.g., Point::move)
            qualified_parts = self._extract_qualified_name_parts(declarator)
            if qualified_parts and len(qualified_parts) > 1:
                # Qualified name: Point::move
                name = qualified_parts[-1]
                full_name = '::'.join(qualified_parts)
            else:
                # Simple name or inside scope
                name = self._extract_function_name(declarator)
                if not name:
                    return
                
                # Build full name with scope
                parent_symbol = '::'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}::{name}" if parent_symbol else name
            
            # Extract parameter types for signature matching (handles overloads)
            parameter_types = self._extract_parameter_types_from_declarator(declarator)
            
            # Check if this is a const method
            is_const = self._is_const_method(declarator)
            
            # Find the symbol for this function - match by signature and const qualifier for overloads
            symbol = self._find_symbol_by_signature(full_name, name, parameter_types, is_const)
            
            # Set current symbol and function node for relationship tracking
            old_symbol = self.current_symbol
            old_function_node = getattr(self, 'current_function_node', None)
            self.current_symbol = symbol
            self.current_function_node = node  # Track for local variable resolution
            
            # Extract DEFINES_BODY relationship if this is an implementation
            if symbol:
                self._extract_defines_body_relationship(node, symbol, declarator)
            
            # Extract REFERENCES from parameter types
            if symbol and declarator:
                param_list = self.parser._find_child_by_type(declarator, 'parameter_list')
                if param_list:
                    self._extract_parameter_references(param_list, symbol)
            
            # Extract REFERENCES from return type
            if symbol:
                self._extract_return_type_references(node, symbol)
            
            # Visit field initializer list (constructor initializers)
            field_init_list = self.parser._find_child_by_type(node, 'field_initializer_list')
            if field_init_list:
                self.visit(field_init_list)
            
            # Visit function body to extract CALLS and REFERENCES
            body = self.parser._find_child_by_type(node, 'compound_statement')
            if body:
                for child in body.children:
                    self.visit(child)
            
            # Restore previous symbol and function node
            self.current_symbol = old_symbol
            self.current_function_node = old_function_node
        
        def visit_call_expression(self, node):
            """Extract CALLS relationships from function calls and CREATES for smart pointer construction"""
            if not self.current_symbol:
                return
            
            # Extract function being called
            function_node = node.children[0] if node.children else None
            if not function_node:
                return
            
            called_name = None
            is_template_function = False
            
            annotations = {}
            
            if function_node.type == 'identifier':
                # Simple function call: foo()
                called_name = self.parser._get_node_text(function_node)
            elif function_node.type == 'field_expression':
                # Method call: obj.method() or ptr->method()
                field = self.parser._find_child_by_type(function_node, 'field_identifier')
                if field:
                    method_name = self.parser._get_node_text(field)
                    # Resolve to qualified name (Class::method) if possible
                    called_name, annotations = self._resolve_method_call_target(function_node, method_name)
            elif function_node.type == 'qualified_identifier':
                # Qualified call: Class::method() or std::make_unique<T>()
                # Use recursive helper for deeply nested qualifiers (outer::inner::func)
                template_func_node = None
                # Check for template_function child first
                for child in function_node.children:
                    if child.type == 'template_function':
                        template_func_node = child
                        is_template_function = True
                        break
                
                if template_func_node:
                    # Collect namespace parts before the template_function
                    parts = []
                    for child in function_node.children:
                        if child.type in ('namespace_identifier', 'identifier', 'type_identifier'):
                            parts.append(self.parser._get_node_text(child))
                        elif child.type == 'qualified_identifier':
                            parts.extend(p for p in _collect_qualified_parts_from_node(child)
                                        if child is not function_node)
                        elif child.type == 'template_function':
                            name_node = self.parser._find_child_by_type(child, 'identifier')
                            if name_node:
                                parts.append(self.parser._get_node_text(name_node))
                    called_name = '::'.join(parts)
                else:
                    parts = _collect_qualified_parts_from_node(function_node)
                    called_name = '::'.join(parts)
                
                # Check for make_unique/make_shared in qualified template functions
                if template_func_node and called_name and ('make_unique' in called_name or 'make_shared' in called_name):
                    self._extract_smart_pointer_creation(template_func_node, called_name, node)
                    return  # Don't create a regular CALLS relationship
            elif function_node.type == 'template_function':
                # Template function call: foo<T>() or std::make_unique<Type>()
                is_template_function = True
                name_node = self.parser._find_child_by_type(function_node, 'identifier')
                if name_node:
                    called_name = self.parser._get_node_text(name_node)
                elif self.parser._find_child_by_type(function_node, 'qualified_identifier'):
                    # Qualified template function: std::make_unique<Type>()
                    qual_id = self.parser._find_child_by_type(function_node, 'qualified_identifier')
                    parts = _collect_qualified_parts_from_node(qual_id)
                    called_name = '::'.join(parts)
                
                # Check for make_unique/make_shared and extract CREATES relationship
                if called_name and ('make_unique' in called_name or 'make_shared' in called_name):
                    self._extract_smart_pointer_creation(function_node, called_name, node)
            
            if called_name and called_name not in self.cpp_keywords:
                # Check if this is a constructor call (looks like a type name)
                # Constructor calls start with uppercase and might be in symbol table as CLASS
                is_constructor = False
                target_symbol = self._get_single_symbol(called_name, called_name)
                
                if target_symbol and target_symbol.symbol_type.value in ['class', 'struct']:
                    # This is a constructor call
                    is_constructor = True
                elif '::' not in called_name and called_name[0].isupper():
                    # Heuristic: unqualified names starting with uppercase are likely constructors
                    # But qualified names like Class::method are method calls
                    is_constructor = True
                
                if is_constructor:
                    # This is object construction, not a regular function call
                    relationship = Relationship(
                        source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                        target_symbol=called_name,
                        relationship_type=RelationshipType.CREATES,
                        source_file=self.file_path,
                        source_range=self.parser._node_to_range(node),
                        annotations={
                            'creation_type': 'stack',
                            'creation_context': 'statement'
                        }
                    )
                else:
                    # Regular function call (or method call)
                    relationship = Relationship(
                        source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                        target_symbol=called_name,
                        relationship_type=RelationshipType.CALLS,
                        source_file=self.file_path,
                        source_range=self.parser._node_to_range(node),
                        annotations=annotations if annotations else None
                    )
                self.relationships.append(relationship)
        
        def visit_declaration(self, node):
            """Extract REFERENCES from variable declarations and CREATES for object construction"""
            if not self.current_symbol:
                return
            
            # First, check if this is a pointer declaration - we need to know this early
            # to decide whether to create INSTANTIATES for templates
            is_pointer_declaration = any(child.type == 'pointer_declarator' for child in node.children)
            
            # Extract the type being declared
            type_name = None
            is_template_instantiation = False
            template_args = []
            template_type_node = None
            
            for child in node.children:
                if child.type == 'type_identifier':
                    type_name = self.parser._get_node_text(child)
                    if type_name and type_name not in self.cpp_keywords:
                        relationship = Relationship(
                            source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                            target_symbol=type_name,
                            relationship_type=RelationshipType.REFERENCES,
                            source_file=self.file_path,
                            source_range=self.parser._node_to_range(child)
                        )
                        self.relationships.append(relationship)
                elif child.type == 'qualified_identifier':
                    # Handle qualified types like std::string OR std::vector<Point>
                    # Check if it contains a template_type
                    template_node = self.parser._find_child_by_type(child, 'template_type')
                    if template_node:
                        # This is a qualified template type like std::vector<Point>
                        is_template_instantiation = True
                        template_type_node = template_node
                        
                        # Extract the qualified template name (e.g., "std::vector")
                        parts = []
                        for subchild in child.children:
                            if subchild.type in ['namespace_identifier']:
                                parts.append(self.parser._get_node_text(subchild))
                            elif subchild.type == 'template_type':
                                # Get template name
                                name_node = self.parser._find_child_by_type(subchild, 'type_identifier')
                                if name_node:
                                    parts.append(self.parser._get_node_text(name_node))
                        type_name = '::'.join(parts) if parts else None
                        
                        if type_name:
                            # Create REFERENCES for the template class
                            relationship = Relationship(
                                source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                                target_symbol=type_name,
                                relationship_type=RelationshipType.REFERENCES,
                                source_file=self.file_path,
                                source_range=self.parser._node_to_range(child),
                                annotations={'reference_type': 'template_instantiation'}
                            )
                            self.relationships.append(relationship)
                            
                            # Extract template argument types
                            self._extract_template_argument_references(template_node, self.current_symbol)
                    else:
                        # Regular qualified type like std::string or ns::Point
                        parts = _collect_qualified_parts_from_node(child)
                        type_name = '::'.join(parts)
                        if type_name:
                            relationship = Relationship(
                                source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                                target_symbol=type_name,
                                relationship_type=RelationshipType.REFERENCES,
                                source_file=self.file_path,
                                source_range=self.parser._node_to_range(child)
                            )
                            self.relationships.append(relationship)
                elif child.type == 'template_type':
                    # Handle template types like Vector<int>, map<string, Point>
                    is_template_instantiation = True
                    template_type_node = child
                    
                    # Extract template name
                    name_node = self.parser._find_child_by_type(child, 'type_identifier')
                    if name_node:
                        type_name = self.parser._get_node_text(name_node)
                        
                        # Create REFERENCES for the template class
                        relationship = Relationship(
                            source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                            target_symbol=type_name,
                            relationship_type=RelationshipType.REFERENCES,
                            source_file=self.file_path,
                            source_range=self.parser._node_to_range(child),
                            annotations={'reference_type': 'template_instantiation'}
                        )
                        self.relationships.append(relationship)
                        
                        # Only create INSTANTIATES if this is NOT a pointer declaration
                        # Vector<int> v; → INSTANTIATES
                        # Vector<int>* v; → NO INSTANTIATES (just a pointer type reference)
                        if not is_pointer_declaration:
                            # Extract concrete type arguments for the annotation
                            inst_type_args = []
                            inst_arg_list = self.parser._find_child_by_type(child, 'template_argument_list')
                            if inst_arg_list:
                                for inst_ac in inst_arg_list.children:
                                    if inst_ac.type in ('type_identifier', 'qualified_identifier'):
                                        inst_type_args.append(self.parser._get_node_text(inst_ac))
                                    elif inst_ac.type == 'type_descriptor':
                                        prim = self.parser._find_child_by_type(inst_ac, 'primitive_type')
                                        tid = self.parser._find_child_by_type(inst_ac, 'type_identifier')
                                        if prim:
                                            inst_type_args.append(self.parser._get_node_text(prim))
                                        elif tid:
                                            inst_type_args.append(self.parser._get_node_text(tid))
                                        else:
                                            inst_type_args.append(self.parser._get_node_text(inst_ac))
                            relationship = Relationship(
                                source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                                target_symbol=type_name,
                                relationship_type=RelationshipType.INSTANTIATES,
                                source_file=self.file_path,
                                source_range=self.parser._node_to_range(child),
                                annotations={'type_args': inst_type_args} if inst_type_args else None
                            )
                            self.relationships.append(relationship)
                        
                        # Extract template argument types using the new helper
                        self._extract_template_argument_references(child, self.current_symbol)
                    
                    # Extract template arguments for metadata (backward compatibility)
                    arg_list = self.parser._find_child_by_type(child, 'template_argument_list')
                    if arg_list:
                        for arg_child in arg_list.children:
                            if arg_child.type == 'type_descriptor':
                                # Extract type from type_descriptor
                                type_node = self.parser._find_child_by_type(arg_child, 'type_identifier')
                                if type_node:
                                    arg_type = self.parser._get_node_text(type_node)
                                    template_args.append(arg_type)
            
            # Check for stack-allocated object creation
            # Three cases:
            # 1. init_declarator: Rectangle rect(args), Rectangle* ptr = new Rectangle()
            # 2. bare identifier: Rectangle rect; (simple default construction)
            # 3. pointer_declarator: Rectangle* ptr; (no object created)
            # 4. template instantiation: Vector<int> v; or Vector<int> v(10);
            if type_name:
                has_declarator = False
                for child in node.children:
                    if child.type == 'init_declarator':
                        self._extract_stack_creation(child, type_name, is_template_instantiation)
                        has_declarator = True
                    elif child.type == 'pointer_declarator':
                        # Pointer declaration without initialization: Rectangle* ptr;
                        # No object created, just REFERENCES already handled
                        has_declarator = True
                
                # If no init_declarator or pointer_declarator, check for bare identifier
                # This is pure default construction: Rectangle rect; or Vector<int> v;
                if not has_declarator:
                    for child in node.children:
                        if child.type == 'identifier':
                            # This is default construction: Rectangle rect; or Vector<int> v;
                            # Create CREATES relationship directly (no pointer/reference to check)
                            if self.current_symbol:
                                annotations = {
                                    'creation_type': 'stack',
                                    'construction': 'default',
                                    'creation_context': 'statement'
                                }
                                if is_template_instantiation:
                                    annotations['template_instantiation'] = True
                                    if template_args:
                                        annotations['template_args'] = template_args
                                
                                relationship = Relationship(
                                    source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                                    target_symbol=type_name,
                                    relationship_type=RelationshipType.CREATES,
                                    source_file=self.file_path,
                                    source_range=self.parser._node_to_range(child),
                                    annotations=annotations
                                )
                                self.relationships.append(relationship)
                            break
            
            # Manually visit children since declaration is in the no-recurse list
            for child in node.children:
                self.visit(child)
        
        def visit_identifier(self, node):
            """Extract REFERENCES relationships from identifier usage"""
            if not self.current_symbol:
                return
            
            name = self.parser._get_node_text(node)
            
            # Skip keywords and common patterns
            if name in self.cpp_keywords:
                return
            
            # Skip if this is the current function name itself
            if name == self.current_symbol.name:
                return
            
            # Check if this references a known symbol
            # Look for variables, classes, etc.
            target_symbol = self._get_single_symbol(name, name)
            
            # Only create reference if we found a symbol or it looks like a type/variable
            if target_symbol or (name and name[0].isupper()):
                relationship = Relationship(
                    source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                    target_symbol=name,
                    relationship_type=RelationshipType.REFERENCES,
                    source_file=self.file_path,
                    source_range=self.parser._node_to_range(node)
                )
                self.relationships.append(relationship)
        
        def visit_template_declaration(self, node):
            """Extract template information and SPECIALIZES relationships.
            
            Handles ``template<typename T> class Foo { … };`` by extracting the
            template parameter list and delegating to the wrapped declaration
            visitor (class_specifier, struct_specifier, or function_definition).
            
            NOTE: This method does NOT handle:
            - Explicit instantiations (``template class Foo<int>;``) — tree-sitter
              node type ``template_instantiation``, no visitor exists.
            - Partial / full specializations — the wrapped declaration is visited
              normally; we do not check whether the parameter list is narrower than
              the primary template's.
            See visit_template_type() docstring for full gap description.
            """
            # Extract template parameters
            params = []
            param_list = self.parser._find_child_by_type(node, 'template_parameter_list')
            if param_list:
                for child in param_list.children:
                    if child.type in ['type_parameter_declaration', 'parameter_declaration']:
                        param_name = self._extract_template_param_name(child)
                        if param_name:
                            params.append(param_name)
            
            # Store template params for use by wrapped declaration
            old_params = self.current_template_params
            self.current_template_params = params
            
            # Visit all significant children with current_template_params set
            # so that visit_field_declaration and visit_template_type can filter
            # template parameters (T, K, V, etc.) from emitted relationships.
            for child in node.children:
                if child.type not in ('template', 'template_parameter_list'):
                    self.visit(child)
            
            # Restore previous template params
            self.current_template_params = old_params
        
        def visit_template_type(self, node):
            """Extract SPECIALIZES relationships for template specializations.
            
            Known gaps and caveats (see PLANNING_FTS_AND_IMPROVEMENTS_V2.md §8):
            
            1. **Over-broad firing**: SPECIALIZES is emitted for ANY template_type
               node found inside a class scope, not only for base-class specializations.
               For example, ``std::vector<int> member_;`` inside a class body will emit
               SPECIALIZES(MyClass → vector) even though MyClass does not *specialise*
               vector — it merely *uses* it.  A more precise approach would limit
               SPECIALIZES to template_type nodes that appear as direct children of
               a base_class_clause (inheritance context) and emit REFERENCES for the
               rest.
            
            2. **Explicit-instantiation handler**: C++ allows
               ``template class Foo<int>;`` (explicit instantiation).  tree-sitter
               parses this as ``template_instantiation`` and ``visit_template_instantiation``
               emits an INSTANTIATES edge with concrete type arguments in annotations.
            
            3. **No partial-specialization detection**: For
               ``template<> class Foo<int> { … };`` (full specialization) or
               ``template<class T> class Foo<T*> { … };`` (partial specialization)
               we do not distinguish between a *primary* template definition and its
               specializations.  Detecting this requires checking whether the
               template_parameter_list is empty or narrower than the primary, and
               whether the class name carries a template_argument_list.
            
            4. **Template-alias propagation untested**: ``using Vec = std::vector;``
               followed by ``Vec<int> v;`` — whether the SPECIALIZES edge targets
               ``Vec`` (alias) or ``vector`` (canonical) depends on how alias_of
               chains interact with this visitor.  No test coverage exists for this
               path.
            """
            # Get template name
            name_node = self.parser._find_child_by_type(node, 'type_identifier')
            if not name_node:
                return
            
            template_name = self.parser._get_node_text(name_node)
            
            # Extract template arguments
            arg_list = self.parser._find_child_by_type(node, 'template_argument_list')
            if not arg_list:
                return
            
            # Determine if this template_type appears in a base_class_clause
            # (inheritance context).  Only emit SPECIALIZES for inheritance;
            # emit REFERENCES for all other usages (fields, locals, etc.).
            in_base_clause = False
            parent = node.parent
            while parent:
                if parent.type == 'base_class_clause':
                    in_base_clause = True
                    break
                if parent.type in ('class_specifier', 'struct_specifier',
                                   'function_definition', 'translation_unit'):
                    break
                parent = parent.parent
            
            if in_base_clause and self.scope_stack:
                current_class = self.scope_stack[-1]
                relationship = Relationship(
                    source_symbol=current_class,
                    target_symbol=template_name,
                    relationship_type=RelationshipType.SPECIALIZES,
                    source_file=self.file_path,
                    source_range=self.parser._node_to_range(node)
                )
                self.relationships.append(relationship)
            elif self.current_symbol:
                relationship = Relationship(
                    source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                    target_symbol=template_name,
                    relationship_type=RelationshipType.REFERENCES,
                    source_file=self.file_path,
                    source_range=self.parser._node_to_range(node)
                )
                self.relationships.append(relationship)
            
            # Create REFERENCES for each template argument that's a type
            template_params = set(self.current_template_params or [])
            for child in arg_list.children:
                if child.type in ['type_identifier', 'qualified_identifier']:
                    arg_type = self.parser._get_node_text(child)
                    if self.current_symbol and arg_type not in self.cpp_keywords and arg_type not in template_params:
                        relationship = Relationship(
                            source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                            target_symbol=arg_type,
                            relationship_type=RelationshipType.REFERENCES,
                            source_file=self.file_path,
                            source_range=self.parser._node_to_range(child)
                        )
                        self.relationships.append(relationship)
        
        def visit_template_instantiation(self, node):
            """Extract INSTANTIATES for explicit template instantiations.
            
            Handles ``template class Foo<int>;`` by finding the template name
            and emitting an INSTANTIATES edge from the current TU/scope to the
            template.  Concrete type arguments are stored in annotations.
            
            tree-sitter AST for ``template class Foo<int>;``::
            
                template_instantiation
                  template
                  class_specifier
                    class
                    template_type       <-- target node is nested
                      type_identifier: Foo
                      template_argument_list: <int>
            """
            template_name = None
            type_args = []
            
            def _find_template_type(n):
                """Search children recursively for template_type."""
                nonlocal template_name, type_args
                for child in n.children:
                    if child.type == 'template_type':
                        name_node = self.parser._find_child_by_type(child, 'type_identifier')
                        if name_node:
                            template_name = self.parser._get_node_text(name_node)
                        arg_list = self.parser._find_child_by_type(child, 'template_argument_list')
                        if arg_list:
                            for ac in arg_list.children:
                                if ac.type in ('type_identifier', 'qualified_identifier'):
                                    type_args.append(self.parser._get_node_text(ac))
                                elif ac.type == 'type_descriptor':
                                    # type_descriptor wraps primitive_type and others
                                    prim = self.parser._find_child_by_type(ac, 'primitive_type')
                                    if prim:
                                        type_args.append(self.parser._get_node_text(prim))
                                    else:
                                        type_args.append(self.parser._get_node_text(ac))
                        return  # Found it
                    elif child.type in ('class_specifier', 'struct_specifier'):
                        _find_template_type(child)
                        if template_name:
                            return
                    elif child.type in ('type_identifier', 'qualified_identifier') and not template_name:
                        template_name = self.parser._get_node_text(child)
            
            _find_template_type(node)
            
            if not template_name:
                return
            
            # Determine source: current function/class if inside one,
            # otherwise the file-level module name (stem) so the graph builder
            # can resolve it via its Strategy 2 (file_name → __file__ node).
            # Using the raw file_path here would create an unresolvable source.
            if self.current_symbol:
                source = self.current_symbol.full_name or self.current_symbol.name
            else:
                source = Path(self.file_path).stem
            
            relationship = Relationship(
                source_symbol=source,
                target_symbol=template_name,
                relationship_type=RelationshipType.INSTANTIATES,
                source_file=self.file_path,
                source_range=self.parser._node_to_range(node),
                annotations={
                    'type_args': type_args,
                } if type_args else None
            )
            self.relationships.append(relationship)
        
        def visit_preproc_def(self, node):
            """Track macro definitions (simple #define)"""
            # Macros are already extracted as symbols in SymbolExtractor
            # Here we could track their usage, but that's complex with preprocessing
            pass
        
        def visit_preproc_function_def(self, node):
            """Track function-like macro definitions"""
            # Similar to preproc_def, already handled as symbols
            pass
        
        def visit_preproc_include(self, node):
            """Create IMPORTS relationships for #include directives
            
            Creates relationships from the current file to included files.
            Tracks both system (<...>) and local ("...") includes.
            """
            # Extract include path
            include_path = None
            is_system = False
            
            # Check for system include: <...>
            system_lib = self.parser._find_child_by_type(node, 'system_lib_string')
            if system_lib:
                include_path = self.parser._get_node_text(system_lib).strip('<>')
                is_system = True
            else:
                # Check for local include: "..."
                string_lit = self.parser._find_child_by_type(node, 'string_literal')
                if string_lit:
                    string_content = self.parser._find_child_by_type(string_lit, 'string_content')
                    if string_content:
                        include_path = self.parser._get_node_text(string_content)
                        is_system = False
            
            if include_path:
                # Get current file as source
                # We'll use the file path itself as the source symbol
                source_file = self.file_path
                
                # Create target symbol name (matches what SymbolExtractor created)
                target_symbol = f"include::{include_path}"
                
                # If we have a current symbol context, use it; otherwise use file
                source_symbol = source_file
                if self.current_symbol:
                    # Use the first namespace or class in the file as source
                    source_symbol = self.current_symbol.full_name or self.current_symbol.name
                
                # Create IMPORTS relationship
                relationship = Relationship(
                    source_symbol=source_symbol,
                    target_symbol=target_symbol,
                    relationship_type=RelationshipType.IMPORTS,
                    source_file=self.file_path,
                    source_range=self.parser._node_to_range(node),
                    annotations={
                        'is_system': is_system,
                        'include_path': include_path,
                        'include_type': 'system' if is_system else 'local'
                    }
                )
                self.relationships.append(relationship)
        
        def _extract_inheritance(self, node, class_name):
            """Extract INHERITANCE relationships from base class clause"""
            base_clause = self.parser._find_child_by_type(node, 'base_class_clause')
            if not base_clause:
                return
            
            for child in base_clause.children:
                base_name = None
                
                if child.type in ['type_identifier', 'identifier']:
                    base_name = self.parser._get_node_text(child)
                elif child.type == 'qualified_identifier':
                    # Extract full qualified name
                    parts = []
                    for subchild in child.children:
                        if subchild.type in ['identifier', 'type_identifier']:
                            parts.append(self.parser._get_node_text(subchild))
                    base_name = '::'.join(parts)
                elif child.type == 'template_type':
                    # Template base class
                    name_node = self.parser._find_child_by_type(child, 'type_identifier')
                    if name_node:
                        base_name = self.parser._get_node_text(name_node)
                
                if base_name:
                    # Build full qualified name for source using scope_stack
                    # scope_stack contains full namespace/class hierarchy including current class
                    source_full_name = '::'.join(self.scope_stack) if self.scope_stack else class_name
                    
                    relationship = Relationship(
                        source_symbol=source_full_name,
                        target_symbol=base_name,
                        relationship_type=RelationshipType.INHERITANCE,
                        source_file=self.file_path,
                        source_range=self.parser._node_to_range(child)
                    )
                    self.relationships.append(relationship)
        
        def _extract_defines_relationships(self, class_name):
            """Extract DEFINES relationships from class to all its members.
            
            Standardized pattern (matches TypeScript, Java, Python):
            - Class/struct to fields
            - Class/struct to methods
            - Class/struct to constructors/destructors
            
            This enables the content expander to find composition:
            Class --DEFINES--> Field --COMPOSITION--> Type
            
            Note: Uses '::' separator for symbol matching because this runs
            BEFORE _normalize_parse_result converts to '.' separator.
            """
            # Build full qualified name for the class (using :: - matches symbol_by_full_name)
            class_full_name = '::'.join(self.scope_stack) if self.scope_stack else class_name
            
            # Find all members that belong to this class
            # symbol_by_full_name.values() returns lists of symbols (for overloads)
            for symbol_list in self.symbol_by_full_name.values():
                for symbol in symbol_list:
                    # Check if this symbol is a member of this class
                    if symbol.symbol_type in [SymbolType.METHOD, SymbolType.CONSTRUCTOR, SymbolType.FIELD]:
                        # Check if parent_symbol matches this class (using :: separator)
                        if symbol.parent_symbol == class_full_name:
                            # This member belongs to this class
                            relationship = Relationship(
                                source_symbol=class_full_name,
                                target_symbol=symbol.full_name or symbol.name,
                                relationship_type=RelationshipType.DEFINES,
                                source_file=self.file_path,
                                source_range=symbol.range,
                                annotations={'member_type': symbol.symbol_type.value}
                            )
                            self.relationships.append(relationship)
        
        def _extract_field_access(self, node):
            """Extract REFERENCES relationship for field access (obj.field or ptr->field)
            
            Args:
                node: field_expression AST node
            """
            if not self.current_symbol:
                return  # Not inside a function/method
            
            # Get field identifier
            field_id_node = self.parser._find_child_by_type(node, 'field_identifier')
            if not field_id_node:
                return
            
            field_name = self.parser._get_node_text(field_id_node)
            
            # Try to resolve to full Class::field name
            # For now, use heuristic: find any FIELD symbol with this name
            target_symbol = None
            for symbol_list in self.symbol_by_name.get(field_name, []):
                for symbol in (symbol_list if isinstance(symbol_list, list) else [symbol_list]):
                    if symbol.symbol_type == SymbolType.FIELD:
                        target_symbol = symbol.full_name
                        break
                if target_symbol:
                    break
            
            # If not resolved, use just the field name
            if not target_symbol:
                target_symbol = field_name
            
            # Create REFERENCES relationship
            relationship = Relationship(
                source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                target_symbol=target_symbol,
                relationship_type=RelationshipType.REFERENCES,
                source_file=self.file_path,
                source_range=self.parser._node_to_range(node),
                annotations={
                    'reference_type': 'field_access',
                    'field_name': field_name
                }
            )
            self.relationships.append(relationship)
        
        def _resolve_method_call_target(self, field_expression_node, method_name):
            """Resolve method call to qualified name (Class::method)
            
            Args:
                field_expression_node: The field_expression AST node (obj.method or ptr->method)
                method_name: The method name being called
            
            Returns:
                tuple: (qualified_name, annotations_dict)
                    - qualified_name: "ClassName::method" if resolved, else just "method"
                    - annotations: Dict with call_type, object_name, access_type, resolved_class
            """
            annotations = {'call_type': 'method_call'}
            
            # Determine access type (. or ->)
            access_type = 'dot'  # default
            for child in field_expression_node.children:
                if child.type == '->':
                    access_type = 'arrow'
                    break
            annotations['access_type'] = access_type
            
            # Get the object/pointer being accessed (left side of . or ->)
            object_node = field_expression_node.children[0] if field_expression_node.children else None
            if not object_node:
                return method_name, annotations
            
            object_name = self.parser._get_node_text(object_node)
            annotations['object_name'] = object_name
            
            # Look up the object's type in our symbol table
            # It could be a local variable, parameter, or field
            object_type = None
            
            # Search in current symbol's scope first (parameters, fields)
            if self.current_symbol:
                # Check parameters - look for PARAMETER symbols with matching parent
                if not object_type:
                    for symbol_list in self.symbol_by_name.get(object_name, []):
                        for symbol in (symbol_list if isinstance(symbol_list, list) else [symbol_list]):
                            if symbol.symbol_type == SymbolType.PARAMETER:
                                if symbol.parent_symbol == self.current_symbol.full_name or symbol.parent_symbol == self.current_symbol.name:
                                    object_type = symbol.return_type
                                    break
                        if object_type:
                            break
                
                # Check fields (if we're in a method, might be accessing a field)
                if not object_type:
                    for symbol_list in self.symbol_by_name.get(object_name, []):
                        for symbol in (symbol_list if isinstance(symbol_list, list) else [symbol_list]):
                            if symbol.symbol_type == SymbolType.FIELD:
                                object_type = symbol.return_type
                                break
                        if object_type:
                            break
                
                # Check local variables by traversing AST if we have the current function node
                # Since local variables aren't extracted as symbols, we need to find their declarations
                if not object_type and hasattr(self, 'current_function_node') and self.current_function_node:
                    object_type = self._find_local_variable_type(self.current_function_node, object_name)
            
            # If we found the object's type, clean it up and add to annotations
            if object_type:
                # Remove pointer/reference markers (*, &, const)
                clean_type = object_type.replace('*', '').replace('&', '').replace('const', '').strip()
                annotations['resolved_class'] = clean_type
                # Keep target_symbol as method name for cross-file resolution compatibility
                # The resolved_class annotation provides the class context for disambiguation
                return method_name, annotations
            
            # Couldn't resolve - return just the method name
            return method_name, annotations
        
        def _find_local_variable_type(self, function_node, var_name):
            """Find the type of a local variable by searching declarations in function body
            
            Args:
                function_node: The function_definition AST node
                var_name: The variable name to search for
            
            Returns:
                str: The variable type, or None if not found
            """
            # Find the compound_statement (function body)
            body = self.parser._find_child_by_type(function_node, 'compound_statement')
            if not body:
                return None
            
            # Search for declaration nodes in the function body
            def search_declarations(node):
                if node.type == 'declaration':
                    # Extract type
                    var_type = None
                    has_var = False
                    
                    for child in node.children:
                        # Get the type
                        if child.type in ['type_identifier', 'qualified_identifier',
                                          'primitive_type', 'template_type',
                                          'sized_type_specifier',
                                          'placeholder_type_specifier']:
                            var_type = self.parser._get_node_text(child)
                        # Check if this declaration includes our variable
                        elif child.type == 'init_declarator':
                            # init_declarator contains the actual declarator (might be pointer_declarator, identifier, etc.)
                            declarator = child.children[0] if child.children else None
                            if declarator:
                                # Check the declarator type
                                if declarator.type == 'pointer_declarator':
                                    # Point* ptr - identifier is inside pointer_declarator
                                    id_node = self.parser._find_child_by_type(declarator, 'identifier')
                                    if id_node and self.parser._get_node_text(id_node) == var_name:
                                        has_var = True
                                        pointer_text = self.parser._get_node_text(declarator)
                                        star_count = pointer_text.count('*')
                                        var_type = var_type + ('*' * star_count) if var_type else None
                                elif declarator.type == 'reference_declarator':
                                    # Point& ref - identifier is inside reference_declarator
                                    id_node = self.parser._find_child_by_type(declarator, 'identifier')
                                    if id_node and self.parser._get_node_text(id_node) == var_name:
                                        has_var = True
                                        var_type = var_type + '&' if var_type else None
                                elif declarator.type == 'identifier':
                                    # Point p - identifier is direct child
                                    if self.parser._get_node_text(declarator) == var_name:
                                        has_var = True
                        elif child.type == 'identifier':
                            # Simple declaration without init_declarator: Point p;
                            if self.parser._get_node_text(child) == var_name:
                                has_var = True
                    
                    if has_var and var_type:
                        return var_type
                
                # Recursively search children
                for child in node.children:
                    result = search_declarations(child)
                    if result:
                        return result
                return None
            
            return search_declarations(body)
        
        def _extract_defines_body_relationship(self, function_def_node, impl_symbol, declarator):
            """Extract DEFINES_BODY relationship linking implementation to declaration.
            
            Creates relationships for:
            - Out-of-line method implementations (Point::move in .cpp → Point::move decl in .h)
            - Out-of-line free function implementations (add in .cpp → add decl in .h)
            - Template implementations (Container<T>::set in .cpp → decl in .h)
            
            Note: Cross-file linking happens at graph building time. This method creates
            the relationship annotation for implementations that have bodies.
            """
            # Check if this function has a body (compound_statement)
            has_body = self.parser._find_child_by_type(function_def_node, 'compound_statement') is not None
            if not has_body:
                return
            
            # Check if this is likely an implementation (has qualified name or explicit body marker)
            has_qualified_name = impl_symbol.metadata and impl_symbol.metadata.get('has_qualified_name', False)
            
            # For implementations, we create a DEFINES_BODY relationship
            # The target is the same symbol name (the declaration)
            # Graph builder will resolve across files
            if has_body:
                # Build annotations including signature for precise matching
                annotations = {
                    'implementation_type': 'out_of_line' if has_qualified_name else 'inline',
                    'has_body': True
                }
                
                # Add signature info for overload resolution
                if impl_symbol.parameter_types:
                    annotations['parameter_types'] = impl_symbol.parameter_types
                if impl_symbol.metadata and impl_symbol.metadata.get('is_const_method'):
                    annotations['is_const_method'] = True
                
                # Create DEFINES_BODY relationship
                # Source: implementation symbol (with body)
                # Target: declaration symbol (same full_name, potentially different file)
                relationship = Relationship(
                    source_symbol=impl_symbol.full_name or impl_symbol.name,
                    target_symbol=impl_symbol.full_name or impl_symbol.name,  # Same name, different file
                    relationship_type=RelationshipType.DEFINES_BODY,
                    source_file=self.file_path,
                    source_range=impl_symbol.range,
                    annotations=annotations
                )
                self.relationships.append(relationship)
        
        def _extract_class_name(self, node):
            """Extract class/struct name from node"""
            for child in node.children:
                if child.type == 'type_identifier':
                    return self.parser._get_node_text(child)
            return None
        
        def _extract_qualified_name_parts(self, declarator_node):
            """Extract all parts from a qualified name (e.g., ['Point', 'move'])"""
            for child in declarator_node.children:
                if child.type == 'qualified_identifier':
                    # Parse the qualified_identifier node structure
                    parts = []
                    self._collect_qualified_parts(child, parts)
                    return parts if parts else None
            return None
        
        def _collect_qualified_parts(self, node, parts):
            """Recursively collect parts from qualified_identifier"""
            for child in node.children:
                if child.type in ['namespace_identifier', 'type_identifier', 'identifier']:
                    parts.append(self.parser._get_node_text(child))
                elif child.type == 'qualified_identifier':
                    # Recursively handle nested qualified identifiers
                    self._collect_qualified_parts(child, parts)
        
        def _extract_function_name(self, declarator):
            """Extract function name from function_declarator"""
            for child in declarator.children:
                if child.type == 'identifier':
                    return self.parser._get_node_text(child)
                elif child.type == 'field_identifier':
                    return self.parser._get_node_text(child)
                elif child.type == 'qualified_identifier':
                    # For qualified names, get the last part
                    parts = _collect_qualified_parts_from_node(child)
                    return parts[-1] if parts else None
            return None
        
        def _extract_template_param_name(self, param_node):
            """Extract template parameter name"""
            for child in param_node.children:
                if child.type in ['type_identifier', 'identifier']:
                    return self.parser._get_node_text(child)
            return None
        
        def _extract_parameter_references(self, param_list_node, function_symbol):
            """Extract REFERENCES relationships from function parameter types"""
            if not function_symbol:
                return
            
            for child in param_list_node.children:
                if child.type == 'parameter_declaration':
                    # Extract the type from the parameter
                    type_name = None
                    
                    for param_child in child.children:
                        if param_child.type == 'type_identifier':
                            # Simple type: Point p
                            type_name = self.parser._get_node_text(param_child)
                            break
                        elif param_child.type == 'qualified_identifier':
                            # Qualified type: ns::Point p OR std::vector<Point> p
                            # Check if it contains a template_type
                            template_node = self.parser._find_child_by_type(param_child, 'template_type')
                            if template_node:
                                # This is a qualified template type like std::vector<Point>
                                # Extract the outer template name for the parameter type
                                parts = []
                                for subchild in param_child.children:
                                    if subchild.type == 'namespace_identifier':
                                        parts.append(self.parser._get_node_text(subchild))
                                    elif subchild.type == 'template_type':
                                        # Get template name
                                        name_node = self.parser._find_child_by_type(subchild, 'type_identifier')
                                        if name_node:
                                            parts.append(self.parser._get_node_text(name_node))
                                        # Extract template argument types
                                        self._extract_template_argument_references(subchild, function_symbol)
                                type_name = '::'.join(parts) if parts else None
                            else:
                                # Regular qualified type: ns::Point p
                                parts = _collect_qualified_parts_from_node(param_child)
                                type_name = '::'.join(parts)
                            break
                        elif param_child.type in ['struct_specifier', 'class_specifier', 'enum_specifier']:
                            # struct Point p or class Point p
                            name_node = self.parser._find_child_by_type(param_child, 'type_identifier')
                            if name_node:
                                type_name = self.parser._get_node_text(name_node)
                                break
                        elif param_child.type == 'template_type':
                            # Template type: std::vector<Point>, map<string, Point>
                            # Extract the outer template name AND template arguments
                            name_node = self.parser._find_child_by_type(param_child, 'type_identifier')
                            if name_node:
                                type_name = self.parser._get_node_text(name_node)
                                # Also extract template argument types
                                self._extract_template_argument_references(param_child, function_symbol)
                                break
                        elif param_child.type == 'pointer_declarator':
                            # Pointer type: Point* p, const Point* p
                            # Look for type_identifier in the pointer_declarator's children
                            # It's NOT nested - the type comes BEFORE pointer_declarator in parameter_declaration
                            # But we need to handle this case for completeness
                            pass  # Type should be in sibling, not inside pointer_declarator
                        elif param_child.type == 'reference_declarator':
                            # Reference type: Point& p, Point&& p
                            # Same as pointer - type should be in sibling
                            pass
                    
                    # Create REFERENCES relationship if we found a type
                    if type_name and type_name not in self.cpp_keywords:
                        relationship = Relationship(
                            source_symbol=function_symbol.full_name or function_symbol.name,
                            target_symbol=type_name,
                            relationship_type=RelationshipType.REFERENCES,
                            source_file=self.file_path,
                            source_range=self.parser._node_to_range(child),
                            annotations={'reference_type': 'parameter_type'}
                        )
                        self.relationships.append(relationship)
        
        def _extract_template_argument_references(self, template_type_node, source_symbol):
            """Extract REFERENCES relationships from template argument types
            
            Handles cases like:
                - std::vector<Point>           → REFERENCES to Point
                - std::map<string, Point>      → REFERENCES to Point (string is primitive/std)
                - std::vector<shared_ptr<Point>> → REFERENCES to Point (nested)
            """
            if not source_symbol:
                return
            
            # Find template_argument_list node
            arg_list = self.parser._find_child_by_type(template_type_node, 'template_argument_list')
            if not arg_list:
                return
            
            # Process each template argument
            for arg_child in arg_list.children:
                if arg_child.type == 'type_descriptor':
                    # Extract type from type_descriptor
                    type_name = None
                    
                    for type_child in arg_child.children:
                        if type_child.type == 'type_identifier':
                            # Simple type: Point
                            type_name = self.parser._get_node_text(type_child)
                            break
                        elif type_child.type == 'qualified_identifier':
                            # Qualified type: ns::Point OR std::shared_ptr<Point>
                            # Check if it contains a template_type (like std::shared_ptr<Point>)
                            template_node = self.parser._find_child_by_type(type_child, 'template_type')
                            if template_node:
                                # Qualified template type: std::shared_ptr<Point>
                                # Extract the outer template name (e.g., "std::shared_ptr")
                                parts = []
                                for subchild in type_child.children:
                                    if subchild.type == 'namespace_identifier':
                                        parts.append(self.parser._get_node_text(subchild))
                                    elif subchild.type == 'template_type':
                                        # Get template name
                                        name_node = self.parser._find_child_by_type(subchild, 'type_identifier')
                                        if name_node:
                                            parts.append(self.parser._get_node_text(name_node))
                                        # Recursively extract nested template arguments
                                        self._extract_template_argument_references(subchild, source_symbol)
                                type_name = '::'.join(parts) if parts else None
                            else:
                                # Regular qualified type: ns::Point
                                parts = _collect_qualified_parts_from_node(type_child)
                                type_name = '::'.join(parts)
                            break
                        elif type_child.type == 'template_type':
                            # Non-qualified nested template: shared_ptr<Point>
                            # Extract outer type
                            name_node = self.parser._find_child_by_type(type_child, 'type_identifier')
                            if name_node:
                                type_name = self.parser._get_node_text(name_node)
                            # Recursively extract nested template arguments
                            self._extract_template_argument_references(type_child, source_symbol)
                            break
                    
                    # Create REFERENCES relationship if we found a user-defined type
                    if type_name and type_name not in self.cpp_keywords:
                        # Filter out common standard library types that aren't user-defined (both qualified and unqualified)
                        std_types = {'string', 'wstring', 'vector', 'list', 'map', 'set', 'unordered_map',
                                    'unordered_set', 'shared_ptr', 'unique_ptr', 'weak_ptr', 'optional',
                                    'variant', 'tuple', 'pair', 'function', 'array', 'deque', 'queue',
                                    'stack', 'priority_queue',
                                    'std::string', 'std::wstring', 'std::vector', 'std::list', 'std::map',
                                    'std::set', 'std::unordered_map', 'std::unordered_set', 'std::shared_ptr',
                                    'std::unique_ptr', 'std::weak_ptr', 'std::optional', 'std::variant',
                                    'std::tuple', 'std::pair', 'std::function', 'std::array', 'std::deque',
                                    'std::queue', 'std::stack', 'std::priority_queue'}
                        
                        if type_name not in std_types:
                            relationship = Relationship(
                                source_symbol=source_symbol.full_name or source_symbol.name,
                                target_symbol=type_name,
                                relationship_type=RelationshipType.REFERENCES,
                                source_file=self.file_path,
                                source_range=self.parser._node_to_range(arg_child),
                                annotations={'reference_type': 'template_argument'}
                            )
                            self.relationships.append(relationship)
        
        def _extract_return_type_references(self, function_node, function_symbol):
            """Extract REFERENCES relationships from function return types
            
            Handles cases like:
                - Point getPoint()              → REFERENCES to Point
                - std::vector<Point> getAll()   → REFERENCES to vector and Point (template arg)
                - Point* find(int id)           → REFERENCES to Point (pointer return)
                - const Point& get() const      → REFERENCES to Point (reference return)
            """
            if not function_symbol:
                return
            
            # Extract return type from function node
            return_type_node = None
            template_type_node = None
            
            for child in function_node.children:
                # Look for return type nodes (appear before function_declarator)
                if child.type == 'type_identifier':
                    # Simple return type: Point getPoint()
                    type_name = self.parser._get_node_text(child)
                    if type_name and type_name not in self.cpp_keywords:
                        relationship = Relationship(
                            source_symbol=function_symbol.full_name or function_symbol.name,
                            target_symbol=type_name,
                            relationship_type=RelationshipType.REFERENCES,
                            source_file=self.file_path,
                            source_range=self.parser._node_to_range(child),
                            annotations={'reference_type': 'return_type'}
                        )
                        self.relationships.append(relationship)
                    return
                
                elif child.type == 'qualified_identifier':
                    # Qualified return type: ns::Point OR std::vector<Point>
                    template_node = self.parser._find_child_by_type(child, 'template_type')
                    if template_node:
                        # Qualified template return type: std::vector<Point>
                        # Extract outer template name
                        parts = []
                        for subchild in child.children:
                            if subchild.type == 'namespace_identifier':
                                parts.append(self.parser._get_node_text(subchild))
                            elif subchild.type == 'template_type':
                                name_node = self.parser._find_child_by_type(subchild, 'type_identifier')
                                if name_node:
                                    parts.append(self.parser._get_node_text(name_node))
                                # Extract template arguments
                                self._extract_template_argument_references(subchild, function_symbol)
                        
                        type_name = '::'.join(parts) if parts else None
                        # Only create relationship for non-std types
                        std_types = {'std::string', 'std::wstring', 'std::vector', 'std::list', 'std::map', 
                                   'std::set', 'std::unordered_map', 'std::unordered_set', 'std::shared_ptr',
                                   'std::unique_ptr', 'std::weak_ptr', 'std::optional', 'std::variant',
                                   'std::tuple', 'std::pair', 'std::function', 'std::array'}
                        
                        if type_name and type_name not in std_types:
                            relationship = Relationship(
                                source_symbol=function_symbol.full_name or function_symbol.name,
                                target_symbol=type_name,
                                relationship_type=RelationshipType.REFERENCES,
                                source_file=self.file_path,
                                source_range=self.parser._node_to_range(child),
                                annotations={'reference_type': 'return_type'}
                            )
                            self.relationships.append(relationship)
                    else:
                        # Regular qualified type: ns::Point or std::string
                        parts = _collect_qualified_parts_from_node(child)
                        type_name = '::'.join(parts)
                        
                        # Filter std types (qualified versions)
                        std_types = {'std::string', 'std::wstring', 'std::vector', 'std::list', 'std::map',
                                   'std::set', 'std::unordered_map', 'std::unordered_set', 'std::shared_ptr',
                                   'std::unique_ptr', 'std::weak_ptr', 'std::optional', 'std::variant',
                                   'std::tuple', 'std::pair', 'std::function', 'std::array', 'std::deque',
                                   'std::queue', 'std::stack', 'std::priority_queue'}
                        
                        if type_name and type_name not in self.cpp_keywords and type_name not in std_types:
                            relationship = Relationship(
                                source_symbol=function_symbol.full_name or function_symbol.name,
                                target_symbol=type_name,
                                relationship_type=RelationshipType.REFERENCES,
                                source_file=self.file_path,
                                source_range=self.parser._node_to_range(child),
                                annotations={'reference_type': 'return_type'}
                            )
                            self.relationships.append(relationship)
                    return
                
                elif child.type == 'template_type':
                    # Non-qualified template return type: vector<Point>
                    # Extract outer template name
                    name_node = self.parser._find_child_by_type(child, 'type_identifier')
                    if name_node:
                        type_name = self.parser._get_node_text(name_node)
                        # Filter std types
                        std_types = {'string', 'wstring', 'vector', 'list', 'map', 'set', 'unordered_map',
                                   'unordered_set', 'shared_ptr', 'unique_ptr', 'weak_ptr', 'optional',
                                   'variant', 'tuple', 'pair', 'function', 'array', 'deque', 'queue',
                                   'stack', 'priority_queue'}
                        
                        if type_name not in std_types:
                            relationship = Relationship(
                                source_symbol=function_symbol.full_name or function_symbol.name,
                                target_symbol=type_name,
                                relationship_type=RelationshipType.REFERENCES,
                                source_file=self.file_path,
                                source_range=self.parser._node_to_range(child),
                                annotations={'reference_type': 'return_type'}
                            )
                            self.relationships.append(relationship)
                    
                    # Extract template arguments
                    self._extract_template_argument_references(child, function_symbol)
                    return
                
                elif child.type == 'pointer_declarator':
                    # Pointer return type: Point* find(), ns::Point* find(), vector<T>* find()
                    # The type is BEFORE the pointer_declarator, look in siblings
                    for i, sibling in enumerate(function_node.children):
                        if sibling == child:
                            if i > 0:
                                prev = function_node.children[i - 1]
                                if prev.type in ['type_identifier', 'qualified_identifier',
                                                  'template_type']:
                                    type_name = self.parser._get_node_text(prev)
                                    if type_name and type_name not in self.cpp_keywords:
                                        relationship = Relationship(
                                            source_symbol=function_symbol.full_name or function_symbol.name,
                                            target_symbol=type_name,
                                            relationship_type=RelationshipType.REFERENCES,
                                            source_file=self.file_path,
                                            source_range=self.parser._node_to_range(prev),
                                            annotations={'reference_type': 'return_type'}
                                        )
                                        self.relationships.append(relationship)
                                    # Also extract template argument references if applicable
                                    if prev.type in ['template_type', 'qualified_identifier']:
                                        self._extract_template_argument_references(prev, function_symbol)
                                    return
                            break
                
                elif child.type == 'reference_declarator':
                    # Reference return type: Point& get(), const Point& get(),
                    # ns::Point& get(), vector<T>& get()
                    for i, sibling in enumerate(function_node.children):
                        if sibling == child:
                            if i > 0:
                                prev = function_node.children[i - 1]
                                if prev.type in ['type_identifier', 'qualified_identifier',
                                                  'template_type']:
                                    type_name = self.parser._get_node_text(prev)
                                    if type_name and type_name not in self.cpp_keywords:
                                        relationship = Relationship(
                                            source_symbol=function_symbol.full_name or function_symbol.name,
                                            target_symbol=type_name,
                                            relationship_type=RelationshipType.REFERENCES,
                                            source_file=self.file_path,
                                            source_range=self.parser._node_to_range(prev),
                                            annotations={'reference_type': 'return_type'}
                                        )
                                        self.relationships.append(relationship)
                                    # Also extract template argument references if applicable
                                    if prev.type in ['template_type', 'qualified_identifier']:
                                        self._extract_template_argument_references(prev, function_symbol)
                                    return
                            break
        
        def _extract_stack_creation(self, init_declarator_node, type_name, is_template_instantiation=False):
            """Extract CREATES relationship for stack-allocated objects
            
            In C++, value declarations create objects even without explicit constructor calls:
            - Rectangle rect;        → CREATES (default construction)
            - Rectangle rect(args);  → CREATES (explicit construction)
            - Vector<int> v;         → CREATES (template instantiation, default construction)
            - Vector<int> v(10);     → CREATES (template instantiation, explicit construction)
            - Rectangle* rect;       → NO CREATES (just a pointer, no object)
            - Rectangle* rect = new Rectangle(); → CREATES handled by visit_new_expression
            """
            if not self.current_symbol or not type_name:
                return
            
            # Check if this is a pointer or reference declaration
            # Pointers don't create objects: Rectangle* p; is just a pointer
            is_pointer = False
            is_reference = False
            for child in init_declarator_node.children:
                if child.type == 'pointer_declarator':
                    is_pointer = True
                    break
                elif child.type == 'reference_declarator':
                    is_reference = True
                    break
            
            # Pointers and references don't create objects
            # Example: Rectangle* p; or Rectangle& r = other;
            if is_pointer or is_reference:
                return
            
            # Check for different initialization patterns:
            # 1. Direct constructor: Point p1(args) or Point p1{args} - has argument_list/initializer_list
            # 2. Copy initialization: Point p = Point(args) - has call_expression (handled by visit_call_expression)
            # 3. Assignment with new: Point* p = new Point() - has new_expression (handled by visit_new_expression)
            
            has_explicit_constructor = False
            has_call_or_new = False
            
            for child in init_declarator_node.children:
                if child.type in ['argument_list', 'initializer_list']:
                    # Direct constructor: Point p1(args) or Point p1{args}
                    has_explicit_constructor = True
                    break
                elif child.type in ['call_expression', 'new_expression']:
                    # Copy initialization or new expression - let those visitors handle it
                    has_call_or_new = True
                    break
            
            # Only create CREATES if not handled by call_expression or new_expression
            if not has_call_or_new:
                # Value declarations create objects in C++:
                # - Rectangle rect(args); → explicit construction
                # - Vector<int> v(10); → template instantiation with explicit construction
                # Note: Pure default construction (Rectangle rect;) is handled in visit_declaration for bare identifiers
                if has_explicit_constructor:
                    annotations = {
                        'creation_type': 'stack',
                        'construction': 'explicit',
                        'creation_context': 'statement'
                    }
                    if is_template_instantiation:
                        annotations['template_instantiation'] = True
                    
                    relationship = Relationship(
                        source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                        target_symbol=type_name,
                        relationship_type=RelationshipType.CREATES,
                        source_file=self.file_path,
                        source_range=self.parser._node_to_range(init_declarator_node),
                        annotations=annotations
                    )
                    self.relationships.append(relationship)
        
        def visit_new_expression(self, node):
            """Extract CREATES relationship for heap allocation with new operator"""
            if not self.current_symbol:
                return
            
            # Extract the type being allocated
            type_name = None
            is_array = False
            is_placement = False
            
            # Check for placement new: new (placement) Type(args)
            # Placement new has an argument_list as second child (after 'new' keyword)
            if len(node.children) > 1 and node.children[1].type == 'argument_list':
                is_placement = True
            
            for child in node.children:
                if child.type == 'type_identifier':
                    type_name = self.parser._get_node_text(child)
                elif child.type == 'qualified_identifier':
                    parts = _collect_qualified_parts_from_node(child)
                    type_name = '::'.join(parts)
                elif child.type == 'template_type':
                    # Template type like new Container<int>()
                    name_node = self.parser._find_child_by_type(child, 'type_identifier')
                    if name_node:
                        type_name = self.parser._get_node_text(name_node)
                elif child.type == 'new_declarator':
                    # Array allocation: new Type[size]
                    is_array = True
            
            if type_name and type_name not in self.cpp_keywords:
                creation_type = 'placement_new' if is_placement else ('new_array' if is_array else 'new')
                
                relationship = Relationship(
                    source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                    target_symbol=type_name,
                    relationship_type=RelationshipType.CREATES,
                    source_file=self.file_path,
                    source_range=self.parser._node_to_range(node),
                    annotations={
                        'creation_type': creation_type,
                        'creation_context': 'statement'
                    }
                )
                self.relationships.append(relationship)
        
        def visit_field_initializer_list(self, node):
            """Extract CREATES relationships for member initialization in constructors"""
            if not self.current_symbol:
                return
            
            # Each field_initializer represents member initialization
            for child in node.children:
                if child.type == 'field_initializer':
                    self._extract_field_initializer_creation(child)
        
        def _extract_field_initializer_creation(self, field_init_node):
            """Extract CREATES from constructor member initializer"""
            if not self.current_symbol:
                return
            
            # Get the field name and check if it has a constructor call
            field_name = None
            has_constructor_call = False
            
            for child in field_init_node.children:
                if child.type == 'field_identifier':
                    field_name = self.parser._get_node_text(child)
                elif child.type in ['argument_list', 'initializer_list']:
                    has_constructor_call = True
            
            if not field_name or not has_constructor_call:
                return
            
            # Try to find the type of this field from our symbols
            # Look for a field symbol with this name in the current class
            class_name = self.scope_stack[-1] if self.scope_stack else None
            if not class_name:
                return
            
            # Find field symbol
            field_full_name = f"{class_name}::{field_name}"
            field_symbol = self._get_single_symbol(field_full_name, field_name)
            
            if field_symbol and field_symbol.return_type:
                # Extract base type (remove pointer/reference markers)
                type_name = field_symbol.return_type.replace('*', '').replace('&', '').strip()
                
                if type_name and type_name not in self.cpp_keywords:
                    relationship = Relationship(
                        source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                        target_symbol=type_name,
                        relationship_type=RelationshipType.CREATES,
                        source_file=self.file_path,
                        source_range=self.parser._node_to_range(field_init_node),
                        annotations={
                            'creation_type': 'initializer_list',
                            'creation_context': 'initializer_list',
                            'member_name': field_name
                        }
                    )
                    self.relationships.append(relationship)
        
        def _extract_smart_pointer_creation(self, template_func_node, called_name, call_expr_node):
            """Extract CREATES relationship for make_unique/make_shared calls"""
            if not self.current_symbol:
                return
            
            # Extract the template argument (the type being created)
            arg_list = self.parser._find_child_by_type(template_func_node, 'template_argument_list')
            if not arg_list:
                return
            
            type_name = None
            for child in arg_list.children:
                if child.type == 'type_identifier':
                    type_name = self.parser._get_node_text(child)
                    break
                elif child.type == 'type_descriptor':
                    # Template argument is wrapped in type_descriptor
                    type_id = self.parser._find_child_by_type(child, 'type_identifier')
                    if type_id:
                        type_name = self.parser._get_node_text(type_id)
                        break
                    # type_descriptor may also contain template_type or qualified_identifier
                    for td_child in child.children:
                        if td_child.type == 'template_type':
                            name_node = self.parser._find_child_by_type(td_child, 'type_identifier')
                            if name_node:
                                type_name = self.parser._get_node_text(name_node)
                            break
                        elif td_child.type == 'qualified_identifier':
                            type_name = self.parser._get_node_text(td_child)
                            break
                elif child.type == 'qualified_identifier':
                    parts = _collect_qualified_parts_from_node(child)
                    type_name = '::'.join(parts)
                    break
                elif child.type == 'template_type':
                    # Bare template type: make_unique<vector<int>>(...)
                    name_node = self.parser._find_child_by_type(child, 'type_identifier')
                    if name_node:
                        type_name = self.parser._get_node_text(name_node)
                    break
            
            if type_name and type_name not in self.cpp_keywords:
                creation_type = 'make_shared' if 'make_shared' in called_name else 'make_unique'
                
                relationship = Relationship(
                    source_symbol=self.current_symbol.full_name or self.current_symbol.name,
                    target_symbol=type_name,
                    relationship_type=RelationshipType.CREATES,
                    source_file=self.file_path,
                    source_range=self.parser._node_to_range(call_expr_node),
                    annotations={
                        'creation_type': creation_type,
                        'creation_context': 'statement'
                    }
                )
                self.relationships.append(relationship)
    
    def extract_relationships(self, ast_node: Any, symbols: List[Symbol], file_path: str) -> List[Relationship]:
        """Extract relationships from C++ tree-sitter AST"""
        if not self._setup_successful:
            return []
        
        # Build symbol lookup index for fast access
        # Store lists of symbols to handle overloaded functions
        symbol_by_name = {}
        symbol_by_full_name = {}
        for symbol in symbols:
            # Store lists to handle overloads (multiple symbols with same name/full_name)
            if symbol.name not in symbol_by_name:
                symbol_by_name[symbol.name] = []
            symbol_by_name[symbol.name].append(symbol)
            
            if symbol.full_name:
                if symbol.full_name not in symbol_by_full_name:
                    symbol_by_full_name[symbol.full_name] = []
                symbol_by_full_name[symbol.full_name].append(symbol)
        
        # Initialize relationship extractor
        extractor = self.RelationshipExtractor(self, symbols, symbol_by_name, symbol_by_full_name, file_path)
        extractor.visit(ast_node)
        
        # Merge field-level relationships (COMPOSITION/AGGREGATION) from symbol extraction
        all_relationships = extractor.relationships[:]
        if hasattr(self, '_field_relationships'):
            all_relationships.extend(self._field_relationships)
            self._field_relationships = []  # Clear for next parse
        
        return all_relationships
    
    # Helper methods (adapted from Python parser)
    
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
        Parse multiple files with cross-file resolution (3-pass approach)
        
        Pass 1: Parse all files in parallel
        Pass 2: Build global symbol registries  
        Pass 3: Enhance relationships with cross-file information
        """
        # Global symbol registries for cross-file resolution
        self._global_class_locations: Dict[str, str] = {}  # class_name -> file_path
        self._global_function_locations: Dict[str, str] = {}  # function_name -> file_path
        self._global_symbol_registry: Dict[str, str] = {}  # symbol_name -> full_path
        
        total = len(file_paths)
        logger.info(f"Parsing {total} C++ files for cross-file analysis with {max_workers} workers")
        
        # PASS 1: Parse all files in parallel
        try:
            results = self._parse_files_parallel(file_paths, max_workers)
            successful_parses = sum(1 for r in results.values() if r.symbols)
            logger.info(f"Pass 1 complete: {successful_parses}/{len(results)} files parsed successfully")
        except Exception as e:
            logger.error(f"Pass 1 (parallel parsing) failed: {type(e).__name__}: {e}", exc_info=True)
            raise
        
        # PASS 2: Build global symbol registries
        logger.info(f"Pass 2: Building global C++ symbol registries from {successful_parses} files")
        try:
            processed = 0
            for file_path, result in results.items():
                if result.symbols:  # Only process files that parsed successfully
                    self._extract_global_symbols(file_path, result)
                    processed += 1
                    if processed % 500 == 0:  # Log progress every 500 files
                        logger.info(f"Pass 2: Processed {processed}/{successful_parses} files")
            logger.info(f"Pass 2 complete: {len(self._global_symbol_registry)} symbols registered")
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
        
        logger.info(f"C++ parsing complete: {len(results)}/{total} files processed, {successful_parses} successful")
        return results
    
    def _parse_files_parallel(self, file_paths: List[str], max_workers: int) -> Dict[str, ParseResult]:
        """
        Parse files in parallel using ThreadPoolExecutor with batching.
        
        ThreadPoolExecutor is used instead of ProcessPoolExecutor because:
        - Lower memory overhead (shared memory)
        - No serialization overhead
        - Tree-sitter releases GIL, so we get true parallelism
        - More stable with large workloads (3000+ files)
        
        For very large codebases (>1000 files), we process in batches to:
        - Prevent memory exhaustion
        - Provide progress feedback
        - Allow graceful degradation if issues occur
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
                # Each worker thread gets a fresh parser via _parse_single_cpp_file
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all files in this batch
                    future_to_path = {}
                    for fp in batch_files:
                        try:
                            future = executor.submit(_parse_single_cpp_file, fp)
                            future_to_path[future] = fp
                        except Exception as e:
                            logger.error(f"Failed to submit file {fp}: {type(e).__name__}: {e}")
                            results[fp] = ParseResult(
                                file_path=fp,
                                language="cpp",
                                symbols=[],
                                relationships=[],
                                errors=[f"Submit failed: {str(e)}"]
                            )
                    
                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(future_to_path):
                        file_path = future_to_path[future]
                        try:
                            # _parse_single_cpp_file returns (file_path, result)
                            parsed_file_path, result = future.result(timeout=60)  # 60 second timeout per file
                            results[parsed_file_path] = result
                        except concurrent.futures.TimeoutError:
                            logger.error(f"Timeout parsing file (>60s): {file_path}")
                            results[file_path] = ParseResult(
                                file_path=file_path,
                                language="cpp",
                                symbols=[],
                                relationships=[],
                                errors=["Timeout during parsing"]
                            )
                        except Exception as e:
                            logger.error(f"Failed to parse file {file_path}: {type(e).__name__}: {e}", exc_info=True)
                            results[file_path] = ParseResult(
                                file_path=file_path,
                                language="cpp",
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
                            language="cpp",
                            symbols=[],
                            relationships=[],
                            errors=[f"Batch processing failed: {str(e)}"]
                        )
        
        successful = sum(1 for r in results.values() if r.symbols)
        logger.info(f"Parallel parsing complete: {successful}/{total} files parsed successfully")
        return results
    
    def _extract_global_symbols(self, file_path: str, result: ParseResult):
        """Extract symbols for global cross-file resolution."""
        from pathlib import Path
        file_name = Path(file_path).stem
        is_header = file_path.endswith(('.h', '.hpp', '.hxx', '.h++'))
        
        # Store class, struct, function, and method locations globally
        for symbol in result.symbols:
            if symbol.symbol_type in [SymbolType.CLASS, SymbolType.STRUCT]:
                # Store class/struct location (prefer header files over implementations)
                if symbol.name not in self._global_class_locations or is_header:
                    self._global_class_locations[symbol.name] = file_path
                
                # Also store full qualified name if available
                if symbol.full_name:
                    if symbol.full_name not in self._global_class_locations or is_header:
                        self._global_class_locations[symbol.full_name] = file_path
                    if symbol.full_name not in self._global_symbol_registry or is_header:
                        self._global_symbol_registry[symbol.full_name] = file_path
                
                # Store simple name
                if symbol.name not in self._global_symbol_registry or is_header:
                    self._global_symbol_registry[symbol.name] = file_path
                
            elif symbol.symbol_type in [SymbolType.FUNCTION, SymbolType.METHOD]:
                # Store function/method location (prefer header files over implementations)
                if symbol.name not in self._global_function_locations or is_header:
                    self._global_function_locations[symbol.name] = file_path
                
                # Also store full qualified name if available
                if symbol.full_name:
                    if symbol.full_name not in self._global_function_locations or is_header:
                        self._global_function_locations[symbol.full_name] = file_path
                    if symbol.full_name not in self._global_symbol_registry or is_header:
                        self._global_symbol_registry[symbol.full_name] = file_path
                
                # Store simple name
                if symbol.name not in self._global_symbol_registry or is_header:
                    self._global_symbol_registry[symbol.name] = file_path
                
                # For methods, also store with just Class::method format (without namespace)
                # This helps resolve method calls like Point::distance even when full name is geometry::Point::distance
                if symbol.parent_symbol and '::' in symbol.full_name:
                    # Extract just the last two parts: Class::method
                    parts = symbol.full_name.split('::')
                    if len(parts) >= 2:
                        short_qualified = '::'.join(parts[-2:])
                        if short_qualified not in self._global_function_locations or is_header:
                            self._global_function_locations[short_qualified] = file_path
                        if short_qualified not in self._global_symbol_registry or is_header:
                            self._global_symbol_registry[short_qualified] = file_path
            
            # Also handle constants - they can have DEFINES_BODY relationships too
            elif symbol.symbol_type == SymbolType.CONSTANT:
                if symbol.name not in self._global_symbol_registry or is_header:
                    self._global_symbol_registry[symbol.name] = file_path
                if symbol.full_name and (symbol.full_name not in self._global_symbol_registry or is_header):
                    self._global_symbol_registry[symbol.full_name] = file_path
            
            # Macros defined in headers are commonly used across files
            elif symbol.symbol_type == SymbolType.MACRO:
                if symbol.name not in self._global_symbol_registry or is_header:
                    self._global_symbol_registry[symbol.name] = file_path
                if symbol.full_name and (symbol.full_name not in self._global_symbol_registry or is_header):
                    self._global_symbol_registry[symbol.full_name] = file_path
    
    def _enhance_cross_file_relationships(self, file_path: str, result: ParseResult):
        """Enhance relationships with cross-file target information."""
        enhanced_relationships = []
        
        for rel in result.relationships:
            enhanced_rel = self._enhance_relationship(rel, file_path)
            enhanced_relationships.append(enhanced_rel)
        
        # Replace with enhanced relationships
        result.relationships.clear()
        result.relationships.extend(enhanced_relationships)
    
    def _enhance_relationship(self, rel: Relationship, source_file: str) -> Relationship:
        """Enhance a single relationship with cross-file target information."""
        
        # For INHERITANCE, CALLS, REFERENCES, and CREATES relationships, try to resolve target to actual file
        if rel.relationship_type in [RelationshipType.INHERITANCE, RelationshipType.CALLS, 
                                     RelationshipType.REFERENCES, RelationshipType.CREATES]:
            target_symbol = rel.target_symbol
            
            # Try to find the target file
            target_file = None
            
            # Check if target is a known class/struct
            if target_symbol in self._global_class_locations:
                target_file = self._global_class_locations[target_symbol]
            # Check if target is a known function
            elif target_symbol in self._global_function_locations:
                target_file = self._global_function_locations[target_symbol]
            # Check the general registry
            elif target_symbol in self._global_symbol_registry:
                target_file = self._global_symbol_registry[target_symbol]
            
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
        
        # For DEFINES_BODY relationships, link implementation (.cpp) to declaration (.h)
        # Both source and target have the same symbol name, but are in different files
        elif rel.relationship_type == RelationshipType.DEFINES_BODY:
            target_symbol = rel.target_symbol
            
            # Try to find the declaration file (typically a header)
            target_file = None
            
            # Check function locations first (most likely for DEFINES_BODY)
            if target_symbol in self._global_function_locations:
                potential_file = self._global_function_locations[target_symbol]
                # Declaration is typically in a different file than implementation
                if potential_file != rel.source_file:
                    target_file = potential_file
            
            # Check general registry as fallback
            if not target_file and target_symbol in self._global_symbol_registry:
                potential_file = self._global_symbol_registry[target_symbol]
                if potential_file != rel.source_file:
                    target_file = potential_file
            
            # If we found the declaration file, update the relationship
            if target_file:
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
