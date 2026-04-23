"""
Enhanced Python Parser implementing the new modular parser architecture

This parser provides comprehensive Python AST analysis with support for:
- Complete symbol extraction (functions, classes, methods, variables, etc.)
- Advanced relationship detection (inheritance, composition, calls, imports)
- Hierarchical context expansion
- Type annotation support
- Decorator and async/await support

KNOWN LIMITATIONS:
==================
1. COMPOSITION EDGES AT FIELD LEVEL (NOT CLASS LEVEL) - FIXED IN THIS SESSION
   - Status: FIXED - Parser now creates field-level composition edges
   - Implementation: `_extract_composition_relationships()` (Pass 3)
   - Edge format: `AppConfig.database` → `DatabaseConfig` (COMPOSITION)
   
   - Remaining work (Content Expander side):
     * Content expander needs to traverse: Class → DEFINES → Field → COMPOSITION → Type
     * Currently only looks at direct successors of class node
     * See: CONTENT_EXPANDER_ANALYSIS.md for details

2. Class-level field type annotations - ALSO FIXED
   - Python 3.6+ class variable type hints are now supported
   - Example: `database: DatabaseConfig` (without assignment)
   - Creates composition edges in `_extract_composition_relationships()`
"""

import ast
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any, Tuple

from .base_parser import (
    BaseParser, LanguageCapabilities, ParseResult, Symbol, Relationship,
    SymbolType, RelationshipType, Scope, Position, Range
)

logger = logging.getLogger(__name__)

# Optional thinking emitter for progress reporting
try:  # pragma: no cover
    from tools import this  # type: ignore
except Exception:  # pragma: no cover
    this = None  # type: ignore


def _parse_single_python_file(file_path: str) -> Tuple[str, ParseResult]:
    """Parse a single Python file - used by parallel workers"""
    try:
        # Create a new parser instance for this worker
        parser = PythonParser()
        result = parser.parse_file(file_path)
        return file_path, result
    except Exception as e:
        logger.warning(f"Failed to parse {file_path}: {e}")
        # Create empty result for failed files
        return file_path, ParseResult(
            file_path=file_path,
            language="python",
            symbols=[], 
            relationships=[]
        )


class PythonParser(BaseParser):
    """
    Advanced Python AST parser implementing the modular parser architecture
    """
    
    def __init__(self):
        super().__init__("python")
        self._current_file = ""
        self._current_content = ""
        self._current_scope_stack: List[str] = []
        
        # Global symbol registries for cross-file analysis
        # Populated during parse_multiple_files(), empty for single-file parsing
        self._global_class_locations: Dict[str, str] = {}  # class_name -> file_path
        self._global_function_locations: Dict[str, str] = {}  # function_name -> file_path
        self._global_symbol_registry: Dict[str, str] = {}  # symbol_name -> qualified_name
    
    def _define_capabilities(self) -> LanguageCapabilities:
        """Define Python parser capabilities"""
        return LanguageCapabilities(
            language="python",
            supported_symbols={
                SymbolType.FUNCTION,
                SymbolType.METHOD,
                SymbolType.CONSTRUCTOR,
                SymbolType.CLASS,
                SymbolType.VARIABLE,
                SymbolType.CONSTANT,
                SymbolType.PROPERTY,
                SymbolType.PARAMETER,
                SymbolType.MODULE,
                SymbolType.DECORATOR
            },
            supported_relationships={
                RelationshipType.IMPORTS,
                RelationshipType.CALLS,
                RelationshipType.INHERITANCE,
                RelationshipType.COMPOSITION,
                RelationshipType.AGGREGATION,
                RelationshipType.DEFINES,
                RelationshipType.DECORATES,
                RelationshipType.ANNOTATES,
                RelationshipType.ASSIGNS,
                RelationshipType.RETURNS,
                RelationshipType.PARAMETER,
                RelationshipType.REFERENCES
            },
            supports_ast_parsing=True,
            supports_type_inference=True,
            supports_cross_file_analysis=True,
            has_classes=True,
            has_interfaces=False,  # Python uses ABC
            has_namespaces=False,  # Python uses modules
            has_generics=True,     # Type hints support generics
            has_decorators=True,
            has_annotations=True,
            max_file_size_mb=5.0,
            typical_parse_time_ms=50.0
        )
    
    def _get_supported_extensions(self) -> Set[str]:
        """Get supported Python file extensions"""
        return {'.py', '.pyx', '.pyi', '.pyw'}
    
    def parse_file(self, file_path: Union[str, Path], content: Optional[str] = None) -> ParseResult:
        """
        Parse a Python file and extract symbols and relationships
        """
        start_time = time.time()
        file_path = str(file_path)
        self._current_file = file_path
        
        try:
            # Read content if not provided
            if content is None:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            self._current_content = content
            
            # Calculate file hash for caching
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Parse AST
            try:
                ast_tree = ast.parse(content, filename=file_path)
            except SyntaxError as e:
                return ParseResult(
                    file_path=file_path,
                    language="python",
                    symbols=[],
                    relationships=[],
                    file_hash=file_hash,
                    parse_time=time.time() - start_time,
                    errors=[f"Syntax error: {e}"]
                )
            
            # Extract symbols and relationships
            symbols = self.extract_symbols(ast_tree, file_path)
            relationships, field_symbols = self.extract_relationships(ast_tree, symbols, file_path)
            
            # Add field symbols discovered during relationship extraction
            symbols.extend(field_symbols)
            
            # Extract module-level information
            imports, exports = self._extract_module_info(ast_tree)
            dependencies = self._extract_dependencies(imports, relationships)
            module_docstring = self._extract_module_docstring(ast_tree)
            
            result = ParseResult(
                file_path=file_path,
                language="python",
                symbols=symbols,
                relationships=relationships,
                imports=imports,
                exports=exports,
                dependencies=dependencies,
                module_docstring=module_docstring,
                file_hash=file_hash,
                parse_time=time.time() - start_time
            )
            
            return self.validate_result(result)
            
        except Exception as e:
            logger.error(f"Failed to parse Python file {file_path}: {e}")
            return ParseResult(
                file_path=file_path,
                language="python",
                symbols=[],
                relationships=[],
                parse_time=time.time() - start_time,
                errors=[f"Parse error: {e}"]
            )
    
    def extract_symbols(self, ast_node: Any, file_path: str) -> List[Symbol]:
        """Extract symbols from Python AST"""
        symbols = []
        self._current_scope_stack = []
        
        class SymbolExtractor(ast.NodeVisitor):
            def __init__(self, parser):
                self.parser = parser
                self.symbols = []
                self.scope_stack = []
                self._class_names = set()  # Pre-computed set for O(1) lookups
            
            def _is_module_level(self) -> bool:
                """Check if we're at module (top) level.
                
                visit_Module pushes the module name onto scope_stack, so
                module-level declarations see len(scope_stack) == 1.
                Anything deeper (class, function) has len > 1.
                """
                return len(self.scope_stack) <= 1
            
            def visit_Module(self, node):
                # Module represents the Python file scope
                # We track it in scope_stack for qualified names but don't emit a MODULE symbol
                # (modules are organizational, not architectural - they don't go in the graph)
                module_name = Path(file_path).stem
                
                # Track module in scope for children (needed for qualified full_name)
                self.scope_stack.append(module_name)
                self.generic_visit(node)
                self.scope_stack.pop()
            
            def visit_ClassDef(self, node):
                # Enter class scope
                parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}.{node.name}" if parent_symbol else node.name
                
                # Extract base classes
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(self.parser._get_full_attribute_name(base))
                    elif isinstance(base, ast.Subscript):
                        # Handle generic inheritance like InMemoryRepository[User]
                        # Extract the base class name from the subscript value
                        if isinstance(base.value, ast.Name):
                            bases.append(base.value.id)
                        elif isinstance(base.value, ast.Attribute):
                            bases.append(self.parser._get_full_attribute_name(base.value))
                
                # Extract decorators
                decorators = [self.parser._get_decorator_name(dec) for dec in node.decorator_list]
                
                symbol = Symbol(
                    name=node.name,
                    symbol_type=SymbolType.CLASS,
                    scope=Scope.GLOBAL if self._is_module_level() else Scope.CLASS,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    docstring=ast.get_docstring(node),
                    source_text=self.parser._extract_node_source(node)
                )
                self.symbols.append(symbol)
                self._class_names.add(node.name)  # Track class names for O(1) method detection
                
                # Enter scope and visit children
                self.scope_stack.append(node.name)
                self.generic_visit(node)
                self.scope_stack.pop()
            
            def visit_FunctionDef(self, node):
                self._visit_function(node, is_async=False)
            
            def visit_AsyncFunctionDef(self, node):
                self._visit_function(node, is_async=True)
            
            def _visit_function(self, node, is_async=False):
                parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}.{node.name}" if parent_symbol else node.name
                
                # Determine if this is a method or function - O(1) lookup using pre-computed set
                is_in_class = bool(self._class_names & set(self.scope_stack))
                if is_in_class and node.name == '__init__':
                    symbol_type = SymbolType.CONSTRUCTOR  # Phase 2: standardise constructor SymbolType
                elif is_in_class:
                    symbol_type = SymbolType.METHOD
                else:
                    symbol_type = SymbolType.FUNCTION
                scope = Scope.FUNCTION
                
                # Extract decorators
                decorators = [self.parser._get_decorator_name(dec) for dec in node.decorator_list]
                
                # Extract return type annotation
                return_type = None
                if node.returns:
                    return_type = self.parser._get_type_annotation(node.returns)
                
                # Extract parameters
                parameters = self.parser._extract_function_parameters(node)
                
                symbol = Symbol(
                    name=node.name,
                    symbol_type=symbol_type,
                    scope=scope,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    docstring=ast.get_docstring(node),
                    return_type=return_type,
                    parameter_types=[p.get('type') for p in parameters if p.get('type')],
                    is_async=is_async,
                    source_text=self.parser._extract_node_source(node),
                    signature=self.parser._build_function_signature(node, return_type),
                    metadata={'decorators': decorators}
                )
                self.symbols.append(symbol)
                
                # Add parameter symbols
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
                
                # Enter scope and visit children (for nested functions)
                self.scope_stack.append(node.name)
                self.generic_visit(node)
                self.scope_stack.pop()
            
            def visit_Assign(self, node):
                # Variable assignments
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                        is_module = self._is_module_level()
                        scope = Scope.GLOBAL if is_module else Scope.FUNCTION
                        
                        # In Python, all module-level variables are effectively constants
                        # regardless of naming convention (logger, CONFIG, etc.)
                        # Only consider case for function-level variables
                        if is_module:
                            symbol_type = SymbolType.CONSTANT
                        else:
                            # For function-level variables, use naming convention
                            symbol_type = SymbolType.CONSTANT if target.id.isupper() else SymbolType.VARIABLE
                        
                        symbol = Symbol(
                            name=target.id,
                            symbol_type=symbol_type,
                            scope=scope,
                            range=self.parser._node_to_range(target),
                            file_path=file_path,
                            parent_symbol=parent_symbol,
                            source_text=self.parser._extract_node_source(node)
                        )
                        self.symbols.append(symbol)
                
                self.generic_visit(node)
            
            def visit_AnnAssign(self, node):
                # Annotated assignments (e.g., DEFAULT_TIMEOUT: int = 30)
                if isinstance(node.target, ast.Name):
                    parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                    is_module = self._is_module_level()
                    scope = Scope.GLOBAL if is_module else Scope.FUNCTION
                    
                    # Extract type annotation
                    type_annotation = self.parser._get_type_annotation(node.annotation) if node.annotation else None
                    
                    # Module-level annotated assignments are constants too
                    if is_module:
                        symbol_type = SymbolType.CONSTANT
                    else:
                        symbol_type = SymbolType.CONSTANT if node.target.id.isupper() else SymbolType.VARIABLE
                    
                    symbol = Symbol(
                        name=node.target.id,
                        symbol_type=symbol_type,
                        scope=scope,
                        range=self.parser._node_to_range(node.target),
                        file_path=file_path,
                        parent_symbol=parent_symbol,
                        return_type=type_annotation,
                        source_text=self.parser._extract_node_source(node)
                    )
                    self.symbols.append(symbol)
                
                self.generic_visit(node)
        
        extractor = SymbolExtractor(self)
        extractor.visit(ast_node)
        return extractor.symbols
    
    def extract_relationships(self, ast_node: Any, symbols: List[Symbol], file_path: str) -> Tuple[List[Relationship], List[Symbol]]:
        """
        Extract relationships from Python AST.
        
        Returns:
            Tuple of (relationships, additional_symbols) where additional_symbols
            are FIELD symbols discovered during composition extraction.
        """
        relationships = []
        symbol_map = {s.name: s for s in symbols}
        
        # Build LOCAL class registry for constructor detection (AST-based, no string heuristics)
        local_class_names = {s.name for s in symbols if s.symbol_type == SymbolType.CLASS}
        
        # Combine with GLOBAL class registry for cross-file detection
        # Global registry is populated during parse_multiple_files()
        all_class_names = local_class_names | set(self._global_class_locations.keys())
        
        class RelationshipExtractor(ast.NodeVisitor):
            def __init__(self, parser, class_registry):
                self.parser = parser
                self.relationships = []
                self.current_symbol = None
                self.scope_stack = []
                self.class_registry = class_registry  # Pure AST-based class detection (local + global)
            
            def visit_ClassDef(self, node):
                self.current_symbol = node.name
                
                # Inheritance relationships
                for base in node.bases:
                    base_name = None
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = self.parser._get_full_attribute_name(base)
                    elif isinstance(base, ast.Subscript):
                        # Handle generic inheritance like InMemoryRepository[User]
                        # Extract the base class name from the subscript value
                        if isinstance(base.value, ast.Name):
                            base_name = base.value.id
                        elif isinstance(base.value, ast.Attribute):
                            base_name = self.parser._get_full_attribute_name(base.value)
                        
                        # Extract generic type parameter(s) and create reference relationships
                        type_params = []
                        if isinstance(base.slice, ast.Name):
                            # Single type parameter: InMemoryRepository[User]
                            type_params.append(base.slice.id)
                        elif isinstance(base.slice, ast.Tuple):
                            # Multiple type parameters: Dict[str, int]
                            for elt in base.slice.elts:
                                if isinstance(elt, ast.Name):
                                    type_params.append(elt.id)
                                elif isinstance(elt, ast.Attribute):
                                    type_params.append(self.parser._get_full_attribute_name(elt))
                        elif isinstance(base.slice, ast.Attribute):
                            # Qualified type parameter: Dict[models.User, str]
                            type_params.append(self.parser._get_full_attribute_name(base.slice))
                        
                        # Create reference relationships for type parameters
                        for type_param in type_params:
                            if type_param:
                                type_ref_relationship = Relationship(
                                    source_symbol=node.name,
                                    target_symbol=type_param,
                                    relationship_type=RelationshipType.REFERENCES,
                                    source_file=file_path,
                                    source_range=self.parser._node_to_range(base.slice),
                                    confidence=0.8,
                                    weight=0.6
                                )
                                self.relationships.append(type_ref_relationship)
                    
                    if base_name:
                        relationship = Relationship(
                            source_symbol=node.name,
                            target_symbol=base_name,
                            relationship_type=RelationshipType.INHERITANCE,
                            source_file=file_path,
                            source_range=self.parser._node_to_range(base),
                            confidence=0.9,
                            weight=0.8
                        )
                        self.relationships.append(relationship)
                
                # Decorator relationships
                for decorator in node.decorator_list:
                    decorator_name = self.parser._get_decorator_name(decorator)
                    if decorator_name:
                        relationship = Relationship(
                            source_symbol=decorator_name,
                            target_symbol=node.name,
                            relationship_type=RelationshipType.DECORATES,
                            source_file=file_path,
                            source_range=self.parser._node_to_range(decorator),
                            confidence=0.95,
                            weight=0.6
                        )
                        self.relationships.append(relationship)
                
                self.scope_stack.append(node.name)
                self.generic_visit(node)
                self.scope_stack.pop()
                self.current_symbol = None
            
            def visit_FunctionDef(self, node):
                self._visit_function(node)
            
            def visit_AsyncFunctionDef(self, node):
                self._visit_function(node)
            
            def _visit_function(self, node):
                old_symbol = self.current_symbol
                self.current_symbol = node.name
                
                # Decorator relationships
                for decorator in node.decorator_list:
                    decorator_name = self.parser._get_decorator_name(decorator)
                    if decorator_name:
                        relationship = Relationship(
                            source_symbol=node.name,
                            target_symbol=decorator_name,
                            relationship_type=RelationshipType.DECORATES,
                            source_file=file_path,
                            source_range=self.parser._node_to_range(decorator),
                            confidence=0.95,
                            weight=0.6
                        )
                        self.relationships.append(relationship)
                
                # Extract REFERENCES from parameter types (recursive decomposition)
                for arg in node.args.args:
                    if arg.annotation:
                        self.parser._extract_type_references_recursive(
                            arg.annotation,
                            source_symbol=node.name,
                            file_path=file_path,
                            reference_type='parameter_type',
                            relationships=self.relationships,
                        )
                
                # Extract REFERENCES from return type (recursive decomposition)
                if node.returns:
                    self.parser._extract_type_references_recursive(
                        node.returns,
                        source_symbol=node.name,
                        file_path=file_path,
                        reference_type='return_type',
                        relationships=self.relationships,
                    )
                
                self.scope_stack.append(node.name)
                self.generic_visit(node)
                self.scope_stack.pop()
                self.current_symbol = old_symbol
            
            def visit_Call(self, node):
                if self.current_symbol:
                    # Function/method calls or constructor calls
                    called_name = None
                    module_prefix = None
                    
                    if isinstance(node.func, ast.Name):
                        called_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        # Module.ClassName() or obj.method()
                        called_name = node.func.attr
                        if isinstance(node.func.value, ast.Name):
                            module_prefix = node.func.value.id
                    
                    if called_name:
                        # Determine if this is a constructor call using pure AST-based detection
                        # Check if called_name exists in our class registry (no string heuristics)
                        is_constructor = called_name in self.class_registry
                        
                        # Choose relationship type
                        if is_constructor:
                            relationship_type = RelationshipType.CREATES
                            confidence = 0.95  # High confidence - AST-based detection
                            annotations = {'creation_context': 'call', 'detection_method': 'ast_registry'}
                            if module_prefix:
                                annotations['module_prefix'] = module_prefix
                        else:
                            relationship_type = RelationshipType.CALLS
                            confidence = 0.8
                            annotations = {}  # Empty dict instead of None
                        
                        relationship = Relationship(
                            source_symbol=self.current_symbol,
                            target_symbol=called_name,
                            relationship_type=relationship_type,
                            source_file=file_path,
                            source_range=self.parser._node_to_range(node),
                            confidence=confidence,
                            weight=0.8 if is_constructor else 0.7,
                            annotations=annotations
                        )
                        self.relationships.append(relationship)
                
                self.generic_visit(node)
            
            def visit_Import(self, node):
                for alias in node.names:
                    imported_name = alias.asname if alias.asname else alias.name
                    relationship = Relationship(
                        source_symbol=Path(file_path).stem,  # Current module name
                        target_symbol=imported_name,
                        relationship_type=RelationshipType.IMPORTS,
                        source_file=file_path,
                        source_range=self.parser._node_to_range(node),
                        confidence=1.0,
                        weight=0.5
                    )
                    self.relationships.append(relationship)
                
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                module_name = node.module or ""
                for alias in node.names:
                    # Use the alias if it exists, otherwise use the actual import name
                    imported_name = alias.asname if alias.asname else alias.name
                    
                    # For relationships, we care about the name as it appears in the code
                    # So if there's an alias, use it; otherwise use the full module.name
                    if alias.asname:
                        target_name = alias.asname
                    else:
                        target_name = f"{module_name}.{alias.name}" if module_name and alias.name != "*" else alias.name
                    
                    relationship = Relationship(
                        source_symbol=Path(file_path).stem,  # Current module name
                        target_symbol=target_name,
                        relationship_type=RelationshipType.IMPORTS,
                        source_file=file_path,
                        source_range=self.parser._node_to_range(node),
                        confidence=1.0,
                        weight=0.5
                    )
                    self.relationships.append(relationship)
                
                self.generic_visit(node)
            
            def visit_Attribute(self, node):
                # NOTE: COMPOSITION edges are now created by _extract_composition_relationships (Pass 3)
                # This old logic created incorrect edges from methods to attributes
                # Keeping this method for potential future use (e.g., USES relationships)
                
                # if self.current_symbol:
                #     # Attribute access (composition/aggregation)
                #     attr_name = self.parser._get_full_attribute_name(node)
                #     if attr_name:
                #         # Determine if this is composition or aggregation based on context
                #         rel_type = RelationshipType.COMPOSITION  # Default to composition
                #         
                #         relationship = Relationship(
                #             source_symbol=self.current_symbol,
                #             target_symbol=attr_name,
                #             relationship_type=rel_type,
                #             source_file=file_path,
                #             source_range=self.parser._node_to_range(node),
                #             confidence=0.6,
                #             weight=0.4
                #         )
                #         self.relationships.append(relationship)
                
                self.generic_visit(node)
            
            def visit_Name(self, node):
                """Detect variable and symbol references"""
                if self.current_symbol and isinstance(node.ctx, ast.Load):
                    # This is a variable/symbol being referenced (loaded/used)
                    # Exclude some common built-ins and keywords to reduce noise
                    excluded_names = {
                        'self', 'cls', 'True', 'False', 'None', 'print', 'len', 'str', 'int', 
                        'float', 'bool', 'list', 'dict', 'tuple', 'set', 'range', 'enumerate',
                        'zip', 'map', 'filter', 'sum', 'min', 'max', 'abs', 'all', 'any'
                    }
                    
                    if node.id not in excluded_names and node.id != self.current_symbol:
                        # Check if this name represents a symbol we know about
                        # This could be a class, function, variable, etc. being referenced
                        relationship = Relationship(
                            source_symbol=self.current_symbol,
                            target_symbol=node.id,
                            relationship_type=RelationshipType.REFERENCES,
                            source_file=file_path,
                            source_range=self.parser._node_to_range(node),
                            confidence=0.7,
                            weight=0.3
                        )
                        self.relationships.append(relationship)
                
                self.generic_visit(node)
        
        extractor = RelationshipExtractor(self, all_class_names)
        extractor.visit(ast_node)
        
        # Extract DEFINES relationships (Pass 2: after symbols extracted)
        defines_relationships = self._extract_defines_relationships(symbols, file_path)
        extractor.relationships.extend(defines_relationships)
        
        # Extract FIELD symbols and COMPOSITION relationships (Pass 3: after symbols and DEFINES extracted)
        field_symbols, composition_relationships = self._extract_fields_and_composition(symbols, file_path, ast_node)
        extractor.relationships.extend(composition_relationships)
        
        return extractor.relationships, field_symbols
    
    def _extract_defines_relationships(self, symbols: List[Symbol], file_path: str) -> List[Relationship]:
        """
        Extract DEFINES relationships (Pass 2 - after symbol extraction).
        
        DEFINES represents containment:
        - MODULE DEFINES top-level classes, functions, constants
        - CLASS DEFINES its methods/fields
        
        Following the pattern from C++/JavaScript/TypeScript parsers.
        
        Rules:
        - CLASS → METHOD (including __init__)
        - CLASS → VARIABLE (class variables)
        
        Note: MODULE → top-level DEFINES relationships are NOT created.
        Modules are organizational, not architectural. Parent-child containment
        is captured by parent_symbol metadata on each symbol.
        """
        relationships = []
        
        # Build parent index for efficient lookup
        parent_index = {}
        for symbol in symbols:
            if symbol.parent_symbol:
                if symbol.parent_symbol not in parent_index:
                    parent_index[symbol.parent_symbol] = []
                parent_index[symbol.parent_symbol].append(symbol)
        
        # Iterate through potential container symbols
        # Only CLASS creates DEFINES edges (not MODULE)
        for symbol in symbols:
            # Only CLASS can DEFINE members
            if symbol.symbol_type != SymbolType.CLASS:
                continue
            
            # Find all members with this class as parent
            # Members use full_name as parent_symbol for nested classes
            members = parent_index.get(symbol.full_name, [])
            if not members:
                # Fallback: try with just name (for top-level classes)
                members = parent_index.get(symbol.name, [])
            
            for member in members:
                # Only create DEFINES for actual members (not parameters, etc.)
                if member.symbol_type not in (SymbolType.METHOD, SymbolType.CONSTRUCTOR, SymbolType.VARIABLE, SymbolType.FIELD, SymbolType.CONSTANT):
                    continue
                
                # Skip members without full_name (shouldn't happen, but defensive)
                if not member.full_name:
                    # Build full_name if missing
                    member_full_name = f"{symbol.full_name}.{member.name}" if symbol.full_name else member.name
                else:
                    member_full_name = member.full_name
                
                # Determine member type for annotation
                if member.symbol_type == SymbolType.METHOD:
                    member_type = 'method'
                elif member.symbol_type == SymbolType.CONSTRUCTOR:
                    member_type = 'constructor'
                elif member.symbol_type == SymbolType.VARIABLE:
                    member_type = 'variable'
                elif member.symbol_type == SymbolType.CONSTANT:
                    member_type = 'variable'  # Treat constants as variables in metadata
                elif member.symbol_type == SymbolType.FIELD:
                    member_type = 'field'
                else:
                    member_type = 'unknown'
                
                # Extract decorator metadata
                metadata = member.metadata if member.metadata else {}
                decorators = metadata.get('decorators', []) or []
                is_static = 'staticmethod' in decorators
                is_classmethod = 'classmethod' in decorators
                is_property = 'property' in decorators
                is_abstract = 'abstractmethod' in decorators
                
                # Create DEFINES relationship
                relationship = Relationship(
                    source_symbol=symbol.full_name,
                    target_symbol=member_full_name,
                    relationship_type=RelationshipType.DEFINES,
                    source_file=file_path,
                    source_range=member.source_code_location if hasattr(member, 'source_code_location') and member.source_code_location else Range(Position(0, 0), Position(0, 0)),
                    annotations={
                        'member_type': member_type,
                        'is_static': is_static,
                        'is_classmethod': is_classmethod,
                        'is_property': is_property,
                        'is_abstract': is_abstract,
                        'decorators': decorators
                    },
                    confidence=1.0,
                    weight=1.0
                )
                relationships.append(relationship)
        
        return relationships
    
    def _extract_fields_and_composition(self, symbols: List[Symbol], file_path: str, ast_node: ast.Module) -> Tuple[List[Symbol], List[Relationship]]:
        """
        Extract FIELD symbols and COMPOSITION relationships (Pass 3 - after symbol and DEFINES extraction).
        
        This method:
        1. Creates FIELD symbols for class instance attributes and class variables
        2. Creates DEFINES relationships from Class → Field
        3. Creates COMPOSITION relationships from Field → FieldType (for non-builtin types)
        
        COMPOSITION represents "has-a" relationships where a class contains instances of other classes.
        This is critical for transitive field access expansion (e.g., config.database.host).
        
        Rules:
        - Analyze __init__ methods for field assignments: self.field = ClassName()
        - Create FIELD symbols for each self.field assignment
        - Create CLASS → FIELD DEFINES edges
        - Create FIELD → FieldType COMPOSITION edges (for non-builtin types)
        - Skip built-in types (str, int, list, dict, etc.) for COMPOSITION
        - Handle type hints: self.field: Type = None
        - Handle module-qualified types: self.field = module.ClassName()
        - Skip parameter assignments: self.field = param (this is aggregation, not composition)
        
        Returns:
            Tuple of (field_symbols, relationships) where relationships includes both DEFINES and COMPOSITION
        """
        field_symbols = []
        relationships = []
        
        # Build class index: map full_name → class symbol
        class_index = {}
        for symbol in symbols:
            if symbol.symbol_type == SymbolType.CLASS:
                class_index[symbol.full_name] = symbol
        
        # Built-in types that should NOT create COMPOSITION edges
        builtin_types = {
            'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set', 
            'frozenset', 'bytes', 'bytearray', 'range', 'complex', 'type',
            'object', 'None', 'NoneType'
        }
        
        # Find all __init__ methods
        init_methods = [s for s in symbols 
                       if s.symbol_type in (SymbolType.METHOD, SymbolType.CONSTRUCTOR) and s.name == '__init__']
        
        # For each __init__, analyze field assignments
        for init_method in init_methods:
            # Find the parent class
            parent_class_name = init_method.parent_symbol
            if not parent_class_name or parent_class_name not in class_index:
                continue
            
            parent_class = class_index[parent_class_name]
            
            # Parse the __init__ method body to find field assignments
            # We need to find the actual FunctionDef node in the AST
            init_node = self._find_function_node(ast_node, init_method)
            if not init_node:
                continue
            
            # Track fields we've seen to avoid duplicates
            seen_fields = set()
            
            # Extract field assignments from __init__ body
            for node in ast.walk(init_node):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                
                # For ast.Assign: self.field = value
                # For ast.AnnAssign: self.field: Type = value
                field_name = None
                field_type = None
                is_instantiation = False
                is_none = False
                is_optional_wrapper = False
                module_prefix = None
                
                if isinstance(node, ast.Assign):
                    # Handle: self.field = ClassName()
                    if len(node.targets) == 1:
                        target = node.targets[0]
                        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                            field_name = target.attr
                            
                            # Check if value is a call (instantiation)
                            if isinstance(node.value, ast.Call):
                                is_instantiation = True
                                # Extract the class name being instantiated
                                if isinstance(node.value.func, ast.Name):
                                    field_type = node.value.func.id
                                elif isinstance(node.value.func, ast.Attribute):
                                    # Module-qualified: module.ClassName()
                                    field_type = node.value.func.attr
                                    if isinstance(node.value.func.value, ast.Name):
                                        module_prefix = node.value.func.value.id
                            elif isinstance(node.value, ast.Constant) and node.value.value is None:
                                is_none = True
                            elif isinstance(node.value, ast.Name):
                                # self.field = param (parameter assignment - skip for now)
                                continue
                
                elif isinstance(node, ast.AnnAssign):
                    # Handle: self.field: Type = value
                    if isinstance(node.target, ast.Attribute) and isinstance(node.target.value, ast.Name) and node.target.value.id == 'self':
                        field_name = node.target.attr
                        
                        # Extract type from annotation
                        if isinstance(node.annotation, ast.Name):
                            field_type = node.annotation.id
                        elif isinstance(node.annotation, ast.Attribute):
                            # Module-qualified: module.ClassName
                            field_type = node.annotation.attr
                            if isinstance(node.annotation.value, ast.Name):
                                module_prefix = node.annotation.value.id
                        elif isinstance(node.annotation, ast.Subscript):
                            # Generic type: Optional[Type], List[Type], etc.
                            # Phase 5: detect Optional wrapper for AGGREGATION
                            outer_type = None
                            if isinstance(node.annotation.value, ast.Name):
                                outer_type = node.annotation.value.id
                            elif isinstance(node.annotation.value, ast.Attribute):
                                outer_type = node.annotation.value.attr
                            if outer_type == 'Optional':
                                is_optional_wrapper = True
                            # Extract the inner type from the subscript
                            if isinstance(node.annotation.slice, ast.Name):
                                field_type = node.annotation.slice.id
                            elif isinstance(node.annotation.slice, ast.Attribute):
                                field_type = node.annotation.slice.attr
                                if isinstance(node.annotation.slice.value, ast.Name):
                                    module_prefix = node.annotation.slice.value.id
                        
                        # Check if value is None or an instantiation
                        if node.value:
                            if isinstance(node.value, ast.Constant) and node.value.value is None:
                                is_none = True
                            elif isinstance(node.value, ast.Call):
                                is_instantiation = True
                
                # Create FIELD symbol and relationships if we found a valid field
                if field_name:
                    # Skip if already seen this field
                    if field_name in seen_fields:
                        continue
                    
                    seen_fields.add(field_name)
                    
                    # Always create FIELD symbol (even for built-in types)
                    field_full_name = f"{parent_class.full_name}.{field_name}"
                    field_symbol = Symbol(
                        name=field_name,
                        symbol_type=SymbolType.FIELD,
                        scope=Scope.CLASS,
                        range=self._node_to_range(node),
                        file_path=file_path,
                        full_name=field_full_name,
                        parent_symbol=parent_class.full_name,
                        metadata={
                            'field_type': field_type if field_type else 'unknown',
                            'is_instance_field': True,
                            'module_prefix': module_prefix
                        }
                    )
                    field_symbols.append(field_symbol)
                    
                    # Create DEFINES relationship: Class → Field
                    defines_rel = Relationship(
                        source_symbol=parent_class.full_name,
                        target_symbol=field_full_name,
                        relationship_type=RelationshipType.DEFINES,
                        source_file=file_path,
                        target_file=file_path,
                        source_range=self._node_to_range(node),
                        annotations={'member_type': 'field', 'container_type': 'class'},
                        confidence=1.0,
                        weight=0.9
                    )
                    relationships.append(defines_rel)
                    
                    # Create COMPOSITION or AGGREGATION relationship for non-builtin types
                    if field_type and field_type not in builtin_types:
                        annotations = {
                            'field_name': field_name,
                        }
                        if is_none:
                            annotations['initialized_to_none'] = True
                        if module_prefix:
                            annotations['module_prefix'] = module_prefix
                        
                        # Phase 5: Optional[Type] or = None → AGGREGATION (weak ownership)
                        rel_type = RelationshipType.AGGREGATION if (is_optional_wrapper or is_none) else RelationshipType.COMPOSITION
                        
                        composition_rel = Relationship(
                            source_symbol=field_full_name,
                            target_symbol=field_type,
                            relationship_type=rel_type,
                            source_file=file_path,
                            target_file=file_path,
                            source_range=self._node_to_range(node),
                            annotations=annotations,
                            confidence=0.9 if is_instantiation else 0.7,
                            weight=0.8
                        )
                        relationships.append(composition_rel)
        
        # Also check class-level variables with type hints (outside __init__)
        for class_symbol in class_index.values():
            class_node = self._find_class_node(ast_node, class_symbol)
            if not class_node:
                continue
            
            for node in class_node.body:
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    # Class variable with type hint: database: Database
                    field_name = node.target.id
                    field_type = None
                    module_prefix = None
                    is_class_optional = False  # Phase 5: reset per iteration
                    
                    if isinstance(node.annotation, ast.Name):
                        field_type = node.annotation.id
                    elif isinstance(node.annotation, ast.Attribute):
                        field_type = node.annotation.attr
                        if isinstance(node.annotation.value, ast.Name):
                            module_prefix = node.annotation.value.id
                    elif isinstance(node.annotation, ast.Subscript):
                        # Generic type: Optional[Type], List[Type], etc.
                        # Phase 5: detect Optional wrapper for AGGREGATION
                        outer_type = None
                        if isinstance(node.annotation.value, ast.Name):
                            outer_type = node.annotation.value.id
                        elif isinstance(node.annotation.value, ast.Attribute):
                            outer_type = node.annotation.value.attr
                        if outer_type == 'Optional':
                            is_class_optional = True
                        if isinstance(node.annotation.slice, ast.Name):
                            field_type = node.annotation.slice.id
                        elif isinstance(node.annotation.slice, ast.Attribute):
                            field_type = node.annotation.slice.attr
                            if isinstance(node.annotation.slice.value, ast.Name):
                                module_prefix = node.annotation.slice.value.id
                    
                    # Create FIELD symbol for class variable
                    field_full_name = f"{class_symbol.full_name}.{field_name}"
                    
                    # Check if we already created this field from __init__
                    already_exists = any(s.full_name == field_full_name for s in field_symbols)
                    if already_exists:
                        continue
                    
                    field_symbol = Symbol(
                        name=field_name,
                        symbol_type=SymbolType.FIELD,
                        scope=Scope.CLASS,
                        range=self._node_to_range(node),
                        file_path=file_path,
                        full_name=field_full_name,
                        parent_symbol=class_symbol.full_name,
                        metadata={
                            'field_type': field_type if field_type else 'unknown',
                            'is_class_variable': True,
                            'module_prefix': module_prefix
                        }
                    )
                    field_symbols.append(field_symbol)
                    
                    # Create DEFINES relationship: Class → Field
                    defines_rel = Relationship(
                        source_symbol=class_symbol.full_name,
                        target_symbol=field_full_name,
                        relationship_type=RelationshipType.DEFINES,
                        source_file=file_path,
                        target_file=file_path,
                        source_range=self._node_to_range(node),
                        annotations={'member_type': 'field', 'container_type': 'class', 'class_variable': True},
                        confidence=1.0,
                        weight=0.9
                    )
                    relationships.append(defines_rel)
                    
                    # Create COMPOSITION or AGGREGATION relationship for non-builtin types
                    if field_type and field_type not in builtin_types:
                        annotations = {
                            'field_name': field_name,
                            'class_variable': True
                        }
                        if module_prefix:
                            annotations['module_prefix'] = module_prefix
                        
                        # Phase 5: Optional class variable → AGGREGATION
                        rel_type = RelationshipType.AGGREGATION if is_class_optional else RelationshipType.COMPOSITION
                        
                        composition_rel = Relationship(
                            source_symbol=field_full_name,
                            target_symbol=field_type,
                            relationship_type=rel_type,
                            source_file=file_path,
                            target_file=file_path,
                            source_range=self._node_to_range(node),
                            annotations=annotations,
                            confidence=0.7,
                            weight=0.6
                        )
                        relationships.append(composition_rel)
        
        return field_symbols, relationships
    
    def _find_function_node(self, ast_node: ast.Module, method_symbol: Symbol) -> ast.FunctionDef:
        """Find the AST FunctionDef node for a given method symbol."""
        # First find the parent class
        parent_class_name = method_symbol.parent_symbol
        if not parent_class_name:
            return None
        
        # Find the class node
        class_node = None
        for node in ast.walk(ast_node):
            if isinstance(node, ast.ClassDef):
                # Match by simple name (last part of full name)
                if node.name == parent_class_name or parent_class_name.endswith(f".{node.name}"):
                    class_node = node
                    break
        
        if not class_node:
            return None
        
        # Now find the method within this class
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == method_symbol.name:
                    return node
        
        return None
    
    def _find_class_node(self, ast_node: ast.Module, class_symbol: Symbol) -> ast.ClassDef:
        """Find the AST ClassDef node for a given class symbol."""
        for node in ast.walk(ast_node):
            if isinstance(node, ast.ClassDef):
                if node.name == class_symbol.name:
                    return node
        return None
    
    def _node_to_range(self, node: ast.AST) -> Range:
        """Convert AST node to Range object"""
        start_line = getattr(node, 'lineno', 1)
        start_col = getattr(node, 'col_offset', 0)
        end_line = getattr(node, 'end_lineno', start_line)
        end_col = getattr(node, 'end_col_offset', start_col + 1)
        
        return Range(
            Position(start_line, start_col),
            Position(end_line, end_col)
        )
    
    def _extract_node_source(self, node: ast.AST) -> str:
        """Extract source code for an AST node"""
        try:
            lines = self._current_content.split('\n')
            start_line = getattr(node, 'lineno', 1) - 1
            end_line = getattr(node, 'end_lineno', start_line + 1)
            
            if start_line < len(lines) and end_line <= len(lines):
                return '\n'.join(lines[start_line:end_line])
        except Exception:
            pass
        return ""
    
    def _get_full_attribute_name(self, node: ast.Attribute) -> str:
        """Get full dotted name for attribute access"""
        parts = []
        current = node
        
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            parts.append(current.id)
        
        return '.'.join(reversed(parts))
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Extract decorator name"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return self._get_full_attribute_name(decorator)
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return self._get_full_attribute_name(decorator.func)
        return ""
    
    def _get_type_annotation(self, annotation: ast.AST) -> str:
        """Extract type annotation as string"""
        try:
            if isinstance(annotation, ast.Name):
                return annotation.id
            elif isinstance(annotation, ast.Attribute):
                return self._get_full_attribute_name(annotation)
            elif isinstance(annotation, ast.Constant):
                return str(annotation.value)
            elif isinstance(annotation, ast.Subscript):
                # Handle generics like List[str], Dict[str, int]
                value = self._get_type_annotation(annotation.value)
                if isinstance(annotation.slice, ast.Index):  # Python < 3.9
                    slice_value = self._get_type_annotation(annotation.slice.value)
                else:  # Python >= 3.9
                    slice_value = self._get_type_annotation(annotation.slice)
                return f"{value}[{slice_value}]"
            else:
                # Fallback to string representation
                return ast.unparse(annotation) if hasattr(ast, 'unparse') else str(annotation)
        except Exception:
            return "Any"
    
    def _extract_function_parameters(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract function parameters with type information"""
        parameters = []
        
        # Regular arguments
        for i, arg in enumerate(node.args.args):
            param_type = None
            if arg.annotation:
                param_type = self._get_type_annotation(arg.annotation)
            
            default_value = None
            defaults_offset = len(node.args.args) - len(node.args.defaults)
            if i >= defaults_offset:
                default_index = i - defaults_offset
                if default_index < len(node.args.defaults):
                    default_value = self._extract_default_value(node.args.defaults[default_index])
            
            parameters.append({
                'name': arg.arg,
                'type': param_type,
                'default': default_value,
                'vararg': False,
                'kwarg': False,
                'range': self._node_to_range(arg)
            })
        
        # *args parameter
        if node.args.vararg:
            param_type = None
            if node.args.vararg.annotation:
                param_type = self._get_type_annotation(node.args.vararg.annotation)
            
            parameters.append({
                'name': node.args.vararg.arg,
                'type': param_type,
                'default': None,
                'vararg': True,
                'kwarg': False,
                'range': self._node_to_range(node.args.vararg)
            })
        
        # **kwargs parameter
        if node.args.kwarg:
            param_type = None
            if node.args.kwarg.annotation:
                param_type = self._get_type_annotation(node.args.kwarg.annotation)
            
            parameters.append({
                'name': node.args.kwarg.arg,
                'type': param_type,
                'default': None,
                'vararg': False,
                'kwarg': True,
                'range': self._node_to_range(node.args.kwarg)
            })
        
        return parameters
    
    def _extract_default_value(self, node: ast.AST) -> str:
        """Extract default parameter value as string"""
        try:
            if isinstance(node, ast.Constant):
                return repr(node.value)
            elif isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return self._get_full_attribute_name(node)
            else:
                return ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
        except Exception:
            return "None"
    
    def _build_function_signature(self, node: ast.FunctionDef, return_type: Optional[str]) -> str:
        """Build function signature string"""
        try:
            params = []
            
            # Regular parameters
            for arg in node.args.args:
                param_str = arg.arg
                if arg.annotation:
                    param_str += f": {self._get_type_annotation(arg.annotation)}"
                params.append(param_str)
            
            # *args
            if node.args.vararg:
                vararg_str = f"*{node.args.vararg.arg}"
                if node.args.vararg.annotation:
                    vararg_str += f": {self._get_type_annotation(node.args.vararg.annotation)}"
                params.append(vararg_str)
            
            # **kwargs
            if node.args.kwarg:
                kwarg_str = f"**{node.args.kwarg.arg}"
                if node.args.kwarg.annotation:
                    kwarg_str += f": {self._get_type_annotation(node.args.kwarg.annotation)}"
                params.append(kwarg_str)
            
            signature = f"{node.name}({', '.join(params)})"
            if return_type:
                signature += f" -> {return_type}"
            
            return signature
        except Exception:
            return node.name
    
    def _extract_module_info(self, ast_tree: ast.Module) -> Tuple[List[str], List[str]]:
        """Extract imports and exports from module"""
        imports = []
        exports = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if alias.name == "*":
                        imports.append(f"{module}.*")
                    else:
                        imports.append(f"{module}.{alias.name}")
        
        # For exports, look for __all__ definition
        for node in ast.walk(ast_tree):
            if (isinstance(node, ast.Assign) and 
                len(node.targets) == 1 and 
                isinstance(node.targets[0], ast.Name) and 
                node.targets[0].id == "__all__"):
                
                if isinstance(node.value, ast.List):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Str):
                            exports.append(elt.s)
                        elif isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            exports.append(elt.value)
        
        return imports, exports
    
    def _extract_dependencies(self, imports: List[str], relationships: List[Relationship]) -> Set[str]:
        """Extract file dependencies from imports and relationships"""
        dependencies = set()
        
        # Add imported modules
        for imp in imports:
            # Extract top-level module name
            top_module = imp.split('.')[0]
            dependencies.add(top_module)
        
        # Add modules from cross-file relationships
        for rel in relationships:
            if '.' in rel.target_symbol:
                module_name = rel.target_symbol.split('.')[0]
                dependencies.add(module_name)
        
        return dependencies
    
    def _extract_type_name_from_annotation(self, annotation: ast.AST) -> Optional[str]:
        """
        Extract the primary type name from a type annotation for REFERENCES relationships.
        
        Handles various annotation patterns:
        - Simple types: User, str, int
        - Generics: List[User], Dict[str, User], Optional[User]
        - Module-qualified: models.User, typing.List
        
        Returns the base type name (e.g., 'User' from 'List[User]', 'User' from 'models.User')
        """
        if isinstance(annotation, ast.Name):
            # Simple type: User
            return annotation.id
        elif isinstance(annotation, ast.Attribute):
            # Module-qualified: models.User → extract 'User'
            return annotation.attr
        elif isinstance(annotation, ast.Subscript):
            # Generic type: List[User], Optional[User], Dict[str, User]
            # Extract from the subscript value (inner type)
            if isinstance(annotation.slice, ast.Name):
                return annotation.slice.id
            elif isinstance(annotation.slice, ast.Attribute):
                return annotation.slice.attr
            elif isinstance(annotation.slice, ast.Tuple):
                # Multiple type parameters: Dict[str, User]
                # Extract first non-builtin type
                for elt in annotation.slice.elts:
                    if isinstance(elt, ast.Name) and elt.id not in {'str', 'int', 'float', 'bool'}:
                        return elt.id
                    elif isinstance(elt, ast.Attribute):
                        return elt.attr
        
        return None
    
    # -- Phase 3: Recursive type decomposition (P-1) --
    # Built-in typing constructs whose base name should NOT emit REFERENCES
    # (their type arguments still get recursed into)
    _TYPING_BUILTINS = frozenset({
        'List', 'Dict', 'Set', 'Tuple', 'FrozenSet', 'Deque',
        'Optional', 'Union', 'Callable', 'Iterator', 'Generator',
        'AsyncIterator', 'AsyncGenerator', 'Coroutine',
        'ClassVar', 'Final', 'Literal', 'Annotated',
        'Type', 'Sequence', 'Mapping', 'MutableMapping', 'MutableSequence',
        'Iterable', 'Collection', 'MutableSet', 'OrderedDict', 'DefaultDict',
        'Counter', 'ChainMap',
        # Lower-case built-in generic forms (Python 3.9+)
        'list', 'dict', 'set', 'tuple', 'frozenset', 'type',
    })
    _PRIMITIVE_TYPES = frozenset({
        'str', 'int', 'float', 'bool', 'bytes', 'bytearray',
        'complex', 'memoryview', 'object', 'None', 'NoneType',
        'Any', 'NoReturn', 'Never',
    })

    def _extract_type_references_recursive(
        self,
        annotation: ast.AST,
        source_symbol: str,
        file_path: str,
        reference_type: str,
        relationships: list,
    ) -> None:
        """
        Recursively walk a Python type annotation AST and emit REFERENCES
        for every user-defined type encountered.

        Mirrors the TypeScript parser's ``_extract_type_references_recursive``.

        Handles:
        - ``ast.Name``  ............. simple type (``User``)
        - ``ast.Attribute``  ........ dotted type (``models.User``)
        - ``ast.Subscript``  ........ generic  (``List[User]``, ``Dict[str, User]``)
        - ``ast.BinOp`` (``|``)  .... PEP-604 union (``User | None``)
        - ``ast.Tuple``  ............ multi-arg (``Dict[str, User]`` slice)
        - ``ast.Constant``  ......... literal / ``None`` — skipped
        """
        if annotation is None:
            return

        if isinstance(annotation, ast.Name):
            name = annotation.id
            if name not in self._PRIMITIVE_TYPES and name not in self._TYPING_BUILTINS:
                relationships.append(Relationship(
                    source_symbol=source_symbol,
                    target_symbol=name,
                    relationship_type=RelationshipType.REFERENCES,
                    source_file=file_path,
                    source_range=self._node_to_range(annotation),
                    confidence=0.9,
                    weight=0.7,
                    annotations={'reference_type': reference_type},
                ))

        elif isinstance(annotation, ast.Attribute):
            # e.g. models.User — emit the leaf attribute
            leaf = annotation.attr
            if leaf not in self._PRIMITIVE_TYPES and leaf not in self._TYPING_BUILTINS:
                relationships.append(Relationship(
                    source_symbol=source_symbol,
                    target_symbol=leaf,
                    relationship_type=RelationshipType.REFERENCES,
                    source_file=file_path,
                    source_range=self._node_to_range(annotation),
                    confidence=0.9,
                    weight=0.7,
                    annotations={'reference_type': reference_type},
                ))

        elif isinstance(annotation, ast.Subscript):
            # Generic: List[User], Dict[str, User], Optional[User]
            # 1) Emit / skip the outer name
            outer = annotation.value
            outer_name = None
            if isinstance(outer, ast.Name):
                outer_name = outer.id
            elif isinstance(outer, ast.Attribute):
                outer_name = outer.attr

            # Emit the outer name only if it is user-defined
            if outer_name and outer_name not in self._TYPING_BUILTINS and outer_name not in self._PRIMITIVE_TYPES:
                relationships.append(Relationship(
                    source_symbol=source_symbol,
                    target_symbol=outer_name,
                    relationship_type=RelationshipType.REFERENCES,
                    source_file=file_path,
                    source_range=self._node_to_range(outer),
                    confidence=0.9,
                    weight=0.7,
                    annotations={'reference_type': f'{reference_type}_generic'},
                ))

            # 2) Recurse into the slice (type arguments)
            slc = annotation.slice
            arg_ref = f'{reference_type}_generic_arg'
            if isinstance(slc, ast.Tuple):
                for elt in slc.elts:
                    self._extract_type_references_recursive(elt, source_symbol, file_path, arg_ref, relationships)
            else:
                self._extract_type_references_recursive(slc, source_symbol, file_path, arg_ref, relationships)

        elif isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            # PEP 604 union: X | Y | Z
            self._extract_type_references_recursive(annotation.left, source_symbol, file_path, reference_type, relationships)
            self._extract_type_references_recursive(annotation.right, source_symbol, file_path, reference_type, relationships)

        elif isinstance(annotation, ast.Tuple):
            # Raw tuple of types (fallback)
            for elt in annotation.elts:
                self._extract_type_references_recursive(elt, source_symbol, file_path, reference_type, relationships)

        elif isinstance(annotation, ast.List):
            # Callable[[ArgType1, ArgType2], ReturnType] uses ast.List for the arg types
            for elt in annotation.elts:
                self._extract_type_references_recursive(elt, source_symbol, file_path, reference_type, relationships)

        elif isinstance(annotation, ast.Constant):
            # Literal / None — nothing to emit
            pass

    def _extract_module_docstring(self, ast_tree: ast.Module) -> Optional[str]:
        """Extract module-level docstring"""
        return ast.get_docstring(ast_tree)
    
    def parse_multiple_files(self, file_paths: List[str], max_workers: int = 4) -> Dict[str, ParseResult]:
        """
        Parse multiple Python files and resolve cross-file relationships.
        
        This method enables proper cross-file symbol resolution by:
        1. First parsing all files individually IN PARALLEL
        2. Building a global symbol registry across all files 
        3. Enhancing relationships with cross-file target information
        
        Args:
            file_paths: List of file paths to parse
            max_workers: Maximum number of parallel workers for file parsing
        """
        results = {}
        
        # Clear and rebuild global symbol registries for cross-file resolution
        # These will be available to extract_relationships() for constructor detection
        self._global_class_locations.clear()
        self._global_function_locations.clear()
        self._global_symbol_registry.clear()
        
        total = len(file_paths)
        # First pass: Parse each file individually IN PARALLEL and build global symbol registry
        logger.info(f"Parsing {total} Python files for cross-file analysis with {max_workers} workers")
        # Plugin infrastructure - only available when running as part of the system
        # if this and getattr(this, 'module', None):
        #     this.module.invocation_thinking(
        #         f"I am on phase parsing\nStarting Python multi-file parsing: {total} files\nReasoning: Need raw AST + symbol inventory (defs, imports, calls) to enable precise cross-file resolution and later relationship enhancement.\nNext: Parse files in parallel (~10% progress intervals) then assemble global symbol registry."
        #     )
        results = self._parse_files_parallel(file_paths, max_workers, emit_thinking=True)
        # Plugin infrastructure - only available when running as part of the system
        # if this and getattr(this, 'module', None):
        #     succeeded = sum(1 for r in results.values() if r.symbols)
        #     pct = int((succeeded/total)*100) if total else 100
        #     this.module.invocation_thinking(
        #         f"I am on phase parsing\nRaw Python parsing complete: {succeeded}/{total} files produced symbols ({pct}%)\nReasoning: We now have enough per-file symbol context to unify names and disambiguate cross-file targets.\nNext: Populate consolidated global symbol tables (classes, functions, variables)."
        #     )
        
        # Build global symbol registry from all parsed results
        # Plugin infrastructure - only available when running as part of the system
        # if this and getattr(this, 'module', None):
        #     this.module.invocation_thinking(
        #         "I am on phase parsing\nBuilding global Python symbol tables (classes, functions, variables)\nReasoning: Central index accelerates cross-file relationship resolution & prevents duplicate ambiguous bindings.\nNext: Resolve cross-file references (imports, call targets)."
        #     )
        for file_path, result in results.items():
            if result.symbols:  # Only process files that parsed successfully
                self._extract_global_symbols(file_path, result)
        
        logger.info(f"Global class registry built: {len(self._global_class_locations)} classes, "
                   f"{len(self._global_function_locations)} functions")
        
        # Re-extract relationships with global class registry for cross-file constructor detection
        logger.info("Re-extracting relationships with global class registry for constructor detection")
        for file_path, result in results.items():
            if result.symbols:  # Only process files that parsed successfully
                # Re-read and parse the file to get AST
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    ast_tree = ast.parse(content, filename=file_path)
                    
                    # Re-extract relationships (now with global class registry populated)
                    new_relationships, new_field_symbols = self.extract_relationships(ast_tree, result.symbols, file_path)
                    
                    # Add new field symbols to result
                    # (avoid duplicates by checking full_name)
                    existing_full_names = {s.full_name for s in result.symbols if s.full_name}
                    for field_sym in new_field_symbols:
                        if field_sym.full_name and field_sym.full_name not in existing_full_names:
                            result.symbols.append(field_sym)
                            existing_full_names.add(field_sym.full_name)
                    
                    # Replace old relationships with new ones
                    result.relationships.clear()
                    result.relationships.extend(new_relationships)
                    
                except Exception as e:
                    logger.warning(f"Failed to re-extract relationships for {file_path}: {e}")
        
        # Plugin infrastructure - only available when running as part of the system
        # if this and getattr(this, 'module', None):
        #     this.module.invocation_thinking(
        #         "I am on phase parsing\nGlobal registry ready\nReasoning: All symbol origins mapped.\nNext: Augment relationships with resolved import & call targets across files."
        #     )
        
        # Second pass: Enhance relationships with cross-file resolution
        logger.info("Enhancing relationships with cross-file resolution")
        # Plugin infrastructure - only available when running as part of the system
        # if this and getattr(this, 'module', None):
        #     this.module.invocation_thinking(
        #         "I am on phase parsing\nEnhancing Python relationships with cross-file targets\nReasoning: Replace generic references with fully qualified targets for accurate graph semantics.\nNext: Finalize results & report aggregate coverage."
        #     )
        for file_path, result in results.items():
            if result.symbols:  # Only process files that parsed successfully
                self._enhance_cross_file_relationships(file_path, result)
        # Plugin infrastructure - only available when running as part of the system
        # if this and getattr(this, 'module', None):
        #     this.module.invocation_thinking(
        #         f"I am on phase parsing\nPython parsing complete: {len(results)}/{total} files (100%)\nReasoning: Multi-pass enrichment finished; symbol + relationship graph is ready for downstream documentation & search.\nNext: Upstream components may now construct language-agnostic graph layers."
        #     )
        
        return results
    
    def _parse_files_parallel(self, file_paths: List[str], max_workers: int, emit_thinking: bool = False) -> Dict[str, ParseResult]:
        """
        Parse multiple Python files in parallel using ThreadPoolExecutor with batching.
        
        ThreadPoolExecutor is used instead of ProcessPoolExecutor because:
        - Lower memory overhead (shared memory)
        - No serialization overhead
        - Tree-sitter and AST release GIL, so we get true parallelism
        - More stable with large workloads (no SystemExit issues)
        
        For very large codebases (>1000 files), we process in batches to:
        - Prevent memory exhaustion
        - Provide progress feedback
        - Allow graceful degradation if issues occur
        
        Args:
            file_paths: List of file paths to parse
            max_workers: Maximum number of parallel workers
            emit_thinking: Whether to emit thinking updates
            
        Returns:
            Dictionary mapping file path to ParseResult
        """
        import concurrent.futures
        import time
        
        results = {}
        total = len(file_paths)
        
        # Batch size: process 500 files at a time to prevent memory issues
        batch_size = 500
        num_batches = (total + batch_size - 1) // batch_size
        
        if num_batches > 1:
            logger.info(f"Processing {total} Python files in {num_batches} batch(es) of up to {batch_size} files each")
        
        start_time = time.time()
        
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
                    future_to_file = {}
                    for fp in batch_files:
                        try:
                            future = executor.submit(_parse_single_python_file, fp)
                            future_to_file[future] = fp
                        except Exception as e:
                            logger.error(f"Failed to submit file {fp}: {type(e).__name__}: {e}")
                            results[fp] = ParseResult(
                                file_path=fp,
                                language="python",
                                symbols=[],
                                relationships=[],
                                errors=[f"Submit failed: {str(e)}"]
                            )
                    
                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            # Plugin infrastructure - only available when running as part of the system
                            # this.module.invocation_stop_checkpoint()
                            
                            # _parse_single_python_file returns (file_path, result)
                            parsed_file_path, result = future.result(timeout=60)  # 60 second timeout per file
                            results[parsed_file_path] = result
                            
                            # Progress reporting
                            completed = len(results)
                            if completed % max(1, int(total / 10)) == 0 or completed == total:
                                elapsed = time.time() - start_time
                                logger.info(f"Parsed {completed}/{total} Python files in {elapsed:.1f}s...")
                                # Plugin infrastructure - only available when running as part of the system
                                # if emit_thinking and this and getattr(this, 'module', None):
                                #     pct = int((completed/total)*100) if total else 100
                                #     this.module.invocation_thinking(
                                #         f"I am on phase parsing\nPython raw parse progress: {completed}/{total} files ({pct}%)\nReasoning: Accumulating per-file symbols to reach critical mass for reliable global registry.\nNext: Continue parallel parsing until 100%, then consolidate symbols."
                                #     )
                        except concurrent.futures.TimeoutError:
                            logger.error(f"Timeout parsing Python file (>60s): {file_path}")
                            results[file_path] = ParseResult(
                                file_path=file_path,
                                language="python",
                                symbols=[],
                                relationships=[],
                                errors=["Timeout during parsing"]
                            )
                        except Exception as e:
                            logger.error(f"Failed to parse Python file {file_path}: {type(e).__name__}: {e}", exc_info=True)
                            results[file_path] = ParseResult(
                                file_path=file_path,
                                language="python",
                                symbols=[],
                                relationships=[],
                                errors=[f"Parse error: {str(e)}"]
                            )
            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} failed: {type(e).__name__}: {e}", exc_info=True)
                # Process remaining files in this batch sequentially as fallback
                for fp in batch_files:
                    if fp not in results:
                        try:
                            parsed_file_path, result = _parse_single_python_file(fp)
                            results[parsed_file_path] = result
                        except Exception as inner_e:
                            logger.error(f"Sequential fallback also failed for {fp}: {inner_e}")
                            results[fp] = ParseResult(
                                file_path=fp,
                                language="python",
                                symbols=[],
                                relationships=[],
                                errors=[f"Batch and fallback failed: {str(inner_e)}"]
                            )
        
        parse_time = time.time() - start_time
        successful_files = len([r for r in results.values() if r.symbols])
        logger.info(f"Parallel Python parsing complete: {successful_files}/{total} files parsed in {parse_time:.2f}s")
        
        return results
    
    def _extract_global_symbols(self, file_path: str, result: ParseResult):
        """Extract symbols for global cross-file resolution."""
        file_name = Path(file_path).stem
        
        # Build import mapping for this file
        imports = {}
        for rel in result.relationships:
            if rel.relationship_type == RelationshipType.IMPORTS:
                # Handle different import patterns
                if "." in rel.target_symbol:
                    # from module.submodule import Symbol or import module.Symbol
                    parts = rel.target_symbol.split(".")
                    symbol_name = parts[-1]
                    imports[symbol_name] = rel.target_symbol
                else:
                    # Simple import: import Symbol
                    imports[rel.target_symbol] = rel.target_symbol
        
        # Store class and function locations globally
        for symbol in result.symbols:
            if symbol.symbol_type == SymbolType.CLASS:
                # Store class location
                self._global_class_locations[symbol.name] = file_path
                
                # Also store in general registry with file qualifier
                self._global_symbol_registry[symbol.name] = f"{file_name}.{symbol.name}"
                
            elif symbol.symbol_type == SymbolType.FUNCTION:
                # Store function location
                self._global_function_locations[symbol.name] = file_path
                
                # Also store in general registry with file qualifier
                self._global_symbol_registry[symbol.name] = f"{file_name}.{symbol.name}"
            
            elif symbol.symbol_type in [SymbolType.VARIABLE, SymbolType.CONSTANT]:
                # Store variables and constants
                self._global_symbol_registry[symbol.name] = f"{file_name}.{symbol.name}"
        
        # Store imported symbols with their file locations
        for local_name, full_name in imports.items():
            # Try to resolve the import to an actual file
            if "." in full_name:
                parts = full_name.split(".")
                module_name = parts[0]
                
                # Check if this corresponds to one of our files
                for other_file_path in [fp for fp in self._global_class_locations.values() 
                                      if Path(fp).stem == module_name]:
                    self._global_symbol_registry[local_name] = f"{module_name}.{local_name}"
    
    def _enhance_cross_file_relationships(self, file_path: str, result: ParseResult):
        """Enhance relationships with cross-file information."""
        enhanced_relationships = []
        file_name = Path(file_path).stem
        
        # Build imports map for this file
        imports = {}
        for rel in result.relationships:
            if rel.relationship_type == RelationshipType.IMPORTS:
                if "." in rel.target_symbol:
                    parts = rel.target_symbol.split(".")
                    symbol_name = parts[-1]
                    imports[symbol_name] = rel.target_symbol
                else:
                    imports[rel.target_symbol] = rel.target_symbol
        
        for rel in result.relationships:
            enhanced_rel = self._enhance_relationship(rel, file_path, imports)
            enhanced_relationships.append(enhanced_rel)
        
        # Replace with enhanced relationships
        result.relationships.clear()
        result.relationships.extend(enhanced_relationships)
    
    def _enhance_relationship(self, rel: Relationship, source_file: str, imports: Dict[str, str]) -> Relationship:
        """Enhance a single relationship with cross-file target information."""
        
        # For inheritance relationships, try to resolve target to actual file
        if rel.relationship_type == RelationshipType.INHERITANCE:
            target_symbol = rel.target_symbol
            
            # Check if the target is an imported symbol
            if target_symbol in imports:
                # This is an imported class being inherited from
                import_path = imports[target_symbol]
                
                # Try to find the actual file that defines this class
                if target_symbol in self._global_class_locations:
                    target_file = self._global_class_locations[target_symbol]
                    
                    # Update relationship with target file information
                    return Relationship(
                        source_symbol=rel.source_symbol,
                        target_symbol=target_symbol,
                        relationship_type=rel.relationship_type,
                        source_file=rel.source_file,
                        target_file=target_file,
                        source_range=rel.source_range,
                        confidence=rel.confidence,
                        weight=rel.weight,
                        context=rel.context,
                        annotations={**rel.annotations, 'cross_file': True, 'import_path': import_path}
                    )
        
        # For calls and references, try to resolve targets
        elif rel.relationship_type in [RelationshipType.CALLS, RelationshipType.REFERENCES]:
            target_symbol = rel.target_symbol
            
            # Check if target is a known symbol from another file
            if target_symbol in self._global_class_locations:
                    target_file = self._global_class_locations[target_symbol]
                    if target_file != source_file:  # Only if it's actually cross-file
                        return Relationship(
                            source_symbol=rel.source_symbol,
                            target_symbol=target_symbol,
                            relationship_type=rel.relationship_type,
                            source_file=rel.source_file,
                            target_file=target_file,
                            source_range=rel.source_range,
                            confidence=rel.confidence,
                            weight=rel.weight,
                            context=rel.context,
                            annotations={**rel.annotations, 'cross_file': True}
                        )
            
            elif target_symbol in self._global_function_locations:
                target_file = self._global_function_locations[target_symbol]
                if target_file != source_file:  # Only if it's actually cross-file
                    return Relationship(
                        source_symbol=rel.source_symbol,
                        target_symbol=target_symbol,
                        relationship_type=rel.relationship_type,
                        source_file=rel.source_file,
                        target_file=target_file,
                        source_range=rel.source_range,
                        confidence=rel.confidence,
                        weight=rel.weight,
                        context=rel.context,
                        annotations={**rel.annotations, 'cross_file': True}
                    )
        
        # Return the original relationship if no enhancement was needed
        return rel
