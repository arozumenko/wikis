"""
Clean Java visitor-based parser using Tree-sitter for Wiki Toolkit Graph Enhancement.

This parser implements a simple visitor pattern for Java AST traversal, focusing on
extracting symbols and relationships for cross-file analysis. Scopes are opened/closed
only for major entities (classes, methods, etc.), not for every block.

Key Features:
- Real tree-sitter AST parsing (no mocks)
- Simple visitor pattern (no cursor API)
- Scope management for major entities only
- Cross-file relationship extraction
- Clean, maintainable code structure
"""

import logging
import time
from typing import List, Dict, Optional, Set, Union, Tuple
from pathlib import Path

from tree_sitter import Node
from tree_sitter_language_pack import get_language, get_parser

# try:
#     from tree_sitter_languages import get_language, get_parser
#     from tree_sitter import Node
#     TREE_SITTER_AVAILABLE = True
# except ImportError:
#     TREE_SITTER_AVAILABLE = False
#     # Create a placeholder for type hints when tree-sitter is not available
#     class Node:
#         pass

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


def _parse_single_java_file(file_path: str) -> Tuple[str, ParseResult]:
    """Parse a single Java file - used by parallel workers"""
    try:
        # Create a new parser instance for this worker
        parser = JavaVisitorParser()
        result = parser.parse_file(file_path)
        return file_path, result
    except Exception as e:
        logger.warning(f"Failed to parse {file_path}: {e}")
        # Create empty result for failed files
        return file_path, ParseResult(
            file_path=file_path,
            language="java",
            symbols=[],
            relationships=[]
        )


class JavaVisitorParser(BaseParser):
    """Clean Java parser using visitor pattern with real Tree-sitter."""

    def __init__(self):
        super().__init__("java")
        self.parser = None
        self.ts_language = None  # Tree-sitter language object
        self._current_file = ""
        self._current_content = ""
        self._symbols = []
        self._relationships = []
        self._scope_stack = []  # Stack of scope names for qualified names

        # Cross-file resolution support
        self._current_package = ""
        self._current_imports = {}  # simple_name -> full_qualified_name
        self._field_types = {}      # field_name -> type_name
        self._variable_types = {}   # variable_name -> type_name

        # Global symbol tables (will be populated by multi-file parsing)
        self._global_class_locations = {}  # class_name -> file_path
        self._global_method_locations = {}  # method_name -> [(class_name, file_path)]

        self._setup_tree_sitter()

    def _define_capabilities(self) -> LanguageCapabilities:
        """Define Java parser capabilities."""
        return LanguageCapabilities(
            language="java",
            supported_symbols={
                SymbolType.CLASS, SymbolType.INTERFACE, SymbolType.METHOD,
                SymbolType.CONSTRUCTOR, SymbolType.FIELD, SymbolType.VARIABLE,
                SymbolType.CONSTANT, SymbolType.ENUM, SymbolType.ANNOTATION,
                SymbolType.PARAMETER, SymbolType.NAMESPACE, SymbolType.FUNCTION,
                SymbolType.MODULE
            },
            supported_relationships={
                RelationshipType.INHERITANCE, RelationshipType.IMPLEMENTATION,
                RelationshipType.CALLS, RelationshipType.IMPORTS,
                RelationshipType.COMPOSITION, RelationshipType.CREATES,
                RelationshipType.DEFINES, RelationshipType.ANNOTATES,
                RelationshipType.REFERENCES,
                RelationshipType.AGGREGATION,  # Phase 5: weak-ownership for Optional/Nullable fields
                RelationshipType.OVERRIDES,    # Phase 5: @Override method detection
            },
            supports_ast_parsing=True,
            supports_type_inference=True,
            supports_cross_file_analysis=True,
            has_classes=True,
            has_interfaces=True,
            has_namespaces=False,
            has_generics=True,
            has_decorators=False,
            has_annotations=True,
            max_file_size_mb=50.0,
            typical_parse_time_ms=25.0
        )

    def _get_supported_extensions(self) -> Set[str]:
        """Get supported Java file extensions."""
        return {'.java'}

    def _setup_tree_sitter(self) -> bool:
        """Setup real tree-sitter Java parser."""
        # if not TREE_SITTER_AVAILABLE:
        #     logger.warning("Tree-sitter not available, falling back to basic parsing")
        #     return False

        try:
            self.ts_language = get_language('java')
            self.parser = get_parser('java')
            return True
        except Exception as e:
            logger.error(f"Failed to initialize tree-sitter Java parser: {e}")
            return False

    def parse_file(self, file_path: Union[str, Path], content: Optional[str] = None) -> ParseResult:
        """
        Parse Java file using tree-sitter visitor pattern.

        Args:
            file_path: Path to the Java file
            content: File content as string

        Returns:
            ParseResult with symbols and relationships
        """
        start_time = time.time()
        file_path = str(file_path)
        self._current_file = file_path

        # Reset state
        self._symbols = []
        self._relationships = []
        self._scope_stack = []
        self._current_package = ""
        self._current_imports = {}
        self._field_types = {}
        self._variable_types = {}
        self._parameter_types = {}

        try:
            # Read content if not provided
            if content is None:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

            self._current_content = content

            # Use tree-sitter if available, otherwise fallback
            if self.parser and self.ts_language:
                tree = self.parser.parse(bytes(content, "utf8"))
                
                # Check if there's a package declaration
                # If not, use file name as the module scope for standardized qualified names
                has_package = self._has_package_declaration(tree.root_node)
                if not has_package:
                    # No package - use file name as scope for qualified names
                    # Don't emit a MODULE symbol (organizational, not architectural)
                    module_name = Path(file_path).stem
                    # Push to scope so classes get proper parent
                    self._enter_scope(module_name)
                
                self.visit_node(tree.root_node)
            else:
                return self._fallback_parse(file_path, content)

            # Extract DEFINES relationships (Pass 2: after symbols extracted)
            defines_relationships = self._extract_defines_relationships(self._symbols, file_path)
            self._relationships.extend(defines_relationships)

            parse_time = time.time() - start_time

            return ParseResult(
                file_path=file_path,
                language="java",
                symbols=self._symbols,
                relationships=self._relationships,
                parse_time=parse_time
            )

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return ParseResult(
                file_path=file_path,
                language="java",
                symbols=[],
                relationships=[],
                parse_time=0.0,
                errors=[f"File not found: {file_path}"]
            )
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return ParseResult(
                file_path=file_path,
                language="java",
                symbols=[],
                relationships=[],
                parse_time=0.0,
                errors=[str(e)]
            )

    # ============================================================================
    # VISITOR PATTERN IMPLEMENTATION
    # ============================================================================

    def visit_node(self, node: Node) -> None:
        """Visit a node and dispatch to appropriate handler."""
        if not node:
            return

        # Dispatch to specific handlers based on node type
        if node.type == "package_declaration":
            self._handle_package_declaration(node)
        elif node.type == "import_declaration":
            self._handle_import_declaration(node)
        elif node.type == "class_declaration":
            self._handle_class_declaration(node)
        elif node.type == "interface_declaration":
            self._handle_interface_declaration(node)
        elif node.type == "enum_declaration":
            self._handle_enum_declaration(node)
        elif node.type == "annotation_type_declaration":
            self._handle_annotation_declaration(node)
        elif node.type == "record_declaration":
            self._handle_record_declaration(node)
        elif node.type == "method_declaration":
            self._handle_method_declaration(node)
        elif node.type == "constructor_declaration":
            self._handle_constructor_declaration(node)
        elif node.type == "compact_constructor_declaration":
            self._handle_compact_constructor_declaration(node)
        elif node.type == "field_declaration":
            self._handle_field_declaration(node)
        elif node.type == "local_variable_declaration":
            self._handle_local_variable_declaration(node)
        elif node.type == "method_invocation":
            self._handle_method_invocation(node)
        elif node.type == "object_creation_expression":
            self._handle_object_creation(node)
        elif node.type == "super_interfaces":
            # Skip - already handled in _extract_inheritance_from_class
            pass
        elif node.type == "superclass":
            # Skip - already handled in _extract_inheritance_from_class
            pass
        else:
            # For other nodes, just visit children
            self._visit_children(node)

    def _visit_children(self, node: Node) -> None:
        """Visit all children of a node."""
        if hasattr(node, 'children'):
            for child in node.children:
                self.visit_node(child)

    # ============================================================================
    # SCOPE MANAGEMENT
    # ============================================================================

    def _enter_scope(self, name: str) -> None:
        """Enter a new scope (only for major entities)."""
        self._scope_stack.append(name)

    def _exit_scope(self) -> None:
        """Exit the current scope."""
        if self._scope_stack:
            self._scope_stack.pop()

    def _get_current_scope_path(self) -> str:
        """Get current qualified scope path."""
        return '.'.join(self._scope_stack) if self._scope_stack else ""

    def _build_qualified_name(self, name: str) -> str:
        """Build qualified name based on current scope."""
        scope_path = self._get_current_scope_path()
        return f"{scope_path}.{name}" if scope_path else name

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def _get_node_text(self, node: Node) -> str:
        """Get text content of a node."""
        if not node or not hasattr(node, 'text'):
            return ""
        return node.text.decode('utf-8')

    def _byte_to_line_col(self, byte_offset: int) -> tuple[int, int]:
        """Convert byte offset to line and column."""
        content = self._current_content
        lines_before = content[:byte_offset].count('\n')
        line_start = content.rfind('\n', 0, byte_offset) + 1
        column = byte_offset - line_start
        return lines_before + 1, column

    def _create_position(self, node: Node) -> Position:
        """Create position from node."""
        line, col = self._byte_to_line_col(node.start_byte)
        return Position(line=line, column=col)

    def _create_range(self, node: Node) -> Range:
        """Create range from node."""
        start_line, start_col = self._byte_to_line_col(node.start_byte)
        end_line, end_col = self._byte_to_line_col(node.end_byte)
        return Range(
            start=Position(line=start_line, column=start_col),
            end=Position(line=end_line, column=end_col)
        )

    def _has_package_declaration(self, root_node: Node) -> bool:
        """Check if the file has a package declaration."""
        if not hasattr(root_node, 'children'):
            return False
        for child in root_node.children:
            if child.type == 'package_declaration':
                return True
        return False

    def _find_child_by_type(self, node: Node, child_type: str) -> Optional[Node]:
        """Find first child of specific type."""
        if not hasattr(node, 'children'):
            return None

        for child in node.children:
            if child.type == child_type:
                return child
        return None

    def _find_children_by_types(self, node: Node, child_types: Set[str]) -> List[Node]:
        """Find all children of specific types."""
        if not hasattr(node, 'children'):
            return []

        result = []
        for child in node.children:
            if child.type in child_types:
                result.append(child)
        return result

    # ============================================================================
    # NODE HANDLERS
    # ============================================================================

    def _handle_package_declaration(self, node: Node) -> None:
        """Handle package declaration.
        
        We track the package in scope for qualified names, but don't emit
        a NAMESPACE symbol (packages are organizational, not architectural).
        """
        # Find the package name
        for child in node.children:
            if child.type in ["scoped_identifier", "identifier"]:
                package_name = self._get_node_text(child)
                if package_name:
                    # Enter package scope for qualified names (but no symbol emitted)
                    self._enter_scope(package_name)
                break

    def _handle_import_declaration(self, node: Node) -> None:
        """Handle import statements."""
        # Find the imported item
        for child in node.children:
            if child.type in ["scoped_identifier", "identifier"]:
                import_path = self._get_node_text(child)
                if import_path:
                    # Store import mapping for type resolution
                    if '.' in import_path:
                        parts = import_path.split('.')
                        class_name = parts[-1]
                        self._current_imports[class_name] = import_path

                    # Create import relationship
                    source_symbol = self._get_current_scope_path() or Path(self._current_file).stem

                    relationship = Relationship(
                        source_symbol=source_symbol,
                        target_symbol=import_path,
                        relationship_type=RelationshipType.IMPORTS,
                        source_file=self._current_file,
                        target_file=self._current_file  # Will be resolved later
                    )
                    self._relationships.append(relationship)
                break

    def _handle_class_declaration(self, node: Node) -> None:
        """Handle class declarations."""
        class_name = None

        # Find class name
        for child in node.children:
            if child.type == "identifier":
                class_name = self._get_node_text(child)
                break

        if not class_name:
            return

        # Create class symbol
        parent_symbol = self._get_current_scope_path() if self._scope_stack else None
        qualified_name = self._build_qualified_name(class_name)

        symbol = Symbol(
            name=class_name,
            symbol_type=SymbolType.CLASS,
            file_path=self._current_file,
            range=self._create_range(node),
            scope=Scope.CLASS,
            parent_symbol=parent_symbol,
            full_name=qualified_name,
            source_text=self._extract_node_source(node)
        )
        self._symbols.append(symbol)

        # Extract inheritance relationships
        self._extract_inheritance_from_class(node, symbol)

        # Enter class scope
        self._enter_scope(class_name)

        # Visit children (methods, fields, etc.)
        self._visit_children(node)

        # Exit class scope
        self._exit_scope()

    def _handle_interface_declaration(self, node: Node) -> None:
        """Handle interface declarations."""
        interface_name = None

        # Find interface name
        for child in node.children:
            if child.type == "identifier":
                interface_name = self._get_node_text(child)
                break

        if not interface_name:
            return

        # Create interface symbol
        parent_symbol = self._get_current_scope_path() if self._scope_stack else None
        qualified_name = self._build_qualified_name(interface_name)

        symbol = Symbol(
            name=interface_name,
            symbol_type=SymbolType.INTERFACE,
            file_path=self._current_file,
            range=self._create_range(node),
            scope=Scope.CLASS,
            parent_symbol=parent_symbol,
            full_name=qualified_name,
            source_text=self._extract_node_source(node)
        )
        self._symbols.append(symbol)

        # Extract inheritance relationships (interfaces extending other interfaces)
        self._extract_inheritance_from_class(node, symbol)

        # Enter interface scope
        self._enter_scope(interface_name)

        # Visit children
        self._visit_children(node)

        # Exit interface scope
        self._exit_scope()

    def _handle_enum_declaration(self, node: Node) -> None:
        """Handle enum declarations."""
        enum_name = None

        # Find enum name
        for child in node.children:
            if child.type == "identifier":
                enum_name = self._get_node_text(child)
                break

        if not enum_name:
            return

        # Create enum symbol
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
            source_text=self._extract_node_source(node)
        )
        self._symbols.append(symbol)

        # Enter enum scope
        self._enter_scope(enum_name)

        # Visit children
        self._visit_children(node)

        # Exit enum scope
        self._exit_scope()

    def _handle_method_declaration(self, node: Node) -> None:
        """Handle method declarations."""
        method_name = None
        return_type = None
        is_override = False

        # Find method name, return type, and @Override annotation
        for child in node.children:
            if child.type == "identifier":
                method_name = self._get_node_text(child)
            elif child.type in ["type_identifier", "generic_type", "array_type", "void_type"]:
                return_type = self._extract_type_name(child)
            elif child.type == "modifiers":
                # Check for @Override marker annotation
                for mod_child in child.children:
                    if mod_child.type == "marker_annotation":
                        ann_name = self._find_child_by_type(mod_child, "identifier")
                        if ann_name and self._get_node_text(ann_name) == "Override":
                            is_override = True

        if not method_name:
            return

        # Extract parameter types before creating symbol
        parameter_types = []
        for child in node.children:
            if child.type == "formal_parameters":
                parameter_types = self._extract_parameter_types(child)
                break

        # Create method symbol
        parent_symbol = self._get_current_scope_path() if self._scope_stack else None
        qualified_name = self._build_qualified_name(method_name)

        metadata = {}
        if is_override:
            metadata['is_override'] = True

        symbol = Symbol(
            name=method_name,
            symbol_type=SymbolType.METHOD,
            file_path=self._current_file,
            range=self._create_range(node),
            scope=Scope.FUNCTION,
            parent_symbol=parent_symbol,
            full_name=qualified_name,
            return_type=return_type,
            parameter_types=parameter_types,
            source_text=self._extract_node_source(node),
            metadata=metadata
        )
        self._symbols.append(symbol)

        # Phase 5: Emit OVERRIDES relationship for @Override methods
        if is_override and parent_symbol:
            # Find the parent class's superclass from existing INHERITANCE relationships
            parent_class_name = self._find_superclass(parent_symbol)
            if parent_class_name:
                overrides_rel = Relationship(
                    source_symbol=qualified_name,
                    target_symbol=f"{parent_class_name}.{method_name}",
                    relationship_type=RelationshipType.OVERRIDES,
                    source_file=self._current_file,
                    target_file=self._current_file,  # Will be resolved in multi-file pass
                    annotations={'override_annotation': '@Override'}
                )
                self._relationships.append(overrides_rel)

        # Create reference relationship for return type (if not built-in and not void)
        if return_type and return_type != "void" and not self._is_builtin_type(return_type):
            relationship = Relationship(
                source_symbol=qualified_name,
                target_symbol=return_type,
                relationship_type=RelationshipType.REFERENCES,
                source_file=self._current_file,
                target_file=self._current_file  # Will be resolved later
            )
            self._relationships.append(relationship)

        # Extract generic type parameters from return type
        for child in node.children:
            if child.type == "generic_type":
                generic_params = self._extract_generic_type_parameters(child)
                for param_type in generic_params:
                    param_relationship = Relationship(
                        source_symbol=qualified_name,
                        target_symbol=param_type,
                        relationship_type=RelationshipType.REFERENCES,
                        source_file=self._current_file,
                        target_file=self._current_file  # Will be resolved later
                    )
                    self._relationships.append(param_relationship)
                break

        # Handle method parameters
        for child in node.children:
            if child.type == "formal_parameters":
                self._handle_method_parameters(child, qualified_name)
                break

        # Enter method scope
        self._enter_scope(method_name)

        # Visit children to find method calls and other relationships
        self._visit_children(node)

        # Exit method scope
        self._exit_scope()

    def _extract_parameter_types(self, formal_params_node: Node) -> List[str]:
        """Extract parameter types from formal parameters node."""
        parameter_types = []
        
        for child in formal_params_node.children:
            if child.type == "formal_parameter":
                # Extract parameter type (including primitives)
                for param_child in child.children:
                    if param_child.type in ["type_identifier", "generic_type", "array_type", 
                                            "boolean_type", "integral_type", "floating_point_type", "void_type"]:
                        param_type = self._extract_type_name(param_child)
                        if param_type:
                            parameter_types.append(param_type)
                        break
        
        return parameter_types

    def _handle_method_parameters(self, formal_params_node: Node, method_qualified_name: str) -> None:
        """Handle method parameters and create reference relationships."""
        for child in formal_params_node.children:
            if child.type == "formal_parameter":
                param_type = None
                param_name = None

                # Extract parameter type and name
                for param_child in child.children:
                    if param_child.type in ["type_identifier", "generic_type", "array_type"]:
                        param_type = self._extract_type_name(param_child)
                    elif param_child.type == "identifier":
                        param_name = self._get_node_text(param_child)

                # Create reference relationship for parameter base type (including standard library types for documentation)
                if param_type:
                    relationship = Relationship(
                        source_symbol=method_qualified_name,
                        target_symbol=param_type,
                        relationship_type=RelationshipType.REFERENCES,
                        source_file=self._current_file,
                        target_file=self._current_file,  # Will be resolved later
                        annotations={'context': 'parameter', 'param_name': param_name or 'unknown'}
                    )
                    self._relationships.append(relationship)

                # Extract generic type parameters from parameter type
                for param_child in child.children:
                    if param_child.type == "generic_type":
                        generic_params = self._extract_generic_type_parameters(param_child)
                        for generic_param_type in generic_params:
                            param_relationship = Relationship(
                                source_symbol=method_qualified_name,
                                target_symbol=generic_param_type,
                                relationship_type=RelationshipType.REFERENCES,
                                source_file=self._current_file,
                                target_file=self._current_file,  # Will be resolved later
                                annotations={'context': 'generic_parameter'}
                            )
                            self._relationships.append(param_relationship)
                        break

    def _handle_constructor_declaration(self, node: Node) -> None:
        """Handle constructor declarations."""
        # Constructor name is the same as the class name
        if not self._scope_stack:
            return

        # Get constructor name (class name)
        constructor_name = self._scope_stack[-1] if self._scope_stack else "unknown"
        parent_symbol = self._get_current_scope_path() if self._scope_stack else None
        qualified_name = self._build_qualified_name(constructor_name)

        # Extract parameter types (for overload disambiguation)
        parameter_types = []
        for child in node.children:
            if child.type == "formal_parameters":
                parameter_types = self._extract_parameter_types(child)
                break

        symbol = Symbol(
            name=constructor_name,
            symbol_type=SymbolType.CONSTRUCTOR,  # Phase 2: standardised constructor SymbolType
            file_path=self._current_file,
            range=self._create_range(node),
            scope=Scope.FUNCTION,
            parent_symbol=parent_symbol,
            full_name=qualified_name,
            parameter_types=parameter_types,
            source_text=self._extract_node_source(node)
        )
        self._symbols.append(symbol)

        # Enter constructor scope
        self._enter_scope(f"<init>")

        # Visit children
        self._visit_children(node)

        # Exit constructor scope
        self._exit_scope()

    def _handle_compact_constructor_declaration(self, node: Node) -> None:
        """Handle compact constructor declarations (Java records)."""
        # Constructor name is the same as the record name
        if not self._scope_stack:
            return

        # Get constructor name (record name)
        constructor_name = self._scope_stack[-1] if self._scope_stack else "unknown"
        qualified_name = self._build_qualified_name(constructor_name)

        symbol = Symbol(
            name=constructor_name,
            symbol_type=SymbolType.CONSTRUCTOR,  # Phase 2: standardised constructor SymbolType
            file_path=self._current_file,
            range=self._create_range(node),
            scope=Scope.FUNCTION
        )
        self._symbols.append(symbol)

        # Enter constructor scope
        self._enter_scope(f"<init>")

        # Visit children
        self._visit_children(node)

        # Exit constructor scope
        self._exit_scope()

    def _handle_field_declaration(self, node: Node) -> None:
        """Handle field declarations."""
        # Extract type and field name
        field_type = None
        field_name = None

        # Look for type and variable declarators
        for child in node.children:
            if child.type in ["type_identifier", "generic_type", "array_type"]:
                field_type = self._extract_type_name(child)
            elif child.type == "variable_declarator":
                # Find field name
                for grandchild in child.children:
                    if grandchild.type == "identifier":
                        field_name = self._get_node_text(grandchild)
                        break

        if field_name:
            # Create field symbol
            parent_symbol = self._get_current_scope_path() if self._scope_stack else None
            qualified_name = self._build_qualified_name(field_name)

            symbol = Symbol(
                name=field_name,
                symbol_type=SymbolType.FIELD,
                file_path=self._current_file,
                range=self._create_range(node),
                scope=Scope.CLASS,
                parent_symbol=parent_symbol,
                full_name=qualified_name,
                source_text=self._extract_node_source(node)
            )
            self._symbols.append(symbol)

            # Track field type for method resolution
            if field_type:
                self._field_types[field_name] = field_type

                # Create type relationship for the main field type (if not built-in)
                if not self._is_builtin_type(field_type):
                    # Determine relationship type based on field characteristics
                    relationship_type = self._determine_field_relationship_type(field_type, field_name, node)

                    relationship = Relationship(
                        source_symbol=qualified_name,
                        target_symbol=field_type,
                        relationship_type=relationship_type,
                        source_file=self._current_file,
                        target_file=self._current_file  # Will be resolved later
                    )
                    self._relationships.append(relationship)

                # Extract and create relationships for generic type parameters
                # Look for the type node to extract generic parameters
                # Phase 5: when outer type is Optional, inner types get AGGREGATION
                is_optional_wrapper = field_type in {'Optional'}
                for child in node.children:
                    if child.type in ["type_identifier", "generic_type", "array_type"]:
                        if child.type == "generic_type":
                            generic_params = self._extract_generic_type_parameters(child)
                            for param_type in generic_params:
                                if self._is_builtin_type(param_type):
                                    param_relationship_type = RelationshipType.REFERENCES
                                elif is_optional_wrapper:
                                    param_relationship_type = RelationshipType.AGGREGATION
                                else:
                                    param_relationship_type = RelationshipType.COMPOSITION

                                param_relationship = Relationship(
                                    source_symbol=qualified_name,
                                    target_symbol=param_type,
                                    relationship_type=param_relationship_type,
                                    source_file=self._current_file,
                                    target_file=self._current_file  # Will be resolved later
                                )
                                self._relationships.append(param_relationship)
                        break

        # Continue visiting children for any nested content
        self._visit_children(node)

    def _handle_method_invocation(self, node: Node) -> None:
        """Handle method calls with improved cross-file resolution."""
        method_name = None
        object_name = None

        # For method_invocation nodes, extract object and method
        # Pattern: object.method() or just method()
        # AST: method_invocation -> [identifier(object)] -> '.' -> identifier(method) -> argument_list

        identifiers = [child for child in node.children if child.type == "identifier"]

        if len(identifiers) >= 2:
            # Object method call: object.method()
            object_name = self._get_node_text(identifiers[0])
            method_name = self._get_node_text(identifiers[1])
        elif len(identifiers) == 1:
            # Direct method call: method()
            method_name = self._get_node_text(identifiers[0])

        if method_name:
            source_method = self._get_current_scope_path() or self._current_file
            target_symbol = method_name

            # If this is an object method call, try to resolve the target class
            if object_name:
                object_type = self._resolve_object_type(object_name)
                if object_type:
                    # Qualified method call
                    target_symbol = f"{object_type}.{method_name}"

            relationship = Relationship(
                source_symbol=source_method,
                target_symbol=target_symbol,
                relationship_type=RelationshipType.CALLS,
                source_file=self._current_file,
                target_file=self._current_file  # Will be resolved later in cross-file enhancement
            )
            self._relationships.append(relationship)

        # Continue visiting children
        self._visit_children(node)

    def _handle_object_creation(self, node: Node) -> None:
        """Handle object creation (new expressions)."""
        # Find the type being instantiated
        for child in node.children:
            if child.type in ["type_identifier", "identifier"]:
                type_name = self._get_node_text(child)
                if type_name:
                    # Create constructor creation relationship (CREATES, not CALLS)
                    source = self._get_current_scope_path() or self._current_file

                    # Create CREATES relationship for constructor calls (matches Python/TypeScript parsers)
                    relationship = Relationship(
                        source_symbol=source,
                        target_symbol=type_name,
                        relationship_type=RelationshipType.CREATES,
                        source_file=self._current_file,
                        target_file=self._current_file,
                        confidence=0.95,  # High confidence for tree-sitter based detection
                        annotations={'creation_type': 'new'}
                    )
                    self._relationships.append(relationship)
                break

        # Continue visiting children
        self._visit_children(node)

    # ============================================================================
    # RELATIONSHIP EXTRACTION
    # ============================================================================

    def _extract_inheritance_from_class(self, class_node: Node, class_symbol: Symbol) -> None:
        """Extract inheritance and implementation relationships from class/interface."""
        # Use qualified name for source to ensure proper node resolution
        source_qualified_name = class_symbol.get_qualified_name() or class_symbol.full_name or class_symbol.name
        
        for child in class_node.children:
            if child.type == "superclass":
                # Handle extends clause
                for grandchild in child.children:
                    if grandchild.type in ["type_identifier", "identifier"]:
                        parent_class = self._get_node_text(grandchild)
                        if parent_class:
                            relationship = Relationship(
                                source_symbol=source_qualified_name,
                                target_symbol=parent_class,
                                relationship_type=RelationshipType.INHERITANCE,
                                source_file=self._current_file,
                                target_file=self._current_file  # Will be resolved later
                            )
                            self._relationships.append(relationship)
                        break

            elif child.type == "super_interfaces":
                # Handle implements clause
                for grandchild in child.children:
                    if grandchild.type == "type_list":
                        # Process all interfaces in the type list
                        for interface_node in grandchild.children:
                            if interface_node.type in ["type_identifier", "identifier"]:
                                interface_name = self._get_node_text(interface_node)
                                if interface_name:
                                    relationship = Relationship(
                                        source_symbol=source_qualified_name,
                                        target_symbol=interface_name,
                                        relationship_type=RelationshipType.IMPLEMENTATION,
                                        source_file=self._current_file,
                                        target_file=self._current_file  # Will be resolved later
                                    )
                                    self._relationships.append(relationship)

    def _find_superclass(self, class_qualified_name: str) -> Optional[str]:
        """Find the superclass of a class from existing INHERITANCE relationships.

        Returns the target symbol of the first INHERITANCE relationship
        where source_symbol matches the given class name.
        """
        for rel in self._relationships:
            if (rel.relationship_type == RelationshipType.INHERITANCE
                    and rel.source_symbol == class_qualified_name):
                return rel.target_symbol
        return None

    def _handle_annotation_declaration(self, node: Node) -> None:
        """Handle annotation type declarations."""
        annotation_name = None

        # Find annotation name
        for child in node.children:
            if child.type == "identifier":
                annotation_name = self._get_node_text(child)
                break

        if not annotation_name:
            return

        # Create annotation symbol
        qualified_name = self._build_qualified_name(annotation_name)

        symbol = Symbol(
            name=annotation_name,
            symbol_type=SymbolType.ANNOTATION,
            file_path=self._current_file,
            range=self._create_range(node),
            scope=Scope.CLASS
        )
        self._symbols.append(symbol)

        # Enter annotation scope
        self._enter_scope(annotation_name)

        # Visit children
        self._visit_children(node)

        # Exit annotation scope
        self._exit_scope()

    def _handle_record_declaration(self, node: Node) -> None:
        """Handle record declarations (Java 14+)."""
        record_name = None

        # Find record name
        for child in node.children:
            if child.type == "identifier":
                record_name = self._get_node_text(child)
                break

        if not record_name:
            return

        # Create record symbol (treat as class with metadata to distinguish)
        parent_symbol = self._get_current_scope_path() if self._scope_stack else None
        qualified_name = self._build_qualified_name(record_name)

        symbol = Symbol(
            name=record_name,
            symbol_type=SymbolType.CLASS,  # Records are special classes
            file_path=self._current_file,
            range=self._create_range(node),
            scope=Scope.CLASS,
            parent_symbol=parent_symbol,
            full_name=qualified_name,
            source_text=self._extract_node_source(node),
            metadata={'is_record': True}  # Mark as record for distinction
        )
        self._symbols.append(symbol)

        # Enter record scope
        self._enter_scope(record_name)

        # Visit children
        self._visit_children(node)

        # Exit record scope
        self._exit_scope()

    def _handle_super_interfaces(self, node: Node) -> None:
        """Handle super interfaces (implements clause)."""
        # Get current class/interface name
        if not self._scope_stack:
            return

        current_class = self._scope_stack[-1]

        # Find type_list containing the interfaces
        type_list = self._find_child_by_type(node, "type_list")
        if type_list:
            for child in type_list.children:
                if child.type in ["type_identifier", "generic_type"]:
                    interface_name = self._get_type_name(child)
                    if interface_name:
                        relationship = Relationship(
                            source_symbol=current_class,
                            target_symbol=interface_name,
                            relationship_type=RelationshipType.IMPLEMENTATION,
                            source_file=self._current_file,
                            source_range=self._create_range(child)
                        )
                        self._relationships.append(relationship)

    def _handle_superclass(self, node: Node) -> None:
        """Handle superclass (extends clause)."""
        # Get current class name
        if not self._scope_stack:
            return

        current_class = self._scope_stack[-1]

        # Find superclass type (should be a direct child)
        for child in node.children:
            if child.type in ["type_identifier", "generic_type"]:
                superclass_name = self._get_type_name(child)
                if superclass_name:
                    relationship = Relationship(
                        source_symbol=current_class,
                        target_symbol=superclass_name,
                        relationship_type=RelationshipType.INHERITANCE,
                        source_file=self._current_file,
                        source_range=self._create_range(child)
                    )
                    self._relationships.append(relationship)
                    break  # Only one superclass in Java

    def _get_type_name(self, node: Node) -> str:
        """Extract type name from a type node, handling generics."""
        if not node:
            return ""

        if node.type == "identifier" or node.type == "type_identifier":
            return self._get_node_text(node)
        elif node.type == "generic_type":
            # For generic types like List<String> or Comparable<Circle>, get the base type
            for child in node.children:
                if child.type in ["type_identifier", "identifier"]:
                    return self._get_node_text(child)
        elif node.type == "scoped_type_identifier":
            # For scoped types like java.util.List
            return self._get_node_text(node)
        elif node.type == "scoped_identifier":
            # For scoped identifiers
            return self._get_node_text(node)

        return ""

    # ============================================================================
    # FALLBACK PARSING
    # ============================================================================

    def _fallback_parse(self, file_path: str, content: str) -> ParseResult:
        """Fallback parsing when tree-sitter is not available."""
        logger.warning(f"Using fallback parsing for {file_path}")

        symbols = []
        relationships = []

        lines = content.split('\n')
        current_class = None

        for i, line in enumerate(lines, 1):
            line = line.strip()

            # Basic package detection
            if line.startswith('package '):
                package_name = line.replace('package ', '').replace(';', '').strip()
                symbol = Symbol(
                    name=package_name,
                    symbol_type=SymbolType.NAMESPACE,
                    file_path=file_path,
                    range=Range(
                        start=Position(line=i, column=0),
                        end=Position(line=i, column=len(line))
                    ),
                    scope=Scope.GLOBAL
                )
                symbols.append(symbol)

            # Basic class detection
            elif 'class ' in line and not line.startswith('//'):
                parts = line.split('class ')
                if len(parts) > 1:
                    class_name = parts[1].split()[0].rstrip('{')
                    current_class = class_name
                    symbol = Symbol(
                        name=class_name,
                        symbol_type=SymbolType.CLASS,
                        file_path=file_path,
                        range=Range(
                            start=Position(line=i, column=0),
                            end=Position(line=i, column=len(line))
                        ),
                        scope=Scope.CLASS
                    )
                    symbols.append(symbol)

            # Basic method detection
            elif ('public ' in line or 'private ' in line) and '(' in line and ')' in line:
                method_name = line.split('(')[0].split()[-1]
                qualified_name = f"{current_class}.{method_name}" if current_class else method_name
                symbol = Symbol(
                    name=method_name,
                    symbol_type=SymbolType.METHOD,
                    file_path=file_path,
                    range=Range(
                        start=Position(line=i, column=0),
                        end=Position(line=i, column=len(line))
                    ),
                    scope=Scope.FUNCTION
                )
                symbols.append(symbol)

        return ParseResult(
            file_path=file_path,
            language="java",
            symbols=symbols,
            relationships=relationships,
            parse_time=0.01,
            warnings=["Using fallback parsing (tree-sitter not available)"]
        )

    # ============================================================================
    # BASE PARSER INTERFACE METHODS
    # ============================================================================

    def parse_code(self, content: str, file_path: str) -> ParseResult:
        """Parse Java code content."""
        return self.parse_file(file_path, content)

    def extract_symbols(self, file_path: str, content: str) -> List[Symbol]:
        """Extract symbols from Java code."""
        result = self.parse_code(content, file_path)
        return result.symbols

    def extract_relationships(self, file_path: str, content: str, symbols: List[Symbol]) -> List[Relationship]:
        """Extract relationships from Java code."""
        result = self.parse_code(content, file_path)
        return result.relationships

    def parse_multiple_files(self, file_paths: List[str], max_workers: int = 4) -> Dict[str, ParseResult]:
        """
        Parse multiple Java files and resolve cross-file relationships.

        Args:
            file_paths: List of file paths to parse
            max_workers: Maximum number of parallel workers for file parsing
        """
        results = {}

        total = len(file_paths)
        # First pass: Parse each file individually IN PARALLEL
        logger.info(f"Parsing {total} Java files for cross-file analysis with {max_workers} workers")
        if this and getattr(this, 'module', None):
            this.module.invocation_thinking(
                f"I am on phase parsing\nStarting Java multi-file parsing: {total} files\nReasoning: Collect class/interface/method symbols & imports to fuel cross-file inheritance and call resolution.\nNext: Parallel parse with ~10% progress checkpoints then construct global symbol registry."
            )
        results = self._parse_files_parallel(file_paths, max_workers, emit_thinking=True)
        if this and getattr(this, 'module', None):
            succeeded = sum(1 for r in results.values() if r.symbols)
            pct = int((succeeded/total)*100) if total else 100
            this.module.invocation_thinking(
                f"I am on phase parsing\nRaw Java parsing complete: {succeeded}/{total} files produced symbols ({pct}%)\nReasoning: Symbol population sufficient for robust cross-file mapping.\nNext: Populate consolidated Java symbol tables (classes -> files, methods -> classes)."
            )

        # Build global symbol registry from all parsed results
        if this and getattr(this, 'module', None):
            this.module.invocation_thinking(
                "I am on phase parsing\nBuilding global Java symbol tables (classes -> files, methods -> classes)\nReasoning: Centralized view enables inheritance + call target resolution across packages.\nNext: Augment relationships using this registry."
            )
        for file_path, result in results.items():
            if result.symbols:  # Only process files that parsed successfully
                self._extract_global_symbols(file_path, result)
        if this and getattr(this, 'module', None):
            this.module.invocation_thinking(
                "I am on phase parsing\nGlobal registry ready\nReasoning: All class & method origins known.\nNext: Enhance relationships (calls, inheritance, imports) with resolved targets."
            )

        # Second pass: Enhance relationships with cross-file resolution
        logger.info("Enhancing Java relationships with cross-file resolution")
        if this and getattr(this, 'module', None):
            this.module.invocation_thinking(
                "I am on phase parsing\nEnhancing Java relationships with cross-file targets\nReasoning: Replace raw textual references with fully-qualified symbols for accurate graph edges.\nNext: Produce final completion summary."
            )
        for file_path, result in results.items():
            if result.symbols:  # Only process files that parsed successfully
                self._enhance_cross_file_relationships(file_path, result)
        if this and getattr(this, 'module', None):
            this.module.invocation_thinking(
                f"I am on phase parsing\nJava parsing complete: {len(results)}/{total} files (100%)\nReasoning: Multi-pass enrichment done; dataset ready for higher-level graph synthesis & docs.\nNext: Upstream analysis may merge multi-language graphs."
            )

        return results

    def _parse_files_parallel(self, file_paths: List[str], max_workers: int, emit_thinking: bool = False) -> Dict[str, ParseResult]:
        """
        Parse multiple Java files in parallel using ThreadPoolExecutor with batching.
        
        ThreadPoolExecutor is used instead of ProcessPoolExecutor because:
        - Lower memory overhead (shared memory)
        - No serialization overhead
        - Tree-sitter releases GIL, so we get true parallelism
        - More stable with large workloads (1000+ files)
        
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
            logger.info(f"Processing {total} Java files in {num_batches} batch(es) of up to {batch_size} files each")

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
                            future = executor.submit(_parse_single_java_file, fp)
                            future_to_file[future] = fp
                        except Exception as e:
                            logger.error(f"Failed to submit file {fp}: {type(e).__name__}: {e}")
                            results[fp] = ParseResult(
                                file_path=fp,
                                language="java",
                                symbols=[],
                                relationships=[],
                                errors=[f"Submit failed: {str(e)}"]
                            )
                    
                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            parsed_file_path, result = future.result(timeout=60)  # 60 second timeout per file
                            results[parsed_file_path] = result
                            
                            # Progress logging
                            completed = len(results)
                            if completed % max(1, int(total/10)) == 0 or completed == total:
                                logger.info(f"Parsed {completed}/{total} Java files...")
                                if emit_thinking and this and getattr(this, 'module', None):
                                    pct = int((completed/total)*100) if total else 100
                                    this.module.invocation_thinking(
                                        f"I am on phase parsing\nJava raw parse progress: {completed}/{total} files ({pct}%)\nReasoning: Gathering class & method inventories to enable inheritance and call target mapping.\nNext: Complete parse phase then build global tables."
                                    )
                        except concurrent.futures.TimeoutError:
                            logger.error(f"Timeout parsing file (>60s): {file_path}")
                            results[file_path] = ParseResult(
                                file_path=file_path,
                                language="java",
                                symbols=[],
                                relationships=[],
                                errors=["Timeout during parsing"]
                            )
                        except Exception as e:
                            logger.error(f"Failed to parse file {file_path}: {type(e).__name__}: {e}", exc_info=True)
                            results[file_path] = ParseResult(
                                file_path=file_path,
                                language="java",
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
                            language="java",
                            symbols=[],
                            relationships=[],
                            errors=[f"Batch processing failed: {str(e)}"]
                        )

        parse_time = time.time() - start_time
        successful_files = len([r for r in results.values() if r.symbols])
        logger.info(f"Parallel Java parsing complete: {successful_files}/{len(file_paths)} files parsed in {parse_time:.2f}s")

        return results

    def _extract_global_symbols(self, file_path: str, result: ParseResult):
        """Extract symbols for global cross-file resolution."""
        # Extract package from relationships
        package = ""
        imports = {}

        for rel in result.relationships:
            if rel.relationship_type == RelationshipType.IMPORTS:
                if "." in rel.target_symbol:
                    parts = rel.target_symbol.split(".")
                    class_name = parts[-1]
                    imports[class_name] = rel.target_symbol

        # Store class locations
        for symbol in result.symbols:
            if symbol.symbol_type in [SymbolType.CLASS, SymbolType.INTERFACE]:
                self._global_class_locations[symbol.name] = file_path

                # Store methods of this class
                class_methods = [s for s in result.symbols
                               if s.symbol_type == SymbolType.METHOD and s.parent_symbol and symbol.name in s.parent_symbol]

                for method in class_methods:
                    if method.name not in self._global_method_locations:
                        self._global_method_locations[method.name] = []
                    self._global_method_locations[method.name].append((symbol.name, file_path))

    def _enhance_cross_file_relationships(self, file_path: str, result: ParseResult):
        """Enhance relationships with cross-file information."""
        enhanced_relationships = []

        # Build imports map for this file
        imports = {}
        for rel in result.relationships:
            if rel.relationship_type == RelationshipType.IMPORTS and "." in rel.target_symbol:
                parts = rel.target_symbol.split(".")
                class_name = parts[-1]
                imports[class_name] = rel.target_symbol

        for rel in result.relationships:
            enhanced_rel = self._enhance_relationship(rel, file_path, imports)
            enhanced_relationships.append(enhanced_rel)

        # Replace with enhanced relationships
        result.relationships.clear()
        result.relationships.extend(enhanced_relationships)

    def _enhance_relationship(self, rel: Relationship, source_file: str, imports: Dict[str, str]) -> Relationship:
        """Enhance a single relationship with cross-file target information."""

        if rel.relationship_type == RelationshipType.CREATES:
            # Resolve constructor (CREATES) target to cross-file
            target_class = rel.target_symbol

            # Check if target class is imported or in global locations
            if target_class in imports:
                target_file = self._global_class_locations.get(target_class)
                if target_file and target_file != source_file:
                    return Relationship(
                        source_symbol=rel.source_symbol,
                        target_symbol=target_class,
                        relationship_type=RelationshipType.CREATES,
                        source_file=source_file,
                        target_file=target_file,
                        source_range=rel.source_range,
                        confidence=rel.confidence,
                        annotations=rel.annotations
                    )
            else:
                # Check if it's a class in the same package or in global locations
                target_file = self._global_class_locations.get(target_class)
                if target_file and target_file != source_file:
                    return Relationship(
                        source_symbol=rel.source_symbol,
                        target_symbol=target_class,
                        relationship_type=RelationshipType.CREATES,
                        source_file=source_file,
                        target_file=target_file,
                        source_range=rel.source_range,
                        confidence=rel.confidence,
                        annotations=rel.annotations
                    )

        elif rel.relationship_type == RelationshipType.CALLS:
            target_symbol = rel.target_symbol

            # Check if it's already a qualified method call (Class.method or Class.<init>)
            if '.' in target_symbol:
                parts = target_symbol.split('.')
                if len(parts) == 2:
                    class_name, method_name = parts

                    # Try to find this class in global locations
                    target_file = self._global_class_locations.get(class_name)

                    if target_file and target_file != source_file:
                        # This is a cross-file method call (including constructors)
                        return Relationship(
                            source_symbol=rel.source_symbol,
                            target_symbol=target_symbol,
                            relationship_type=RelationshipType.CALLS,
                            source_file=source_file,
                            target_file=target_file,
                            source_range=rel.source_range
                        )
            else:
                # Simple method name - try to resolve using global method locations
                method_name = target_symbol

                if method_name in self._global_method_locations:
                    method_locations = self._global_method_locations[method_name]

                    # Try to find the most likely target
                    target_class, target_file = self._find_best_method_target(
                        method_name, method_locations, source_file, imports)

                    if target_file and target_file != source_file:
                        # This is a cross-file method call
                        return Relationship(
                            source_symbol=rel.source_symbol,
                            target_symbol=f"{target_class}.{method_name}",
                            relationship_type=RelationshipType.CALLS,
                            source_file=source_file,
                            target_file=target_file,
                            source_range=rel.source_range
                        )

        elif rel.relationship_type == RelationshipType.INHERITANCE:
            # Resolve inheritance target
            parent_class = rel.target_symbol

            # Check if parent class is imported
            if parent_class in imports:
                target_file = self._global_class_locations.get(parent_class)
                if target_file and target_file != source_file:
                    return Relationship(
                        source_symbol=rel.source_symbol,
                        target_symbol=parent_class,
                        relationship_type=RelationshipType.INHERITANCE,
                        source_file=source_file,
                        target_file=target_file,
                        source_range=rel.source_range
                    )
            else:
                # Check if parent class is in global locations (same package or other visible class)
                target_file = self._global_class_locations.get(parent_class)
                if target_file and target_file != source_file:
                    return Relationship(
                        source_symbol=rel.source_symbol,
                        target_symbol=parent_class,
                        relationship_type=RelationshipType.INHERITANCE,
                        source_file=source_file,
                        target_file=target_file,
                        source_range=rel.source_range
                    )

        elif rel.relationship_type == RelationshipType.IMPLEMENTATION:
            # Resolve implementation target (interface)
            interface_name = rel.target_symbol

            # Check if interface is imported
            if interface_name in imports:
                target_file = self._global_class_locations.get(interface_name)
                if target_file and target_file != source_file:
                    return Relationship(
                        source_symbol=rel.source_symbol,
                        target_symbol=interface_name,
                        relationship_type=RelationshipType.IMPLEMENTATION,
                        source_file=source_file,
                        target_file=target_file,
                        source_range=rel.source_range
                    )
            else:
                # Check if interface is in global locations (same package or other visible interface)
                target_file = self._global_class_locations.get(interface_name)
                if target_file and target_file != source_file:
                    return Relationship(
                        source_symbol=rel.source_symbol,
                        target_symbol=interface_name,
                        relationship_type=RelationshipType.IMPLEMENTATION,
                        source_file=source_file,
                        target_file=target_file,
                        source_range=rel.source_range
                    )

        elif rel.relationship_type == RelationshipType.IMPORTS:
            # Resolve import target to cross-file
            import_path = rel.target_symbol

            # Extract class name from import path (e.g., com.example.util.Helper -> Helper)
            if '.' in import_path:
                parts = import_path.split('.')
                class_name = parts[-1]

                # Find the target file for this class
                target_file = self._global_class_locations.get(class_name)

                if target_file and target_file != source_file:
                    # This is a cross-file import
                    return Relationship(
                        source_symbol=rel.source_symbol,
                        target_symbol=import_path,
                        relationship_type=RelationshipType.IMPORTS,
                        source_file=source_file,
                        target_file=target_file,
                        source_range=rel.source_range
                    )

        elif rel.relationship_type == RelationshipType.REFERENCES:
            # Resolve reference target to cross-file
            referenced_type = rel.target_symbol

            # Check if referenced type is imported
            if referenced_type in imports:
                target_file = self._global_class_locations.get(referenced_type)
                if target_file and target_file != source_file:
                    return Relationship(
                        source_symbol=rel.source_symbol,
                        target_symbol=referenced_type,
                        relationship_type=RelationshipType.REFERENCES,
                        source_file=source_file,
                        target_file=target_file,
                        source_range=rel.source_range
                    )
            else:
                # Check if it's a class in the same package or in global locations
                target_file = self._global_class_locations.get(referenced_type)
                if target_file and target_file != source_file:
                    return Relationship(
                        source_symbol=rel.source_symbol,
                        target_symbol=referenced_type,
                        relationship_type=RelationshipType.REFERENCES,
                        source_file=source_file,
                        target_file=target_file,
                        source_range=rel.source_range
                    )

        # Return original relationship if no enhancement
        return rel

    def _find_best_method_target(self, method_name: str, locations: List[tuple],
                                source_file: str, imports: Dict[str, str]) -> tuple:
        """Find the best target for a method call."""

        # Prefer methods from imported classes
        for class_name, file_path in locations:
            if class_name in imports:
                return class_name, file_path

        # Return first available
        if locations:
            return locations[0]

        return None, None

    def _determine_field_relationship_type(self, field_type: str, field_name: str, node) -> RelationshipType:
        """Determine the appropriate relationship type for a field declaration."""
        # Extract the base type (remove generics like List<Role> -> Role)
        base_type = field_type
        if '<' in field_type:
            # For generics like List<Role>, Set<Permission>, extract the parameter type
            generic_start = field_type.find('<')
            generic_end = field_type.rfind('>')
            if generic_start != -1 and generic_end != -1:
                generic_content = field_type[generic_start + 1:generic_end]
                # Handle simple cases like List<Role> or Set<Permission>
                if ',' not in generic_content and not '<' in generic_content:
                    base_type = generic_content.strip()

        # Check if it's a collection type (List, Set, Map, etc.)
        collection_types = {'List', 'Set', 'Map', 'Collection', 'ArrayList', 'HashSet', 'HashMap'}
        is_collection = any(field_type.startswith(ctype) for ctype in collection_types)

        # Check for interface types
        interface_indicators = {'Interface', 'Service', 'Repository', 'Controller', 'Validator'}
        is_interface_like = any(indicator in field_type for indicator in interface_indicators)

        # Check for primitive wrapper or basic types that should be references
        basic_types = {'String', 'Integer', 'Long', 'Double', 'Float', 'Boolean', 'Date', 'LocalDateTime', 'LocalDate', 'UUID'}
        is_basic_type = base_type in basic_types

        # Phase 5: Check for Optional wrapper (weak ownership → AGGREGATION)
        optional_types = {'Optional'}
        is_optional = any(field_type.startswith(otype) for otype in optional_types)

        # Check for @Nullable annotation on the field node
        is_nullable = False
        if node is not None:
            for child in node.children:
                if child.type == 'modifiers':
                    for mod_child in child.children:
                        if mod_child.type == 'marker_annotation':
                            ann_id = self._find_child_by_type(mod_child, 'identifier')
                            if ann_id and self._get_node_text(ann_id) == 'Nullable':
                                is_nullable = True
                    break

        # Determine relationship type based on field characteristics
        if is_interface_like:
            return RelationshipType.REFERENCES  # Dependencies/services
        elif is_basic_type:
            return RelationshipType.REFERENCES  # Basic type references
        elif is_optional or is_nullable:
            return RelationshipType.AGGREGATION  # Weak ownership (may be absent)
        elif is_collection:
            return RelationshipType.COMPOSITION  # Collections indicate ownership
        else:
            # Single object field - likely composition (owned relationship)
            return RelationshipType.COMPOSITION

    def _resolve_object_type(self, object_name: str) -> str:
        """Resolve the type of an object (field, variable, or parameter)."""
        # Check if it's a field
        if object_name in self._field_types:
            field_type = self._field_types[object_name]
            # Return the simple type name for lookup
            return field_type.split('.')[-1] if '.' in field_type else field_type

        # Check if it's a local variable (would need variable declaration tracking)
        if object_name in self._variable_types:
            var_type = self._variable_types[object_name]
            return var_type.split('.')[-1] if '.' in var_type else var_type

        # Check if it's a parameter (would need parameter tracking)
        if object_name in self._parameter_types:
            param_type = self._parameter_types[object_name]
            return param_type.split('.')[-1] if '.' in param_type else param_type

        return None

    def _is_builtin_type(self, type_name: str) -> bool:
        """Check if a type is a built-in Java type that we shouldn't create references for."""
        if not type_name:
            return True

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
        if '.' in type_name:
            for package in java_std_packages:
                if type_name.startswith(package + '.'):
                    return True

        builtin_types = {
            # Primitive types
            'byte', 'short', 'int', 'long', 'float', 'double', 'char', 'boolean', 'void',
            # Wrapper types
            'Byte', 'Short', 'Integer', 'Long', 'Float', 'Double', 'Character', 'Boolean', 'Void',
            # Common built-in classes
            'String', 'Object', 'Class', 'Number', 'Enum',
            # Collections (java.util)
            'List', 'ArrayList', 'LinkedList', 'Set', 'HashSet', 'LinkedHashSet', 'TreeSet',
            'Map', 'HashMap', 'LinkedHashMap', 'TreeMap', 'Queue', 'Deque', 'ArrayDeque',
            'Collection', 'Iterator', 'Iterable', 'Optional', 'Objects',
            # Common exceptions
            'Exception', 'RuntimeException', 'Error', 'Throwable',
            'IllegalArgumentException', 'IllegalStateException', 'NullPointerException',
            # I/O and other common types
            'File', 'Path', 'URI', 'URL', 'Date', 'Calendar', 'LocalDate', 'LocalDateTime',
            'BigInteger', 'BigDecimal', 'UUID', 'Random', 'Thread', 'Runnable',
            'Pattern', 'Matcher', 'Serializable',  # Added missing types
            # Annotations
            'Override', 'Deprecated', 'SuppressWarnings', 'FunctionalInterface'
        }
        return type_name in builtin_types

    def _extract_generic_type_parameters(self, node: Node) -> List[str]:
        """Extract generic type parameters from a generic type node (recursively).

        Descends into nested ``generic_type`` / ``array_type`` / ``wildcard``
        nodes so that deeply-nested user types are not missed.
        e.g. ``Map<String, List<User>>`` → ``['User']``
        """
        parameters: List[str] = []
        if not node or node.type != "generic_type":
            return parameters

        # Look for type_arguments node
        for child in node.children:
            if child.type == "type_arguments":
                for type_arg_child in child.children:
                    self._collect_user_types_from_type_node(type_arg_child, parameters)
                break
        return parameters

    def _collect_user_types_from_type_node(self, node: 'Node', out: List[str]) -> None:
        """Recursively collect user-defined type names from a type AST node.

        Handles ``type_identifier``, ``generic_type``, ``array_type`` and
        ``wildcard`` (``? extends T`` / ``? super T``).
        """
        if node is None:
            return

        if node.type == "type_identifier":
            name = self._get_node_text(node)
            if name and not self._is_builtin_type(name):
                out.append(name)

        elif node.type == "generic_type":
            # Emit the base type if user-defined
            base = self._extract_type_name(node)
            if base and not self._is_builtin_type(base):
                out.append(base)
            # Recurse into type_arguments
            for child in node.children:
                if child.type == "type_arguments":
                    for arg in child.children:
                        self._collect_user_types_from_type_node(arg, out)
                    break

        elif node.type == "array_type":
            # e.g. User[] — recurse into element type
            for child in node.children:
                if child.type in ("type_identifier", "generic_type", "array_type"):
                    self._collect_user_types_from_type_node(child, out)

        elif node.type == "wildcard":
            # ? extends T / ? super T
            for child in node.children:
                if child.type in ("type_identifier", "generic_type", "array_type"):
                    self._collect_user_types_from_type_node(child, out)

    def _extract_type_name(self, node: Node) -> str:
        """Extract type name from a type node."""
        if not node:
            return None

        if node.type == "type_identifier":
            return self._get_node_text(node)
        elif node.type == "generic_type":
            # For generic types like List<String>, get the base type
            for child in node.children:
                if child.type == "type_identifier":
                    return self._get_node_text(child)
        elif node.type == "array_type":
            # For array types like String[], get the element type
            for child in node.children:
                if child.type in ["type_identifier", "generic_type"]:
                    return self._extract_type_name(child)
        elif node.type == "identifier":
            return self._get_node_text(node)
        elif node.type == "wildcard":
            # Handle wildcards in generic types
            for child in node.children:
                if child.type in ["type_identifier", "generic_type"]:
                    return self._extract_type_name(child)
        elif node.type in ["boolean_type", "integral_type", "floating_point_type", "void_type"]:
            # Handle primitive types (boolean, int, long, float, double, etc.)
            return self._get_node_text(node)

        return None

    def _handle_variable_declarator(self, node: Node) -> None:
        """Handle local variable declarations to track types."""
        # This method should be called from local_variable_declaration handler
        variable_name = None

        for child in node.children:
            if child.type == "identifier":
                variable_name = self._get_node_text(child)
                break

        # The parent should be local_variable_declaration which has the type
        if variable_name and node.parent:
            parent = node.parent
            for child in parent.children:
                if child.type in ["type_identifier", "generic_type", "array_type"]:
                    var_type = self._extract_type_name(child)
                    if var_type:
                        self._variable_types[variable_name] = var_type
                    break

    def _handle_local_variable_declaration(self, node: Node) -> None:
        """Handle local variable declarations to track types."""
        variable_type = None
        variable_type_node = None

        # Extract type from the declaration
        for child in node.children:
            if child.type in ["type_identifier", "generic_type", "array_type"]:
                variable_type = self._extract_type_name(child)
                variable_type_node = child
                break

        # Extract variable names from variable_declarator children
        if variable_type:
            for child in node.children:
                if child.type == "variable_declarator":
                    self._handle_variable_declarator_with_type(child, variable_type, variable_type_node)

        # Continue visiting children
        self._visit_children(node)

    def _handle_variable_declarator_with_type(self, node: Node, variable_type: str, variable_type_node: Node) -> None:
        """Handle variable declarator with known type."""
        for child in node.children:
            if child.type == "identifier":
                variable_name = self._get_node_text(child)
                if variable_name:
                    # Track the variable type for method resolution
                    self._variable_types[variable_name] = variable_type

                    # Create reference relationship for variable type (if not built-in)
                    if not self._is_builtin_type(variable_type):
                        current_method = self._get_current_method()
                        source_symbol = current_method if current_method else self._build_qualified_name(variable_name)

                        relationship = Relationship(
                            source_symbol=source_symbol,
                            target_symbol=variable_type,
                            relationship_type=RelationshipType.REFERENCES,
                            source_file=self._current_file,
                            target_file=self._current_file  # Will be resolved later
                        )
                        self._relationships.append(relationship)

                    # Extract generic type parameters from variable type
                    if variable_type_node and variable_type_node.type == "generic_type":
                        generic_params = self._extract_generic_type_parameters(variable_type_node)
                        for param_type in generic_params:
                            current_method = self._get_current_method()
                            source_symbol = current_method if current_method else self._build_qualified_name(variable_name)

                            param_relationship = Relationship(
                                source_symbol=source_symbol,
                                target_symbol=param_type,
                                relationship_type=RelationshipType.REFERENCES,
                                source_file=self._current_file,
                                target_file=self._current_file  # Will be resolved later
                            )
                            self._relationships.append(param_relationship)
                break

    def _get_current_method(self) -> str:
        """Get the current method context for reference relationships."""
        if len(self._scope_stack) >= 2:
            # Usually: [package/class, method]
            return self._build_qualified_name(self._scope_stack[-1])
        elif self._scope_stack:
            # Fall back to current scope
            return self._build_qualified_name(self._scope_stack[-1])
        return None

    def _extract_node_source(self, node: Node) -> Optional[str]:
        """Extract source code text for a tree-sitter node."""
        if not node or not self._current_content:
            return None

        try:
            # Get the byte range of the node
            start_byte = node.start_byte
            end_byte = node.end_byte

            # Extract text from the source content
            source_bytes = self._current_content.encode('utf-8')
            node_bytes = source_bytes[start_byte:end_byte]
            node_text = node_bytes.decode('utf-8')

            return node_text
        except Exception:
            return None

    def _extract_defines_relationships(self, symbols: List[Symbol], file_path: str) -> List[Relationship]:
        """
        Extract DEFINES relationships (Pass 2 - after symbol extraction).
        
        DEFINES represents containment:
        - PACKAGE (NAMESPACE) DEFINES top-level classes, interfaces, enums
        - CLASS/INTERFACE DEFINES its methods/fields
        
        Following the pattern from C++/JavaScript/TypeScript/Python parsers.
        
        Rules:
        - CLASS → METHOD (including constructors as ClassName.<init>)
        - CLASS → FIELD
        - INTERFACE → METHOD (method signatures)
        
        Note: PACKAGE → top-level DEFINES relationships are NOT created.
        Packages are organizational, not architectural. Parent-child containment
        is captured by parent_symbol metadata on each symbol.
        
        Java-specific features:
        - Constructors use <init> naming convention
        - Overloaded methods tracked with parameter_types metadata
        - Static members tracked with is_static flag
        - Visibility modifiers (public/private/protected/package) in metadata
        - Abstract methods tracked with is_abstract flag
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
        # Only CLASS and INTERFACE create DEFINES edges (not PACKAGE/NAMESPACE)
        for symbol in symbols:
            # Only CLASS and INTERFACE can DEFINE members
            if symbol.symbol_type not in (SymbolType.CLASS, SymbolType.INTERFACE):
                continue
            
            # Find all members with this class/interface as parent
            # Use full_name for nested class support
            members = parent_index.get(symbol.full_name, [])
            if not members:
                # Fallback: try with just name (for top-level classes)
                members = parent_index.get(symbol.name, [])
            
            for member in members:
                # Only DEFINE actual members (methods and fields)
                if member.symbol_type not in (SymbolType.METHOD, SymbolType.FIELD):
                    continue
                
                # Determine member type and build target symbol name
                member_type = self._get_member_type(member)
                
                # Get parameter types for signature (if method/constructor)
                param_types = getattr(member, 'parameter_types', []) if member.symbol_type == SymbolType.METHOD else []
                
                # For constructors, use <init> convention with signature for overloads
                if member_type == 'constructor':
                    if param_types:
                        # Include signature for overloaded constructors
                        param_str = ', '.join(param_types)
                        target_symbol = f"{symbol.full_name}.<init>({param_str})"
                    else:
                        # No-arg constructor
                        target_symbol = f"{symbol.full_name}.<init>()"
                else:
                    # For regular methods, check if we need signature (detect overloads)
                    # Count methods with same name in this class
                    same_name_methods = [m for m in members 
                                        if m.symbol_type == SymbolType.METHOD 
                                        and m.name == member.name]
                    
                    if len(same_name_methods) > 1:
                        # Overloaded method - include signature
                        param_str = ', '.join(param_types) if param_types else ''
                        target_symbol = f"{member.full_name}({param_str})"
                    else:
                        # Unique method name - no signature needed
                        target_symbol = member.full_name
                
                # Extract modifiers from source text
                modifiers = self._extract_modifiers(member)
                
                # Build metadata
                annotations = {
                    'member_type': member_type,
                    'is_static': 'static' in modifiers,
                    'is_abstract': 'abstract' in modifiers,
                    'visibility': self._extract_visibility(modifiers),
                }
                
                # Add parameter types for methods (including overloaded methods)
                if member.symbol_type == SymbolType.METHOD and hasattr(member, 'parameter_types'):
                    # Always include parameter_types, even if empty list (for zero-param methods)
                    annotations['parameter_types'] = member.parameter_types
                    # Build signature for disambiguation
                    param_str = ', '.join(member.parameter_types) if member.parameter_types else ''
                    annotations['signature'] = f"{member.name}({param_str})"
                
                # Create DEFINES relationship
                relationship = Relationship(
                    source_symbol=symbol.full_name,
                    target_symbol=target_symbol,
                    relationship_type=RelationshipType.DEFINES,
                    source_file=file_path,
                    target_file=file_path,
                    annotations=annotations
                )
                relationships.append(relationship)
        
        return relationships

    def _get_member_type(self, symbol: Symbol) -> str:
        """Determine the member type string for DEFINES metadata."""
        if symbol.symbol_type == SymbolType.FIELD:
            return 'field'
        elif symbol.symbol_type == SymbolType.METHOD:
            # Check if it's a constructor by checking if name matches parent class
            if symbol.name == symbol.parent_symbol or '<init>' in (symbol.full_name or ''):
                return 'constructor'
            return 'method'
        return 'unknown'

    def _extract_modifiers(self, symbol: Symbol) -> List[str]:
        """Extract modifiers from symbol's source text."""
        modifiers = []
        
        if not symbol.source_text:
            return modifiers
        
        # Common Java modifiers
        modifier_keywords = ['public', 'private', 'protected', 'static', 'final', 
                            'abstract', 'synchronized', 'native', 'transient', 'volatile']
        
        # Extract first line to check for modifiers (they usually appear at the start)
        first_line = symbol.source_text.split('\n')[0].lower()
        
        for keyword in modifier_keywords:
            if keyword in first_line:
                modifiers.append(keyword)
        
        return modifiers

    def _extract_visibility(self, modifiers: List[str]) -> str:
        """Extract visibility modifier from modifiers list."""
        if 'public' in modifiers:
            return 'public'
        elif 'private' in modifiers:
            return 'private'
        elif 'protected' in modifiers:
            return 'protected'
        else:
            return 'package'  # Default package-private in Java
