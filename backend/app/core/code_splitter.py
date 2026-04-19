"""
Advanced code splitter with AST parsing and code_graph relationship building
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

from tree_sitter import Parser, Node
from tree_sitter_language_pack import get_parser

# TREE_SITTER_AVAILABLE = True

import networkx as nx
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)


class SymbolInfo:
    """Information about a code symbol"""
    def __init__(self, name: str, node_type: str, start_line: int, end_line: int, 
                 file_path: str, language: str):
        self.name = name
        self.node_type = node_type
        self.start_line = start_line
        self.end_line = end_line
        self.file_path = file_path
        self.language = language
        self.imports: Set[str] = set()
        self.calls: Set[str] = set()
        self.code_content: str = ""
        self.docstring: Optional[str] = None
        self.parameters: List[str] = []
        self.return_type: Optional[str] = None


class GraphAwareCodeSplitter:
    """Advanced code splitter with AST parsing and code_graph relationship building"""
    
    SUPPORTED_LANGUAGES = {
        '.py': 'python',
        '.js': 'javascript', 
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.cs': 'c_sharp',
        '.cpp': 'cpp',
        '.c': 'c',
        '.rb': 'ruby',
        '.php': 'php',
        '.rs': 'rust',
        '.kt': 'kotlin',
        '.scala': 'scala'
    }
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.symbol_table: Dict[str, SymbolInfo] = {}
        self.call_graph = nx.DiGraph()
        self.import_graph = nx.DiGraph()
        self.combined_graph = nx.DiGraph()
        
        # Store file-level imports for cross-file relationship building
        self.file_imports = {}  # file_path -> set of file-level imports
        
        # Initialize parsers if tree-sitter is available
        self.parsers: Dict[str, Parser] = {}
        self._initialize_parsers()
            
        # Markdown splitter for documentation
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"), 
                ("###", "Header 3"),
                ("####", "Header 4"),
            ]
        )
    
    def _initialize_parsers(self):
        """Initialize tree-sitter parsers for supported languages"""
        parser_mapping = {
            'python': 'python',
            'javascript': 'javascript',
            'typescript': 'typescript', 
            'java': 'java',
            'go': 'go',
            'c_sharp': 'csharp',
            'cpp': 'cpp',
            'c': 'c',
            'ruby': 'ruby',
            'php': 'php',
            'rust': 'rust',
            'kotlin': 'kotlin',
            'scala': 'scala'
        }
        
        for lang_name, ts_name in parser_mapping.items():
            try:
                # language = tree_sitter_languages.get_language(ts_name)
                parser = get_parser(ts_name)
                self.parsers[lang_name] = parser
                logger.debug(f"Initialized parser for {lang_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize parser for {lang_name}: {e}")
    
    def split_repository(self, repo_path: str, file_patterns: Optional[List[str]] = None) -> Tuple[List[Document], List[Document], nx.DiGraph]:
        """
        Split repository into documents and build relationship code_graph
        
        Returns:
            - Text documents (README, docs)
            - Code documents (with metadata)
            - Combined relationship code_graph
        """
        if file_patterns is None:
            file_patterns = ['**/*']
        
        text_docs = []
        code_docs = []
        
        # Process all files in repository
        for root, dirs, files in os.walk(repo_path):
            # Skip common build/cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                      {'node_modules', '__pycache__', 'build', 'dist', 'target', 'vendor'}]
            
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_path)
                
                try:
                    if self._is_text_file(file):
                        text_docs.extend(self._process_text_file(file_path, rel_path))
                    elif self._is_code_file(file):
                        code_docs.extend(self._process_code_file(file_path, rel_path))
                except Exception as e:
                    logger.warning(f"Failed to process {rel_path}: {e}")
        
        # Build combined code_graph
        self._build_combined_graph()
        
        logger.info(f"Processed {len(text_docs)} text documents and {len(code_docs)} code documents")
        logger.info(f"Built code_graph with {self.combined_graph.number_of_nodes()} nodes and {self.combined_graph.number_of_edges()} edges")
        
        return text_docs, code_docs, self.combined_graph
    
    def _is_text_file(self, filename: str) -> bool:
        """Check if file is a text/documentation file"""
        text_extensions = {'.md', '.txt', '.rst', '.org', '.adoc', '.markdown'}
        text_names = {'README', 'CHANGELOG', 'LICENSE', 'CONTRIBUTING'}
        
        ext = Path(filename).suffix.lower()
        name = Path(filename).stem.upper()
        
        return ext in text_extensions or name in text_names
    
    def _is_code_file(self, filename: str) -> bool:
        """Check if file is a source code file"""
        ext = Path(filename).suffix.lower()
        return ext in self.SUPPORTED_LANGUAGES
    
    def _process_text_file(self, file_path: str, rel_path: str) -> List[Document]:
        """Process text/documentation file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if Path(file_path).suffix.lower() in {'.md', '.markdown'}:
                # Use markdown splitter
                docs = self.markdown_splitter.split_text(content)
                return [
                    Document(
                        page_content=doc.page_content,
                        metadata={
                            'source': rel_path,
                            'file_type': 'markdown',
                            'header': doc.metadata.get('Header 1', ''),
                            'section': doc.metadata.get('Header 2', ''),
                            'subsection': doc.metadata.get('Header 3', ''),
                            'chunk_type': 'text'
                        }
                    )
                    for doc in docs
                ]
            else:
                # Simple text splitting
                return [Document(
                    page_content=content,
                    metadata={
                        'source': rel_path,
                        'file_type': 'text',
                        'chunk_type': 'text'
                    }
                )]
                
        except Exception as e:
            logger.error(f"Error processing text file {rel_path}: {e}")
            return []
    
    def _process_code_file(self, file_path: str, rel_path: str) -> List[Document]:
        """Process source code file with AST parsing"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            language = self._get_language(file_path)
            logger.info(f"Detected language - {language}")
            
            if language in self.parsers:
                logger.info(f"Parsing with ast parser - {language}, {file_path}")
                return self._process_with_ast(content, rel_path, language)
            else:
                logger.info(f"Parsing with regex - {language}, {file_path}")
                return self._process_with_regex(content, rel_path, language)
                
        except Exception as e:
            logger.error(f"Error processing code file {rel_path}: {e}")
            return []
    
    def _get_language(self, file_path: str) -> str:
        """Get language from file extension"""
        ext = Path(file_path).suffix.lower()
        return self.SUPPORTED_LANGUAGES.get(ext, 'unknown')
    
    def _get_extension(self, language: str) -> str:
        """Get file extension for a language"""
        extension_map = {
            'python': 'py',
            'javascript': 'js',
            'typescript': 'ts',
            'java': 'java',
            'go': 'go',
            'c_sharp': 'cs',
            'rust': 'rs',
            'ruby': 'rb',
            'kotlin': 'kt',
            'scala': 'scala',
            'cpp': 'cpp',
            'c': 'c',
            'php': 'php'
        }
        return extension_map.get(language, language)

    def _process_with_ast(self, content: str, file_path: str, language: str) -> List[Document]:
        """Process code file using AST parsing"""
        parser = self.parsers[language]
        tree = parser.parse(bytes(content, 'utf8'))
        logger.info(f"Parsed file tree is:\n{tree}")
        
        documents = []
        lines = content.split('\n')
        
        # Define node types to extract for each language
        target_nodes = {
            'python': ['function_def', 'function_definition', 'class_definition', 'import_statement', 'import_from_statement'],
            'javascript': ['function_declaration', 'method_definition', 'class_declaration', 'import_statement', 'arrow_function', 'lexical_declaration'],
            'typescript': ['function_declaration', 'method_definition', 'class_declaration', 'import_statement', 'arrow_function', 'lexical_declaration'],
            'java': ['method_declaration', 'class_declaration', 'import_declaration', 'constructor_declaration'],
            'go': ['function_declaration', 'method_declaration', 'type_declaration', 'import_spec', 'import_declaration'],
            'c_sharp': ['method_declaration', 'class_declaration', 'using_directive', 'constructor_declaration'],
            'rust': ['function_item', 'impl_item', 'struct_item', 'enum_item', 'trait_item', 'use_declaration', 'mod_item'],
            'ruby': ['method', 'class', 'module', 'singleton_method', 'constant_assignment'],
            'kotlin': ['function_declaration', 'class_declaration', 'object_declaration', 'companion_object', 'property_declaration', 'import_header'],
            'scala': ['function_definition', 'class_definition', 'object_definition', 'trait_definition', 'import_expression', 'val_definition'],
        }
        
        nodes_to_extract = target_nodes.get(language, ['function_declaration', 'class_declaration'])
        
        # First, extract file-level imports
        file_imports = self._extract_file_imports(tree.root_node, language)
        logger.info(f"Extracted {len(file_imports)} imports from {file_path}")
        
        # Store file-level imports for relationship building
        self.file_imports[file_path] = file_imports
        
        # Extract symbols and build documents
        for node in self._traverse_tree(tree.root_node):
            if node.type in nodes_to_extract:
                symbol_info = self._extract_symbol_info(node, lines, file_path, language)
                if symbol_info:
                    # Symbols should only have their own local imports, not file-level imports
                    # File-level imports are handled separately and shouldn't be attributed to individual symbols
                    
                    self.symbol_table[f"{file_path}:{symbol_info.name}:{symbol_info.start_line}"] = symbol_info
                    
                    # Create document for this symbol
                    doc = Document(
                        page_content=symbol_info.code_content,
                        metadata={
                            'source': file_path,
                            'symbol': symbol_info.name,
                            'node_type': symbol_info.node_type,
                            'start_line': symbol_info.start_line,
                            'end_line': symbol_info.end_line,
                            'language': language,
                            'imports': list(symbol_info.imports),
                            'calls': list(symbol_info.calls),
                            'chunk_type': 'code',
                            'parameters': symbol_info.parameters,
                            'return_type': symbol_info.return_type,
                            'docstring': symbol_info.docstring
                        }
                    )
                    documents.append(doc)
        
        # Also create file-level document if it has content not covered by symbols
        if not documents:
            documents.append(Document(
                page_content=content,
                metadata={
                    'source': file_path,
                    'language': language,
                    'chunk_type': 'code',
                    'symbol': 'file_level',
                    'imports': list(file_imports)
                }
            ))
        
        return documents
    
    def _process_with_regex(self, content: str, file_path: str, language: str) -> List[Document]:
        """Fallback processing using regex patterns"""
        documents = []
        
        # Simple regex patterns for common languages
        patterns = {
            'python': [
                r'^def\s+(\w+)\s*\(',
                r'^class\s+(\w+)\s*[:(\[]',
                r'^import\s+(\w+)',
                r'^from\s+(\w+)'
            ],
            'javascript': [
                r'function\s+(\w+)\s*\(',
                r'class\s+(\w+)\s*{',
                r'const\s+(\w+)\s*=\s*\(',
                r'import.*from\s+["\']([^"\']+)["\']'
            ]
        }
        
        lang_patterns = patterns.get(language, patterns['python'])
        
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_symbol = None
        
        for i, line in enumerate(lines):
            current_chunk.append(line)
            
            # Check for symbol definitions
            for pattern in lang_patterns:
                match = re.search(pattern, line)
                if match:
                    if current_chunk and len('\n'.join(current_chunk)) > 50:
                        # Save previous chunk
                        chunk_content = '\n'.join(current_chunk[:-1])
                        if chunk_content.strip():
                            documents.append(Document(
                                page_content=chunk_content,
                                metadata={
                                    'source': file_path,
                                    'language': language,
                                    'chunk_type': 'code',
                                    'symbol': current_symbol or f'chunk_{len(documents)}',
                                    'start_line': max(0, i - len(current_chunk) + 1),
                                    'end_line': i
                                }
                            ))
                    current_chunk = [line]
                    current_symbol = match.group(1)
                    break
            
            # Chunk size limit
            if len('\n'.join(current_chunk)) > self.chunk_size:
                chunk_content = '\n'.join(current_chunk)
                documents.append(Document(
                    page_content=chunk_content,
                    metadata={
                        'source': file_path,
                        'language': language,
                        'chunk_type': 'code',
                        'symbol': current_symbol or f'chunk_{len(documents)}',
                        'start_line': max(0, i - len(current_chunk)),
                        'end_line': i
                    }
                ))
                current_chunk = []
                current_symbol = None
        
        # Add final chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            if chunk_content.strip():
                documents.append(Document(
                    page_content=chunk_content,
                    metadata={
                        'source': file_path,
                        'language': language,
                        'chunk_type': 'code',
                        'symbol': current_symbol or f'chunk_{len(documents)}',
                        'start_line': max(0, len(lines) - len(current_chunk)),
                        'end_line': len(lines)
                    }
                ))
        
        return documents
    
    def _traverse_tree(self, node: 'Node'):
        """Traverse AST tree nodes"""
        yield node
        for child in node.children:
            yield from self._traverse_tree(child)
    
    def _extract_symbol_info(self, node: 'Node', lines: List[str], file_path: str, language: str) -> Optional[SymbolInfo]:
        """Extract symbol information from AST node"""
        try:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            # Extract symbol name
            name = self._get_symbol_name(node, language)
            logger.info(f"Extract symbol information for {name} in {file_path}")
            if not name:
                return None
            
            # Extract code content
            code_lines = lines[start_line-1:end_line]
            code_content = '\n'.join(code_lines)
            
            # Refine Go type_declaration sub-types (struct, interface, type_alias)
            refined_node_type = node.type
            if language == 'go' and node.type == 'type_declaration':
                refined_node_type = self._classify_go_type_declaration(node)

            symbol_info = SymbolInfo(
                name=name,
                node_type=refined_node_type,
                start_line=start_line,
                end_line=end_line,
                file_path=file_path,
                language=language
            )
            symbol_info.code_content = code_content
            
            # Extract additional information
            symbol_info.docstring = self._extract_docstring(node, lines, language)
            symbol_info.parameters = self._extract_parameters(node, language)
            symbol_info.return_type = self._extract_return_type(node, language)
            
            # Extract imports and calls
            symbol_info.imports = self._extract_imports(node, language)
            symbol_info.calls = self._extract_calls(node, language)
            logger.info(f"Extract symbol information {symbol_info} for {symbol_info.name} in {file_path}")
            
            return symbol_info
            
        except Exception as e:
            logger.warning(f"Failed to extract symbol info: {e}")
            return None
    
    def _get_symbol_name(self, node: 'Node', language: str) -> Optional[str]:
        """Extract symbol name from node"""
        if language == 'python':
            if node.type in ['function_def', 'function_definition', 'class_definition']:
                for child in node.children:
                    if child.type == 'identifier':
                        return child.text.decode('utf8')
        elif language in ['javascript', 'typescript']:
            if node.type in ['function_declaration', 'class_declaration']:
                for child in node.children:
                    if child.type == 'identifier':
                        return child.text.decode('utf8')
            elif node.type == 'method_definition':
                for child in node.children:
                    if child.type == 'property_identifier':
                        return child.text.decode('utf8')
            elif node.type == 'lexical_declaration':
                # Handle const/let function assignments
                for child in node.children:
                    if child.type == 'variable_declarator':
                        for var_child in child.children:
                            if var_child.type == 'identifier':
                                return var_child.text.decode('utf8')
            elif node.type == 'arrow_function':
                # For arrow functions, we need to get the name from parent context
                return f"arrow_function_{node.start_point[0]}"
        elif language == 'java':
            if node.type in ['method_declaration', 'class_declaration', 'constructor_declaration']:
                for child in node.children:
                    if child.type == 'identifier':
                        return child.text.decode('utf8')
        elif language == 'go':
            if node.type == 'function_declaration':
                for child in node.children:
                    if child.type == 'identifier':
                        return child.text.decode('utf8')
            elif node.type == 'type_declaration':
                # Go type_declaration nests name under type_spec/type_identifier
                for child in node.children:
                    if child.type == 'identifier':
                        return child.text.decode('utf8')
                    if child.type == 'type_spec':
                        for gc in child.children:
                            if gc.type == 'type_identifier':
                                return gc.text.decode('utf8')
            elif node.type == 'method_declaration':
                for child in node.children:
                    if child.type == 'field_identifier':
                        return child.text.decode('utf8')
            elif node.type == 'import_spec':
                for child in node.children:
                    if child.type == 'interpreted_string_literal':
                        return child.text.decode('utf8').strip('"')
        elif language == 'c_sharp':
            if node.type in ['method_declaration', 'class_declaration', 'constructor_declaration']:
                for child in node.children:
                    if child.type == 'identifier':
                        return child.text.decode('utf8')
            elif node.type == 'using_directive':
                for child in node.children:
                    if child.type in ['identifier', 'qualified_name']:
                        return child.text.decode('utf8')
        elif language == 'rust':
            if node.type in ['function_item', 'mod_item']:
                for child in node.children:
                    if child.type == 'identifier':
                        return child.text.decode('utf8')
            elif node.type in ['struct_item', 'enum_item', 'trait_item']:
                for child in node.children:
                    if child.type == 'type_identifier':
                        return child.text.decode('utf8')
            elif node.type == 'impl_item':
                # For impl blocks, create a descriptive name
                type_name = None
                trait_name = None
                for child in node.children:
                    if child.type == 'type_identifier':
                        type_name = child.text.decode('utf8')
                    elif child.type == 'generic_type' and child.children:
                        for gc in child.children:
                            if gc.type == 'type_identifier':
                                trait_name = gc.text.decode('utf8')
                                break
                if trait_name and type_name:
                    return f"{trait_name} for {type_name}"
                elif type_name:
                    return f"impl {type_name}"
                return "impl"
        elif language == 'ruby':
            if node.type in ['method', 'singleton_method', 'class', 'module']:
                for child in node.children:
                    if child.type in ['identifier', 'constant']:
                        return child.text.decode('utf8')
            elif node.type == 'constant_assignment':
                for child in node.children:
                    if child.type == 'constant':
                        return child.text.decode('utf8')
        elif language == 'kotlin':
            if node.type in ['function_declaration']:
                for child in node.children:
                    if child.type == 'simple_identifier':
                        return child.text.decode('utf8')
            elif node.type in ['class_declaration', 'object_declaration']:
                for child in node.children:
                    if child.type == 'type_identifier':
                        return child.text.decode('utf8')
            elif node.type == 'companion_object':
                # Check if companion object has a name (e.g., "companion object Factory")
                for child in node.children:
                    if child.type == 'type_identifier':
                        return f"companion object {child.text.decode('utf8')}"
                # Anonymous companion object
                return "companion object"
            elif node.type == 'property_declaration':
                for child in node.children:
                    if child.type == 'variable_declaration':
                        for var_child in child.children:
                            if var_child.type == 'simple_identifier':
                                return var_child.text.decode('utf8')
        elif language == 'scala':
            if node.type in ['function_definition', 'class_definition', 'object_definition', 'trait_definition']:
                for child in node.children:
                    if child.type == 'identifier':
                        return child.text.decode('utf8')
            elif node.type == 'val_definition':
                for child in node.children:
                    if child.type == 'identifier':
                        return child.text.decode('utf8')
        
        return None
    
    @staticmethod
    def _classify_go_type_declaration(node: 'Node') -> str:
        """Inspect a Go type_declaration AST node and return a refined type.

        Go ``type Foo struct { ... }`` produces a tree-sitter tree:
            type_declaration
              type_spec
                type_identifier  (name)
                struct_type      (body)

        This method looks at the body child of the first ``type_spec``
        to determine whether this is a struct, interface, or type alias.
        """
        for child in node.children:
            if child.type == 'type_spec':
                for gc in child.children:
                    if gc.type == 'struct_type':
                        return 'struct_item'
                    if gc.type == 'interface_type':
                        return 'interface_item'
                # No struct/interface body → type alias (e.g. type MyInt int)
                return 'type_alias_item'
        return 'type_declaration'  # fallback

    def _extract_docstring(self, node: 'Node', lines: List[str], language: str) -> Optional[str]:
        """Extract docstring/comments for symbol"""
        # Implementation depends on language-specific patterns
        # This is a simplified version
        if language == 'python':
            # Look for string literal right after function/class definition
            for child in node.children:
                if child.type == 'block':
                    for stmt in child.children:
                        if stmt.type == 'expression_statement':
                            for expr_child in stmt.children:
                                if expr_child.type == 'string':
                                    return expr_child.text.decode('utf8').strip('"\' ')
        return None
    
    def _extract_parameters(self, node: 'Node', language: str) -> List[str]:
        """Extract function parameters for multiple languages"""
        params = []
        
        if language == 'python' and node.type == 'function_def':
            for child in node.children:
                if child.type == 'parameters':
                    for param_child in child.children:
                        if param_child.type == 'identifier':
                            params.append(param_child.text.decode('utf8'))
                        elif param_child.type == 'typed_parameter':
                            # Handle typed parameters like 'name: str'
                            for typed_child in param_child.children:
                                if typed_child.type == 'identifier':
                                    params.append(typed_child.text.decode('utf8'))
                                    break
        
        elif language in ['javascript', 'typescript'] and node.type in ['function_declaration', 'method_definition']:
            for child in node.children:
                if child.type == 'formal_parameters':
                    for param_child in child.children:
                        if param_child.type == 'identifier':
                            params.append(param_child.text.decode('utf8'))
                        elif param_child.type == 'required_parameter':
                            # Handle typed parameters in TypeScript
                            for typed_child in param_child.children:
                                if typed_child.type == 'identifier':
                                    params.append(typed_child.text.decode('utf8'))
                                    break
        
        elif language == 'java' and node.type == 'method_declaration':
            for child in node.children:
                if child.type == 'formal_parameters':
                    for param_child in child.children:
                        if param_child.type == 'formal_parameter':
                            # Java parameters have type and name
                            identifiers = [c for c in param_child.children if c.type == 'identifier']
                            if identifiers:
                                params.append(identifiers[-1].text.decode('utf8'))  # Last identifier is the name
        
        elif language == 'go' and node.type == 'function_declaration':
            for child in node.children:
                if child.type == 'parameter_list':
                    for param_child in child.children:
                        if param_child.type == 'parameter_declaration':
                            # Go parameters: name type
                            identifiers = [c for c in param_child.children if c.type == 'identifier']
                            if identifiers:
                                params.append(identifiers[0].text.decode('utf8'))  # First identifier is the name
        
        elif language == 'c_sharp' and node.type == 'method_declaration':
            for child in node.children:
                if child.type == 'parameter_list':
                    for param_child in child.children:
                        if param_child.type == 'parameter':
                            # C# parameters: type name
                            identifiers = [c for c in param_child.children if c.type == 'identifier']
                            if identifiers:
                                params.append(identifiers[-1].text.decode('utf8'))  # Last identifier is the name
        
        elif language == 'rust' and node.type == 'function_item':
            for child in node.children:
                if child.type == 'parameters':
                    for param_child in child.children:
                        if param_child.type == 'parameter':
                            # Rust parameters: name: type
                            for param_part in param_child.children:
                                if param_part.type == 'identifier':
                                    params.append(param_part.text.decode('utf8'))
                                    break
        
        elif language == 'ruby' and node.type in ['method', 'singleton_method']:
            for child in node.children:
                if child.type == 'method_parameters':
                    for param_child in child.children:
                        if param_child.type == 'identifier':
                            params.append(param_child.text.decode('utf8'))
                        elif param_child.type == 'optional_parameter':
                            # Ruby optional parameters: name = default
                            for param_part in param_child.children:
                                if param_part.type == 'identifier':
                                    params.append(param_part.text.decode('utf8'))
                                    break
        
        elif language == 'kotlin' and node.type == 'function_declaration':
            for child in node.children:
                if child.type == 'function_value_parameters':
                    for param_child in child.children:
                        if param_child.type == 'parameter':
                            # Kotlin parameters: name: type
                            for param_part in param_child.children:
                                if param_part.type == 'simple_identifier':
                                    params.append(param_part.text.decode('utf8'))
                                    break
        
        elif language == 'scala' and node.type == 'function_definition':
            for child in node.children:
                if child.type == 'parameters':
                    for param_child in child.children:
                        if param_child.type == 'parameter':
                            # Scala parameters: name: type
                            for param_part in param_child.children:
                                if param_part.type == 'identifier':
                                    params.append(param_part.text.decode('utf8'))
                                    break
        
        return params
    
    def _extract_return_type(self, node: 'Node', language: str) -> Optional[str]:
        """Extract return type annotation for multiple languages"""
        
        if language == 'python' and node.type == 'function_def':
            # Look for return type annotation after ->
            for child in node.children:
                if child.type == 'type':
                    return child.text.decode('utf8')
        
        elif language == 'typescript' and node.type in ['function_declaration', 'method_definition']:
            # TypeScript: function name(): ReturnType
            for child in node.children:
                if child.type == 'type_annotation':
                    return child.text.decode('utf8').lstrip(': ')
        
        elif language == 'java' and node.type == 'method_declaration':
            # Java: ReturnType methodName()
            for child in node.children:
                if child.type in ['type_identifier', 'primitive_type', 'generic_type']:
                    return child.text.decode('utf8')
        
        elif language == 'go' and node.type == 'function_declaration':
            # Go: func name() ReturnType
            for child in node.children:
                if child.type in ['type_identifier', 'pointer_type']:
                    return child.text.decode('utf8')
        
        elif language == 'c_sharp' and node.type == 'method_declaration':
            # C#: ReturnType MethodName()
            for child in node.children:
                if child.type in ['predefined_type', 'identifier', 'generic_name']:
                    return child.text.decode('utf8')
        
        elif language == 'rust' and node.type == 'function_item':
            # Rust: fn name() -> ReturnType
            for child in node.children:
                if child.type == 'function_type':
                    return child.text.decode('utf8')
                elif child.type == 'type_identifier':
                    # Look for return type after arrow
                    return child.text.decode('utf8')
        
        elif language == 'ruby' and node.type in ['method', 'singleton_method']:
            # Ruby typically doesn't have explicit return types in syntax
            # But we can look for type annotations if they exist
            return None
        
        elif language == 'kotlin' and node.type == 'function_declaration':
            # Kotlin: fun name(): ReturnType
            for child in node.children:
                if child.type == 'type':
                    return child.text.decode('utf8')
        
        elif language == 'scala' and node.type == 'function_definition':
            # Scala: def name(): ReturnType
            for child in node.children:
                if child.type == 'type':
                    return child.text.decode('utf8')
        
        return None
    
    def _extract_imports(self, node: 'Node', language: str) -> Set[str]:
        """Extract import statements within the symbol (only local imports, not file-level)"""
        imports = set()
        
        # Only extract imports that are truly LOCAL to this symbol
        # (e.g., imports inside function bodies), not file-level imports
        # File-level imports are handled separately in _extract_file_imports
        
        for child_node in self._traverse_tree(node):
            # Only capture imports that are inside function/method bodies or other scopes
            # Skip imports that are at the top level of the symbol definition
            if self._is_local_import(child_node, node, language):
                if language == 'python':
                    if child_node.type == 'import_statement':
                    # import module or import module as alias
                        for child in child_node.children:
                            if child.type == 'dotted_name':
                                imports.add(child.text.decode('utf8'))
                            elif child.type == 'identifier':
                                imports.add(child.text.decode('utf8'))
                            elif child.type == 'aliased_import':
                                # import module as alias
                                for alias_child in child.children:
                                    if alias_child.type == 'dotted_name':
                                        imports.add(alias_child.text.decode('utf8'))
                                    elif alias_child.type == 'identifier':
                                        imports.add(alias_child.text.decode('utf8'))
                    
                    elif child_node.type == 'import_from_statement':
                        # from module import name
                        module_name = None
                        for child in child_node.children:
                            if child.type == 'dotted_name':
                                module_name = child.text.decode('utf8')
                            elif child.type == 'import_list':
                                for import_child in child.children:
                                    if import_child.type == 'identifier':
                                        imports.add(import_child.text.decode('utf8'))
                                    elif import_child.type == 'aliased_import':
                                        # from module import name as alias
                                        for alias_child in import_child.children:
                                            if alias_child.type == 'identifier':
                                                imports.add(alias_child.text.decode('utf8'))
                                                break
                            elif child.type == 'wildcard_import':
                                # from module import *
                                if module_name:
                                    imports.add(f"{module_name}.*")
                        
                        # Add the module itself
                        if module_name:
                            imports.add(module_name)
                elif language in ['javascript', 'typescript']:
                    if child_node.type == 'import_statement':
                        # import { name } from 'module' or import name from 'module'
                        module_name = None
                        for child in child_node.children:
                            if child.type == 'string':
                                # Extract module name from string
                                module_name = child.text.decode('utf8').strip('\'"')
                                imports.add(module_name)
                            elif child.type == 'import_clause':
                                # Extract imported names
                                for import_child in self._traverse_tree(child):
                                    if import_child.type == 'identifier':
                                        imports.add(import_child.text.decode('utf8'))
                                    elif import_child.type == 'import_specifier':
                                        # Named imports
                                        for spec_child in import_child.children:
                                            if spec_child.type == 'identifier':
                                                imports.add(spec_child.text.decode('utf8'))
                
                    # Dynamic imports
                    elif child_node.type == 'call_expression':
                        for child in child_node.children:
                            if child.type == 'identifier' and child.text.decode('utf8') == 'import':
                                # Dynamic import() calls
                                for arg_child in child_node.children:
                                    if arg_child.type == 'string':
                                        imports.add(arg_child.text.decode('utf8').strip('\'"'))
                    
                    # require() calls (CommonJS)
                    elif child_node.type == 'call_expression':
                        for child in child_node.children:
                            if child.type == 'identifier' and child.text.decode('utf8') == 'require':
                                for arg_child in child_node.children:
                                    if arg_child.type == 'string':
                                        imports.add(arg_child.text.decode('utf8').strip('\'"'))
            
            elif language == 'java':
                if child_node.type == 'import_declaration':
                    # import package.Class;
                    for child in child_node.children:
                        if child.type in ['scoped_identifier', 'identifier']:
                            imports.add(child.text.decode('utf8'))
                        elif child.type == 'asterisk':
                            # import package.*;
                            imports.add('wildcard_import')
                
                # Static imports
                elif child_node.type == 'import_declaration':
                    import_text = child_node.text.decode('utf8')
                    if 'static' in import_text:
                        imports.add('static_import')
            
            elif language == 'go':
                if child_node.type == 'import_spec':
                    # import "package"
                    for child in child_node.children:
                        if child.type == 'interpreted_string_literal':
                            package_name = child.text.decode('utf8').strip('"')
                            imports.add(package_name)
                        elif child.type == 'identifier':
                            # Aliased imports: alias "package"
                            imports.add(child.text.decode('utf8'))
                
                elif child_node.type == 'import_declaration':
                    # Handle import blocks
                    for child in child_node.children:
                        if child.type == 'import_spec_list':
                            for spec_child in child.children:
                                if spec_child.type == 'import_spec':
                                    for spec_inner in spec_child.children:
                                        if spec_inner.type == 'interpreted_string_literal':
                                            package_name = spec_inner.text.decode('utf8').strip('"')
                                            imports.add(package_name)
            
            elif language == 'c_sharp':
                if child_node.type == 'using_directive':
                    # using System.Collections;
                    for child in child_node.children:
                        if child.type in ['qualified_name', 'identifier']:
                            imports.add(child.text.decode('utf8'))
                        elif child.type == 'name_equals':
                            # using alias = Namespace.Type;
                            imports.add('alias_import')
                
                # Static using
                elif child_node.type == 'using_directive':
                    using_text = child_node.text.decode('utf8')
                    if 'static' in using_text:
                        imports.add('static_using')
            
            elif language == 'rust':
                if child_node.type == 'use_declaration':
                    # use crate::module::Item; or use std::collections::HashMap;
                    for child in child_node.children:
                        if child.type == 'scoped_identifier':
                            # use std::collections::HashMap;
                            imports.add(child.text.decode('utf8'))
                        elif child.type == 'identifier':
                            # Simple use statements
                            imports.add(child.text.decode('utf8'))
                        elif child.type == 'scoped_use_list':
                            # use serde::{Deserialize, Serialize};
                            for scope_child in child.children:
                                if scope_child.type == 'identifier':
                                    imports.add(scope_child.text.decode('utf8'))
                                elif scope_child.type == 'use_list':
                                    # Extract items from the list
                                    for list_child in scope_child.children:
                                        if list_child.type == 'identifier':
                                            imports.add(list_child.text.decode('utf8'))
                        elif child.type == 'use_list':
                            # use {Item1, Item2};
                            for use_child in child.children:
                                if use_child.type == 'identifier':
                                    imports.add(use_child.text.decode('utf8'))
                        elif child.type == 'use_as_clause':
                            # use module::Item as Alias;
                            for alias_child in child.children:
                                if alias_child.type == 'identifier':
                                    imports.add(alias_child.text.decode('utf8'))
                                    break
            
            elif language == 'ruby':
                if child_node.type == 'call':
                    # Ruby require/load calls
                    for child in child_node.children:
                        if child.type == 'identifier' and child.text.decode('utf8') in ['require', 'load', 'require_relative']:
                            # Look for the argument (string literal)
                            for arg_child in child_node.children:
                                if arg_child.type == 'argument_list':
                                    # Extract from argument list
                                    for arg_item in arg_child.children:
                                        if arg_item.type == 'string':
                                            # Remove quotes and add to imports
                                            module_name = arg_item.text.decode('utf8').strip('\'"')
                                            imports.add(module_name)
                                elif arg_child.type == 'string':
                                    # Direct string argument
                                    module_name = arg_child.text.decode('utf8').strip('\'"')
                                    imports.add(module_name)
                            break
            
            elif language == 'kotlin':
                if child_node.type == 'import_list':
                    # Kotlin import list contains multiple import_header nodes
                    for import_child in child_node.children:
                        if import_child.type == 'import_header':
                            for header_child in import_child.children:
                                if header_child.type == 'identifier':
                                    imports.add(header_child.text.decode('utf8'))
                                elif header_child.type == 'import_alias':
                                    # import package.Class as Alias
                                    for alias_child in header_child.children:
                                        if alias_child.type == 'simple_identifier':
                                            imports.add(alias_child.text.decode('utf8'))
                                            break
                elif child_node.type == 'import_header':
                    # Single import statement
                    for header_child in child_node.children:
                        if header_child.type == 'identifier':
                            imports.add(header_child.text.decode('utf8'))
                        elif header_child.type == 'import_alias':
                            # import package.Class as Alias
                            for alias_child in header_child.children:
                                if alias_child.type == 'simple_identifier':
                                    imports.add(alias_child.text.decode('utf8'))
                                    break
            
            elif language == 'scala':
                if child_node.type == 'import_declaration':
                    # import package.Class or import package.{Class1, Class2}
                    import_parts = []
                    
                    # Collect all identifiers and handle different import patterns
                    for child in child_node.children:
                        if child.type == 'identifier':
                            import_parts.append(child.text.decode('utf8'))
                        elif child.type == 'namespace_selectors':
                            # import package.{Class1, Class2}
                            base_import = '.'.join(import_parts) if import_parts else ''
                            for selector_child in child.children:
                                if selector_child.type == 'identifier':
                                    if base_import:
                                        imports.add(f"{base_import}.{selector_child.text.decode('utf8')}")
                                    else:
                                        imports.add(selector_child.text.decode('utf8'))
                                elif selector_child.type == 'renamed_identifier':
                                    # import package.{Class => Alias}
                                    for rename_child in selector_child.children:
                                        if rename_child.type == 'identifier':
                                            if base_import:
                                                imports.add(f"{base_import}.{rename_child.text.decode('utf8')}")
                                            else:
                                                imports.add(rename_child.text.decode('utf8'))
                                            break
                        elif child.type == 'namespace_wildcard':
                            # import package._
                            base_import = '.'.join(import_parts) if import_parts else ''
                            if base_import:
                                imports.add(f"{base_import}._")
        
        return imports

    def _extract_file_imports(self, root_node: 'Node', language: str) -> Set[str]:
        """Extract import statements at the file level"""
        imports = set()
        
        # Traverse only top-level nodes for imports
        for child_node in root_node.children:
            if language == 'rust':
                if child_node.type == 'use_declaration':
                    # use crate::module::Item; or use std::collections::HashMap;
                    for child in child_node.children:
                        if child.type == 'scoped_identifier':
                            # use std::collections::HashMap;
                            imports.add(child.text.decode('utf8'))
                        elif child.type == 'identifier':
                            # Simple use statements
                            imports.add(child.text.decode('utf8'))
                        elif child.type == 'scoped_use_list':
                            # use serde::{Deserialize, Serialize};
                            for scope_child in child.children:
                                if scope_child.type == 'identifier':
                                    imports.add(scope_child.text.decode('utf8'))
                                elif scope_child.type == 'use_list':
                                    # Extract items from the list
                                    for list_child in scope_child.children:
                                        if list_child.type == 'identifier':
                                            imports.add(list_child.text.decode('utf8'))
                        elif child.type == 'use_list':
                            # use {Item1, Item2};
                            for use_child in child.children:
                                if use_child.type == 'identifier':
                                    imports.add(use_child.text.decode('utf8'))
                        elif child.type == 'use_as_clause':
                            # use module::Item as Alias;
                            for alias_child in child.children:
                                if alias_child.type == 'identifier':
                                    imports.add(alias_child.text.decode('utf8'))
                                    break
            
            elif language == 'ruby':
                if child_node.type == 'call':
                    # Ruby require/load calls
                    for child in child_node.children:
                        if child.type == 'identifier' and child.text.decode('utf8') in ['require', 'load', 'require_relative']:
                            # Look for the argument (string literal)
                            for arg_child in child_node.children:
                                if arg_child.type == 'argument_list':
                                    # Extract from argument list
                                    for arg_item in arg_child.children:
                                        if arg_item.type == 'string':
                                            # Remove quotes and add to imports
                                            module_name = arg_item.text.decode('utf8').strip('\'"')
                                            imports.add(module_name)
                                elif arg_child.type == 'string':
                                    # Direct string argument
                                    module_name = arg_child.text.decode('utf8').strip('\'"')
                                    imports.add(module_name)
                            break
            
            elif language == 'kotlin':
                if child_node.type == 'import_list':
                    # Kotlin import list contains multiple import_header nodes
                    for import_child in child_node.children:
                        if import_child.type == 'import_header':
                            for header_child in import_child.children:
                                if header_child.type == 'identifier':
                                    imports.add(header_child.text.decode('utf8'))
                                elif header_child.type == 'import_alias':
                                    # import package.Class as Alias
                                    for alias_child in header_child.children:
                                        if alias_child.type == 'simple_identifier':
                                            imports.add(alias_child.text.decode('utf8'))
                                            break
                elif child_node.type == 'import_header':
                    # Single import statement
                    for header_child in child_node.children:
                        if header_child.type == 'identifier':
                            imports.add(header_child.text.decode('utf8'))
                        elif header_child.type == 'import_alias':
                            # import package.Class as Alias
                            for alias_child in header_child.children:
                                if alias_child.type == 'simple_identifier':
                                    imports.add(alias_child.text.decode('utf8'))
                                    break
            
            elif language == 'scala':
                if child_node.type == 'import_declaration':
                    # import package.Class or import package.{Class1, Class2}
                    import_parts = []
                    
                    # Collect all identifiers and handle different import patterns
                    for child in child_node.children:
                        if child.type == 'identifier':
                            import_parts.append(child.text.decode('utf8'))
                        elif child.type == 'namespace_selectors':
                            # import package.{Class1, Class2}
                            base_import = '.'.join(import_parts) if import_parts else ''
                            for selector_child in child.children:
                                if selector_child.type == 'identifier':
                                    if base_import:
                                        imports.add(f"{base_import}.{selector_child.text.decode('utf8')}")
                                    else:
                                        imports.add(selector_child.text.decode('utf8'))
                                elif selector_child.type == 'renamed_identifier':
                                    # import package.{Class => Alias}
                                    for rename_child in selector_child.children:
                                        if rename_child.type == 'identifier':
                                            if base_import:
                                                imports.add(f"{base_import}.{rename_child.text.decode('utf8')}")
                                            else:
                                                imports.add(rename_child.text.decode('utf8'))
                                            break
                        elif child.type == 'namespace_wildcard':
                            # import package._
                            base_import = '.'.join(import_parts) if import_parts else ''
                            if base_import:
                                imports.add(f"{base_import}._")
            
            # Handle other languages (Python, JavaScript, etc.) at file level too
            elif language == 'python':
                if child_node.type == 'import_statement':
                    # import module or import module as alias
                    for child in child_node.children:
                        if child.type == 'dotted_name':
                            imports.add(child.text.decode('utf8'))
                        elif child.type == 'identifier':
                            imports.add(child.text.decode('utf8'))
                        elif child.type == 'aliased_import':
                            # import module as alias
                            for alias_child in child.children:
                                if alias_child.type == 'dotted_name':
                                    imports.add(alias_child.text.decode('utf8'))
                                elif alias_child.type == 'identifier':
                                    imports.add(alias_child.text.decode('utf8'))
                
                elif child_node.type == 'import_from_statement':
                    # from module import name
                    module_name = None
                    for child in child_node.children:
                        if child.type == 'dotted_name':
                            module_name = child.text.decode('utf8')
                            imports.add(module_name)
                        elif child.type == 'import_list':
                            for import_child in child.children:
                                if import_child.type == 'identifier':
                                    imports.add(import_child.text.decode('utf8'))
                                elif import_child.type == 'aliased_import':
                                    # from module import name as alias
                                    for alias_child in import_child.children:
                                        if alias_child.type == 'identifier':
                                            imports.add(alias_child.text.decode('utf8'))
                                            break
                        elif child.type == 'wildcard_import':
                            # from module import *
                            if module_name:
                                imports.add(f"{module_name}.*")
            
            # Handle other languages at file level
            elif language in ['javascript', 'typescript']:
                if child_node.type == 'import_statement':
                    # import { name } from 'module' or import name from 'module'
                    for child in child_node.children:
                        if child.type == 'string':
                            # Extract module name from string
                            module_name = child.text.decode('utf8').strip('\'"')
                            imports.add(module_name)
                        elif child.type == 'import_clause':
                            # Extract imported names
                            for import_child in self._traverse_tree(child):
                                if import_child.type == 'identifier':
                                    imports.add(import_child.text.decode('utf8'))
            
            elif language == 'java':
                if child_node.type == 'import_declaration':
                    # import package.Class;
                    for child in child_node.children:
                        if child.type in ['scoped_identifier', 'identifier']:
                            imports.add(child.text.decode('utf8'))
            
            elif language == 'go':
                if child_node.type == 'import_declaration':
                    # Handle import blocks
                    for child in child_node.children:
                        if child.type == 'import_spec_list':
                            for spec_child in child.children:
                                if spec_child.type == 'import_spec':
                                    for spec_inner in spec_child.children:
                                        if spec_inner.type == 'interpreted_string_literal':
                                            package_name = spec_inner.text.decode('utf8').strip('"')
                                            imports.add(package_name)
                elif child_node.type == 'import_spec':
                    # Single import
                    for child in child_node.children:
                        if child.type == 'interpreted_string_literal':
                            package_name = child.text.decode('utf8').strip('"')
                            imports.add(package_name)
            
            elif language == 'c_sharp':
                if child_node.type == 'using_directive':
                    # using System.Collections;
                    for child in child_node.children:
                        if child.type in ['qualified_name', 'identifier']:
                            imports.add(child.text.decode('utf8'))
        
        return imports

    def get_symbol_neighbors(self, symbol_key: str, hops: int = 1) -> Set[str]:
        """Get neighboring symbols in the relationship code_graph"""
        if symbol_key not in self.combined_graph:
            return set()
        
        neighbors = set()
        current_nodes = {symbol_key}
        
        for _ in range(hops):
            next_nodes = set()
            for node in current_nodes:
                # Get both incoming and outgoing neighbors
                next_nodes.update(self.combined_graph.predecessors(node))
                next_nodes.update(self.combined_graph.successors(node))
            
            neighbors.update(next_nodes)
            current_nodes = next_nodes - neighbors  # Only expand to new nodes
        
        return neighbors - {symbol_key}  # Exclude the original symbol
    
    def export_graph(self) -> Dict:
        """Export code_graph in serializable format"""
        return {
            'nodes': [
                {
                    'id': node_id,
                    **data
                }
                for node_id, data in self.combined_graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': source,
                    'target': target,
                    **data
                }
                for source, target, data in self.combined_graph.edges(data=True)
            ]
        }
    
    def extract_symbols(self, file_path: str, language: str) -> List[SymbolInfo]:
        """Extract symbols from a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.splitlines()
            symbols = []
            
            # Try AST-based extraction first
            if language in self.parsers:
                try:
                    tree = self.parsers[language].parse(content.encode('utf-8'))
                    root_node = tree.root_node
                    
                    # Get target nodes for this language (same as _process_with_ast)
                    target_nodes = {
                        'python': ['function_def', 'function_definition', 'class_definition', 'import_statement', 'import_from_statement'],
                        'javascript': ['function_declaration', 'method_definition', 'class_declaration', 'import_statement', 'arrow_function', 'lexical_declaration'],
                        'typescript': ['function_declaration', 'method_definition', 'class_declaration', 'import_statement', 'arrow_function', 'lexical_declaration'],
                        'java': ['method_declaration', 'class_declaration', 'import_declaration', 'constructor_declaration'],
                        'go': ['function_declaration', 'method_declaration', 'type_declaration', 'import_spec', 'import_declaration'],
                        'c_sharp': ['method_declaration', 'class_declaration', 'using_directive', 'constructor_declaration'],
                        'rust': ['function_item', 'impl_item', 'struct_item', 'enum_item', 'trait_item', 'use_declaration', 'mod_item'],
                        'ruby': ['method', 'class', 'module', 'singleton_method', 'constant_assignment'],
                        'kotlin': ['function_declaration', 'class_declaration', 'object_declaration', 'companion_object', 'property_declaration', 'import_header'],
                        'scala': ['function_definition', 'class_definition', 'object_definition', 'trait_definition', 'import_expression', 'val_definition'],
                    }
                    
                    nodes_to_find = target_nodes.get(language, [])
                    
                    # Find all matching nodes
                    for node in self._traverse_tree(root_node):
                        if node.type in nodes_to_find:
                            symbol_info = self._extract_symbol_info(node, lines, file_path, language)
                            if symbol_info:
                                symbols.append(symbol_info)
                except Exception as e:
                    print(f"AST extraction failed for {language}: {e}")
                    
            return symbols
            
        except Exception as e:
            print(f"Error extracting symbols from {file_path}: {e}")
            return []
    
    def add_symbol(self, symbol: SymbolInfo):
        """Add a symbol to the symbol table"""
        # Create a unique key for the symbol
        key = f"{symbol.file_path}:{symbol.name}:{symbol.start_line}"
        self.symbol_table[key] = symbol
    
    def split_file(self, file_path: str, language: str) -> List[Document]:
        """Split a single file into code documents"""
        if not os.path.isfile(file_path):
            logger.warning(f"File not found: {file_path}")
            return []
        
        rel_path = os.path.relpath(file_path, os.getcwd())
        
        try:
            if self._is_code_file(file_path):
                return self._process_code_file(file_path, rel_path)
            else:
                return self._process_text_file(file_path, rel_path)
        except Exception as e:
            logger.error(f"Error processing file {rel_path}: {e}")
            return []
    
    def process_repository(self, repo_path: str) -> Tuple[List[Document], List[Document], nx.DiGraph]:
        """
        Process entire repository: split documents, build symbols, and create relationship code_graph
        """
        text_docs, code_docs, _ = self.split_repository(repo_path)
        
        # Now symbols are already extracted in split_repository -> _process_code_file -> _process_with_ast
        # No need to extract them again, just build the code_graph
        
        # After processing all files, we build the combined code_graph
        self._build_combined_graph()
        
        logger.info(f"Processed repository with {len(text_docs)} text documents and {len(code_docs)} code documents")
        logger.info(f"Extracted {len(self.symbol_table)} symbols")
        logger.info(f"Built combined code_graph with {self.combined_graph.number_of_nodes()} nodes and {self.combined_graph.number_of_edges()} edges")
        
        return text_docs, code_docs, self.combined_graph
    
    def _extract_calls(self, node: 'Node', language: str) -> Set[str]:
        """Extract function/method calls within the symbol"""
        calls = set()
        
        # Traverse the entire subtree to find function calls
        for child_node in self._traverse_tree(node):
            if language == 'python':
                if child_node.type == 'call':
                    # function_name(args) or object.method()
                    for child in child_node.children:
                        if child.type == 'identifier':
                            calls.add(child.text.decode('utf8'))
                        elif child.type == 'attribute':
                            # object.method() - get the full chain
                            calls.add(child.text.decode('utf8'))
                        elif child.type == 'subscript':
                            # dict_obj['key']() type calls
                            calls.add(child.text.decode('utf8'))
                
                # Constructor calls through class instantiation
                elif child_node.type == 'call' and child_node.children:
                    first_child = child_node.children[0]
                    if first_child.type == 'identifier':
                        # Direct constructor call like User()
                        calls.add(first_child.text.decode('utf8'))
                    elif first_child.type == 'attribute':
                        # Module.Class() calls
                        calls.add(first_child.text.decode('utf8'))
            
            elif language in ['javascript', 'typescript']:
                if child_node.type == 'call_expression':
                    # function(args) or object.method(args)
                    for child in child_node.children:
                        if child.type == 'identifier':
                            calls.add(child.text.decode('utf8'))
                        elif child.type == 'member_expression':
                            # Extract the method name from member expression
                            calls.add(child.text.decode('utf8'))
                
                # Constructor calls with 'new' keyword
                elif child_node.type == 'new_expression':
                    for child in child_node.children:
                        if child.type == 'identifier':
                            calls.add(f"new {child.text.decode('utf8')}")
                        elif child.type == 'member_expression':
                            calls.add(f"new {child.text.decode('utf8')}")
            
            elif language == 'java':
                if child_node.type == 'method_invocation':
                    # method() or object.method()
                    for child in child_node.children:
                        if child.type == 'identifier':
                            calls.add(child.text.decode('utf8'))
                        elif child.type == 'field_access':
                            # Extract method from field access
                            calls.add(child.text.decode('utf8'))
                
                # Constructor calls with 'new'
                elif child_node.type == 'object_creation_expression':
                    for child in child_node.children:
                        if child.type == 'type_identifier':
                            calls.add(f"new {child.text.decode('utf8')}")
                        elif child.type == 'generic_type':
                            calls.add(f"new {child.text.decode('utf8')}")
            
            elif language == 'go':
                if child_node.type == 'call_expression':
                    # function(args)
                    for child in child_node.children:
                        if child.type == 'identifier':
                            calls.add(child.text.decode('utf8'))
                        elif child.type == 'selector_expression':
                            # package.Function()
                            calls.add(child.text.decode('utf8'))
                
                # Struct literal creation
                elif child_node.type == 'composite_literal':
                    for child in child_node.children:
                        if child.type == 'type_identifier':
                            calls.add(f"new {child.text.decode('utf8')}")
                        elif child.type == 'struct_type':
                            calls.add('struct_literal')
            
            elif language == 'c_sharp':
                if child_node.type == 'invocation_expression':
                    # Method() or object.Method()
                    for child in child_node.children:
                        if child.type == 'identifier':
                            calls.add(child.text.decode('utf8'))
                        elif child.type == 'member_access_expression':
                            # Extract method from member access
                            calls.add(child.text.decode('utf8'))
                
                # Constructor calls with 'new'
                elif child_node.type == 'object_creation_expression':
                    for child in child_node.children:
                        if child.type == 'identifier':
                            calls.add(f"new {child.text.decode('utf8')}")
                        elif child.type == 'generic_name':
                            calls.add(f"new {child.text.decode('utf8')}")
            
            elif language == 'rust':
                if child_node.type == 'call_expression':
                    # function(args) or object.method(args)
                    for child in child_node.children:
                        if child.type == 'identifier':
                            calls.add(child.text.decode('utf8'))
                        elif child.type == 'field_expression':
                            # struct.method()
                            calls.add(child.text.decode('utf8'))
                        elif child.type == 'scoped_identifier':
                            # module::function()
                            calls.add(child.text.decode('utf8'))
                
                # Struct instantiation
                elif child_node.type == 'struct_expression':
                    for child in child_node.children:
                        if child.type in ['type_identifier', 'scoped_type_identifier']:
                            calls.add(f"new {child.text.decode('utf8')}")
                
                # Macro calls
                elif child_node.type == 'macro_invocation':
                    for child in child_node.children:
                        if child.type == 'identifier':
                            calls.add(f"macro {child.text.decode('utf8')}")
                
                # Method calls on types
                elif child_node.type == 'method_call_expression':
                    for child in child_node.children:
                        if child.type == 'field_identifier':
                            calls.add(child.text.decode('utf8'))
            
            elif language == 'ruby':
                if child_node.type == 'call':
                    # method() or object.method()
                    for child in child_node.children:
                        if child.type == 'identifier':
                            calls.add(child.text.decode('utf8'))
                        elif child.type == 'method_call':
                            calls.add(child.text.decode('utf8'))
                        elif child.type == 'constant':
                            calls.add(child.text.decode('utf8'))
                
                # Block calls (iterator methods)
                elif child_node.type == 'method_call' and 'block' in [c.type for c in child_node.children]:
                    for child in child_node.children:
                        if child.type == 'identifier':
                            calls.add(f"block {child.text.decode('utf8')}")
                
                # String interpolation
                elif child_node.type == 'string':
                    if '#{' in child_node.text.decode('utf8'):
                        calls.add('string_interpolation')
            
            elif language == 'kotlin':
                if child_node.type == 'call_expression':
                    # function(args) or object.method(args)
                    for child in child_node.children:
                        if child.type == 'simple_identifier':
                            calls.add(child.text.decode('utf8'))
                        elif child.type == 'navigation_expression':
                            calls.add(child.text.decode('utf8'))
                
                # Constructor calls
                elif child_node.type == 'constructor_invocation':
                    for child in child_node.children:
                        if child.type == 'simple_identifier':
                            calls.add(f"new {child.text.decode('utf8')}")
                
                # Lambda expressions
                elif child_node.type == 'lambda_literal':
                    calls.add('lambda')
            
            elif language == 'scala':
                if child_node.type == 'call_expression':
                    # function(args) or object.method(args)
                    for child in child_node.children:
                        if child.type == 'identifier':
                            calls.add(child.text.decode('utf8'))
                        elif child.type == 'field_expression':
                            calls.add(child.text.decode('utf8'))
                
                # Constructor calls (object instantiation)
                elif child_node.type == 'instance_expression':
                    for child in child_node.children:
                        if child.type == 'identifier':
                            calls.add(f"new {child.text.decode('utf8')}")
                
                # Pattern matching
                elif child_node.type == 'match_expression':
                    calls.add('pattern_match')
                
                # For comprehensions
                elif child_node.type == 'for_expression':
                    calls.add('for_comprehension')
                
                # Function literals
                elif child_node.type == 'function_literal':
                    calls.add('function_literal')
        
        return calls

    def _build_combined_graph(self):
        """Build combined relationship code_graph from symbol table"""
        # Add nodes for all symbols
        for symbol_key, symbol_info in self.symbol_table.items():
            self.combined_graph.add_node(
                symbol_key,
                name=symbol_info.name,
                file_path=symbol_info.file_path,
                node_type=symbol_info.node_type,
                language=symbol_info.language,
                start_line=symbol_info.start_line,
                end_line=symbol_info.end_line
            )
        
        # Add edges for imports and calls (only within the same language)
        for symbol_key, symbol_info in self.symbol_table.items():
            # Import relationships (both symbol-level and file-level imports)
            all_imports = set(symbol_info.imports)  # Symbol-level imports (local/in-scope)
            
            # Add file-level imports for this symbol's file
            if symbol_info.file_path in self.file_imports:
                all_imports.update(self.file_imports[symbol_info.file_path])
            
            for imported in all_imports:
                # Try to find the imported symbol in any file of the same language
                for target_key, target_info in self.symbol_table.items():
                    if (target_info.language == symbol_info.language and  # Same language check
                        target_key != symbol_key and
                        (target_info.name == imported or 
                         target_info.file_path.endswith(f"{imported}.{self._get_extension(symbol_info.language)}") or
                         imported in target_key)):
                        self.combined_graph.add_edge(symbol_key, target_key, relationship='imports')
            
            # Call relationships (same file and cross-file, same language only)
            for called in symbol_info.calls:
                # First try same file (language is guaranteed to be the same)
                same_file_key = f"{symbol_info.file_path}:{called}"
                if same_file_key in self.symbol_table:
                    self.combined_graph.add_edge(symbol_key, same_file_key, relationship='calls')
                else:
                    # Handle module.function calls (Python: utils.load_config)
                    if '.' in called:
                        parts = called.split('.')
                        if len(parts) == 2:
                            module_name, function_name = parts
                            # Look for function in files that contain the module name (same language)
                            for target_key, target_info in self.symbol_table.items():
                                if (target_info.language == symbol_info.language and  # Same language check
                                    target_info.name == function_name and 
                                    module_name in target_info.file_path and 
                                    target_key != symbol_key):
                                    self.combined_graph.add_edge(symbol_key, target_key, relationship='calls')
                                    break
                    
                    # Handle constructor calls (new ApiClient)
                    elif called.startswith('new '):
                        class_name = called[4:]  # Remove 'new ' prefix
                        for target_key, target_info in self.symbol_table.items():
                            if (target_info.language == symbol_info.language and  # Same language check
                                target_info.name == class_name and 
                                target_info.node_type in ['class_declaration', 'class_definition', 'struct_item', 'class'] and
                                target_key != symbol_key):
                                self.combined_graph.add_edge(symbol_key, target_key, relationship='calls')
                                break
                    
                    # Direct function calls across files (same language only)
                    else:
                        for target_key, target_info in self.symbol_table.items():
                            if (target_info.language == symbol_info.language and  # Same language check
                                target_info.name == called and 
                                target_info.file_path != symbol_info.file_path and
                                target_info.node_type in ['function_definition', 'function_declaration', 'method_definition', 'method_declaration', 'function_item', 'method'] and
                                target_key != symbol_key):
                                self.combined_graph.add_edge(symbol_key, target_key, relationship='calls')
                                break
    
    def get_symbol_neighbors(self, symbol_key: str, hops: int = 1) -> Set[str]:
        """Get neighboring symbols in the relationship code_graph"""
        if symbol_key not in self.combined_graph:
            return set()
        
        neighbors = set()
        current_nodes = {symbol_key}
        
        for _ in range(hops):
            next_nodes = set()
            for node in current_nodes:
                # Get both incoming and outgoing neighbors
                next_nodes.update(self.combined_graph.predecessors(node))
                next_nodes.update(self.combined_graph.successors(node))
            
            neighbors.update(next_nodes)
            current_nodes = next_nodes - neighbors  # Only expand to new nodes
        
        return neighbors - {symbol_key}  # Exclude the original symbol
    
    def export_graph(self) -> Dict:
        """Export code_graph in serializable format"""
        return {
            'nodes': [
                {
                    'id': node_id,
                    **data
                }
                for node_id, data in self.combined_graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': source,
                    'target': target,
                    **data
                }
                for source, target, data in self.combined_graph.edges(data=True)
            ]
        }
    
    def add_symbol(self, symbol: SymbolInfo):
        """Add a symbol to the symbol table"""
        # Create a unique key for the symbol
        key = f"{symbol.file_path}:{symbol.name}:{symbol.start_line}"
        self.symbol_table[key] = symbol
    
    def split_file(self, file_path: str, language: str) -> List[Document]:
        """Split a single file into code documents"""
        if not os.path.isfile(file_path):
            logger.warning(f"File not found: {file_path}")
            return []
        
        rel_path = os.path.relpath(file_path, os.getcwd())
        
        try:
            if self._is_code_file(file_path):
                return self._process_code_file(file_path, rel_path)
            else:
                return self._process_text_file(file_path, rel_path)
        except Exception as e:
            logger.error(f"Error processing file {rel_path}: {e}")
            return []
    
    def _get_refined_node_type(self, node: 'Node', language: str, code_content: str) -> str:
        """Get a more specific node type based on language and content analysis"""
        base_type = node.type
        
        if language == 'kotlin' and base_type == 'class_declaration':
            # Analyze the content to determine the specific type
            code_lower = code_content.lower().strip()
            
            if code_lower.startswith('interface'):
                return 'interface_declaration'
            elif 'data class' in code_content:
                return 'dataclass_declaration'
            elif code_lower.startswith('abstract class'):
                return 'abstract_class_declaration'
            elif code_lower.startswith('sealed class'):
                return 'sealed_class_declaration'
            elif code_lower.startswith('enum class'):
                return 'enum_class_declaration'
            else:
                return 'class_declaration'
        
        elif language == 'rust' and base_type == 'impl_item':
            # Distinguish between trait implementations and inherent implementations
            if ' for ' in code_content:
                return 'trait_impl_declaration'
            else:
                return 'inherent_impl_declaration'
        
        elif language == 'scala' and base_type == 'class_definition':
            # Distinguish Scala class types
            code_lower = code_content.lower().strip()
            
            if code_lower.startswith('case class'):
                return 'case_class_definition'
            elif code_lower.startswith('abstract class'):
                return 'abstract_class_definition'
            elif code_lower.startswith('sealed class'):
                return 'sealed_class_definition'
            else:
                return 'class_definition'
        
        elif language == 'python' and base_type == 'class_definition':
            # Check for abstract classes or dataclasses
            if '@dataclass' in code_content or 'from dataclasses import' in code_content:
                return 'dataclass_definition'
            elif 'ABC' in code_content or 'abstractmethod' in code_content:
                return 'abstract_class_definition'
            else:
                return 'class_definition'
        
        # For functions, we could also distinguish suspend functions, async functions, etc.
        elif language == 'kotlin' and base_type == 'function_declaration':
            if 'suspend ' in code_content:
                return 'suspend_function_declaration'
            elif 'inline ' in code_content:
                return 'inline_function_declaration'
            else:
                return 'function_declaration'
        
        elif language == 'python' and base_type in ['function_def', 'function_definition']:
            if 'async def' in code_content:
                return 'async_function_definition'
            else:
                return base_type
        
        elif language == 'javascript' and base_type == 'function_declaration':
            if 'async ' in code_content:
                return 'async_function_declaration'
            else:
                return 'function_declaration'
        
        # Return the original type if no refinement is needed
        return base_type
    
    def _is_local_import(self, child_node: 'Node', symbol_node: 'Node', language: str) -> bool:
        """Check if an import is local to the symbol (inside function body, not at file level)"""
        
        # Import types by language
        import_types = {
            'python': ['import_statement', 'import_from_statement'],
            'javascript': ['import_statement'],
            'typescript': ['import_statement'],
            'java': ['import_declaration'],
            'go': ['import_spec', 'import_declaration'],
            'c_sharp': ['using_directive'],
            'rust': ['use_declaration'],
            'kotlin': ['import_header'],
            'scala': ['import_declaration'],
        }
        
        lang_imports = import_types.get(language, [])
        if child_node.type not in lang_imports:
            return False
        
        # For most cases, we don't want to extract any imports from symbols
        # since file-level imports are handled separately
        # Only extract imports that are clearly inside function bodies
        current = child_node.parent
        while current and current != symbol_node:
            # If we find a function body, block, or similar construct,
            # this import is likely local to the function
            if current.type in ['block', 'function_body', 'compound_statement', 'suite']:
                return True
            current = current.parent
        
        # If we reach here, the import is likely at the symbol/file level (not local)
        return False
