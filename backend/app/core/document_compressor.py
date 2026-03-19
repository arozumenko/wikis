"""
Document Compression for Hierarchical Context Management.

This module provides intelligent compression of code documents into multiple detail levels:
- Tier 1: Full detail (no compression)
- Tier 2: Signatures and public API only (~40% of original size)
- Tier 3: Minimal summaries (~10% of original size)

Purpose: Fit 100-130 expanded documents into token budget by compressing lower-priority
documents while preserving full detail for most relevant components.
"""

import logging
import re

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentCompressor:
    """
    Compresses code documents based on tier level for hierarchical context management.

    Compression preserves essential information while reducing token count:
    - Tier 2: Extract signatures, docstrings, public APIs (remove implementations)
    - Tier 3: Extract symbol metadata only (name, type, members list)
    """

    def __init__(self):
        """Initialize document compressor."""
        self.logger = logger

    def compress_document(self, doc: Document, tier: int) -> Document:
        """
        Compress document based on tier level.

        Args:
            doc: Document to compress
            tier: Compression tier (1=full, 2=signatures, 3=summary)

        Returns:
            Compressed document with tier metadata
        """
        if tier == 1:
            # No compression - return full detail
            return doc

        content = doc.page_content
        metadata = doc.metadata.copy()
        symbol_type = metadata.get("symbol_type", "").lower()
        language = metadata.get("language", "").lower()
        file_type = metadata.get("file_type", "").lower()

        # IMPORTANT: Don't compress documentation files - they're already human-readable prose
        # Documentation files contain WHY and HOW explanations which are critical
        if self._is_documentation_file(symbol_type, language, file_type):
            self.logger.debug(f"Skipping compression for documentation: {metadata.get('symbol_name', 'unknown')}")
            # Add metadata to indicate it's uncompressed documentation
            metadata["compression_tier"] = tier
            metadata["compression_skipped"] = True
            metadata["skip_reason"] = "documentation_file"
            return Document(page_content=content, metadata=metadata)

        try:
            if tier == 2:
                # Signature level - extract API shape without implementations
                compressed_content = self._compress_to_signature(content, symbol_type, language)

            elif tier == 3:
                # Summary level - minimal metadata only
                compressed_content = self._compress_to_summary(content, metadata)

            else:
                # Invalid tier - return original
                self.logger.warning(f"Invalid compression tier {tier}, returning original")
                return doc

            # Add compression metadata
            metadata["compression_tier"] = tier
            metadata["original_length"] = len(content)
            metadata["compressed_length"] = len(compressed_content)

            return Document(page_content=compressed_content, metadata=metadata)

        except Exception as e:
            self.logger.warning(f"Compression failed for {metadata.get('symbol_name', 'unknown')}: {e}")
            # Return original on error
            return doc

    def _is_documentation_file(self, symbol_type: str, language: str, file_type: str) -> bool:
        """
        Check if document is a documentation file that should not be compressed.

        Documentation files are already human-readable prose explaining architecture,
        design decisions, and usage patterns. Compressing them loses valuable context.

        Args:
            symbol_type: Type of symbol (text_chunk, readme, documentation, etc.)
            language: Language/format (markdown, text, yaml, json, toml)
            file_type: File type (text, code)

        Returns:
            True if this is a documentation file that should remain uncompressed
        """
        # Check symbol types that indicate documentation
        doc_symbol_types = {
            "readme",
            "documentation",
            "markdown",
            "text_chunk",
            "config",
            "configuration",
            "text",
            "plain_text",
        }
        if symbol_type in doc_symbol_types:
            return True

        # Check language/format indicators
        doc_languages = {
            "markdown",
            "text",
            "plain_text",
            "restructuredtext",
            "yaml",
            "yml",
            "json",
            "toml",
            "xml",
            "ini",
            "cfg",
        }
        if language in doc_languages:
            return True

        # Check file_type
        if file_type == "text":
            return True

        return False

    def _compress_to_signature(self, content: str, symbol_type: str, language: str) -> str:
        """
        Compress to Tier 2: signatures and public API only.

        Extracts:
        - Class declarations with docstrings
        - Method signatures without implementations
        - Field declarations without initializers
        - Interface/enum definitions (already concise)
        - Constants with full value and comments (important context)

        Args:
            content: Original source code
            symbol_type: Type of symbol (class, struct, trait, function, interface, impl, etc.)
            language: Programming language

        Returns:
            Compressed content with signatures only
        """
        # Constants: Keep full definition (value + comments are both important)
        if symbol_type == "constant":
            return content  # Don't compress - constants define system behavior

        # Handle class-like structures (class, struct, trait, impl)
        if symbol_type in ["class", "struct", "trait", "impl", "protocol"]:
            return self._extract_class_signature(content, language)

        elif symbol_type in ["function", "method"]:
            return self._extract_function_signature(content, language)

        elif symbol_type == "interface":
            # Interfaces are already concise (no implementations)
            return content

        elif symbol_type == "enum":
            # Enums are already concise
            return content

        elif symbol_type in ["module", "file", "namespace"]:
            # Extract module-level signatures
            return self._extract_module_signatures(content, language)

        elif symbol_type in ["type_alias", "macro"]:
            # Type aliases and macros are usually concise
            return content[:800] + "\n..." if len(content) > 800 else content

        else:
            # For other types, keep first 500 characters
            return content[:500] + "\n..." if len(content) > 500 else content

    def _compress_to_summary(self, content: str, metadata: dict) -> str:
        """
        Compress to Tier 3: minimal summary only.

        Creates brief summary with:
        - Symbol name and type
        - File path and language
        - List of key members (methods/fields for classes)
        - Brief description if available

        Args:
            content: Original source code
            metadata: Document metadata

        Returns:
            Minimal summary string
        """
        symbol_name = metadata.get("symbol_name", "Unknown")
        symbol_type = metadata.get("symbol_type", "unknown")
        file_path = metadata.get("file_path", "")
        language = metadata.get("language", "")
        parent = metadata.get("parent_symbol", "")

        # Build summary header
        summary_lines = [f"# {symbol_type.title()}: {symbol_name}", f"File: {file_path}", f"Language: {language}"]

        if parent:
            summary_lines.append(f"Parent: {parent}")

        # Add type-specific summary
        if symbol_type == "class":
            methods = self._extract_method_names(content, language)
            if methods:
                summary_lines.append(f"Methods: {', '.join(methods[:10])}")
                if len(methods) > 10:
                    summary_lines.append(f"... and {len(methods) - 10} more")

        elif symbol_type == "interface":
            methods = self._extract_method_names(content, language)
            if methods:
                summary_lines.append(f"Methods: {', '.join(methods)}")

        elif symbol_type in ["function", "method"]:
            signature = self._extract_first_line_signature(content)
            if signature:
                summary_lines.append(f"Signature: {signature}")

        elif symbol_type == "enum":
            values = self._extract_enum_values(content, language)
            if values:
                summary_lines.append(f"Values: {', '.join(values[:10])}")

        return "\n".join(summary_lines)

    def _extract_class_signature(self, content: str, language: str) -> str:
        """
        Extract class/struct/trait signature: declaration, docstring, public method signatures.

        Handles:
        - Python: class with method signatures
        - TypeScript/JavaScript: class/interface with method signatures
        - Java: class with annotations and method signatures
        - C++: class/struct with method declarations (handles struct::method implementations)
        - Rust: struct/trait/impl with method signatures

        Args:
            content: Full source code
            language: Programming language

        Returns:
            Structure with signatures only (no implementations)
        """
        if language == "python":
            return self._extract_python_class_signature(content)
        elif language in ["typescript", "javascript", "ts", "js"]:
            return self._extract_typescript_class_signature(content)
        elif language == "java":
            return self._extract_java_class_signature(content)
        elif language in ["cpp", "c++", "cxx"]:
            return self._extract_cpp_class_signature(content)
        elif language == "rust":
            return self._extract_rust_signature(content)
        else:
            # Fallback: keep first 600 chars
            return content[:600] + "\n..." if len(content) > 600 else content

    def _extract_python_class_signature(self, content: str) -> str:
        """Extract Python class signature with method signatures only."""
        lines = content.split("\n")
        result = []
        in_method = False
        in_class_docstring = False
        in_method_docstring = False
        docstring_delimiter = None
        indent_level = 0
        class_docstring_done = False

        for line in lines:
            stripped = line.strip()

            # Keep class declaration
            if stripped.startswith("class "):
                result.append(line)
                indent_level = len(line) - len(line.lstrip())
                continue

            # Track class-level docstring (keep it)
            if not class_docstring_done and ('"""' in line or "'''" in line):
                if not in_class_docstring:
                    in_class_docstring = True
                    docstring_delimiter = '"""' if '"""' in line else "'''"
                    result.append(line)
                    # Check if docstring closes on same line
                    if line.count(docstring_delimiter) >= 2:
                        in_class_docstring = False
                        class_docstring_done = True
                else:
                    result.append(line)
                    in_class_docstring = False
                    class_docstring_done = True
                continue

            if in_class_docstring:
                result.append(line)
                continue

            # Keep method/property declarations
            if re.match(r"\s+(def |@property|@staticmethod|@classmethod)", line):
                result.append(line)
                in_method = True
                in_method_docstring = False
                continue

            # Skip method docstrings (we only keep class docstring)
            if in_method and ('"""' in line or "'''" in line):
                if not in_method_docstring:
                    in_method_docstring = True
                    # Check if closes on same line
                    if line.count('"""') >= 2 or line.count("'''") >= 2:
                        in_method_docstring = False
                else:
                    in_method_docstring = False
                continue

            if in_method_docstring:
                continue  # Skip docstring content

            # Keep method signature continuation (parameters on multiple lines)
            if in_method and ("(" in line or ")" in line or "->" in line):
                result.append(line)
                if ")" in line:
                    result.append(" " * (indent_level + 4) + "...")
                    in_method = False
                continue

            # Skip method body
            if in_method:
                result.append(" " * (indent_level + 4) + "...")
                in_method = False
                continue

        return "\n".join(result)

    def _extract_typescript_class_signature(self, content: str) -> str:
        """Extract TypeScript/JavaScript class signature."""
        lines = content.split("\n")
        result = []
        in_method = False

        for line in lines:
            stripped = line.strip()

            # Keep class declaration and extends/implements
            if (
                stripped.startswith("class ")
                or stripped.startswith("export class ")
                or stripped.startswith("export default class ")
            ):
                result.append(line)
                line.count("{") - line.count("}")
                continue

            # Keep interface/type declarations
            if (
                stripped.startswith("interface ")
                or stripped.startswith("type ")
                or stripped.startswith("extends ")
                or stripped.startswith("implements ")
            ):
                result.append(line)
                continue

            # Keep field declarations
            if re.match(r"\s*(public|private|protected|readonly)?\s+\w+\s*:", line):
                # Field declaration - keep without initializer
                field_line = re.sub(r"\s*=.*?;", ";", line)
                result.append(field_line)
                continue

            # Keep method signatures
            if re.match(r"\s*(public|private|protected|static|async)?\s*\w+\s*\(", line):
                result.append(line)
                in_method = True
                # Check if method signature closes on same line
                if "{" in line:
                    result.append("  ...")
                    in_method = False
                continue

            # Skip method body
            if in_method and "{" in line:
                result.append("  ...")
                in_method = False
                continue

        return "\n".join(result)

    def _extract_java_class_signature(self, content: str) -> str:
        """Extract Java class signature."""
        lines = content.split("\n")
        result = []
        in_method = False

        for line in lines:
            stripped = line.strip()

            # Keep class declaration
            if (
                stripped.startswith("public class ")
                or stripped.startswith("class ")
                or stripped.startswith("public interface ")
            ):
                result.append(line)
                continue

            # Keep annotations
            if stripped.startswith("@"):
                result.append(line)
                continue

            # Keep field declarations
            if re.match(r"\s*(public|private|protected)?\s+(static|final)?\s+\w+\s+\w+", line):
                # Check if it's a field (has semicolon or assignment)
                if ";" in line or "=" in line:
                    field_line = re.sub(r"\s*=.*?;", ";", line)
                    result.append(field_line)
                    continue

            # Keep method signatures
            if re.match(r"\s*(public|private|protected)?\s+(static|final)?\s+\w+\s+\w+\s*\(", line):
                result.append(line)
                in_method = True
                # Check if signature closes
                if "{" in line:
                    result.append("  ...")
                    in_method = False
                continue

            # Skip method body
            if in_method and "{" in line:
                result.append("  ...")
                in_method = False
                continue

        return "\n".join(result)

    def _extract_cpp_class_signature(self, content: str) -> str:
        """
        Extract C++ class/struct signature.

        Note: ContentExpander already concatenates struct declarations with their
        method implementations (struct::method() from .cpp files), so this receives
        a complete merged document.
        """
        lines = content.split("\n")
        result = []
        in_method = False

        for line in lines:
            stripped = line.strip()

            # Keep class declaration
            if stripped.startswith("class ") or stripped.startswith("struct "):
                result.append(line)
                continue

            # Keep public/private/protected markers
            if stripped in ["public:", "private:", "protected:"]:
                result.append(line)
                continue

            # Keep member declarations (fields and method signatures)
            if re.match(r"\s*\w+[\s\*&]+\w+", line):
                # Check if it's a method (has parentheses)
                if "(" in line:
                    result.append(line)
                    in_method = True
                    if "{" in line:
                        result.append("  ...")
                        in_method = False
                else:
                    # Field declaration
                    field_line = re.sub(r"\s*=.*?;", ";", line)
                    result.append(field_line)
                continue

            # Skip method body
            if in_method and "{" in line:
                result.append("  ...")
                in_method = False
                continue

        return "\n".join(result)

    def _extract_rust_signature(self, content: str) -> str:
        """
        Extract Rust struct/trait/impl signature.

        Note: Currently using basic parser. Will be enhanced when rich Rust parser is added.
        For now, extracts struct/trait declarations and method signatures, removes implementations.
        """
        lines = content.split("\n")
        result = []
        in_block = False
        brace_count = 0

        for line in lines:
            stripped = line.strip()

            # Keep struct/trait/impl/enum declarations
            if any(
                stripped.startswith(kw)
                for kw in ["pub struct", "struct", "pub trait", "trait", "impl", "pub enum", "enum"]
            ):
                result.append(line)
                if "{" in line:
                    brace_count = line.count("{") - line.count("}")
                    in_block = True
                continue

            if in_block:
                # Update brace count
                brace_count += line.count("{") - line.count("}")

                # Keep field declarations and method signatures
                if "fn " in line or "pub fn " in line:
                    result.append(line)
                    if "{" not in line:
                        # Multi-line signature
                        continue
                    else:
                        result.append("    ...")
                elif ":" in line and "{" not in line and ";" in line:
                    # Field declaration
                    result.append(line)

                # Check if block ended
                if brace_count == 0:
                    in_block = False
                    result.append("}")

        return "\n".join(result) if result else content[:600]

    def _extract_function_signature(self, content: str, language: str) -> str:
        """Extract function signature with docstring, no implementation."""
        lines = content.split("\n")
        result = []
        in_docstring = False
        docstring_delimiter = None
        signature_done = False

        for line in lines:
            stripped = line.strip()

            # Keep function declaration
            if (
                stripped.startswith("def ")
                or stripped.startswith("function ")
                or stripped.startswith("async ")
                or stripped.startswith("export ")
            ):
                result.append(line)
                continue

            # Keep decorators (Python)
            if stripped.startswith("@"):
                result.append(line)
                continue

            # Track docstrings
            if '"""' in line or "'''" in line or "/*" in line:
                if not in_docstring:
                    in_docstring = True
                    docstring_delimiter = '"""' if '"""' in line else ("'''" if "'''" in line else "*/")
                    result.append(line)
                    if line.count(docstring_delimiter) >= 2:
                        in_docstring = False
                else:
                    result.append(line)
                    in_docstring = False
                continue

            if in_docstring:
                result.append(line)
                continue

            # Keep signature continuation (multi-line parameters)
            if not signature_done and ("(" in line or ")" in line or "->" in line or ":" in line):
                result.append(line)
                if ")" in line or ":" in line:
                    signature_done = True
                    result.append("  ...")
                continue

            # Stop after signature
            if signature_done:
                break

        return "\n".join(result)

    def _extract_module_signatures(self, content: str, language: str) -> str:
        """Extract module-level function signatures and exports."""
        # For modules, extract all top-level function signatures
        lines = content.split("\n")
        result = []

        for line in lines:
            stripped = line.strip()

            # Keep imports/exports
            if stripped.startswith("import ") or stripped.startswith("from ") or stripped.startswith("export "):
                result.append(line)
                continue

            # Keep top-level function signatures (no indentation)
            if line.startswith("def ") or line.startswith("function ") or line.startswith("async function "):
                result.append(line)
                result.append("  ...\n")
                continue

        return "\n".join(result) if result else content[:500]

    def _extract_method_names(self, content: str, language: str) -> list[str]:
        """Extract method names from class content."""
        method_names = []

        if language == "python":
            # Match: def method_name(
            matches = re.findall(r"def\s+(\w+)\s*\(", content)
            method_names = [m for m in matches if not m.startswith("_")]  # Public only

        elif language in ["typescript", "javascript"]:
            # Match: methodName( or async methodName(
            matches = re.findall(r"(?:async\s+)?(\w+)\s*\(", content)
            method_names = [m for m in matches if m not in ["if", "while", "for", "switch"]]

        elif language == "java":
            # Match: public/private returnType methodName(
            matches = re.findall(r"(?:public|private|protected)\s+\w+\s+(\w+)\s*\(", content)
            method_names = matches

        elif language in ["cpp", "c++"]:
            # Match: returnType methodName(
            matches = re.findall(r"\w+\s+(\w+)\s*\(", content)
            method_names = matches

        return method_names[:20]  # Limit to first 20

    def _extract_first_line_signature(self, content: str) -> str:
        """Extract first line of function signature."""
        lines = content.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("function ") or stripped.startswith("async "):
                # Remove body if on same line
                if "{" in line:
                    line = line[: line.index("{")].strip()
                if ":" in line and "->" not in line:
                    line = line[: line.rindex(":")].strip()
                return line.strip()
        return content.split("\n")[0][:100]

    def _extract_enum_values(self, content: str, language: str) -> list[str]:
        """Extract enum value names."""
        values = []

        if language == "python":
            # Match: NAME = value
            matches = re.findall(r"(\w+)\s*=", content)
            values = matches

        elif language in ["typescript", "javascript"]:
            # Match: NAME = value or NAME,
            matches = re.findall(r"(\w+)\s*[=,]", content)
            values = matches

        elif language == "java":
            # Match: NAME(args) or NAME,
            matches = re.findall(r"(\w+)(?:\([^)]*\))?\s*[,;]", content)
            values = matches

        return values[:15]  # Limit to first 15


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses rough approximation: 1 token ≈ 4 characters for code.
    For more accuracy, install tiktoken library.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    # Try tiktoken if available
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # Fallback: rough approximation (1 token ≈ 4 chars for code)
        return len(text) // 4


def calculate_compression_ratio(original: Document, compressed: Document) -> float:
    """
    Calculate compression ratio.

    Args:
        original: Original document
        compressed: Compressed document

    Returns:
        Compression ratio (compressed_size / original_size)
    """
    original_size = len(original.page_content)
    compressed_size = len(compressed.page_content)

    if original_size == 0:
        return 1.0

    return compressed_size / original_size
