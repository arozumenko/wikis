from __future__ import annotations

from langchain_core.documents import Document

from app.core.code_graph.graph_query_service import RelationshipResult, SymbolResult
from app.core.deep_research.research_tools import create_codebase_tools
from app.services.research_service import _extract_tool_source_references


class _ProjectQueryService:
    def __init__(self) -> None:
        self.nodes = {
            "configurations::n-cfg": {
                "node_id": "n-cfg",
                "project_node_id": "configurations::n-cfg",
                "wiki_id": "configurations",
                "symbol_name": "ConfigurationAPI",
                "symbol_type": "class",
                "rel_path": "api/v2/models.py",
                "start_line": 10,
                "end_line": 40,
                "content": "class ConfigurationAPI:\n    def models(self):\n        return self.client.models()",
                "docstring": "Configuration plugin API.",
            },
            "sdk::n-sdk": {
                "node_id": "n-sdk",
                "project_node_id": "sdk::n-sdk",
                "wiki_id": "sdk",
                "symbol_name": "EliteAClient.models",
                "symbol_type": "method",
                "rel_path": "elitea_sdk/client.py",
                "start_line": 50,
                "end_line": 70,
                "content": "def models(self):\n    return self.get('/configurations/models/{project_id}')",
                "docstring": "SDK client method for configuration models.",
            },
            "configurations::n-var": {
                "node_id": "n-var",
                "project_node_id": "configurations::n-var",
                "wiki_id": "configurations",
                "symbol_name": "public_configurations",
                "symbol_type": "variable",
                "rel_path": "utils_models.py",
                "start_line": 5,
                "end_line": 5,
                "content": "public_configurations = []",
                "docstring": "",
            },
        }

    def search(self, query, k=20, symbol_types=None, exclude_types=None, layer=None, path_prefix=None):
        results = [
            SymbolResult(
                node_id="configurations::n-var",
                symbol_name="public_configurations",
                symbol_type="variable",
                rel_path="utils_models.py",
                connections=0,
                score=1.0,
            ),
            SymbolResult(
                node_id="configurations::n-cfg",
                symbol_name="ConfigurationAPI",
                symbol_type="class",
                rel_path="api/v2/models.py",
                connections=2,
                score=0.9,
            ),
            SymbolResult(
                node_id="sdk::n-sdk",
                symbol_name="EliteAClient.models",
                symbol_type="method",
                rel_path="elitea_sdk/client.py",
                connections=3,
                score=0.8,
            ),
        ]
        if symbol_types:
            results = [result for result in results if result.symbol_type in symbol_types]
        if exclude_types:
            results = [result for result in results if result.symbol_type not in exclude_types]
        if path_prefix:
            results = [result for result in results if result.rel_path.startswith(path_prefix)]
        for result in results:
            wiki_id = result.node_id.split("::", 1)[0]
            setattr(result, "wiki_id", wiki_id)
            setattr(result, "source_wiki_id", wiki_id)
        return results[:k]

    def wiki_ids(self):
        return ["configurations", "sdk"]

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def resolve_symbol(self, symbol_name, file_path="", language=""):
        if symbol_name in self.nodes:
            return symbol_name
        for node_id, data in self.nodes.items():
            if data["symbol_name"] == symbol_name:
                return node_id
        return None

    def get_relationships(self, node_id, direction="both", max_depth=2, max_results=50):
        if node_id == "configurations::n-cfg":
            return [
                RelationshipResult(
                    source_name="ConfigurationAPI",
                    target_name="EliteAClient.models",
                    relationship_type="cross_repo_api_surface",
                    source_type="class",
                    target_type="method",
                    source_node_id="configurations::n-cfg",
                    target_node_id="sdk::n-sdk",
                    source_wiki_id="configurations",
                    target_wiki_id="sdk",
                    weight=0.63,
                    provenance={"surface": "GET /configurations/models/{var}", "matcher": "api_surface:rest"},
                )
            ]
        return []

    def cross_repo_edges(self, node_id, direction="outgoing"):
        row = {
            "source_node_id": "configurations::n-cfg",
            "target_node_id": "sdk::n-sdk",
            "edge_class": "cross_repo",
            "weight": 0.63,
            "provenance": {
                "source_relationship_type": "cross_repo_api_surface",
                "surface": "GET /configurations/models/{var}",
                "matcher": "api_surface:rest",
                "source_wiki_id": "configurations",
                "target_wiki_id": "sdk",
            },
        }
        if direction in ("outgoing", "both") and node_id == row["source_node_id"]:
            return [row]
        if direction in ("incoming", "both") and node_id == row["target_node_id"]:
            return [row]
        return []

    def resolve_and_traverse(self, symbol_name, direction="both", max_depth=2, max_results=50, file_path="", language=""):
        node_id = self.resolve_symbol(symbol_name, file_path=file_path, language=language)
        if not node_id:
            return None, []
        return node_id, self.get_relationships(
            node_id,
            direction=direction,
            max_depth=max_depth,
            max_results=max_results,
        )

    def query(self, expression):
        return [result for result in self.search(expression) if result.symbol_type in {"class", "method"}]


class _MixedRetriever:
    def search_repository(self, query, k=15, apply_expansion=True):
        return [
            Document(
                page_content="Configuration documentation",
                metadata={"source": "README.md", "symbol_type": "file_doc", "is_doc": 1},
            ),
            Document(
                page_content="def configure(): pass",
                metadata={"source": "src/config.py", "symbol_type": "function", "is_doc": 0},
            ),
        ][:k]


def _tool_by_name(tools, name: str):
    return next(tool for tool in tools if tool.name == name)


def test_search_codebase_uses_project_query_service_without_primary_graph(monkeypatch):
    monkeypatch.delenv("WIKIS_PROGRESSIVE_TOOLS", raising=False)
    tools = create_codebase_tools(
        retriever_stack=None,
        graph_manager=None,
        code_graph=None,
        query_service=_ProjectQueryService(),
    )

    output = _tool_by_name(tools, "search_codebase").invoke({"query": "configuration models", "k": 5})

    assert "configurations:api/v2/models.py" in output
    assert "sdk:elitea_sdk/client.py" in output
    assert "EliteAClient.models" in output


def test_search_graph_uses_project_query_service_without_primary_graph(monkeypatch):
    monkeypatch.delenv("WIKIS_PROGRESSIVE_TOOLS", raising=False)
    tools = create_codebase_tools(
        retriever_stack=None,
        graph_manager=None,
        code_graph=None,
        query_service=_ProjectQueryService(),
    )

    output = _tool_by_name(tools, "search_graph").invoke({"query": "ConfigurationAPI", "k": 5})

    assert "ConfigurationAPI" in output
    assert "EliteAClient.models" in output
    assert "cross_repo" in output


def test_progressive_tools_expose_and_accept_federated_node_ids(monkeypatch):
    monkeypatch.setenv("WIKIS_PROGRESSIVE_TOOLS", "1")
    tools = create_codebase_tools(
        retriever_stack=None,
        graph_manager=None,
        code_graph=None,
        query_service=_ProjectQueryService(),
    )
    tool_names = {tool.name for tool in tools}

    assert {
        "search_symbols",
        "find_cross_repo_links",
        "get_relationships",
        "get_code",
        "search_docs",
        "query_graph",
        "think",
    } <= tool_names
    assert "search_codebase" not in tool_names

    symbol_output = _tool_by_name(tools, "search_symbols").invoke({"query": "models", "k": 5})
    assert "[wiki: sdk]" in symbol_output
    assert "[id: `sdk::n-sdk`]" in symbol_output
    assert "public_configurations" not in symbol_output
    assert "(variable" not in symbol_output

    graph_output = _tool_by_name(tools, "query_graph").invoke({"expression": "type:method"})
    assert "[wiki: sdk]" in graph_output
    assert "[id: `sdk::n-sdk`]" in graph_output

    rel_output = _tool_by_name(tools, "get_relationships").invoke({"symbol_name": "configurations::n-cfg"})
    assert "EliteAClient.models" in rel_output
    assert "cross_repo_api_surface" in rel_output
    assert "[wiki: sdk]" in rel_output
    assert "[id: `sdk::n-sdk`]" in rel_output
    assert "[surface: `GET /configurations/models/{var}`]" in rel_output

    link_output = _tool_by_name(tools, "find_cross_repo_links").invoke({"symbol_name": "configurations::n-cfg"})
    assert "Found 1 direct cross-repo links" in link_output
    assert "ConfigurationAPI" in link_output
    assert "EliteAClient.models" in link_output
    assert "GET /configurations/models/{var}" in link_output
    assert "api_surface:rest" in link_output

    code_output = _tool_by_name(tools, "get_code").invoke({"symbol_name": "sdk::n-sdk"})
    assert "elitea_sdk/client.py" in code_output
    assert "/configurations/models/{project_id}" in code_output


def test_codemap_tool_output_parser_preserves_project_node_ids():
    seen_symbols: set[str] = set()
    output = """
1. `EliteAClient.models` (method) in elitea_sdk/client.py [3 refs] [wiki: sdk] [id: `sdk::n-sdk`]
- **ConfigurationAPI** (class) — `api/v2/models.py` (2 connections) [wiki: configurations] [id: `configurations::n-cfg`]
### `Config-Loader` (function) — src/config-loader.py:12-44
"""

    refs = _extract_tool_source_references(output, seen_symbols)

    assert [(ref.symbol, ref.symbol_type, ref.file_path, ref.wiki_id, ref.node_id) for ref in refs] == [
        ("EliteAClient.models", "method", "elitea_sdk/client.py", "sdk", "sdk::n-sdk"),
        ("ConfigurationAPI", "class", "api/v2/models.py", "configurations", "configurations::n-cfg"),
        ("Config-Loader", "function", "src/config-loader.py", None, None),
    ]


def test_progressive_file_tools_search_all_project_repo_roots(monkeypatch, tmp_path):
    monkeypatch.setenv("WIKIS_PROGRESSIVE_TOOLS", "1")
    configurations_root = tmp_path / "configurations"
    sdk_root = tmp_path / "sdk"
    (configurations_root / "methods").mkdir(parents=True)
    (configurations_root / "methods" / "config.py").write_text("CONFIG = True\n")
    (sdk_root / "elitea_sdk" / "tools" / "jira").mkdir(parents=True)
    (sdk_root / "elitea_sdk" / "__init__.py").write_text("SDK = True\n")
    (sdk_root / "elitea_sdk" / "tools" / "jira" / "__init__.py").write_text("class JiraToolkit:\n    pass\n")

    tools = create_codebase_tools(
        retriever_stack=None,
        graph_manager=None,
        code_graph=None,
        query_service=_ProjectQueryService(),
        repo_path={"configurations": str(configurations_root), "sdk": str(sdk_root)},
    )

    root_output = _tool_by_name(tools, "list_repo_files").invoke({"directory": ".", "pattern": "*"})
    assert "# configurations:./" in root_output
    assert "# sdk:./" in root_output

    sdk_output = _tool_by_name(tools, "list_repo_files").invoke({"directory": "elitea_sdk", "pattern": "*.py"})
    assert "# sdk:elitea_sdk/" in sdk_output
    assert "__init__.py" in sdk_output
    assert "Directory not found" not in sdk_output

    source_output = _tool_by_name(tools, "read_source_file").invoke({
        "file_path": "elitea_sdk/tools/jira/__init__.py",
        "offset": 0,
        "limit": 20,
    })
    assert "# sdk:elitea_sdk/tools/jira/__init__.py" in source_output
    assert "class JiraToolkit" in source_output


def test_search_docs_returns_documentation_only(monkeypatch):
    monkeypatch.setenv("WIKIS_PROGRESSIVE_TOOLS", "1")
    tools = create_codebase_tools(
        retriever_stack=_MixedRetriever(),
        graph_manager=None,
        code_graph=None,
        query_service=_ProjectQueryService(),
    )

    output = _tool_by_name(tools, "search_docs").invoke({"query": "configuration", "k": 5})

    assert "Configuration documentation" in output
    assert "def configure" not in output
    assert "src/config.py" not in output
