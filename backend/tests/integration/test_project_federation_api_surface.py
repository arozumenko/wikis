"""Project-federation API-surface regressions over real indexed repos.

These tests intentionally build tiny repositories on disk and run them
through ``EnhancedUnifiedGraphBuilder`` before merging them as a project.
That catches parser/indexing details synthetic NetworkX-only tests miss:
Pylon ``metadata.json`` plugin names, ``class API(APIBase)`` source text,
and SDK f-string path literals.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import pytest

from app.core.code_graph.api_surface_extractor import extract_api_surfaces_for_graph
from app.core.code_graph.cross_repo_linker import (
    build_cross_repo_edges,
    make_project_node_id,
)
from app.core.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
from app.core.filesystem_indexer import FilesystemRepositoryIndexer
from app.core.repo_providers.models import LocalPathConfig
from app.core.storage import open_storage


def _write_repo(root: Path, files: Dict[str, str]) -> Path:
    for rel_path, content in files.items():
        full_path = root / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(textwrap.dedent(content), encoding="utf-8")
    return root


_LANGUAGE_BY_EXT = {
    ".proto": "proto",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".feature": "gherkin",
}


def _inject_file_nodes(g: nx.MultiDiGraph, root: Path, extensions: Tuple[str, ...]) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in extensions:
            continue
        rel_path = path.relative_to(root).as_posix()
        node_id = f"file::{rel_path}"
        if g.has_node(node_id):
            continue
        g.add_node(
            node_id,
            language=_LANGUAGE_BY_EXT.get(path.suffix, ""),
            rel_path=rel_path,
            file_name=path.stem,
            symbol_name=path.stem,
            symbol_type="file",
            source_text=path.read_text(encoding="utf-8", errors="ignore"),
        )


def _index_repo(root: Path, *, inject_extensions: Tuple[str, ...] = ()) -> nx.MultiDiGraph:
    builder = EnhancedUnifiedGraphBuilder(max_workers=1)
    analysis = builder.analyze_repository(str(root))
    graph = analysis.unified_graph
    if inject_extensions:
        _inject_file_nodes(graph, root, inject_extensions)
    return graph


def _merge_project_graph(graphs: Dict[str, nx.MultiDiGraph]) -> nx.MultiDiGraph:
    merged = nx.MultiDiGraph()
    for wiki_id, graph in graphs.items():
        for node_id, data in graph.nodes(data=True):
            project_node_id = make_project_node_id(wiki_id, str(node_id))
            attrs = dict(data)
            attrs["wiki_id"] = wiki_id
            attrs["raw_node_id"] = str(node_id)
            merged.add_node(project_node_id, **attrs)
        for source_id, target_id, edge_data in graph.edges(data=True):
            merged.add_edge(
                make_project_node_id(wiki_id, str(source_id)),
                make_project_node_id(wiki_id, str(target_id)),
                **edge_data,
            )
    return merged


def test_pylon_configurations_plugin_links_to_sdk_client_from_real_indexed_repos(tmp_path):
    configurations_root = _write_repo(
        tmp_path / "configurations",
        {
            "metadata.json": """
                {"name": "configurations", "version": "0.1.0"}
            """,
            "api/v2/models.py": """
                from api.base import APIBase

                class API(APIBase):
                    url_params = [
                        '<int:project_id>',
                    ]

                    def get(self, project_id: int, **kwargs):
                        return {"project_id": project_id}
            """,
        },
    )
    sdk_root = _write_repo(
        tmp_path / "elitea-sdk",
        {
            "elitea_sdk/runtime/clients/client.py": """
                import requests

                class EliteAClient:
                    def __init__(self, base_url: str, project_id: int):
                        self.base_url = base_url
                        self.project_id = project_id

                    def models(self):
                        api_path = "/api/v2"
                        url = f"{self.base_url}{api_path}/configurations/models/{self.project_id}"
                        return requests.get(url).json()
            """,
        },
    )

    merged_graph = _merge_project_graph({
        "configurations-wiki": _index_repo(configurations_root),
        "sdk-wiki": _index_repo(sdk_root),
    })

    surfaces_by_node = extract_api_surfaces_for_graph(merged_graph)
    all_surfaces = {
        surface["surface"]
        for node_surfaces in surfaces_by_node.values()
        for surface in node_surfaces
    }
    assert "* /api/v2/configurations/models/{var}" in all_surfaces
    assert "* /configurations/models/{var}" in all_surfaces

    edges = build_cross_repo_edges(
        wiki_ids=["configurations-wiki", "sdk-wiki"],
        merged_graph=merged_graph,
        relatedness={("configurations-wiki", "sdk-wiki"): 0.18},
        surfaces_by_node=surfaces_by_node,
        threshold=0.15,
    )

    contract_edges = [
        edge
        for edge in edges
        if (edge.get("provenance") or {}).get("surface")
        == "* /configurations/models/{var}"
    ]
    assert contract_edges, (
        "Expected the Pylon producer route and SDK f-string consumer URL "
        "to create at least one cross-repo API-surface edge"
    )
    assert all(edge["edge_class"] == "cross_repo" for edge in contract_edges)


@pytest.mark.parametrize(
    ("case_name", "left_files", "right_files", "expected_surface", "inject_left"),
    [
        (
            "rest_fastapi_to_typescript_client",
            {
                "backend/app/api/items.py": """
                    from fastapi import APIRouter
                    router = APIRouter()

                    @router.get('/items')
                    def list_items():
                        return []
                """,
            },
            {
                "web/src/api/items.ts": """
                    export async function listItems() {
                        return fetch('/items', { method: 'GET' });
                    }
                """,
            },
            "GET /items",
            (),
        ),
        (
            "dto_python_to_typescript",
            {
                "backend/app/models.py": """
                    from dataclasses import dataclass

                    @dataclass
                    class UserPublic:
                        id: str
                        email: str
                        full_name: str
                """,
            },
            {
                "web/src/types.ts": """
                    export interface UserPublic {
                        id: string;
                        email: string;
                        full_name: string;
                    }
                """,
            },
            "obj:user_public#email,full_name,id",
            (),
        ),
        (
            "ffi_rust_to_python_ctypes",
            {
                "native/src/lib.rs": """
                    #[no_mangle]
                    pub extern "C" fn compute_hash(data: *const u8, len: usize) -> u64 { 0 }
                """,
            },
            {
                "py_client/bridge.py": """
                    import ctypes

                    def compute_hash(data: bytes) -> int:
                        lib = ctypes.CDLL('libnative.so')
                        lib.compute_hash.restype = ctypes.c_uint64
                        return lib.compute_hash(data, len(data))
                """,
            },
            "ffi:compute_hash",
            (),
        ),
        (
            "grpc_proto_to_python_servicer",
            {
                "proto/user.proto": """
                    syntax = "proto3";
                    service UserSvc {
                      rpc GetUser (GetUserRequest) returns (GetUserResponse);
                    }
                """,
            },
            {
                "server/user_pb2_grpc.py": """
                    class UserSvcServicer:
                        def GetUser(self, request, context):
                            return None
                """,
            },
            "grpc:UserSvc/GetUser",
            (".proto",),
        ),
        (
            "graphql_sdl_to_python_resolver",
            {
                "schema/schema.graphql": """
                    type Query {
                      user(id: ID!): User
                    }
                """,
            },
            {
                "server/resolvers.py": """
                    from strawberry import type

                    @Query
                    def user(id: str):
                        return None
                """,
            },
            "gql:query",
            (".graphql",),
        ),
        (
            "bdd_feature_to_python_step",
            {
                "features/login.feature": """
                    Feature: Login
                      Scenario: Successful login
                        Given the user is logged in
                """,
            },
            {
                "steps/login_steps.py": """
                    from behave import given

                    @given('the user is logged in')
                    def user_logged_in(context):
                        pass
                """,
            },
            "bdd:the user is logged in",
            (".feature",),
        ),
        (
            "cli_python_to_go",
            {
                "cli/main.py": """
                    import click

                    @click.command('deploy')
                    def deploy():
                        pass
                """,
            },
            {
                "cmd/main.go": """
                    package main

                    import "github.com/spf13/cobra"

                    var deployCmd = &cobra.Command{Use: "deploy"}
                """,
            },
            "cli:deploy",
            (),
        ),
    ],
)
def test_synthetic_project_repos_emit_cross_repo_edges_for_api_surface_matrix(
    tmp_path,
    case_name,
    left_files,
    right_files,
    expected_surface,
    inject_left,
):
    left_root = _write_repo(tmp_path / f"left-{case_name}", left_files)
    right_root = _write_repo(tmp_path / f"right-{case_name}", right_files)
    merged_graph = _merge_project_graph({
        "left-wiki": _index_repo(left_root, inject_extensions=inject_left),
        "right-wiki": _index_repo(right_root),
    })

    surfaces_by_node = extract_api_surfaces_for_graph(merged_graph)
    edges = build_cross_repo_edges(
        wiki_ids=["left-wiki", "right-wiki"],
        merged_graph=merged_graph,
        relatedness={("left-wiki", "right-wiki"): 1.0},
        surfaces_by_node=surfaces_by_node,
        threshold=0.15,
    )

    matched_edges = [
        edge
        for edge in edges
        if expected_surface in str((edge.get("provenance") or {}).get("surface", ""))
    ]
    assert matched_edges, (
        f"Expected cross-repo edge for {case_name} on {expected_surface}; "
        f"surfaces={sorted({surface['surface'] for node_surfaces in surfaces_by_node.values() for surface in node_surfaces})}"
    )
    assert all(edge["edge_class"] == "cross_repo" for edge in matched_edges)
    assert all(edge["source_node_id"].split("::", 1)[0] != edge["target_node_id"].split("::", 1)[0] for edge in matched_edges)


def test_name_surfaces_do_not_link_shared_pydantic_dependency_from_real_indexed_repos(tmp_path):
    """Shared external imports like ``SecretStr`` are not repo contracts."""
    left_root = _write_repo(
        tmp_path / "left",
        {
            "models.py": """
                from pydantic import BaseModel, SecretStr

                class Configuration(BaseModel):
                    token: SecretStr
            """,
        },
    )
    right_root = _write_repo(
        tmp_path / "right",
        {
            "settings.py": """
                from pydantic import BaseModel, SecretStr

                class LocalSettings(BaseModel):
                    token: SecretStr
            """,
        },
    )
    merged_graph = _merge_project_graph({
        "left-wiki": _index_repo(left_root),
        "right-wiki": _index_repo(right_root),
    })

    surfaces_by_node = extract_api_surfaces_for_graph(merged_graph)
    all_surfaces = {
        surface["surface"]
        for node_surfaces in surfaces_by_node.values()
        for surface in node_surfaces
    }

    assert "name:secret_str" not in all_surfaces
    assert "name:base_model" not in all_surfaces


@pytest.mark.skipif(
    os.getenv("WIKIS_RUN_REAL_EMBEDDINGS") != "1",
    reason="set WIKIS_RUN_REAL_EMBEDDINGS=1 to run the real HuggingFace embedding smoke test",
)
def test_filesystem_indexer_populates_unified_db_with_real_huggingface_embeddings(tmp_path):
    """Opt-in smoke test for the real embedding path, no mocks involved.

    Normal CI should not download embedding models. To run locally with a
    cached model:

        WIKIS_RUN_REAL_EMBEDDINGS=1 pytest tests/integration/test_project_federation_api_surface.py -q

    Set ``WIKIS_TEST_HF_LOCAL_ONLY=0`` if downloading the model is acceptable.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    repo_root = _write_repo(
        tmp_path / "embedded-repo",
        {
            "app.py": """
                class ApiClient:
                    def models(self):
                        return "/configurations/models/{project_id}"
            """,
        },
    )

    model_kwargs = {}
    if os.getenv("WIKIS_TEST_HF_LOCAL_ONLY", "1") != "0":
        model_kwargs["local_files_only"] = True
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv(
            "WIKIS_TEST_HF_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        model_kwargs=model_kwargs,
    )

    indexer = FilesystemRepositoryIndexer(
        cache_dir=str(tmp_path / "cache"),
        clone_config=LocalPathConfig(local_path=str(repo_root)),
        cleanup_repos_on_exit=False,
        embeddings=embeddings,
        max_workers=1,
    )
    indexer.index_repository(str(repo_root), force_rebuild=True, max_files=4)

    storage = open_storage(
        repo_id=getattr(indexer, "_unified_db_cache_key", "test"),
        db_path=getattr(indexer, "_unified_db_path"),
        readonly=True,
    )
    try:
        if not storage.vec_available():
            pytest.skip("sqlite-vec is unavailable in this environment")
        embedded_rows = storage.conn.execute("SELECT count(*) FROM repo_vec").fetchone()[0]
    finally:
        storage.close()

    assert embedded_rows > 0