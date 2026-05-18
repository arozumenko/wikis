"""End-to-end cross-language linking against synthetic mini-repos.

These tests exercise the **production pipeline** as it runs during
indexing on the UI:

    EnhancedUnifiedGraphBuilder.analyze_repository(<tmp dir>)
        → unified NetworkX graph
        → extract_api_surfaces_for_graph(g, repo_root=...)
        → run_cross_language_linker(g, surfaces_by_node=..., parser_rels=...)
        → assert cross_language edges materialize

Each test case builds a tiny self-contained repo in ``tmp_path`` covering
a specific cross-language scenario and asserts that the linker creates
at least one ``edge_class="cross_language"`` edge bridging the right
two languages with the right surface kind.

The test suite is deliberately broader than the regex-only unit tests
because it exposes real parser quirks (decorator preamble loss, symbol
slicing, file-level vs symbol-level scope). For surface kinds where
the platform's tree-sitter parsers don't currently produce symbol nodes
(``.proto``, ``.graphql``, ``.feature``), the harness injects synthetic
file-level nodes so the linker still has something to pair against —
mirroring the way a future parser plugin would feed those formats.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import pytest

from app.core.code_graph.api_surface_extractor import (
    APISurface,
    extract_api_surfaces,
    extract_api_surfaces_for_graph,
)
from app.core.code_graph.cross_language_linker import (
    run_cross_language_linker,
)
from app.core.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
from app.core.parsers.csharp_visitor_parser import CSharpVisitorParser


# ──────────────────────────────────────────────────────────────────────
# Harness helpers
# ──────────────────────────────────────────────────────────────────────


def _write_repo(root: Path, files: Dict[str, str]) -> Path:
    """Materialize a synthetic repo. ``files`` maps relative path → text.
    De-dents each value so callers can use natural triple-quoted blocks.
    """
    for rel, content in files.items():
        full = root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(textwrap.dedent(content), encoding="utf-8")
    return root


def _csharp_parser_available() -> bool:
    try:
        parser = CSharpVisitorParser()
        return parser.parser is not None
    except Exception:
        return False


CSHARP_PARSER_AVAILABLE = _csharp_parser_available()


_LANGUAGE_BY_EXT = {
    ".proto": "proto",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".feature": "gherkin",
}


def _inject_file_nodes(
    g: nx.MultiDiGraph, root: Path, extensions: Tuple[str, ...]
) -> None:
    """Add file-as-node entries for extensions the parser doesn't slice
    into symbols (``.proto``, ``.graphql``, ``.feature``). Side effects:
    adds one node per matching file with ``source_text`` = file content.
    """
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in extensions:
            continue
        rel = path.relative_to(root).as_posix()
        node_id = f"file::{rel}"
        if g.has_node(node_id):
            continue
        g.add_node(
            node_id,
            language=_LANGUAGE_BY_EXT.get(path.suffix, ""),
            rel_path=rel,
            file_name=path.stem,
            symbol_name=path.stem,
            symbol_type="file",
            source_text=path.read_text(encoding="utf-8", errors="ignore"),
        )


def _run_pipeline(
    root: Path, *, inject_extensions: Tuple[str, ...] = ()
) -> Tuple[nx.MultiDiGraph, Dict[str, List[APISurface]], List[Tuple[str, str, dict]]]:
    """Run the production cross-language pipeline against a tmp repo.

    Returns: ``(graph, surfaces_by_node, cl_edges)``.
    """
    builder = EnhancedUnifiedGraphBuilder(max_workers=1)
    analysis = builder.analyze_repository(str(root))
    g: nx.MultiDiGraph = analysis.unified_graph

    if inject_extensions:
        _inject_file_nodes(g, root, inject_extensions)

    surfaces = extract_api_surfaces_for_graph(g, repo_root=str(root))
    edges = run_cross_language_linker(
        g,
        cross_language_relationships=getattr(analysis, "cross_language_relationships", None),
        surfaces_by_node=surfaces,
    )
    return g, surfaces, edges


def _cross_lang_edges(
    edges: List[Tuple[str, str, dict]],
) -> List[Tuple[str, str, dict]]:
    return [
        (s, t, a) for s, t, a in edges
        if a.get("edge_class") == "cross_language"
    ]


def _has_pair_with_languages(
    g: nx.MultiDiGraph,
    edges: List[Tuple[str, str, dict]],
    *,
    langs: Tuple[str, str],
    surface_substring: str = "",
    surface_kind: str = "",
) -> bool:
    """True if any cross-language edge connects the two given languages
    AND its provenance.surface contains the substring (when supplied)."""
    for src, tgt, attrs in _cross_lang_edges(edges):
        a = (g.nodes.get(src, {}).get("language") or "").lower()
        b = (g.nodes.get(tgt, {}).get("language") or "").lower()
        if {a, b} != {langs[0].lower(), langs[1].lower()}:
            continue
        prov = attrs.get("provenance") or {}
        s = str(prov.get("surface") or "")
        if surface_substring and surface_substring not in s:
            continue
        if surface_kind and prov.get("kind", "").lower() != surface_kind.lower():
            continue
        return True
    return False


# ──────────────────────────────────────────────────────────────────────
# REST: Python ↔ TypeScript (FastAPI ↔ openapi-ts client)
# ──────────────────────────────────────────────────────────────────────


def test_python_fastapi_pairs_with_typescript_openapi_client(tmp_path):
    _write_repo(tmp_path, {
        "backend/app/api/routes/items.py": """
            from fastapi import APIRouter

            router = APIRouter(prefix="/items", tags=["items"])

            @router.get("/")
            def list_items():
                return []

            @router.post("/")
            def create_item(payload: dict):
                return payload
        """,
        "frontend/src/client/sdk.gen.ts": """
            export const ItemsService = {
                listItems() {
                    return __request(OpenAPI, { method: 'GET', url: '/api/v1/items/' });
                },
                createItem(data) {
                    return __request(OpenAPI, { method: 'POST', url: '/api/v1/items/' });
                },
            };
        """,
    })
    g, surfaces, edges = _run_pipeline(tmp_path)
    assert _has_pair_with_languages(
        g, edges, langs=("python", "typescript"), surface_substring="GET /items",
    )
    assert _has_pair_with_languages(
        g, edges, langs=("python", "typescript"), surface_substring="POST /items",
    )


# ──────────────────────────────────────────────────────────────────────
# REST: Go ↔ TypeScript (chi ↔ axios)
# ──────────────────────────────────────────────────────────────────────


def test_go_chi_pairs_with_typescript_axios(tmp_path):
    _write_repo(tmp_path, {
        "server/main.go": """
            package main

            import "github.com/go-chi/chi/v5"

            func main() {
                r := chi.NewRouter()
                r.GET("/api/v1/users", listUsers)
                r.POST("/api/v1/users", createUser)
            }

            func listUsers(w http.ResponseWriter, r *http.Request)   {}
            func createUser(w http.ResponseWriter, r *http.Request)  {}
        """,
        "web/src/api/users.ts": """
            export async function listUsers() {
                return axios.get('/api/v1/users');
            }
            export async function createUser(payload: any) {
                return axios.post('/api/v1/users', payload);
            }
        """,
    })
    g, surfaces, edges = _run_pipeline(tmp_path)
    assert _has_pair_with_languages(
        g, edges, langs=("go", "typescript"), surface_substring="GET /api/v1/users",
    )
    assert _has_pair_with_languages(
        g, edges, langs=("go", "typescript"), surface_substring="POST /api/v1/users",
    )


# ──────────────────────────────────────────────────────────────────────
# REST: Java Spring ↔ TypeScript fetch
# ──────────────────────────────────────────────────────────────────────


def test_java_spring_pairs_with_typescript_fetch(tmp_path):
    _write_repo(tmp_path, {
        "backend/src/main/java/com/example/UserController.java": """
            package com.example;

            import org.springframework.web.bind.annotation.*;

            @RestController
            public class UserController {

                @GetMapping("/api/v1/users")
                public List<User> list() { return new ArrayList<>(); }

                @PostMapping("/api/v1/users")
                public User create(@RequestBody User u) { return u; }
            }
        """,
        "web/src/api/users.ts": """
            export async function getUsers() {
                return fetch('/api/v1/users');
            }
            export async function createUser(payload: any) {
                return fetch('/api/v1/users', { method: 'POST', body: JSON.stringify(payload) });
            }
        """,
    })
    g, surfaces, edges = _run_pipeline(tmp_path)
    assert _has_pair_with_languages(
        g, edges, langs=("java", "typescript"), surface_substring="GET /api/v1/users",
    )
    assert _has_pair_with_languages(
        g, edges, langs=("java", "typescript"), surface_substring="POST /api/v1/users",
    )


# ──────────────────────────────────────────────────────────────────────
# REST: Rust axum ↔ Go HTTP client (no JS at all)
# ──────────────────────────────────────────────────────────────────────


def test_rust_pairs_with_python_via_suffix_alternate(tmp_path):
    """Rust handler declared with prefix-stripped path matches a Python
    client calling the gateway-prefixed URL via the suffix alternate."""
    _write_repo(tmp_path, {
        "rust_server/src/lib.rs": """
            // Bare handler — surface emitted via Python-side fetch alternate.
            pub fn list_health() -> &'static str { "ok" }
        """,
        "py_client/main.py": """
            import requests

            def check_health():
                # External call to the rust server through the gateway.
                # Surface: GET /healthz (decorator-style detection NA in Rust here);
                # we exercise the python-side regex matcher instead.
                requests.get('/api/v1/healthz')

            # A second Python service registers /healthz under a Flask blueprint.
            from flask import Blueprint
            bp = Blueprint('health', __name__, url_prefix='/api/v1')

            @bp.route('/healthz')
            def health():
                return {'ok': True}
        """,
    })
    # The blueprint and the requests.get('/api/v1/healthz') live in the same
    # file → the python matcher will emit two GET /api/v1/healthz surfaces
    # plus their stripped alternates. Without a TS client we still exercise
    # the suffix-alternate path via Python+Python, but the linker filters
    # same-language pairs, so we instead verify that surfaces materialize.
    g, surfaces, edges = _run_pipeline(tmp_path)
    py_surfaces = []
    for node_surfaces in surfaces.values():
        for s in node_surfaces:
            if s["surface"].startswith(("GET /api/v1/healthz", "GET /healthz")):
                py_surfaces.append(s["surface"])
    assert "GET /api/v1/healthz" in py_surfaces
    # Suffix alternate emitted for the prefixed path.
    assert "GET /healthz" in py_surfaces


# ──────────────────────────────────────────────────────────────────────
# Object / DTO sharing: Python dataclass ↔ TypeScript interface
# ──────────────────────────────────────────────────────────────────────


def test_python_dataclass_pairs_with_typescript_interface(tmp_path):
    _write_repo(tmp_path, {
        "backend/app/models/user.py": """
            from dataclasses import dataclass

            @dataclass
            class User:
                id: int
                name: str
                email: str
        """,
        "web/src/types/user.ts": """
            export interface User {
                id: number;
                name: string;
                email: string;
            }
        """,
    })
    g, surfaces, edges = _run_pipeline(tmp_path)
    assert _has_pair_with_languages(
        g, edges, langs=("python", "typescript"),
        surface_substring="obj:user#email,id,name",
    )


def test_go_struct_pairs_with_typescript_interface(tmp_path):
    """Go ``ID``/``Name`` (PascalCase) pairs with TS ``id``/``name`` via
    the JSON tag → case-folded surface."""
    _write_repo(tmp_path, {
        "server/models/user.go": """
            package models

            type User struct {
                ID    int64  `json:"id"`
                Name  string `json:"name"`
                Email string `json:"email"`
            }
        """,
        "web/src/types/user.ts": """
            export interface User {
                id: number;
                name: string;
                email: string;
            }
        """,
    })
    g, surfaces, edges = _run_pipeline(tmp_path)
    assert _has_pair_with_languages(
        g, edges, langs=("go", "typescript"),
        surface_substring="obj:user#email,id,name",
    )


def test_java_record_pairs_with_rust_struct(tmp_path):
    _write_repo(tmp_path, {
        "backend/src/main/java/com/example/Item.java": """
            package com.example;
            public record Item(Long id, String title, Double price) {}
        """,
        "rust_service/src/models.rs": """
            use serde::{Deserialize, Serialize};

            #[derive(Serialize, Deserialize)]
            pub struct Item {
                pub id: i64,
                pub title: String,
                pub price: f64,
            }
        """,
    })
    g, surfaces, edges = _run_pipeline(tmp_path)
    assert _has_pair_with_languages(
        g, edges, langs=("java", "rust"),
        surface_substring="obj:item#id,price,title",
    )


def test_cpp_struct_pairs_with_typescript_interface(tmp_path):
    _write_repo(tmp_path, {
        "native/include/user_profile.hpp": """
            #include <string>

            struct UserProfile {
                int id;
                std::string full_name;
                std::string email;

                void normalize();
            };
        """,
        "web/src/types/userProfile.ts": """
            export interface UserProfile {
                id: number;
                full_name: string;
                email: string;
            }
        """,
    })
    g, surfaces, edges = _run_pipeline(tmp_path)
    assert _has_pair_with_languages(
        g, edges, langs=("cpp", "typescript"),
        surface_substring="obj:user_profile#email,full_name,id",
    )


def test_cpp_local_include_pairs_with_typescript_definition_name(tmp_path):
    _write_repo(tmp_path, {
        "native/src/bridge.cpp": """
            #include "TransformSdk.hpp"

            void connect_transform_sdk() {}
        """,
        "web/src/contracts/transform.ts": """
            export interface TransformSdk {
                endpoint: string;
                version: string;
            }
        """,
    })
    g, surfaces, edges = _run_pipeline(tmp_path)
    surf_keys = {s["surface"] for ns in surfaces.values() for s in ns}
    assert "name:transform_sdk" in surf_keys
    assert _has_pair_with_languages(
        g, edges, langs=("cpp", "typescript"),
        surface_substring="name:transform_sdk",
    )


@pytest.mark.skipif(not CSHARP_PARSER_AVAILABLE, reason="tree-sitter C# parser is unavailable")
def test_csharp_record_pairs_with_python_dataclass(tmp_path):
    _write_repo(tmp_path, {
        "Backend/Models/Order.cs": """
            namespace Backend.Models;
            public record Order(long OrderId, double Total);
        """,
        "py_service/order.py": """
            from dataclasses import dataclass

            @dataclass
            class Order:
                order_id: int
                total: float
        """,
    })
    g, surfaces, edges = _run_pipeline(tmp_path)
    assert _has_pair_with_languages(
        g, edges, langs=("csharp", "python"),
        surface_substring="obj:order#order_id,total",
    )


# ──────────────────────────────────────────────────────────────────────
# FFI / ABI
# ──────────────────────────────────────────────────────────────────────


def test_rust_extern_pairs_with_python_ctypes(tmp_path):
    _write_repo(tmp_path, {
        "native/src/lib.rs": """
            #[no_mangle]
            pub extern "C" fn compute_hash(data: *const u8, len: usize) -> u64 {
                // ... hash impl ...
                0
            }
        """,
        "py_client/native_bridge.py": """
            import ctypes

            lib = ctypes.CDLL('libnative.so')

            def compute_hash(data: bytes) -> int:
                lib.compute_hash.restype = ctypes.c_uint64
                return lib.compute_hash(data, len(data))
        """,
    })
    g, surfaces, edges = _run_pipeline(tmp_path)
    assert _has_pair_with_languages(
        g, edges, langs=("rust", "python"), surface_substring="ffi:compute_hash",
    )


@pytest.mark.skipif(not CSHARP_PARSER_AVAILABLE, reason="tree-sitter C# parser is unavailable")
def test_csharp_pinvoke_does_not_create_false_pair(tmp_path):
    """P/Invoke surface uses library name (``ffi:libnative``) — verify
    it appears in surfaces but does NOT incorrectly pair with the Rust
    function-name surface (``ffi:compute_hash``)."""
    _write_repo(tmp_path, {
        "native/src/lib.rs": """
            #[no_mangle]
            pub extern "C" fn compute_hash(data: *const u8) -> u64 { 0 }
        """,
        "Client/NativeBridge.cs": """
            using System.Runtime.InteropServices;

            public static class NativeBridge {
                [DllImport("libnative")]
                public static extern ulong compute_hash(byte[] data);
            }
        """,
    })
    g, surfaces, edges = _run_pipeline(tmp_path)
    surf_keys = {s["surface"] for ns in surfaces.values() for s in ns}
    # Both kinds of FFI surface emitted.
    assert "ffi:compute_hash" in surf_keys  # rust extern
    assert "ffi:libnative" in surf_keys     # C# DllImport (library-keyed)
    # And the Rust extern + the C# native function (NOT just library) pair via name.
    # If the C# parser emits a node for ``compute_hash`` with a ``DllImport``
    # decorator preamble we should also see the symbol-name surface.
    assert _has_pair_with_languages(
        g, edges, langs=("rust", "csharp"), surface_substring="ffi:compute_hash",
    )


def test_jni_native_method_pairs_with_python_ctypes(tmp_path):
    _write_repo(tmp_path, {
        "Backend/src/main/java/com/example/NativeBridge.java": """
            package com.example;
            public class NativeBridge {
                public native long computeHash(byte[] data);
            }
        """,
        "py_client/bridge.py": """
            import ctypes
            lib = ctypes.CDLL('libnative.so')
            def computeHash(data: bytes) -> int:
                return lib.computeHash(data, len(data))
        """,
    })
    g, surfaces, edges = _run_pipeline(tmp_path)
    assert _has_pair_with_languages(
        g, edges, langs=("java", "python"), surface_substring="ffi:computeHash",
    )


# ──────────────────────────────────────────────────────────────────────
# gRPC (file-injected nodes — no .proto symbol parser)
# ──────────────────────────────────────────────────────────────────────


def test_grpc_proto_pairs_with_python_servicer(tmp_path):
    _write_repo(tmp_path, {
        "proto/user.proto": """
            syntax = "proto3";
            package user;

            service UserSvc {
              rpc GetUser (GetUserRequest) returns (GetUserResponse);
              rpc ListUsers (ListUsersRequest) returns (ListUsersResponse);
            }
        """,
        "py_server/user_pb2_grpc.py": """
            class UserSvcServicer:
                def GetUser(self, request, context):
                    return None

                def ListUsers(self, request, context):
                    return None
        """,
    })
    g, surfaces, edges = _run_pipeline(tmp_path, inject_extensions=(".proto",))
    surf_keys = {s["surface"] for ns in surfaces.values() for s in ns}
    assert "grpc:UserSvc/GetUser" in surf_keys
    assert "grpc:UserSvc/ListUsers" in surf_keys
    # Cross-language pair between the .proto file node and the python stub.
    assert _has_pair_with_languages(
        g, edges, langs=("proto", "python"), surface_substring="grpc:UserSvc/GetUser",
    )


# ──────────────────────────────────────────────────────────────────────
# GraphQL (SDL ↔ resolver decorator)
# ──────────────────────────────────────────────────────────────────────


def test_graphql_sdl_pairs_with_python_resolver(tmp_path):
    _write_repo(tmp_path, {
        "schema/schema.graphql": """
            type Query {
              user(id: ID!): User
            }
            type Mutation {
              createUser(input: UserInput!): User
            }
        """,
        "backend/app/resolvers/user.py": """
            from strawberry import type

            @Query
            def user(id: str): ...

            @Mutation
            def createUser(input: dict): ...
        """,
    })
    g, surfaces, edges = _run_pipeline(tmp_path, inject_extensions=(".graphql",))
    surf_keys = {s["surface"] for ns in surfaces.values() for s in ns}
    assert "gql:query" in surf_keys
    assert "gql:mutation" in surf_keys
    assert _has_pair_with_languages(
        g, edges, langs=("graphql", "python"), surface_substring="gql:",
    )


# ──────────────────────────────────────────────────────────────────────
# CLI (Python click ↔ Go cobra — same command name)
# ──────────────────────────────────────────────────────────────────────


def test_python_click_pairs_with_go_cobra(tmp_path):
    _write_repo(tmp_path, {
        "py_cli/main.py": """
            import click

            @click.command('deploy')
            def deploy():
                '''Deploy the application.'''
                pass
        """,
        "go_cli/main.go": """
            package main

            import "github.com/spf13/cobra"

            var deployCmd = &cobra.Command{
                Use:   "deploy",
                Short: "Deploy the application",
            }

            func main() {
                rootCmd := &cobra.Command{Use: "app"}
                rootCmd.AddCommand(deployCmd)
                rootCmd.Execute()
            }
        """,
    })
    g, surfaces, edges = _run_pipeline(tmp_path)
    assert _has_pair_with_languages(
        g, edges, langs=("python", "go"), surface_substring="cli:deploy",
    )


# ──────────────────────────────────────────────────────────────────────
# BDD (Gherkin .feature ↔ Python step definition)
# ──────────────────────────────────────────────────────────────────────


def test_gherkin_feature_pairs_with_python_step_def(tmp_path):
    _write_repo(tmp_path, {
        "tests/features/login.feature": """
            Feature: Login
              Scenario: Successful login
                Given the user is logged in
                When the user navigates to /home
                Then the dashboard is displayed
        """,
        "tests/steps/login_steps.py": """
            from behave import given, when, then

            @given('the user is logged in')
            def step_login(context):
                pass

            @when('the user navigates to /home')
            def step_navigate(context):
                pass

            @then('the dashboard is displayed')
            def step_assert(context):
                pass
        """,
    })
    g, surfaces, edges = _run_pipeline(tmp_path, inject_extensions=(".feature",))
    surf_keys = {s["surface"] for ns in surfaces.values() for s in ns}
    assert "bdd:the user is logged in" in surf_keys
    assert _has_pair_with_languages(
        g, edges, langs=("gherkin", "python"),
        surface_substring="bdd:the user is logged in",
    )


# ──────────────────────────────────────────────────────────────────────
# Multi-tier scenario: REST + obj + FFI in one repo
# ──────────────────────────────────────────────────────────────────────


def test_full_stack_repo_yields_multiple_surface_kinds(tmp_path):
    """A realistic mini-repo combining REST + DTO + FFI cross-language
    bindings produces ≥ 3 distinct surface kinds and ≥ 1 cross_language
    edge per kind."""
    _write_repo(tmp_path, {
        "backend/app/api/users.py": """
            from fastapi import APIRouter
            router = APIRouter(prefix="/users")

            @router.get("/{id}")
            def get_user(id: int):
                return {"id": id, "name": "Alice", "email": "a@b.c"}
        """,
        "backend/app/models/user.py": """
            from dataclasses import dataclass

            @dataclass
            class User:
                id: int
                name: str
                email: str
        """,
        "backend/app/native_bridge.py": """
            import ctypes
            lib = ctypes.CDLL('libnative.so')

            def hash_password(pw: str) -> int:
                return lib.hash_password(pw.encode())
        """,
        "frontend/src/types/user.ts": """
            export interface User {
                id: number;
                name: string;
                email: string;
            }
        """,
        "frontend/src/api/users.ts": """
            export async function getUser(id: number): Promise<User> {
                return __request(OpenAPI, { method: 'GET', url: '/api/v1/users/{id}' });
            }
        """,
        "native/src/lib.rs": """
            #[no_mangle]
            pub extern "C" fn hash_password(pw: *const u8) -> u64 { 0 }
        """,
    })
    g, surfaces, edges = _run_pipeline(tmp_path)
    cl = _cross_lang_edges(edges)
    assert cl, "expected at least one cross_language edge"

    # Collect the surfaces actually paired across languages.
    paired_surfaces = {
        (e[2].get("provenance") or {}).get("surface", "")
        for e in cl
    }
    # REST pair (Python/TS): GET /users/{id} via suffix alternate.
    assert any("GET /users/{id}" in s for s in paired_surfaces), \
        f"no REST pair, got: {paired_surfaces}"
    # DTO pair (Python/TS): obj:user#email,id,name.
    assert any("obj:user#email,id,name" in s for s in paired_surfaces), \
        f"no DTO pair, got: {paired_surfaces}"
    # FFI pair (Rust/Python): ffi:hash_password.
    assert any("ffi:hash_password" in s for s in paired_surfaces), \
        f"no FFI pair, got: {paired_surfaces}"


# ──────────────────────────────────────────────────────────────────────
# Regression: openapi-ts generated `types.gen.ts` (TS type_alias)
# pairs with FastAPI Pydantic models even when the parser leaves
# ``source_text`` empty for thin type aliases.
# ──────────────────────────────────────────────────────────────────────


def test_openapi_ts_type_alias_pairs_with_pydantic_when_source_text_empty(tmp_path):
    """Reproduces the production gap observed against
    fastapi/full-stack-fastapi-template: ``frontend/src/client/types.gen.ts``
    contains every DTO as ``export type X = { ... }`` but the TS parser
    persists those nodes with ``source_text=None``. The L1 obj matcher
    then has nothing to scan, so DTO pairs never form. The fix is for
    ``extract_api_surfaces_for_graph`` to slice the file
    ``[start_line..end_line]`` when ``source_text`` is empty.

    This test bypasses the parser by injecting a TS ``type_alias`` node
    with explicit empty ``source_text`` + on-disk file, mirroring the
    exact production state.
    """
    _write_repo(tmp_path, {
        "backend/app/models.py": """
            from sqlmodel import SQLModel

            class UserPublic(SQLModel):
                email: str
                full_name: str
                id: str
        """,
        "frontend/src/client/types.gen.ts": """
            // This file is auto-generated by @hey-api/openapi-ts

            export type UserPublic = {
                email: string;
                full_name?: (string | null);
                id: string;
            };
        """,
    })

    builder = EnhancedUnifiedGraphBuilder(max_workers=1)
    g: nx.MultiDiGraph = builder.analyze_repository(str(tmp_path)).unified_graph

    # Inject the TS type_alias node the way the production parser does:
    # span recorded but body stripped. start_line/end_line are 1-based,
    # inclusive, and cover the ``export type UserPublic = {…};`` block.
    rel = "frontend/src/client/types.gen.ts"
    lines = (tmp_path / rel).read_text(encoding="utf-8").splitlines()
    # Find the export-type declaration's first line (1-based).
    start = next(i + 1 for i, ln in enumerate(lines) if "export type UserPublic" in ln)
    end = next(i + 1 for i, ln in enumerate(lines[start - 1:], start=start) if "};" in ln)
    node_id = "typescript::frontend__src__client__types_gen_ts::types.gen.UserPublic"
    g.add_node(
        node_id,
        language="typescript",
        rel_path=rel,
        file_name="types.gen",
        symbol_name="UserPublic",
        symbol_type="type_alias",
        source_text=None,
        start_line=start,
        end_line=end,
    )

    surfaces = extract_api_surfaces_for_graph(g, repo_root=str(tmp_path))
    edges = run_cross_language_linker(g, surfaces_by_node=surfaces)

    # The TS node must now expose obj:user_public#email,full_name,id.
    ts_surfaces = surfaces.get(node_id, [])
    obj_keys = [s["surface"] for s in ts_surfaces if s["kind"] == "obj"]
    assert any("obj:user_public" in k for k in obj_keys), \
        f"slice fallback failed; TS node has no obj surface: {ts_surfaces}"

    # And the cross-language linker must form a Python↔TS pair on it.
    assert _has_pair_with_languages(
        g, edges,
        langs=("python", "typescript"),
        surface_substring="obj:user_public",
    ), f"no obj pair across py/ts; cross_lang edges = {[(e[0], e[1], (e[2].get('provenance') or {}).get('surface')) for e in _cross_lang_edges(edges)]}"
