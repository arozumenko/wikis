"""API surface extractor (Phase 6 / Action 6.2).

Extracts canonical API-surface keys from parser output so that the
cross-language linker can pair endpoints across languages
(e.g. a Python ``@app.post("/api/users")`` handler with the TypeScript
client that calls ``fetch("/api/users", {method: "POST"})``).

This module is **side-effect free** — it inspects node attributes and
returns a list of :class:`APISurface` dicts. Persistence and graph
mutation happen in the linker / pipeline integration layer.

The matchers are intentionally lightweight regex/heuristic matchers
keyed by ``language``. The dispatcher returns an empty list when no
matcher applies, so callers can run it on every node without guarding.

Supported surfaces
------------------
* REST (Python: Flask, FastAPI; TS/JS: Express, NestJS; Java: JAX-RS,
  Spring; Go: chi/gin) — canonical key ``"<METHOD> <path>"``.
* gRPC — canonical key ``"grpc:<service>/<method>"`` from the proto
  service / rpc definition or generated stubs.
* GraphQL — canonical key ``"gql:<operation>:<field>"``.
* FFI — canonical key ``"ffi:<symbol>"`` (matches ``extern "C"``,
  ``ctypes``, JNI, P/Invoke, wasm-bindgen).
* BDD — canonical key ``"bdd:<step text>"``.
* CLI — canonical key ``"cli:<command path>"`` for argparse / click /
  cobra subcommand registration.

A node may yield multiple surfaces (e.g. a FastAPI handler decorated
with both ``@router.get("/")`` and ``@router.head("/")``).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, TypedDict


class APISurface(TypedDict):
    kind: str
    surface: str
    weight_hint: float
    metadata: dict


# ──────────────────────────────────────────────────────────────────────
# REST matchers
# ──────────────────────────────────────────────────────────────────────

# Common HTTP method tokens used across decorators.
_HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")

# Python: @app.get("/x"), @router.post("/x"), @blueprint.route("/x", methods=["GET"])
_PY_REST_DECORATOR = re.compile(
    r"@\s*(?:\w+\.)?(?P<method>get|post|put|patch|delete|head|options|route)"
    r"\s*\(\s*(?P<args>[^)]+)\)",
    re.IGNORECASE,
)
_PY_ROUTE_METHODS = re.compile(r"methods\s*=\s*\[([^\]]+)\]", re.IGNORECASE)
_QUOTED_PATH = re.compile(r"""['"]([^'"]+)['"]""")

# TypeScript/JavaScript: @Get("/x"), @Post("/x"), app.get("/x", ...)
_TS_NEST_DECORATOR = re.compile(
    r"@\s*(?P<method>Get|Post|Put|Patch|Delete|Head|Options)"
    r"\s*\(\s*(?P<args>[^)]*)\)",
)
_TS_EXPRESS_CALL = re.compile(
    r"\b(?:app|router)\s*\.\s*(?P<method>get|post|put|patch|delete|head|options)"
    r"\s*\(\s*(?P<args>[^,]+),",
    re.IGNORECASE,
)

# Java: @GET / @POST + @Path("/x"); Spring: @GetMapping("/x")
_JAVA_PATH = re.compile(r"""@\s*Path\s*\(\s*['"]([^'"]+)['"]""")
_JAVA_METHOD = re.compile(r"@\s*(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b")
_JAVA_SPRING = re.compile(
    r"@\s*(?P<method>Get|Post|Put|Patch|Delete|Head|Options)Mapping"
    r"\s*\(\s*['\"]?(?P<path>[^'\")\s,]*)",
)
_JAVA_REQUEST = re.compile(
    r"@\s*RequestMapping\s*\(\s*['\"]?(?P<path>[^'\")\s,]*)",
)

# Go: chi/gin r.GET("/x", ...)
_GO_REST = re.compile(
    r"\b\w+\s*\.\s*(?P<method>GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)"
    r"\s*\(\s*['\"](?P<path>[^'\"]+)",
)


def _normalize_path(path: str) -> str:
    p = path.strip().strip("\"'")
    if not p:
        return "/"
    if not p.startswith("/"):
        p = "/" + p
    # Drop trailing slash except for the root path.
    if len(p) > 1 and p.endswith("/"):
        p = p[:-1]
    return p


def _surface_rest(method: str, path: str, weight_hint: float = 0.7) -> APISurface:
    return APISurface(
        kind="rest",
        surface=f"{method.upper()} {_normalize_path(path)}",
        weight_hint=weight_hint,
        metadata={"method": method.upper(), "path": _normalize_path(path)},
    )


def _match_rest_python(text: str) -> List[APISurface]:
    out: List[APISurface] = []
    for m in _PY_REST_DECORATOR.finditer(text):
        method = m.group("method").lower()
        args = m.group("args")
        path_match = _QUOTED_PATH.search(args)
        if not path_match:
            continue
        path = path_match.group(1)
        if method == "route":
            methods_match = _PY_ROUTE_METHODS.search(args)
            methods = []
            if methods_match:
                methods = [
                    s.strip().strip("\"'").upper()
                    for s in methods_match.group(1).split(",")
                    if s.strip()
                ]
            if not methods:
                methods = ["GET"]
            for met in methods:
                if met in _HTTP_METHODS:
                    out.append(_surface_rest(met, path))
        else:
            out.append(_surface_rest(method, path))
    return out


def _match_rest_typescript(text: str) -> List[APISurface]:
    out: List[APISurface] = []
    for m in _TS_NEST_DECORATOR.finditer(text):
        method = m.group("method")
        args = m.group("args") or ""
        path_match = _QUOTED_PATH.search(args)
        path = path_match.group(1) if path_match else "/"
        out.append(_surface_rest(method, path))
    for m in _TS_EXPRESS_CALL.finditer(text):
        method = m.group("method")
        args = m.group("args") or ""
        path_match = _QUOTED_PATH.search(args)
        if path_match:
            out.append(_surface_rest(method, path_match.group(1)))
    return out


def _match_rest_java(text: str) -> List[APISurface]:
    out: List[APISurface] = []

    # JAX-RS: @Path + @GET/POST/...
    paths = [m.group(1) for m in _JAVA_PATH.finditer(text)]
    methods = [m.group(1) for m in _JAVA_METHOD.finditer(text)]
    if paths and methods:
        for p in paths:
            for met in methods:
                out.append(_surface_rest(met, p))

    # Spring: @GetMapping("/x"), @PostMapping("/x"), @RequestMapping("/x")
    for m in _JAVA_SPRING.finditer(text):
        out.append(_surface_rest(m.group("method"), m.group("path") or "/"))
    for m in _JAVA_REQUEST.finditer(text):
        # @RequestMapping defaults to GET when no method= specified.
        out.append(_surface_rest("GET", m.group("path") or "/"))

    return out


def _match_rest_go(text: str) -> List[APISurface]:
    out: List[APISurface] = []
    for m in _GO_REST.finditer(text):
        out.append(_surface_rest(m.group("method"), m.group("path")))
    return out


# ──────────────────────────────────────────────────────────────────────
# gRPC
# ──────────────────────────────────────────────────────────────────────

# proto: ``rpc Foo(BarRequest) returns (BarResponse);`` inside ``service Svc { ... }``
_PROTO_SERVICE = re.compile(r"\bservice\s+(\w+)\s*{", re.MULTILINE)
_PROTO_RPC = re.compile(r"\brpc\s+(\w+)\s*\(", re.MULTILINE)


def _match_grpc(text: str, language: str) -> List[APISurface]:
    """Detect gRPC services across .proto definitions and Python/Java stubs."""
    out: List[APISurface] = []

    if language == "proto" or "service " in text and "rpc " in text:
        services = _PROTO_SERVICE.findall(text)
        rpcs = _PROTO_RPC.findall(text)
        for svc in services or [""]:
            for rpc in rpcs:
                out.append(APISurface(
                    kind="grpc",
                    surface=f"grpc:{svc}/{rpc}" if svc else f"grpc:{rpc}",
                    weight_hint=0.8,
                    metadata={"service": svc, "method": rpc},
                ))

    return out


# ──────────────────────────────────────────────────────────────────────
# GraphQL
# ──────────────────────────────────────────────────────────────────────

_GQL_FIELD = re.compile(
    r"\b(?P<op>type|extend\s+type)\s+(?P<root>Query|Mutation|Subscription)\s*{",
    re.IGNORECASE,
)
_GQL_RESOLVER_DEC = re.compile(
    r"@\s*(?P<op>Query|Mutation|Subscription|Resolver|FieldResolver)\s*\(",
)


def _match_graphql(text: str) -> List[APISurface]:
    out: List[APISurface] = []
    if "type Query" in text or "type Mutation" in text or "type Subscription" in text:
        # Coarse SDL detection — emit a single root surface so the linker
        # can still pair files. Field-level surfaces would need a real
        # GraphQL parser.
        for m in _GQL_FIELD.finditer(text):
            out.append(APISurface(
                kind="graphql",
                surface=f"gql:{m.group('root').lower()}",
                weight_hint=0.6,
                metadata={"root": m.group("root").lower()},
            ))
    for m in _GQL_RESOLVER_DEC.finditer(text):
        out.append(APISurface(
            kind="graphql",
            surface=f"gql:{m.group('op').lower()}",
            weight_hint=0.5,
            metadata={"resolver": m.group("op").lower()},
        ))
    return out


# ──────────────────────────────────────────────────────────────────────
# FFI
# ──────────────────────────────────────────────────────────────────────

_FFI_EXTERN_C = re.compile(r"""extern\s+["']C["']""")
_FFI_CTYPES = re.compile(r"\bctypes\.(?:CDLL|WinDLL|cdll|windll)\b")
_FFI_JNI = re.compile(r"\bnative\s+\w+\s+\w+\s*\(")
_FFI_PINVOKE = re.compile(r"\[\s*DllImport\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")
_FFI_WASM = re.compile(r"#\s*\[\s*wasm_bindgen\s*\]")


def _match_ffi(text: str, symbol_name: str) -> List[APISurface]:
    out: List[APISurface] = []
    triggers = (
        bool(_FFI_EXTERN_C.search(text))
        or bool(_FFI_CTYPES.search(text))
        or bool(_FFI_JNI.search(text))
        or bool(_FFI_WASM.search(text))
    )
    if triggers and symbol_name:
        out.append(APISurface(
            kind="ffi",
            surface=f"ffi:{symbol_name}",
            weight_hint=0.6,
            metadata={"symbol": symbol_name},
        ))
    for m in _FFI_PINVOKE.finditer(text):
        out.append(APISurface(
            kind="ffi",
            surface=f"ffi:{m.group(1)}",
            weight_hint=0.7,
            metadata={"library": m.group(1)},
        ))
    return out


# ──────────────────────────────────────────────────────────────────────
# BDD (Gherkin step → step definition)
# ──────────────────────────────────────────────────────────────────────

_BDD_DECORATOR = re.compile(
    r"@\s*(?P<kind>given|when|then|step)\s*\(\s*['\"](?P<text>[^'\"]+)['\"]\s*\)",
    re.IGNORECASE,
)
_BDD_GHERKIN = re.compile(
    r"^\s*(?P<kind>Given|When|Then|And|But)\s+(?P<text>.+)$",
    re.MULTILINE,
)


def _match_bdd(text: str) -> List[APISurface]:
    out: List[APISurface] = []
    for m in _BDD_DECORATOR.finditer(text):
        out.append(APISurface(
            kind="bdd",
            surface=f"bdd:{m.group('text').strip().lower()}",
            weight_hint=0.7,
            metadata={"kind": m.group("kind").lower()},
        ))
    for m in _BDD_GHERKIN.finditer(text):
        out.append(APISurface(
            kind="bdd",
            surface=f"bdd:{m.group('text').strip().lower()}",
            weight_hint=0.6,
            metadata={"kind": m.group("kind").lower()},
        ))
    return out


# ──────────────────────────────────────────────────────────────────────
# CLI (argparse / click / cobra)
# ──────────────────────────────────────────────────────────────────────

_CLI_CLICK = re.compile(
    r"@\s*(?:\w+\.)?(?:command|group)\s*\(\s*(?:name\s*=\s*)?['\"]([^'\"]+)['\"]",
)
_CLI_ARGPARSE = re.compile(
    r"add_subparsers\s*\(.*?\)\.add_parser\s*\(\s*['\"]([^'\"]+)['\"]",
    re.DOTALL,
)
_CLI_COBRA = re.compile(r"&cobra\.Command\s*{[^}]*?Use:\s*['\"]([^'\"]+)['\"]", re.DOTALL)


def _match_cli(text: str) -> List[APISurface]:
    out: List[APISurface] = []
    for m in _CLI_CLICK.finditer(text):
        out.append(APISurface(
            kind="cli",
            surface=f"cli:{m.group(1)}",
            weight_hint=0.6,
            metadata={"framework": "click"},
        ))
    for m in _CLI_ARGPARSE.finditer(text):
        out.append(APISurface(
            kind="cli",
            surface=f"cli:{m.group(1)}",
            weight_hint=0.6,
            metadata={"framework": "argparse"},
        ))
    for m in _CLI_COBRA.finditer(text):
        out.append(APISurface(
            kind="cli",
            surface=f"cli:{m.group(1).split()[0]}",
            weight_hint=0.6,
            metadata={"framework": "cobra"},
        ))
    return out


# ──────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────

_REST_BY_LANGUAGE: Dict[str, Callable[[str], List[APISurface]]] = {
    "python": _match_rest_python,
    "typescript": _match_rest_typescript,
    "javascript": _match_rest_typescript,
    "java": _match_rest_java,
    "kotlin": _match_rest_java,
    "go": _match_rest_go,
}


def extract_api_surfaces(
    node_data: dict,
    parser_metadata: Optional[dict] = None,
) -> List[APISurface]:
    """Return all API surfaces visible in *node_data*.

    The matchers operate on ``source_text`` plus a few normalised
    attributes (``language``, ``symbol_name``). ``parser_metadata`` is
    accepted but currently unused — reserved for matchers that need
    decorator AST detail beyond what survives in ``source_text``.
    """
    text = (node_data.get("source_text") or "")
    if not text:
        return []
    language = (node_data.get("language") or "").lower()
    symbol_name = node_data.get("symbol_name") or ""

    surfaces: List[APISurface] = []

    rest_fn = _REST_BY_LANGUAGE.get(language)
    if rest_fn:
        surfaces.extend(rest_fn(text))

    surfaces.extend(_match_grpc(text, language))
    surfaces.extend(_match_graphql(text))
    surfaces.extend(_match_ffi(text, symbol_name))
    surfaces.extend(_match_bdd(text))
    surfaces.extend(_match_cli(text))

    # De-duplicate while preserving order; tag the first occurrence wins.
    seen = set()
    unique: List[APISurface] = []
    for s in surfaces:
        key = (s["kind"], s["surface"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return unique


# ──────────────────────────────────────────────────────────────────────
# Phase 1c orchestrator
# ──────────────────────────────────────────────────────────────────────


def extract_api_surfaces_for_graph(
    g: "Any",  # nx.MultiDiGraph — annotation lazy to avoid import cost
    *,
    parser_metadata_by_node: Optional[Dict[str, dict]] = None,
) -> Dict[str, List[APISurface]]:
    """Walk *g* and attach API-surface metadata to every node.

    Side effects: each node whose ``source_text`` exposes any surface
    gets its ``api_surface`` attribute set to the list of ``APISurface``
    dicts. Nodes without surfaces are left untouched (no key written).

    Returns a mapping ``{node_id: [APISurface, ...]}`` containing
    only the nodes for which at least one surface was detected. The
    return value is what :func:`run_cross_language_linker` expects as
    ``surfaces_by_node`` for its L1 pass.

    Pure with respect to edges and to nodes that produce no surfaces.
    """
    parser_metadata_by_node = parser_metadata_by_node or {}
    out: Dict[str, List[APISurface]] = {}
    for node_id, data in g.nodes(data=True):
        try:
            surfaces = extract_api_surfaces(
                data, parser_metadata=parser_metadata_by_node.get(str(node_id))
            )
        except Exception:  # pragma: no cover — defensive; matchers are regex-only
            continue
        if not surfaces:
            continue
        # Mutate the in-memory node attrs so the SQLite/Postgres
        # serialisers persist the surfaces via the ``api_surface``
        # column.
        data["api_surface"] = surfaces
        out[str(node_id)] = surfaces
    return out
