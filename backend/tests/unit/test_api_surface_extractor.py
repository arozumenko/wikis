"""Phase 6 / Action 6.2 — API surface extractor."""

from __future__ import annotations

import pytest

from app.core.code_graph.api_surface_extractor import extract_api_surfaces


def _node(text: str, *, language: str = "python", symbol_name: str = "handler") -> dict:
    return {"source_text": text, "language": language, "symbol_name": symbol_name}


class TestRESTPython:
    def test_fastapi_decorator(self):
        s = extract_api_surfaces(_node("@app.post('/api/users')\ndef create_user(): ..."))
        # Canonical surface is always emitted; ``/api`` prefix triggers a
        # suffix-alternate so cross-language pairing works against routes
        # declared without the gateway prefix.
        assert {
            "kind": "rest",
            "surface": "POST /api/users",
            "weight_hint": 0.7,
            "metadata": {"method": "POST", "path": "/api/users"},
        } in s
        assert any(
            x["surface"] == "POST /users" and x["metadata"].get("prefix_stripped")
            for x in s
        )

    def test_flask_route_with_methods(self):
        s = extract_api_surfaces(_node(
            "@blueprint.route('/items', methods=['GET', 'POST'])\ndef items(): ..."
        ))
        kinds = sorted((x["surface"] for x in s))
        assert kinds == ["GET /items", "POST /items"]

    def test_flask_route_default_get(self):
        s = extract_api_surfaces(_node("@app.route('/health')\ndef health(): ..."))
        assert s[0]["surface"] == "GET /health"


class TestRESTTypeScript:
    def test_nest_decorator(self):
        s = extract_api_surfaces(_node(
            "@Get('/users/:id')\ngetUser(id) {}", language="typescript",
        ))
        assert s[0]["surface"] == "GET /users/:id"

    def test_express_call(self):
        s = extract_api_surfaces(_node(
            "app.post('/login', (req, res) => {});", language="javascript",
        ))
        assert s[0]["surface"] == "POST /login"

    def test_axios_client_call(self):
        s = extract_api_surfaces(_node(
            "axios.get('/api/users');", language="typescript",
        ))
        assert any(x["surface"] == "GET /api/users" for x in s)

    def test_generated_client_post(self):
        # OpenAPI-style generated client: ``client.post("/items", body)``.
        s = extract_api_surfaces(_node(
            "client.post('/items', body);", language="typescript",
        ))
        assert any(x["surface"] == "POST /items" for x in s)

    def test_fetch_default_get(self):
        s = extract_api_surfaces(_node(
            "fetch('/api/health');", language="typescript",
        ))
        assert any(x["surface"] == "GET /api/health" for x in s)

    def test_fetch_with_method_option(self):
        s = extract_api_surfaces(_node(
            'fetch("/api/users", {method: "POST", body: payload});',
            language="typescript",
        ))
        assert any(x["surface"] == "POST /api/users" for x in s)

    def test_openapi_ts_request_options(self):
        # @hey-api/openapi-ts generates ``__request(OpenAPI, {method:'X', url:'/y'})``
        text = """
        public static readItems(): CancelablePromise<X> {
            return __request(OpenAPI, {
                method: 'GET',
                url: '/api/v1/items/',
                query: {},
            });
        }
        """
        s = extract_api_surfaces(_node(text, language="typescript"))
        # Canonical full surface present.
        assert any(x["surface"] == "GET /api/v1/items" for x in s)
        # Suffix-alternate (with /api/v1 stripped) lets the linker pair this
        # against a Python ``APIRouter(prefix="/items")`` route.
        assert any(
            x["surface"] == "GET /items" and x["metadata"].get("prefix_stripped")
            for x in s
        )

    def test_openapi_ts_url_first(self):
        # openapi-typescript-codegen sometimes emits ``url`` before ``method``.
        text = "request({ url: '/api/v1/login/access-token', method: 'POST' })"
        s = extract_api_surfaces(_node(text, language="typescript"))
        assert any(x["surface"] == "POST /api/v1/login/access-token" for x in s)
        assert any(
            x["surface"] == "POST /login/access-token"
            and x["metadata"].get("prefix_stripped")
            for x in s
        )


class TestRouterPrefix:
    def test_python_router_prefix_in_same_text(self):
        # When the symbol slice exposes both the APIRouter declaration and
        # the route decorator, the matcher prepends the prefix to all paths.
        text = (
            'router = APIRouter(prefix="/items", tags=["items"])\n'
            "@router.get('/{id}')\n"
            "def read_item(id: int): ..."
        )
        s = extract_api_surfaces(_node(text, language="python"))
        assert any(x["surface"] == "GET /items/{id}" for x in s)


class TestSuffixAlternate:
    @pytest.mark.parametrize(
        "raw_path, expected_alt",
        [
            ("/api/v1/items", "/items"),
            ("/api/v2beta/users/{id}", "/users/{id}"),
            ("/rest/orders", "/orders"),
            ("/graphql/v1/me", "/me"),
            ("/api/health-check", "/health-check"),
        ],
    )
    def test_strips_common_api_prefix(self, raw_path, expected_alt):
        s = extract_api_surfaces(_node(
            f"@app.get('{raw_path}')\ndef h(): ...", language="python",
        ))
        assert any(
            x["metadata"].get("prefix_stripped") and x["surface"].endswith(expected_alt)
            for x in s
        )

    def test_no_alternate_for_root(self):
        # Path ``/`` has no extractable suffix \u2014 don't emit an alternate.
        s = extract_api_surfaces(_node("@app.get('/')\ndef h(): ...", language="python"))
        assert all(not x["metadata"].get("prefix_stripped") for x in s)


class TestExtractForGraphRouterPrefix:
    def test_repo_root_picks_up_router_prefix(self, tmp_path):
        from app.core.code_graph.api_surface_extractor import (
            extract_api_surfaces_for_graph,
        )
        import networkx as nx

        # Realistic FastAPI module: APIRouter(prefix="/items") at module
        # scope, decorator+def slice in the symbol's source_text.
        repo_root = tmp_path
        rel_path = "app/api/routes/items.py"
        full = repo_root / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(
            'from fastapi import APIRouter\n'
            'router = APIRouter(prefix="/items", tags=["items"])\n'
            '\n'
            '@router.get("/{id}")\n'
            'def read_item(id: int): ...\n',
            encoding="utf-8",
        )

        g = nx.MultiDiGraph()
        g.add_node(
            "python::items::read_item",
            language="python",
            rel_path=rel_path,
            symbol_name="read_item",
            source_text='@router.get("/{id}")\ndef read_item(id: int): ...',
        )
        out = extract_api_surfaces_for_graph(g, repo_root=str(repo_root))
        surfaces = out["python::items::read_item"]
        assert any(s["surface"] == "GET /items/{id}" for s in surfaces)
        # Original source_text restored (no synthetic prefix line leaked).
        node_data = g.nodes["python::items::read_item"]
        assert "APIRouter" not in node_data["source_text"]


class TestRESTJava:
    def test_jaxrs_path_plus_method(self):
        text = "@Path('/users')\n@GET\npublic Response list() {}"
        s = extract_api_surfaces(_node(text, language="java"))
        assert any(x["surface"] == "GET /users" for x in s)

    def test_spring_get_mapping(self):
        s = extract_api_surfaces(_node("@GetMapping('/api/v1/x')", language="java"))
        assert s[0]["surface"] == "GET /api/v1/x"


class TestGRPC:
    def test_proto_service_rpc(self):
        text = "service UserSvc {\n  rpc GetUser (Req) returns (Resp);\n}"
        s = extract_api_surfaces(_node(text, language="proto"))
        assert s[0]["kind"] == "grpc"
        assert s[0]["surface"] == "grpc:UserSvc/GetUser"


class TestGraphQL:
    def test_sdl_root(self):
        s = extract_api_surfaces(_node("type Query { user(id: ID!): User }"))
        assert any(x["kind"] == "graphql" and x["surface"] == "gql:query" for x in s)


class TestFFI:
    def test_extern_c_marks_ffi(self):
        s = extract_api_surfaces(_node('extern "C" int do_thing(void);', symbol_name="do_thing"))
        assert s[0] == {
            "kind": "ffi", "surface": "ffi:do_thing", "weight_hint": 0.6,
            "metadata": {"symbol": "do_thing"},
        }

    def test_pinvoke_extracts_library(self):
        text = '[DllImport("kernel32.dll")]\nstatic extern void Beep();'
        s = extract_api_surfaces(_node(text, language="csharp", symbol_name="Beep"))
        assert any(x["surface"] == "ffi:kernel32.dll" for x in s)


class TestBDD:
    def test_step_decorator(self):
        s = extract_api_surfaces(_node('@given("the user is logged in")\ndef step(): ...'))
        assert s[0]["kind"] == "bdd"
        assert s[0]["surface"] == "bdd:the user is logged in"

    def test_gherkin_text(self):
        s = extract_api_surfaces(_node("Given the system is ready"))
        assert any(x["surface"] == "bdd:the system is ready" for x in s)


class TestCLI:
    def test_click_command(self):
        s = extract_api_surfaces(_node("@click.command('hello')\ndef cmd(): ..."))
        assert s[0] == {
            "kind": "cli", "surface": "cli:hello", "weight_hint": 0.6,
            "metadata": {"framework": "click"},
        }


class TestDispatcher:
    def test_no_text_returns_empty(self):
        assert extract_api_surfaces({"language": "python"}) == []

    def test_dedup_preserves_first(self):
        # Two REST decorators with the same canonical key should collapse.
        text = "@app.get('/x')\n@app.get('/x')\ndef f(): ..."
        s = extract_api_surfaces(_node(text))
        assert len(s) == 1


class TestExtractApiSurfacesForGraph:
    """PR-12: Phase 1c orchestrator that walks the graph and persists
    api_surface attributes on each node."""

    def test_attaches_surfaces_and_returns_mapping(self):
        import networkx as nx

        from app.core.code_graph.api_surface_extractor import (
            extract_api_surfaces_for_graph,
        )

        g = nx.MultiDiGraph()
        g.add_node(
            "rest_node",
            source_text="@app.post('/api/users')\ndef create_user(): ...",
            language="python",
            symbol_name="create_user",
        )
        g.add_node(
            "plain_node",
            source_text="x = 1",
            language="python",
            symbol_name="x",
        )

        out = extract_api_surfaces_for_graph(g)

        assert "rest_node" in out
        assert "plain_node" not in out
        assert any(s["kind"] == "rest" for s in out["rest_node"])
        # Mutation: surfaces attached to node attrs for downstream
        # serialisation by sqlite/postgres _nx_node_to_dict.
        assert g.nodes["rest_node"]["api_surface"] == out["rest_node"]
        assert "api_surface" not in g.nodes["plain_node"]

    def test_empty_graph_returns_empty_mapping(self):
        import networkx as nx

        from app.core.code_graph.api_surface_extractor import (
            extract_api_surfaces_for_graph,
        )

        assert extract_api_surfaces_for_graph(nx.MultiDiGraph()) == {}

    def test_pylon_plugin_mount_is_scoped_per_wiki(self):
        import networkx as nx

        from app.core.code_graph.api_surface_extractor import (
            extract_api_surfaces_for_graph,
        )

        g = nx.MultiDiGraph()
        g.add_node(
            "cfg::metadata",
            wiki_id="cfg",
            language="json",
            rel_path="metadata.json",
            symbol_name="metadata",
            symbol_type="file",
            source_text='{"name": "configurations"}',
        )
        g.add_node(
            "inv::metadata",
            wiki_id="inv",
            language="json",
            rel_path="metadata.json",
            symbol_name="metadata",
            symbol_type="file",
            source_text='{"name": "inventory"}',
        )
        api_source = (
            "class API(APIBase):\n"
            "    url_params = ['<int:project_id>']\n"
            "    def get(self, project_id: int, **kwargs):\n"
            "        return {}\n"
        )
        g.add_node(
            "cfg::models-api",
            wiki_id="cfg",
            language="python",
            rel_path="api/v2/models.py",
            symbol_name="API",
            symbol_type="class",
            source_text=api_source,
        )
        g.add_node(
            "inv::models-api",
            wiki_id="inv",
            language="python",
            rel_path="api/v2/models.py",
            symbol_name="API",
            symbol_type="class",
            source_text=api_source,
        )

        surfaces = extract_api_surfaces_for_graph(g)

        cfg_surfaces = {s["surface"] for s in surfaces["cfg::models-api"]}
        inv_surfaces = {s["surface"] for s in surfaces["inv::models-api"]}
        assert "* /api/v2/configurations/models/{var}" in cfg_surfaces
        assert "* /api/v2/inventory/models/{var}" in inv_surfaces
        assert "* /api/v2/inventory/models/{var}" not in cfg_surfaces
        assert "* /api/v2/configurations/models/{var}" not in inv_surfaces


class TestRESTJavaSuffixAlternate:
    """Java REST routes must emit suffix-stripped alternates so a Spring
    controller declared on ``/api/v1/users`` pairs against a TS client
    calling ``/users`` (proxy-stripped) or vice-versa."""

    def test_spring_emits_suffix(self):
        s = extract_api_surfaces(_node("@GetMapping('/api/v1/users')", language="java"))
        surfaces = {x["surface"] for x in s}
        assert "GET /api/v1/users" in surfaces
        assert "GET /users" in surfaces

    def test_jaxrs_emits_suffix(self):
        text = "@Path('/api/v2/items')\n@POST\npublic Response create() {}"
        s = extract_api_surfaces(_node(text, language="java"))
        surfaces = {x["surface"] for x in s}
        assert "POST /api/v2/items" in surfaces
        assert "POST /items" in surfaces


class TestRESTGoSuffixAlternate:
    def test_chi_get_emits_suffix(self):
        s = extract_api_surfaces(_node('r.GET("/api/v1/users", handler)', language="go"))
        surfaces = {x["surface"] for x in s}
        assert "GET /api/v1/users" in surfaces
        assert "GET /users" in surfaces

    def test_gin_post_no_suffix_when_no_prefix(self):
        s = extract_api_surfaces(_node('engine.POST("/healthz", handler)', language="go"))
        surfaces = {x["surface"] for x in s}
        assert surfaces == {"POST /healthz"}


# ──────────────────────────────────────────────────────────────────────
# Object / data-shape matcher (cross-language DTO pairing)
# ──────────────────────────────────────────────────────────────────────


class TestObjectsPython:
    def test_dataclass_fields(self):
        text = (
            "@dataclass\n"
            "class User:\n"
            "    id: int\n"
            "    name: str\n"
            "    email: str = ''\n"
        )
        s = extract_api_surfaces(_node(text, language="python", symbol_name="User"))
        objs = [x for x in s if x["kind"] == "obj"]
        assert objs
        assert objs[0]["surface"] == "obj:user#email,id,name"
        assert objs[0]["metadata"]["fields"] == ["email", "id", "name"]

    def test_pydantic_basemodel(self):
        text = (
            "class Item(BaseModel):\n"
            "    title: str\n"
            "    price: float\n"
        )
        s = extract_api_surfaces(_node(text, language="python"))
        objs = [x for x in s if x["kind"] == "obj"]
        assert objs[0]["surface"] == "obj:item#price,title"

    def test_typed_dict(self):
        text = (
            "class Order(TypedDict):\n"
            "    order_id: str\n"
            "    total: float\n"
        )
        s = extract_api_surfaces(_node(text, language="python"))
        objs = [x for x in s if x["kind"] == "obj"]
        assert objs[0]["surface"] == "obj:order#order_id,total"

    def test_class_with_methods_skips_method_lines(self):
        text = (
            "class Service:\n"
            "    name: str\n"
            "    def run(self) -> None: ...\n"
        )
        s = extract_api_surfaces(_node(text, language="python"))
        objs = [x for x in s if x["kind"] == "obj"]
        assert objs[0]["surface"] == "obj:service#name"


class TestObjectsTypeScript:
    def test_interface(self):
        text = "export interface User { id: number; name: string; email?: string; }"
        s = extract_api_surfaces(_node(text, language="typescript"))
        objs = [x for x in s if x["kind"] == "obj"]
        assert objs[0]["surface"] == "obj:user#email,id,name"

    def test_type_alias(self):
        text = "export type Item = { title: string; price: number; };"
        s = extract_api_surfaces(_node(text, language="typescript"))
        objs = [x for x in s if x["kind"] == "obj"]
        assert objs[0]["surface"] == "obj:item#price,title"

    def test_javascript_interface_via_jsdoc_form(self):
        # JSDoc style — also handled by the same TS interface regex.
        # Field names are snake-cased by _obj_surface so casing
        # differences across languages don't break cross-language
        # pairing (orderId ↔ order_id).
        text = "interface Order { orderId: string; total: number; }"
        s = extract_api_surfaces(_node(text, language="javascript"))
        objs = [x for x in s if x["kind"] == "obj"]
        assert objs[0]["surface"] == "obj:order#order_id,total"


class TestObjectsGo:
    def test_struct_uses_json_tag(self):
        text = (
            "type User struct {\n"
            "    ID    int    `json:\"id\"`\n"
            "    Name  string `json:\"name\"`\n"
            "    Email string `json:\"email,omitempty\"`\n"
            "}\n"
        )
        s = extract_api_surfaces(_node(text, language="go"))
        objs = [x for x in s if x["kind"] == "obj"]
        assert objs[0]["surface"] == "obj:user#email,id,name"

    def test_struct_falls_back_to_field_name(self):
        text = (
            "type Item struct {\n"
            "    Title string\n"
            "    Price float64\n"
            "}\n"
        )
        s = extract_api_surfaces(_node(text, language="go"))
        objs = [x for x in s if x["kind"] == "obj"]
        assert objs[0]["surface"] == "obj:item#price,title"


class TestObjectsJava:
    def test_class_fields(self):
        text = (
            "public class User {\n"
            "    private Long id;\n"
            "    private String name;\n"
            "    private String email;\n"
            "}\n"
        )
        s = extract_api_surfaces(_node(text, language="java"))
        objs = [x for x in s if x["kind"] == "obj"]
        assert objs[0]["surface"] == "obj:user#email,id,name"

    def test_record_params(self):
        text = "public record Item(Long id, String title, Double price) {}"
        s = extract_api_surfaces(_node(text, language="java"))
        objs = [x for x in s if x["kind"] == "obj"]
        assert objs[0]["surface"] == "obj:item#id,price,title"


class TestObjectsRust:
    def test_struct_serde_rename(self):
        text = (
            "pub struct User {\n"
            "    #[serde(rename = \"id\")]\n"
            "    user_id: i64,\n"
            "    name: String,\n"
            "}\n"
        )
        s = extract_api_surfaces(_node(text, language="rust"))
        objs = [x for x in s if x["kind"] == "obj"]
        assert objs[0]["surface"] == "obj:user#id,name"

    def test_struct_plain(self):
        text = "struct Item { title: String, price: f64 }"
        s = extract_api_surfaces(_node(text, language="rust"))
        objs = [x for x in s if x["kind"] == "obj"]
        assert objs[0]["surface"] == "obj:item#price,title"


class TestObjectsCpp:
    def test_struct_fields(self):
        text = (
            "struct User {\n"
            "    int id;\n"
            "    std::string name;\n"
            "    std::string email = {};\n"
            "    void normalize();\n"
            "};\n"
        )
        s = extract_api_surfaces(_node(text, language="cpp"))
        objs = [x for x in s if x["kind"] == "obj"]
        assert objs[0]["surface"] == "obj:user#email,id,name"

    def test_class_fields_with_access_labels(self):
        text = (
            "class Item {\n"
            "public:\n"
            "    int id;\n"
            "    double price;\n"
            "private:\n"
            "    std::string title;\n"
            "};\n"
        )
        s = extract_api_surfaces(_node(text, language="cpp"))
        objs = [x for x in s if x["kind"] == "obj"]
        assert objs[0]["surface"] == "obj:item#id,price,title"


class TestObjectsCSharp:
    def test_class_properties(self):
        text = (
            "public class User {\n"
            "    public long Id { get; set; }\n"
            "    public string Name { get; set; }\n"
            "}\n"
        )
        s = extract_api_surfaces(_node(text, language="csharp"))
        objs = [x for x in s if x["kind"] == "obj"]
        assert objs[0]["surface"] == "obj:user#id,name"

    def test_record(self):
        text = "public record Item(long Id, string Title, double Price);"
        s = extract_api_surfaces(_node(text, language="csharp"))
        objs = [x for x in s if x["kind"] == "obj"]
        assert objs[0]["surface"] == "obj:item#id,price,title"


class TestObjectsCrossLanguagePairing:
    """Validates the *intent* of obj: surfaces — same TypeName + same
    field-set (case-folded) yields the same surface key, so the L1
    linker can pair the records across any language combination."""

    def test_python_dataclass_matches_ts_interface(self):
        py = extract_api_surfaces(_node(
            "@dataclass\nclass User:\n    id: int\n    name: str\n",
            language="python",
        ))
        ts = extract_api_surfaces(_node(
            "interface User { id: number; name: string; }",
            language="typescript",
        ))
        py_obj = next(x["surface"] for x in py if x["kind"] == "obj")
        ts_obj = next(x["surface"] for x in ts if x["kind"] == "obj")
        assert py_obj == ts_obj == "obj:user#id,name"

    def test_go_struct_matches_python_dataclass_via_json_tag(self):
        go = extract_api_surfaces(_node(
            "type User struct {\n    ID int `json:\"id\"`\n    Name string `json:\"name\"`\n}",
            language="go",
        ))
        py = extract_api_surfaces(_node(
            "@dataclass\nclass User:\n    id: int\n    name: str\n",
            language="python",
        ))
        go_obj = next(x["surface"] for x in go if x["kind"] == "obj")
        py_obj = next(x["surface"] for x in py if x["kind"] == "obj")
        assert go_obj == py_obj

    def test_rust_struct_matches_java_record_via_serde(self):
        rs = extract_api_surfaces(_node(
            "pub struct Item {\n    #[serde(rename = \"id\")]\n    item_id: i64,\n    title: String,\n}",
            language="rust",
        ))
        java = extract_api_surfaces(_node(
            "public record Item(Long id, String title) {}",
            language="java",
        ))
        rs_obj = next(x["surface"] for x in rs if x["kind"] == "obj")
        java_obj = next(x["surface"] for x in java if x["kind"] == "obj")
        assert rs_obj == java_obj == "obj:item#id,title"


class TestFFIComprehensive:
    def test_rust_extern_c(self):
        text = 'extern "C" {\n    fn compute_hash(data: *const u8) -> u64;\n}'
        s = extract_api_surfaces(_node(text, language="rust", symbol_name="compute_hash"))
        ffi = [x for x in s if x["kind"] == "ffi"]
        assert any(x["surface"] == "ffi:compute_hash" for x in ffi)

    def test_python_ctypes(self):
        text = "lib = ctypes.CDLL('libcompute.so')\nlib.compute_hash.restype = ctypes.c_uint64"
        s = extract_api_surfaces(_node(text, language="python", symbol_name="compute_hash"))
        ffi = [x for x in s if x["kind"] == "ffi"]
        assert any(x["surface"] == "ffi:compute_hash" for x in ffi)

    def test_csharp_pinvoke(self):
        text = '[DllImport("libcompute")]\npublic static extern ulong ComputeHash(byte[] data);'
        s = extract_api_surfaces(_node(text, language="csharp", symbol_name="ComputeHash"))
        ffi = [x for x in s if x["kind"] == "ffi"]
        # P/Invoke: surface keyed on library name.
        assert any(x["surface"] == "ffi:libcompute" for x in ffi)

    def test_jni_native(self):
        text = "public native long computeHash(byte[] data);"
        s = extract_api_surfaces(_node(text, language="java", symbol_name="computeHash"))
        ffi = [x for x in s if x["kind"] == "ffi"]
        assert any(x["surface"] == "ffi:computeHash" for x in ffi)

    def test_wasm_bindgen(self):
        text = "#[wasm_bindgen]\npub fn compute_hash(data: &[u8]) -> u64 { 0 }"
        s = extract_api_surfaces(_node(text, language="rust", symbol_name="compute_hash"))
        ffi = [x for x in s if x["kind"] == "ffi"]
        assert any(x["surface"] == "ffi:compute_hash" for x in ffi)


class TestCLIComprehensive:
    def test_argparse_subparser(self):
        text = "sub = parser.add_subparsers().add_parser('build')"
        s = extract_api_surfaces(_node(text, language="python"))
        assert any(x["surface"] == "cli:build" for x in s)

    def test_cobra_command(self):
        text = 'var rootCmd = &cobra.Command{\n    Use: "deploy",\n    Short: "Deploy the app",\n}'
        s = extract_api_surfaces(_node(text, language="go"))
        assert any(x["surface"] == "cli:deploy" for x in s)


class TestBDDComprehensive:
    def test_python_step_def(self):
        text = "@given('the user is logged in')\ndef step_impl(context): ..."
        s = extract_api_surfaces(_node(text, language="python"))
        bdd = [x for x in s if x["kind"] == "bdd"]
        assert bdd and bdd[0]["surface"] == "bdd:the user is logged in"

    def test_gherkin_steps(self):
        text = (
            "Feature: Login\n"
            "  Scenario: Successful login\n"
            "    Given the user is logged in\n"
            "    When the user navigates to /home\n"
            "    Then the dashboard is displayed\n"
        )
        s = extract_api_surfaces(_node(text, language="gherkin"))
        surfaces = {x["surface"] for x in s if x["kind"] == "bdd"}
        assert "bdd:the user is logged in" in surfaces
        assert "bdd:the user navigates to /home" in surfaces
        assert "bdd:the dashboard is displayed" in surfaces


class TestNameImports:
    def test_rust_use_named_imports(self):
        text = "use crate::sdk::{TransformSdk, StreamConfig as Config};\n"
        s = extract_api_surfaces(_node(text, language="rust"))
        surfaces = {x["surface"] for x in s if x["kind"] == "name"}
        assert "name:transform_sdk" in surfaces
        assert "name:stream_config" in surfaces

    def test_cpp_local_include_import(self):
        text = '#include "sdk/TransformSdk.hpp"\n#include <vector>\n'
        s = extract_api_surfaces(_node(text, language="cpp"))
        surfaces = {x["surface"] for x in s if x["kind"] == "name"}
        assert "name:transform_sdk" in surfaces
        assert "name:vector" not in surfaces
