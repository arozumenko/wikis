"""Phase 6 / Action 6.2 — API surface extractor."""

from __future__ import annotations

import pytest

from app.core.code_graph.api_surface_extractor import extract_api_surfaces


def _node(text: str, *, language: str = "python", symbol_name: str = "handler") -> dict:
    return {"source_text": text, "language": language, "symbol_name": symbol_name}


class TestRESTPython:
    def test_fastapi_decorator(self):
        s = extract_api_surfaces(_node("@app.post('/api/users')\ndef create_user(): ..."))
        assert s == [{
            "kind": "rest",
            "surface": "POST /api/users",
            "weight_hint": 0.7,
            "metadata": {"method": "POST", "path": "/api/users"},
        }]

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
