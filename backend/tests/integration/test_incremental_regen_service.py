"""End-to-end test for #116 PR 3's :class:`IncrementalRegenService`.

Wires a real :class:`UnifiedWikiDB` (PR 1 tables) + a stubbed LLM +
in-memory page-body callbacks + a stubbed structural handler, then
drives one incremental regen and asserts each regime fired correctly.

The structural handler is a callback (not a real agent invocation) so
this test runs without LLM access or the full generation pipeline.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.core.agents.page_patcher import PagePatcher
from app.core.storage.incremental import compute_content_hash
from app.core.unified_db import UnifiedWikiDB
from app.services.incremental_regen import Regime
from app.services.incremental_regen_service import IncrementalRegenService


class _StubLLM:
    def __init__(self, output: str) -> None:
        self._output = output
        self.call_count = 0

    def invoke(self, messages, **_kwargs):  # type: ignore[no-untyped-def]
        self.call_count += 1
        return AIMessage(content=self._output)


# ---------------------------------------------------------------------------
# Fixture: seeded wiki with three pages exercising all three regimes
# ---------------------------------------------------------------------------


@pytest.fixture()
def fixture(tmp_path):
    """Seed a wiki:

    * page-trivial — cites sym-moved (will move file paths)
    * page-edit — cites sym-modified (will have content change)
    * page-struct — cites sym-deleted (will get deleted → primary-symbol-deleted override)
    """
    db = UnifiedWikiDB(tmp_path / "regen.wiki.db", embedding_dim=8)

    def _seed_node(nid: str, *, source: str, path: str = "x.py", name: str | None = None) -> None:
        db.upsert_node(
            nid,
            rel_path=path,
            file_name=path.rsplit("/", 1)[-1],
            language="python",
            symbol_name=name or nid,
            symbol_type="function",
            is_architectural=1,
            source_text=source,
        )

    _seed_node("sym-moved", source="def moved(): pass", path="old_path.py")
    _seed_node("sym-modified", source="def modify_me(): pass", path="x.py", name="modify_me")
    _seed_node("sym-deleted", source="def goner(): pass", path="x.py")
    _seed_node("sym-bystander", source="def b(): pass", path="x.py")

    db.upsert_wiki_page_with_symbols(
        {
            "page_id": "page-trivial",
            "wiki_id": "w",
            "title": "Movers",
            "anchor_slug": "movers",
            "primary_symbol_id": "sym-moved",
        },
        [("sym-moved", "primary")],
    )
    db.upsert_wiki_page_with_symbols(
        {
            "page_id": "page-edit",
            "wiki_id": "w",
            "title": "Editables",
            "anchor_slug": "editables",
            "primary_symbol_id": "sym-modified",
        },
        [("sym-modified", "primary")],
    )
    db.upsert_wiki_page_with_symbols(
        {
            "page_id": "page-struct",
            "wiki_id": "w",
            "title": "Structurals",
            "anchor_slug": "structurals",
            "primary_symbol_id": "sym-deleted",
        },
        [("sym-deleted", "primary")],
    )

    yield db
    db.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_service(
    db: UnifiedWikiDB,
    *,
    page_bodies: dict[str, str],
    llm_output: str,
    structural_pages: list[str] | None = None,
) -> tuple[IncrementalRegenService, list[str]]:
    """Construct the service with in-memory body storage + structural log."""
    structural_log: list[str] = []
    if structural_pages is None:
        structural_pages = structural_log

    def _read(pid: str) -> str | None:
        return page_bodies.get(pid)

    def _write(pid: str, body: str) -> None:
        page_bodies[pid] = body

    def _structural(page) -> bool:
        structural_pages.append(page.page_id)
        return True

    svc = IncrementalRegenService(
        storage=db,
        page_patcher=PagePatcher(_StubLLM(llm_output)),
        read_page_body=_read,
        write_page_body=_write,
        structural_handler=_structural,
    )
    return svc, structural_pages


def _parsed(node_id: str, source: str, path: str = "x.py") -> dict:
    return {
        "node_id": node_id,
        "source_text": source,
        "rel_path": path,
        "file_name": path.rsplit("/", 1)[-1],
        "language": "python",
        "symbol_name": node_id,
        "symbol_type": "function",
        "is_architectural": 1,
        "content_hash": compute_content_hash(source),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIncrementalRegenServiceE2E:
    def test_no_changes_yields_all_unchanged(self, fixture) -> None:
        # Re-parse with identical content → no changes → nothing happens.
        parsed = [
            _parsed("sym-moved", "def moved(): pass", "old_path.py"),
            _parsed("sym-modified", "def modify_me(): pass"),
            _parsed("sym-deleted", "def goner(): pass"),
            _parsed("sym-bystander", "def b(): pass"),
        ]
        page_bodies = {"page-trivial": "x", "page-edit": "x", "page-struct": "x"}
        svc, struct_log = _build_service(
            fixture, page_bodies=page_bodies, llm_output="DO NOT USE",
        )

        stats = svc.run(parsed)

        assert stats.trivial_patched == 0
        assert stats.edit_applied == 0
        assert stats.structural_regenerated == 0
        assert struct_log == []

    def test_trivial_regime_patches_paths(self, fixture) -> None:
        # sym-moved gets a new rel_path — should go through trivial.
        # The page body has a code_context block referring to the old path.
        page_bodies = {
            "page-trivial": (
                '# Movers\n\n'
                '<code_context path="old_path.py">def moved(): pass</code_context>'
            ),
            "page-edit": "no changes here",
            "page-struct": "no changes here",
        }
        parsed = [
            _parsed("sym-moved", "def moved(): pass", "new_path.py"),
            _parsed("sym-modified", "def modify_me(): pass"),
            _parsed("sym-deleted", "def goner(): pass"),
            _parsed("sym-bystander", "def b(): pass"),
        ]
        svc, _ = _build_service(
            fixture, page_bodies=page_bodies, llm_output="UNUSED",
        )

        stats = svc.run(parsed)

        assert stats.trivial_patched == 1
        assert 'path="new_path.py"' in page_bodies["page-trivial"]
        assert "old_path.py" not in page_bodies["page-trivial"]

    def test_edit_regime_calls_llm_and_persists_when_quality_gate_passes(
        self, fixture,
    ) -> None:
        # sym-modified gets a body change. Page-edit routes to edit regime;
        # stub LLM returns a low-diff output → quality gate accepts.
        original_body = (
            "# Editables\n\nThe modify_me function does work. "
            "It takes no arguments and returns nothing."
        )
        llm_output = (
            "# Editables\n\nThe modify_me function does work now. "
            "It takes no arguments and returns a result."
        )
        page_bodies = {
            "page-trivial": "x",
            "page-edit": original_body,
            "page-struct": "x",
        }
        parsed = [
            _parsed("sym-moved", "def moved(): pass", "old_path.py"),
            _parsed("sym-modified", "def modify_me(): return 1"),  # changed
            _parsed("sym-deleted", "def goner(): pass"),
            _parsed("sym-bystander", "def b(): pass"),
        ]
        svc, _ = _build_service(
            fixture, page_bodies=page_bodies, llm_output=llm_output,
        )

        stats = svc.run(parsed)

        assert stats.edit_applied == 1
        assert page_bodies["page-edit"] == llm_output

    def test_edit_demoted_to_structural_when_diff_too_high(self, fixture) -> None:
        original_body = "The modify_me function does specific work."
        runaway = "completely unrelated quux baz wibble wobble"
        page_bodies = {
            "page-trivial": "x",
            "page-edit": original_body,
            "page-struct": "x",
        }
        parsed = [
            _parsed("sym-moved", "def moved(): pass", "old_path.py"),
            _parsed("sym-modified", "def modify_me(): return 1"),
            _parsed("sym-deleted", "def goner(): pass"),
            _parsed("sym-bystander", "def b(): pass"),
        ]
        svc, struct_log = _build_service(
            fixture, page_bodies=page_bodies, llm_output=runaway,
        )

        stats = svc.run(parsed)

        # Quality gate rejected → demoted to structural.
        assert stats.edit_applied == 0
        assert stats.edit_demoted_to_structural == 1
        assert stats.structural_regenerated == 1
        assert "page-edit" in struct_log
        # Original body unchanged (didn't write rejected output).
        assert page_bodies["page-edit"] == original_body

    def test_structural_regime_for_deleted_primary_symbol(self, fixture) -> None:
        # sym-deleted is omitted from parsed → DELETED change → primary-symbol
        # override routes page-struct to structural.
        page_bodies = {
            "page-trivial": "x",
            "page-edit": "x",
            "page-struct": "x",
        }
        parsed = [
            _parsed("sym-moved", "def moved(): pass", "old_path.py"),
            _parsed("sym-modified", "def modify_me(): pass"),
            _parsed("sym-bystander", "def b(): pass"),
            # sym-deleted absent
        ]
        svc, struct_log = _build_service(
            fixture, page_bodies=page_bodies, llm_output="UNUSED",
        )

        stats = svc.run(parsed)

        assert stats.structural_regenerated == 1
        assert "page-struct" in struct_log

    def test_structural_handler_failure_is_counted_not_raised(self, fixture) -> None:
        # A raising structural handler must not bubble; it counts in
        # structural_failed and the rest of the run continues.
        def _raise(page):
            raise RuntimeError("simulated handler crash")

        def _read(pid: str) -> str | None:
            return "x"

        def _write(pid: str, body: str) -> None:
            pass

        svc = IncrementalRegenService(
            storage=fixture,
            page_patcher=PagePatcher(_StubLLM("UNUSED")),
            read_page_body=_read,
            write_page_body=_write,
            structural_handler=_raise,
        )
        parsed = [
            _parsed("sym-moved", "def moved(): pass", "old_path.py"),
            _parsed("sym-modified", "def modify_me(): pass"),
            _parsed("sym-bystander", "def b(): pass"),
            # sym-deleted absent → structural
        ]
        stats = svc.run(parsed)
        assert stats.structural_failed == 1
        assert stats.structural_regenerated == 0
