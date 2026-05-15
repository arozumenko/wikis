"""Unit tests for the trivial-regime citation patcher.

The patcher's contract:

* Rewrites only ``<code_context path=...>`` attributes for MOVED changes.
* Idempotent — running twice on the same input yields the same output.
* Preserves quoting style (double vs. single).
* Ignores non-MOVED change kinds even if they leak through.
* Recomputes ``content_hash`` even when no paths were rewritten so
  callers always get a consistent hash value back.
"""

from __future__ import annotations

from app.core.storage.incremental import compute_content_hash
from app.services.change_detector import ChangeKind, NodeChange
from app.services.trivial_patcher import patch_trivial_page


def _moved(node_id: str, old_path: str, new_path: str) -> NodeChange:
    return NodeChange(
        kind=ChangeKind.MOVED,
        node_id=node_id,
        old_path=old_path,
        new_path=new_path,
        old_hash="h",
        new_hash="h",
    )


def test_rewrites_double_quoted_path() -> None:
    content = '<code_context path="old/path.py">def x(): ...</code_context>'
    changes = [_moved("n1", "old/path.py", "new/path.py")]

    result = patch_trivial_page("p1", content, changes)

    assert 'path="new/path.py"' in result.new_content
    assert "old/path.py" not in result.new_content
    assert result.paths_rewritten == 1


def test_rewrites_single_quoted_path() -> None:
    content = "<code_context path='old/path.py'>def x(): ...</code_context>"
    changes = [_moved("n1", "old/path.py", "new/path.py")]

    result = patch_trivial_page("p1", content, changes)

    assert "path='new/path.py'" in result.new_content
    assert result.paths_rewritten == 1


def test_preserves_other_attributes() -> None:
    content = '<code_context lang="python" path="old.py" start_line="42">body</code_context>'
    changes = [_moved("n1", "old.py", "new.py")]

    result = patch_trivial_page("p1", content, changes)

    # Other attributes remain in place; only path swapped.
    assert 'lang="python"' in result.new_content
    assert 'path="new.py"' in result.new_content


def test_idempotent_on_second_run() -> None:
    content = '<code_context path="old.py">x</code_context>'
    changes = [_moved("n1", "old.py", "new.py")]

    once = patch_trivial_page("p1", content, changes)
    # Re-running with the SAME (old → new) on already-patched content:
    # the old path is gone, so the rewrite finds nothing to do.
    twice = patch_trivial_page("p1", once.new_content, changes)

    assert twice.paths_rewritten == 0
    assert twice.new_content == once.new_content


def test_multiple_paths_rewritten_in_one_pass() -> None:
    content = (
        '<code_context path="a.py">x</code_context>\n'
        '<code_context path="b.py">y</code_context>\n'
        '<code_context path="c.py">z</code_context>'
    )
    changes = [
        _moved("na", "a.py", "A.py"),
        _moved("nb", "b.py", "B.py"),
        # c.py not moved → stays as-is.
    ]

    result = patch_trivial_page("p1", content, changes)

    assert result.paths_rewritten == 2
    assert 'path="A.py"' in result.new_content
    assert 'path="B.py"' in result.new_content
    assert 'path="c.py"' in result.new_content


def test_ignores_non_moved_changes() -> None:
    content = '<code_context path="old.py">x</code_context>'
    changes = [
        NodeChange(kind=ChangeKind.MODIFIED, node_id="n1", old_hash="a", new_hash="b"),
        NodeChange(kind=ChangeKind.DELETED, node_id="n2", old_hash="c"),
    ]

    result = patch_trivial_page("p1", content, changes)

    assert result.paths_rewritten == 0
    assert result.new_content == content


def test_returns_content_hash_even_when_no_rewrites() -> None:
    content = "no code_context here, just prose"
    changes = [_moved("n1", "irrelevant.py", "elsewhere.py")]

    result = patch_trivial_page("p1", content, changes)

    assert result.paths_rewritten == 0
    assert result.new_content == content
    assert result.new_content_hash == compute_content_hash(content)


def test_move_to_same_path_is_a_noop() -> None:
    # Sentinel case: NodeChange has old_path == new_path. Shouldn't
    # happen in practice (the detector filters those), but defensively
    # skip rather than rewriting.
    content = '<code_context path="a.py">x</code_context>'
    changes = [_moved("n1", "a.py", "a.py")]

    result = patch_trivial_page("p1", content, changes)

    assert result.paths_rewritten == 0
    assert result.new_content == content


def test_hash_changes_when_content_changes() -> None:
    content = '<code_context path="old.py">x</code_context>'
    changes = [_moved("n1", "old.py", "new.py")]

    result = patch_trivial_page("p1", content, changes)

    # New hash != hash of original content.
    original_hash = compute_content_hash(content)
    assert result.new_content_hash != original_hash
