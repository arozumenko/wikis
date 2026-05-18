"""#181 Bug A — prefix-match-aware classification for KNOWN_FILENAMES.

The motivating bug: a freshly-refreshed wiki for a Node.js repo with a
``Dockerfile.prod`` at the repo root produced zero infrastructure_document
nodes in the graph.  ``Dockerfile.prod`` falls into the no-extension
branch and didn't match the exact-only ``KNOWN_FILENAMES`` lookup, so the
indexer dropped it.

These tests pin the new behavior:
- exact matches still classify the same way they always did,
- conventional ``<stem>.<variant>`` suffixes route to the same doc_type,
- and ``PREFIX_MATCH_FILENAMES`` is kept in sync with ``KNOWN_FILENAMES``
  so a typo in either set surfaces as a test failure rather than a silent
  ingestion regression.
"""

from __future__ import annotations

from app.core.constants import (
    KNOWN_FILENAMES,
    PREFIX_MATCH_FILENAMES,
    classify_known_filename,
)


class TestExactMatchUnchanged:
    """Pre-existing behavior must keep working."""

    def test_dockerfile_classifies_as_infrastructure(self):
        assert classify_known_filename("Dockerfile") == "infrastructure"

    def test_makefile_classifies_as_build_config(self):
        assert classify_known_filename("Makefile") == "build_config"

    def test_unknown_filename_returns_none(self):
        assert classify_known_filename("randomthing.weird") is None

    def test_empty_string_returns_none(self):
        assert classify_known_filename("") is None


class TestPrefixMatchVariants:
    """New behavior — conventional suffixes route to the same doc_type."""

    def test_dockerfile_prod_classifies_as_infrastructure(self):
        # The actual reproducer from onetest-ai/core.
        assert classify_known_filename("Dockerfile.prod") == "infrastructure"

    def test_dockerfile_dev_classifies_as_infrastructure(self):
        assert classify_known_filename("Dockerfile.dev") == "infrastructure"

    def test_dockerfile_staging_classifies_as_infrastructure(self):
        assert classify_known_filename("Dockerfile.staging") == "infrastructure"

    def test_containerfile_prod_classifies_as_infrastructure(self):
        assert classify_known_filename("Containerfile.prod") == "infrastructure"

    def test_makefile_local_classifies_as_build_config(self):
        assert classify_known_filename("Makefile.local") == "build_config"

    def test_makefile_am_classifies_as_build_config(self):
        # automake convention — real-world pattern in C/C++ projects.
        assert classify_known_filename("Makefile.am") == "build_config"

    def test_jenkinsfile_deploy_classifies_as_script(self):
        assert classify_known_filename("Jenkinsfile.deploy") == "script"

    def test_env_staging_classifies_as_config(self):
        # ``.env.staging`` isn't in KNOWN_FILENAMES explicitly but should
        # match the ``.env`` prefix stem.
        assert classify_known_filename(".env.staging") == "config"


class TestPrefixDoesNotOverreach:
    """Guard against false positives."""

    def test_dockerfile_without_dot_separator_does_not_match(self):
        # ``Dockerfileold`` is a typo, not a Dockerfile variant.
        assert classify_known_filename("Dockerfileold") is None

    def test_makefile_without_dot_separator_does_not_match(self):
        assert classify_known_filename("Makefilejunk") is None

    def test_file_with_known_extension_is_not_classified_here(self):
        # Extension-based dispatch happens BEFORE this helper in the
        # graph builder.  ``README.md`` must not match ``README`` (which
        # exists in KNOWN_FILENAMES via no-extension form) — and there
        # isn't a "README" stem in PREFIX_MATCH_FILENAMES either, so
        # this returns None and the .md extension lookup wins upstream.
        assert classify_known_filename("README.md") is None


class TestPrefixMatchFilenamesContract:
    """Drift-guard: every entry in PREFIX_MATCH_FILENAMES must exist in
    KNOWN_FILENAMES, otherwise the helper would raise KeyError at runtime.
    """

    def test_every_prefix_stem_has_a_known_filename_entry(self):
        missing = [stem for stem in PREFIX_MATCH_FILENAMES if stem not in KNOWN_FILENAMES]
        assert not missing, (
            f"PREFIX_MATCH_FILENAMES entries not in KNOWN_FILENAMES: {missing}. "
            "Either add them to KNOWN_FILENAMES or remove them from "
            "PREFIX_MATCH_FILENAMES."
        )
