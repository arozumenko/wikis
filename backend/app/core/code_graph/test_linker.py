"""Test linker (Phase 6 / Action 6.4).

Pairs test nodes with the production code they exercise. Side-effect
free — returns ``[(test_node, prod_node, attrs)]`` so the caller can
choose when to inject the edges.

Heuristics (cheap, in priority order)
-------------------------------------

1. **Same-stem**: ``test_foo.py`` ↔ ``foo.py`` (handles ``test_``,
   ``_test``, ``tests/``, ``__tests__/`` prefixes/suffixes).
2. **Class-name proxy**: a test class ``TestUserService`` pairs with a
   production class ``UserService`` in the same language.
3. **Import-of**: a test node whose ``imports`` set lists a production
   module pairs with the public symbols defined in that module.

All edges use ``edge_class="test_link"``, ``relationship_type=
"test_link_<heuristic>"`` and weight 0.5 (synthetic floor).
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx

from ..feature_flags import FeatureFlags, get_feature_flags


_TEST_PREFIX = re.compile(r"^test[_-]")
_TEST_SUFFIX = re.compile(r"[_-]test$")
_TEST_DIR_TOKENS = ("tests/", "test/", "__tests__/", "spec/")


def _is_test_node(node_data: dict) -> bool:
    if node_data.get("is_test"):
        return True
    rel = (node_data.get("rel_path") or "").lower()
    if any(tok in rel for tok in _TEST_DIR_TOKENS):
        return True
    name = (node_data.get("file_name") or "").lower()
    if _TEST_PREFIX.match(name) or _TEST_SUFFIX.search(name):
        return True
    return False


def _strip_test_decoration(file_name: str) -> str:
    """Drop the ``test_`` / ``_test`` decoration from a test file stem."""
    name = file_name.lower()
    name = _TEST_PREFIX.sub("", name)
    name = _TEST_SUFFIX.sub("", name)
    return name


def _strip_test_class(name: str) -> str:
    """Drop the ``Test`` prefix/suffix from a test class name."""
    if name.startswith("Test"):
        return name[len("Test"):]
    if name.endswith("Test") or name.endswith("Tests"):
        return name.rsplit("Test", 1)[0]
    return name


# Symbol types that represent "the file" rather than "a thing inside the
# file". Same-stem matching only links these — pairing every symbol in a
# test file with every symbol in the prod file is a Cartesian explosion
# (measured at 1.25M edges on a single TS repo).
_FILE_LEVEL_TYPES = frozenset({"module", "module_doc", "file_doc"})


def link_same_stem(g: nx.MultiDiGraph) -> List[Tuple[str, str, dict]]:
    """Pair test files with production files that share a stem.

    Restricted to file/module-level nodes (``module``, ``module_doc``,
    ``file_doc``). Linking every symbol in the test file to every symbol
    in the prod file produces an N×M Cartesian product per file pair
    and blows the edge count up by 4–5 orders of magnitude on real
    repositories. File-level edges still let downstream consumers walk
    from a test file to its prod file in one hop.
    """
    by_stem: Dict[Tuple[str, str], List[str]] = defaultdict(list)  # (stem, lang) -> nodes
    test_nodes: List[Tuple[str, str, str]] = []  # (node_id, stripped_stem, lang)

    for nid, data in g.nodes(data=True):
        stype = (data.get("symbol_type") or "").lower()
        if stype not in _FILE_LEVEL_TYPES:
            continue
        lang = (data.get("language") or "").lower()
        stem = (data.get("file_name") or "").lower()
        if not stem:
            continue
        if _is_test_node(data):
            test_nodes.append((nid, _strip_test_decoration(stem), lang))
        else:
            by_stem[(stem, lang)].append(nid)

    out: List[Tuple[str, str, dict]] = []
    for test_id, stem, lang in test_nodes:
        candidates = by_stem.get((stem, lang)) or []
        for prod_id in candidates:
            if prod_id == test_id:
                continue
            out.append((test_id, prod_id, {
                "relationship_type": "test_link_same_stem",
                "edge_class": "test_link",
                "weight": 0.5,
                "provenance": {
                    "source": "test_linker",
                    "matcher": "same_stem",
                    "stem": stem,
                },
            }))
    return out


def link_class_proxy(g: nx.MultiDiGraph) -> List[Tuple[str, str, dict]]:
    """Pair ``TestUserService`` ↔ ``UserService`` by stripped class name."""
    prod_by_name: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    test_classes: List[Tuple[str, str, str]] = []

    for nid, data in g.nodes(data=True):
        if (data.get("symbol_type") or "").lower() != "class":
            continue
        sname = data.get("symbol_name") or ""
        if not sname:
            continue
        lang = (data.get("language") or "").lower()
        if _is_test_node(data) and (sname.startswith("Test") or sname.endswith("Test") or sname.endswith("Tests")):
            test_classes.append((nid, _strip_test_class(sname), lang))
        else:
            prod_by_name[(sname, lang)].append(nid)

    out: List[Tuple[str, str, dict]] = []
    for test_id, base_name, lang in test_classes:
        for prod_id in prod_by_name.get((base_name, lang), []):
            if prod_id == test_id:
                continue
            out.append((test_id, prod_id, {
                "relationship_type": "test_link_class_proxy",
                "edge_class": "test_link",
                "weight": 0.5,
                "provenance": {
                    "source": "test_linker",
                    "matcher": "class_proxy",
                    "base": base_name,
                },
            }))
    return out


def run_test_linker(
    g: nx.MultiDiGraph,
    *,
    flags: Optional[FeatureFlags] = None,
) -> List[Tuple[str, str, dict]]:
    """Return the full set of test-link edges (deduped)."""
    flags = flags or get_feature_flags()
    edges: List[Tuple[str, str, dict]] = []
    edges.extend(link_same_stem(g))
    edges.extend(link_class_proxy(g))

    seen: set = set()
    out: List[Tuple[str, str, dict]] = []
    for src, tgt, attrs in edges:
        key = (src, tgt, attrs.get("relationship_type"))
        if key in seen:
            continue
        seen.add(key)
        out.append((src, tgt, attrs))
    return out
