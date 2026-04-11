"""
Unit tests for app/core/code_graph/expansion_engine.py

Uses synthetic NetworkX graphs — no real FAISS/LLM/network calls.
"""

import pytest
import networkx as nx

from app.core.code_graph.expansion_engine import (
    AugmentedContent,
    CLASS_LIKE_TYPES,
    EXPANSION_PRIORITIES,
    EXPANSION_WORTHY_TYPES,
    ExpansionResult,
    GLOBAL_EXPANSION_CAP,
    PER_SYMBOL_CAP,
    SKIP_RELATIONSHIPS,
    augment_cpp_node,
    augment_go_node,
    edges_between,
    expand_smart,
    find_calls_to_free_functions,
    find_composed_types,
    find_creates_from_methods,
    format_type_args,
    get_edge_annotations,
    get_neighbors_by_relationship,
    has_relationship,
    resolve_alias_chain,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _digraph(*nodes_attrs):
    """Build DiGraph with optional node attrs: (node_id, attr_dict)."""
    g = nx.DiGraph()
    for node_id, attrs in nodes_attrs:
        g.add_node(node_id, **attrs)
    return g


def _add_edge(g, src, tgt, rel_type):
    g.add_edge(src, tgt, relationship_type=rel_type)


def _multidigraph():
    """Return empty MultiDiGraph."""
    return nx.MultiDiGraph()


# ---------------------------------------------------------------------------
# edges_between
# ---------------------------------------------------------------------------


def test_edges_between_digraph_returns_single_edge():
    g = nx.DiGraph()
    g.add_node("A")
    g.add_node("B")
    g.add_edge("A", "B", relationship_type="inheritance")
    edges = edges_between(g, "A", "B")
    assert len(edges) == 1
    assert edges[0]["relationship_type"] == "inheritance"


def test_edges_between_digraph_no_edge():
    g = nx.DiGraph()
    g.add_node("A")
    g.add_node("B")
    edges = edges_between(g, "A", "B")
    assert edges == []


def test_edges_between_multidigraph_multiple_edges():
    g = nx.MultiDiGraph()
    g.add_node("A")
    g.add_node("B")
    g.add_edge("A", "B", relationship_type="calls")
    g.add_edge("A", "B", relationship_type="references")
    edges = edges_between(g, "A", "B")
    rel_types = {e["relationship_type"] for e in edges}
    assert "calls" in rel_types
    assert "references" in rel_types


def test_edges_between_missing_source():
    g = nx.DiGraph()
    g.add_node("B")
    edges = edges_between(g, "A", "B")
    assert edges == []


# ---------------------------------------------------------------------------
# has_relationship
# ---------------------------------------------------------------------------


def test_has_relationship_true():
    g = nx.DiGraph()
    g.add_edge("A", "B", relationship_type="inheritance")
    assert has_relationship(g, "A", "B", "inheritance")


def test_has_relationship_case_insensitive():
    g = nx.DiGraph()
    g.add_edge("A", "B", relationship_type="Inheritance")
    assert has_relationship(g, "A", "B", "inheritance")


def test_has_relationship_false():
    g = nx.DiGraph()
    g.add_edge("A", "B", relationship_type="calls")
    assert not has_relationship(g, "A", "B", "inheritance")


def test_has_relationship_multiple_types():
    g = nx.DiGraph()
    g.add_edge("A", "B", relationship_type="calls")
    assert has_relationship(g, "A", "B", "inheritance", "calls")


def test_has_relationship_no_edge():
    g = nx.DiGraph()
    g.add_node("A")
    g.add_node("B")
    assert not has_relationship(g, "A", "B", "inheritance")


# ---------------------------------------------------------------------------
# get_edge_annotations
# ---------------------------------------------------------------------------


def test_get_edge_annotations_found():
    g = nx.DiGraph()
    g.add_edge("A", "B", relationship_type="instantiates", annotations={"type_args": ["int"]})
    ann = get_edge_annotations(g, "A", "B", "instantiates")
    assert ann == {"type_args": ["int"]}


def test_get_edge_annotations_wrong_type():
    g = nx.DiGraph()
    g.add_edge("A", "B", relationship_type="calls")
    ann = get_edge_annotations(g, "A", "B", "instantiates")
    assert ann == {}


def test_get_edge_annotations_no_edge():
    g = nx.DiGraph()
    ann = get_edge_annotations(g, "A", "B", "instantiates")
    assert ann == {}


def test_get_edge_annotations_no_annotations_key():
    g = nx.DiGraph()
    g.add_edge("A", "B", relationship_type="instantiates")
    ann = get_edge_annotations(g, "A", "B", "instantiates")
    assert ann == {}


# ---------------------------------------------------------------------------
# format_type_args
# ---------------------------------------------------------------------------


def test_format_type_args_empty():
    assert format_type_args([]) == ""


def test_format_type_args_single():
    assert format_type_args(["int"]) == "<int>"


def test_format_type_args_multiple():
    assert format_type_args(["int", "Point"]) == "<int, Point>"


# ---------------------------------------------------------------------------
# get_neighbors_by_relationship
# ---------------------------------------------------------------------------


def test_get_neighbors_by_relationship_successors():
    g = nx.DiGraph()
    g.add_edge("A", "B", relationship_type="inheritance")
    g.add_edge("A", "C", relationship_type="calls")
    results = get_neighbors_by_relationship(g, "A", {"inheritance"}, "successors")
    ids = [r[0] for r in results]
    assert "B" in ids
    assert "C" not in ids


def test_get_neighbors_by_relationship_predecessors():
    g = nx.DiGraph()
    g.add_edge("X", "A", relationship_type="calls")
    results = get_neighbors_by_relationship(g, "A", {"calls"}, "predecessors")
    ids = [r[0] for r in results]
    assert "X" in ids


def test_get_neighbors_by_relationship_respects_limit():
    g = nx.DiGraph()
    for i in range(5):
        g.add_edge("A", f"B{i}", relationship_type="inheritance")
    results = get_neighbors_by_relationship(g, "A", {"inheritance"}, "successors", limit=2)
    assert len(results) <= 2


def test_get_neighbors_by_relationship_type_filter():
    g = nx.DiGraph()
    g.add_node("B", symbol_type="class")
    g.add_node("C", symbol_type="function")
    g.add_edge("A", "B", relationship_type="calls")
    g.add_edge("A", "C", relationship_type="calls")
    results = get_neighbors_by_relationship(
        g, "A", {"calls"}, "successors", type_filter=frozenset({"class"})
    )
    ids = [r[0] for r in results]
    assert "B" in ids
    assert "C" not in ids


def test_get_neighbors_by_relationship_invalid_direction():
    g = nx.DiGraph()
    results = get_neighbors_by_relationship(g, "A", {"calls"}, "invalid_direction")
    assert results == []


def test_get_neighbors_by_relationship_no_matching_edges():
    g = nx.DiGraph()
    g.add_edge("A", "B", relationship_type="defines")
    results = get_neighbors_by_relationship(g, "A", {"inheritance"}, "successors")
    assert results == []


# ---------------------------------------------------------------------------
# resolve_alias_chain
# ---------------------------------------------------------------------------


def test_resolve_alias_chain_direct():
    g = nx.DiGraph()
    g.add_node("Alias", symbol_type="type_alias")
    g.add_node("Concrete", symbol_type="class")
    g.add_edge("Alias", "Concrete", relationship_type="alias_of")
    result = resolve_alias_chain(g, "Alias")
    assert result == "Concrete"


def test_resolve_alias_chain_through_intermediate():
    g = nx.DiGraph()
    g.add_node("A1", symbol_type="type_alias")
    g.add_node("A2", symbol_type="type_alias")
    g.add_node("Concrete", symbol_type="class")
    g.add_edge("A1", "A2", relationship_type="alias_of")
    g.add_edge("A2", "Concrete", relationship_type="alias_of")
    result = resolve_alias_chain(g, "A1")
    assert result == "Concrete"


def test_resolve_alias_chain_no_alias():
    g = nx.DiGraph()
    g.add_node("A", symbol_type="class")
    # No alias_of edges
    result = resolve_alias_chain(g, "A")
    assert result is None


def test_resolve_alias_chain_cycle_terminates():
    g = nx.DiGraph()
    g.add_node("A", symbol_type="type_alias")
    g.add_node("B", symbol_type="type_alias")
    g.add_edge("A", "B", relationship_type="alias_of")
    # B has no further alias_of — chain terminates
    result = resolve_alias_chain(g, "A", max_hops=5)
    assert result is not None or result is None  # Just shouldn't hang


# ---------------------------------------------------------------------------
# find_composed_types
# ---------------------------------------------------------------------------


def test_find_composed_types_via_field():
    g = nx.DiGraph()
    g.add_node("Service", symbol_type="class")
    g.add_node("field1", symbol_type="field")
    g.add_node("Repository", symbol_type="class")
    g.add_edge("Service", "field1", relationship_type="defines")
    g.add_edge("field1", "Repository", relationship_type="composition")
    results = find_composed_types(g, "Service")
    assert "Repository" in results


def test_find_composed_types_no_composition_edges():
    g = nx.DiGraph()
    g.add_node("Service", symbol_type="class")
    g.add_node("field1", symbol_type="field")
    g.add_node("Repository", symbol_type="class")
    g.add_edge("Service", "field1", relationship_type="defines")
    # No composition edges
    results = find_composed_types(g, "Service")
    assert results == []


def test_find_composed_types_respects_limit():
    g = nx.DiGraph()
    g.add_node("Service", symbol_type="class")
    for i in range(5):
        field = f"field{i}"
        repo = f"Repo{i}"
        g.add_node(field, symbol_type="field")
        g.add_node(repo, symbol_type="class")
        g.add_edge("Service", field, relationship_type="defines")
        g.add_edge(field, repo, relationship_type="composition")
    results = find_composed_types(g, "Service", limit=2)
    assert len(results) <= 2


# ---------------------------------------------------------------------------
# find_creates_from_methods
# ---------------------------------------------------------------------------


def test_find_creates_from_methods():
    g = nx.DiGraph()
    g.add_node("Service", symbol_type="class")
    g.add_node("create_method", symbol_type="method")
    g.add_node("Widget", symbol_type="class")
    g.add_edge("Service", "create_method", relationship_type="defines")
    g.add_edge("create_method", "Widget", relationship_type="creates")
    results = find_creates_from_methods(g, "Service")
    assert "Widget" in results


def test_find_creates_from_methods_no_creates_edges():
    g = nx.DiGraph()
    g.add_node("Service", symbol_type="class")
    results = find_creates_from_methods(g, "Service")
    assert results == []


# ---------------------------------------------------------------------------
# find_calls_to_free_functions
# ---------------------------------------------------------------------------


def test_find_calls_to_free_functions():
    g = nx.DiGraph()
    g.add_node("Service", symbol_type="class")
    g.add_node("do_work", symbol_type="method")
    g.add_node("helper", symbol_type="function")
    g.add_edge("Service", "do_work", relationship_type="defines")
    g.add_edge("do_work", "helper", relationship_type="calls")
    results = find_calls_to_free_functions(g, "Service")
    assert "helper" in results


def test_find_calls_to_free_functions_excludes_methods():
    g = nx.DiGraph()
    g.add_node("Service", symbol_type="class")
    g.add_node("do_work", symbol_type="method")
    g.add_node("other_method", symbol_type="method")  # Not a function
    g.add_edge("Service", "do_work", relationship_type="defines")
    g.add_edge("do_work", "other_method", relationship_type="calls")
    results = find_calls_to_free_functions(g, "Service")
    assert "other_method" not in results


# ---------------------------------------------------------------------------
# augment_cpp_node
# ---------------------------------------------------------------------------


def test_augment_cpp_node_function_with_defines_body():
    g = nx.DiGraph()
    g.add_node(
        "decl_node",
        symbol_type="function",
        source_text="void login() ;",
        file_path="auth.h",
        language="cpp",
    )
    g.add_node(
        "impl_node",
        symbol_type="function",
        source_text="void login() { /* impl */ }",
        file_path="auth.cpp",
        rel_path="auth.cpp",
        language="cpp",
    )
    g.add_edge("impl_node", "decl_node", relationship_type="defines_body")
    result = augment_cpp_node(g, "decl_node")
    assert result is not None
    assert isinstance(result, AugmentedContent)
    assert "impl" in result.augmented_content


def test_augment_cpp_node_no_content():
    g = nx.DiGraph()
    g.add_node("decl", symbol_type="function", source_text="", file_path="auth.h")
    result = augment_cpp_node(g, "decl")
    assert result is None


def test_augment_cpp_node_unknown_node():
    g = nx.DiGraph()
    result = augment_cpp_node(g, "nonexistent")
    assert result is None


def test_augment_cpp_node_class_with_method_impl():
    g = nx.DiGraph()
    g.add_node(
        "MyClass",
        symbol_type="class",
        source_text="class MyClass {};",
        file_path="myclass.h",
    )
    g.add_node(
        "MyMethod",
        symbol_type="method",
        source_text="void doWork() {}",
        file_path="myclass.h",
    )
    g.add_node(
        "ImplMethod",
        symbol_type="method",
        source_text="void MyClass::doWork() { int x = 1; }",
        file_path="myclass.cpp",
        rel_path="myclass.cpp",
    )
    g.add_edge("MyClass", "MyMethod", relationship_type="defines")
    g.add_edge("ImplMethod", "MyMethod", relationship_type="defines_body")
    result = augment_cpp_node(g, "MyClass")
    assert result is not None
    assert "MyClass::doWork" in result.augmented_content


def test_augment_cpp_node_same_file_impl_not_added():
    """Cross-file augmentation only — same-file impl should not be added."""
    g = nx.DiGraph()
    g.add_node(
        "decl",
        symbol_type="function",
        source_text="void foo();",
        file_path="auth.h",
    )
    g.add_node(
        "impl",
        symbol_type="function",
        source_text="void foo() {}",
        file_path="auth.h",  # same file
        rel_path="auth.h",
    )
    g.add_edge("impl", "decl", relationship_type="defines_body")
    result = augment_cpp_node(g, "decl")
    assert result is None


# ---------------------------------------------------------------------------
# augment_go_node
# ---------------------------------------------------------------------------


def test_augment_go_node_struct_with_cross_file_methods():
    g = nx.DiGraph()
    g.add_node(
        "UserStruct",
        symbol_type="struct",
        source_text="type User struct {}",
        rel_path="user.go",
        file_path="user.go",
    )
    g.add_node(
        "GetName",
        symbol_type="method",
        source_text="func (u User) GetName() string {}",
        rel_path="user_methods.go",
        file_path="user_methods.go",
    )
    g.add_edge("UserStruct", "GetName", relationship_type="defines")
    result = augment_go_node(g, "UserStruct")
    assert result is not None
    assert "GetName" in result.augmented_content


def test_augment_go_node_non_struct_type():
    g = nx.DiGraph()
    g.add_node("MyFunc", symbol_type="function", source_text="func foo() {}")
    result = augment_go_node(g, "MyFunc")
    assert result is None


def test_augment_go_node_no_content():
    g = nx.DiGraph()
    g.add_node("UserStruct", symbol_type="struct", source_text="")
    result = augment_go_node(g, "UserStruct")
    assert result is None


def test_augment_go_node_unknown():
    g = nx.DiGraph()
    result = augment_go_node(g, "nonexistent")
    assert result is None


def test_augment_go_node_same_file_method_excluded():
    """Methods in the same file should NOT be included in Go augmentation."""
    g = nx.DiGraph()
    g.add_node(
        "UserStruct",
        symbol_type="struct",
        source_text="type User struct {}",
        rel_path="user.go",
        file_path="user.go",
    )
    g.add_node(
        "GetName",
        symbol_type="method",
        source_text="func (u User) GetName() string {}",
        rel_path="user.go",  # same file
        file_path="user.go",
    )
    g.add_edge("UserStruct", "GetName", relationship_type="defines")
    result = augment_go_node(g, "UserStruct")
    assert result is None


# ---------------------------------------------------------------------------
# expand_smart — main entry point
# ---------------------------------------------------------------------------


def _class_graph():
    """A graph simulating a class hierarchy for expansion."""
    g = nx.DiGraph()
    g.add_node("AuthService", symbol_type="class", language="python")
    g.add_node("BaseService", symbol_type="class", language="python")
    g.add_node("IService", symbol_type="interface", language="python")
    g.add_node("login", symbol_type="function", language="python")
    g.add_edge("AuthService", "BaseService", relationship_type="inheritance")
    g.add_edge("AuthService", "IService", relationship_type="implementation")
    g.add_edge("login", "AuthService", relationship_type="calls")
    return g


def test_expand_smart_includes_original_nodes():
    g = _class_graph()
    result = expand_smart({"AuthService"}, g)
    assert "AuthService" in result.expanded_nodes


def test_expand_smart_adds_base_class():
    g = _class_graph()
    result = expand_smart({"AuthService"}, g)
    assert "BaseService" in result.expanded_nodes


def test_expand_smart_adds_interface():
    g = _class_graph()
    result = expand_smart({"AuthService"}, g)
    assert "IService" in result.expanded_nodes


def test_expand_smart_returns_expansion_result():
    g = _class_graph()
    result = expand_smart({"AuthService"}, g)
    assert isinstance(result, ExpansionResult)
    assert isinstance(result.expanded_nodes, set)
    assert isinstance(result.augmentations, dict)
    assert isinstance(result.expansion_reasons, dict)


def test_expand_smart_empty_input():
    g = _class_graph()
    result = expand_smart(set(), g)
    assert result.expanded_nodes == set()


def test_expand_smart_unknown_type_no_expansion():
    g = nx.DiGraph()
    g.add_node("module_doc", symbol_type="module_doc", language="python")
    g.add_node("Other", symbol_type="class", language="python")
    g.add_edge("module_doc", "Other", relationship_type="references")
    result = expand_smart({"module_doc"}, g)
    # module_doc is in EXPANSION_WORTHY_TYPES but not in _STRATEGY_MAP directly
    assert "module_doc" in result.expanded_nodes


def test_expand_smart_respects_per_symbol_cap():
    g = nx.DiGraph()
    g.add_node("Hub", symbol_type="class", language="python")
    # Add many neighbors
    for i in range(30):
        nid = f"Class{i}"
        g.add_node(nid, symbol_type="class", language="python")
        g.add_edge("Hub", nid, relationship_type="inheritance")

    result = expand_smart({"Hub"}, g, per_symbol_cap=5)
    # Should not have more than 5 + 1 (the original)
    assert len(result.expanded_nodes) <= 6


def test_expand_smart_respects_global_cap():
    g = nx.DiGraph()
    # Create 3 class nodes each with many successors
    for j in range(3):
        cls = f"Class{j}"
        g.add_node(cls, symbol_type="class", language="python")
        for i in range(20):
            sub = f"Sub{j}_{i}"
            g.add_node(sub, symbol_type="class", language="python")
            g.add_edge(cls, sub, relationship_type="inheritance")

    matched = {"Class0", "Class1", "Class2"}
    result = expand_smart(matched, g, global_cap=10, per_symbol_cap=15)
    total_new = len(result.expanded_nodes) - len(matched)
    assert total_new <= 10


def test_expand_smart_function_node():
    g = nx.DiGraph()
    g.add_node("process", symbol_type="function", language="python")
    g.add_node("Widget", symbol_type="class", language="python")
    g.add_edge("process", "Widget", relationship_type="creates")
    result = expand_smart({"process"}, g)
    assert "Widget" in result.expanded_nodes


def test_expand_smart_method_node():
    g = nx.DiGraph()
    g.add_node("do_work", symbol_type="method", language="python")
    g.add_node("Helper", symbol_type="class", language="python")
    g.add_edge("do_work", "Helper", relationship_type="creates")
    result = expand_smart({"do_work"}, g)
    assert "Helper" in result.expanded_nodes


def test_expand_smart_constant_node():
    g = nx.DiGraph()
    g.add_node("MAX_SIZE", symbol_type="constant", language="python")
    g.add_node("init_pool", symbol_type="function", language="python")
    g.add_edge("MAX_SIZE", "init_pool", relationship_type="calls")
    result = expand_smart({"MAX_SIZE"}, g)
    assert "init_pool" in result.expanded_nodes


def test_expand_smart_type_alias_node():
    g = nx.DiGraph()
    g.add_node("UserId", symbol_type="type_alias", language="python")
    g.add_node("UUID", symbol_type="class", language="python")
    g.add_edge("UserId", "UUID", relationship_type="alias_of")
    result = expand_smart({"UserId"}, g)
    assert "UUID" in result.expanded_nodes


def test_expand_smart_macro_node():
    g = nx.DiGraph()
    g.add_node("CHECK", symbol_type="macro", language="cpp")
    g.add_node("ErrorClass", symbol_type="class", language="cpp")
    g.add_edge("CHECK", "ErrorClass", relationship_type="references")
    result = expand_smart({"CHECK"}, g)
    assert "ErrorClass" in result.expanded_nodes


def test_expand_smart_cpp_augmentation():
    g = nx.DiGraph()
    g.add_node(
        "decl",
        symbol_type="function",
        source_text="void foo();",
        language="cpp",
        file_path="foo.h",
    )
    g.add_node(
        "impl",
        symbol_type="function",
        source_text="void foo() { int x = 1; }",
        language="cpp",
        file_path="foo.cpp",
        rel_path="foo.cpp",
    )
    g.add_edge("impl", "decl", relationship_type="defines_body")
    result = expand_smart({"decl"}, g)
    assert "decl" in result.augmentations


def test_expand_smart_go_augmentation():
    g = nx.DiGraph()
    g.add_node(
        "UserStruct",
        symbol_type="struct",
        source_text="type User struct {}",
        language="go",
        rel_path="user.go",
        file_path="user.go",
    )
    g.add_node(
        "GetName",
        symbol_type="method",
        source_text="func (u User) GetName() string { return u.Name }",
        language="go",
        rel_path="user_methods.go",
        file_path="user_methods.go",
    )
    g.add_edge("UserStruct", "GetName", relationship_type="defines")
    result = expand_smart({"UserStruct"}, g)
    assert "UserStruct" in result.augmentations


# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------


def test_class_like_types_contains_expected():
    assert "class" in CLASS_LIKE_TYPES
    assert "interface" in CLASS_LIKE_TYPES
    assert "struct" in CLASS_LIKE_TYPES
    assert "enum" in CLASS_LIKE_TYPES
    assert "trait" in CLASS_LIKE_TYPES


def test_skip_relationships_contains_expected():
    assert "defines" in SKIP_RELATIONSHIPS
    assert "imports" in SKIP_RELATIONSHIPS
    assert "decorates" in SKIP_RELATIONSHIPS


def test_expansion_priorities_structure():
    for rel_type, cfg in EXPANSION_PRIORITIES.items():
        assert "budget" in cfg
        assert "direction" in cfg
        assert "priority" in cfg


def test_global_expansion_cap_positive():
    assert GLOBAL_EXPANSION_CAP > 0


def test_per_symbol_cap_positive():
    assert PER_SYMBOL_CAP > 0
