"""Tests for Pydantic state models and LangGraph compatibility."""

from __future__ import annotations

from pydantic import BaseModel

from app.core.state.wiki_state import (
    EnhancementState,
    PageGenerationState,
    QualityAssessmentState,
    WikiState,
)


class TestStateModelsArePydantic:
    def test_wiki_state_is_basemodel(self):
        assert issubclass(WikiState, BaseModel)

    def test_page_generation_state_is_basemodel(self):
        assert issubclass(PageGenerationState, BaseModel)

    def test_quality_assessment_state_is_basemodel(self):
        assert issubclass(QualityAssessmentState, BaseModel)

    def test_enhancement_state_is_basemodel(self):
        assert issubclass(EnhancementState, BaseModel)


class TestWikiStateDefaults:
    def test_all_fields_have_defaults(self):
        state = WikiState()
        assert state.current_phase == ""
        assert state.wiki_pages == []
        assert state.messages == []
        assert state.errors == []
        assert state.structure_planning_complete is False
        assert state.export_complete is False
        assert state.retry_counts == {}

    def test_json_schema_generation(self):
        schema = WikiState.model_json_schema()
        assert "properties" in schema
        assert "wiki_pages" in schema["properties"]
        assert "current_phase" in schema["properties"]


class TestLangGraphCompatibility:
    def test_operator_add_accumulation(self):
        from langgraph.graph import END, START, StateGraph

        def add_message(state):
            return {"messages": ["hello"], "current_phase": "done"}

        graph = StateGraph(WikiState)
        graph.add_node("step", add_message)
        graph.add_edge(START, "step")
        graph.add_edge("step", END)
        compiled = graph.compile()

        result = compiled.invoke(WikiState())
        assert result["messages"] == ["hello"]
        assert result["current_phase"] == "done"

    def test_accumulation_appends(self):
        from langgraph.graph import END, START, StateGraph

        def step1(state):
            return {"messages": ["first"]}

        def step2(state):
            return {"messages": ["second"]}

        graph = StateGraph(WikiState)
        graph.add_node("s1", step1)
        graph.add_node("s2", step2)
        graph.add_edge(START, "s1")
        graph.add_edge("s1", "s2")
        graph.add_edge("s2", END)
        compiled = graph.compile()

        result = compiled.invoke(WikiState())
        assert result["messages"] == ["first", "second"]
