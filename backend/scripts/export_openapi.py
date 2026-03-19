"""Export OpenAPI schema to JSON, including all Pydantic model schemas.

Routes aren't wired yet (Phase 2-3), so this script explicitly adds
all API/event models to the OpenAPI components.schemas section.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app
from app.models import (
    AskRequest,
    AskResponse,
    ChatMessage,
    DeleteWikiResponse,
    ErrorEvent,
    ErrorResponse,
    GenerateWikiRequest,
    GenerateWikiResponse,
    HealthResponse,
    PageCompleteEvent,
    ProgressEvent,
    ResearchRequest,
    ResearchResponse,
    SourceReference,
    # SSE events
    SSEEvent,
    WikiCompleteEvent,
    WikiListResponse,
    WikiSummary,
)

ALL_MODELS = [
    GenerateWikiRequest,
    GenerateWikiResponse,
    ChatMessage,
    SourceReference,
    AskRequest,
    AskResponse,
    ResearchRequest,
    ResearchResponse,
    WikiSummary,
    WikiListResponse,
    DeleteWikiResponse,
    HealthResponse,
    ErrorResponse,
    SSEEvent,
    ProgressEvent,
    PageCompleteEvent,
    WikiCompleteEvent,
    ErrorEvent,
]


def export() -> None:
    schema = app.openapi()

    # Ensure components.schemas exists
    schema.setdefault("components", {}).setdefault("schemas", {})

    for model in ALL_MODELS:
        json_schema = model.model_json_schema(mode="serialization")

        # Hoist any nested $defs into top-level schemas
        defs = json_schema.pop("$defs", {})
        for def_name, def_schema in defs.items():
            schema["components"]["schemas"][def_name] = def_schema

        # Replace internal $ref paths to use OpenAPI component refs
        raw = json.dumps(json_schema)
        raw = raw.replace("#/$defs/", "#/components/schemas/")
        json_schema = json.loads(raw)

        schema["components"]["schemas"][model.__name__] = json_schema

    out = Path(__file__).resolve().parent.parent / "openapi.json"
    out.write_text(json.dumps(schema, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    export()
