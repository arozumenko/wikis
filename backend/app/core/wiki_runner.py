"""Subprocess entry-point for wiki generation.

Spawned by :meth:`app.services.wiki_service.WikiService._run_wiki_subprocess` as::

    python -u -m app.core.wiki_runner --input <path> --output <path>

Why a subprocess
----------------
Wiki generation does heavy CPU + blocking IO work (git clone, tree-sitter
parsing, NetworkX graph construction, Leiden clustering, LangGraph page
generation).  Running it inside the uvicorn process — even via
``asyncio.to_thread`` — blocks the event loop for long stretches because
the workload holds the GIL for hundreds of milliseconds at a time.  That
starves SSE keep-alive pings and every other concurrent request, which
users see as "UI frozen" and "Stream timeout" on big repositories.

Moving the work to a dedicated Python subprocess gives us true OS-level
isolation: the API's event loop only does async I/O (read pipe, write
artifacts), so the UI stays responsive regardless of repo size.

Protocol
--------
* ``--input``   JSON file with the :class:`GenerateWikiRequest` fields
                plus the resolved ``planner_type`` / ``exclude_tests``.
* ``--output``  JSON file written with the ``generate_wiki`` result dict.
                Any ``artifacts[].data`` that was bytes is base64-encoded
                with ``_b64: true`` so it round-trips through JSON.
* ``stdout``    One JSON object per line.  The parent reads these via
                ``await proc.stdout.readline()`` and translates them into
                SSE events.
* ``stderr``    Python logging output — captured by docker / k8s logs.
* Exit code     ``0`` on success, non-zero on failure.  The output file's
                ``success`` field is the canonical truth for the parent.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import traceback


def _emit(payload: dict) -> None:
    """Write one JSON line to stdout (parent reads via ``readline``)."""
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _configure_logging() -> None:
    level_name = os.getenv("WIKIS_WORKER_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    _configure_logging()
    log = logging.getLogger("wiki_runner")

    try:
        with open(args.input, encoding="utf-8") as f:
            payload = json.load(f)

        from app.config import get_settings
        from app.core.hybrid_wiki_toolkit_wrapper import HybridWikiToolkitWrapper
        from app.core.repo_providers.factory import RepoProviderFactory
        from app.services.llm_factory import create_embeddings, create_llm

        settings = get_settings()

        llm_overrides = {"model": payload["llm_model"]} if payload.get("llm_model") else {}
        emb_overrides = {"model": payload["embedding_model"]} if payload.get("embedding_model") else {}

        llm = create_llm(settings, tier="high", **llm_overrides)
        llm_low = create_llm(settings, tier="low")
        embeddings = create_embeddings(settings, **emb_overrides)

        clone_config = RepoProviderFactory.from_url(
            url=payload["repo_url"],
            token=payload.get("access_token"),
            branch=payload.get("branch") or "main",
        )

        def _progress(phase: str, progress: float, message: str) -> None:
            _emit({"t": "progress", "phase": phase, "progress": progress, "message": message})

        toolkit = HybridWikiToolkitWrapper(
            clone_config=clone_config,
            llm=llm,
            embeddings=embeddings,
            cache_dir=settings.cache_dir,
            force_rebuild_index=payload.get("force_rebuild_index", True),
            llm_low=llm_low,
            progress_callback=_progress,
            # Keep the cloned repo on disk so Deep Research's FilesystemBackend
            # can read source files post-generation.  Parent deletes via delete_wiki.
            cleanup_repos_on_exit=False,
            max_concurrent_pages=settings.llm_max_concurrency,
        )

        # Token pre-estimation (mirrors logic previously inline in wiki_service).
        try:
            from app.services.context_limits import get_context_limit

            model_name = payload.get("llm_model") or settings.llm_model or ""
            estimated = 0
            if hasattr(toolkit, "estimate_index_tokens"):
                estimated = toolkit.estimate_index_tokens()
            if not estimated and getattr(toolkit, "indexer", None):
                summary = toolkit.get_index_summary() if hasattr(toolkit, "get_index_summary") else {}
                total_chars = summary.get("total_chars", 0) if isinstance(summary, dict) else 0
                estimated = total_chars // 4
            if estimated > 0:
                _emit({
                    "t": "token_estimate",
                    "estimated_tokens": estimated,
                    "model_context_limit": get_context_limit(model_name),
                })
        except Exception as exc:  # noqa: BLE001 — estimation is best-effort
            log.debug("Token estimation skipped: %s", exc)

        result = toolkit.generate_wiki(
            query=payload.get("wiki_title") or "Generate comprehensive wiki",
            include_research=payload.get("include_research", True),
            include_diagrams=payload.get("include_diagrams", True),
            planner_type=payload["planner_type"],
            exclude_tests=payload.get("exclude_tests", False),
        )

        # Normalize bytes → base64 so the dict survives JSON round-trip.
        for art in (result or {}).get("artifacts") or []:
            data = art.get("data")
            if isinstance(data, (bytes, bytearray)):
                art["data"] = base64.b64encode(data).decode("ascii")
                art["_b64"] = True

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f)

        return 0 if result and result.get("success") else 1

    except BaseException as exc:  # noqa: BLE001 — includes SystemExit/KeyboardInterrupt for subprocess cleanup
        log.exception("wiki_runner failed")
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                    f,
                )
        except Exception:  # noqa: BLE001
            pass
        _emit({"t": "error", "error": f"{type(exc).__name__}: {exc}"})
        return 2


if __name__ == "__main__":
    sys.exit(main())
