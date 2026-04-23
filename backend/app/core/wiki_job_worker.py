#!/usr/bin/env python3

"""
Wiki Job Worker - Entry point for K8s Job-based wiki generation.

This module is the entry point for K8s Jobs that perform wiki generation.
It reads input from a shared PVC, runs the wiki generation, and writes
the result back to the PVC.

Error Handling:
- Writes result.json with success=false on exceptions
- Writes to /dev/termination-log for K8s-native error reporting
- Exit code 1 signals failure to K8s Job controller

Usage:
    python -m plugin_implementation.wiki_job_worker --job-id=<job_id>
"""

import argparse
import json
import logging
import os
import sys
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# K8s termination log path (K8s reads this on container termination)
TERMINATION_LOG_PATH = "/dev/termination-log"


def write_termination_message(message: str, max_length: int = 4096):
    """
    Write termination message to K8s termination log.

    K8s reads /dev/termination-log when a container exits and makes it
    available in pod.status.containerStatuses[].state.terminated.message.

    This is useful for crash scenarios where result.json may not be written.

    Args:
        message: Error message to write
        max_length: Max message length (K8s default limit is 4096 bytes)
    """
    try:
        # Truncate message if too long
        if len(message) > max_length:
            message = message[: max_length - 20] + "\n... (truncated)"

        with open(TERMINATION_LOG_PATH, "w", encoding="utf-8") as f:
            f.write(message)
    except Exception:  # noqa: S110
        # /dev/termination-log may not exist in non-K8s environments
        pass


# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("wiki_job_worker")


def load_input(job_id: str) -> dict[str, Any] | None:
    """Load input data from the jobs directory."""
    base_path = Path(os.environ.get("WIKIS_BASE_PATH", "/data"))
    input_file = base_path / "jobs" / job_id / "input.json"

    if not input_file.exists():
        log.error(f"Input file not found: {input_file}")
        return None

    try:
        with open(input_file, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.error(f"Failed to load input: {e}")
        return None


def save_result(job_id: str, result: dict[str, Any]):
    """Save result data to the jobs directory."""
    base_path = Path(os.environ.get("WIKIS_BASE_PATH", "/data"))
    result_file = base_path / "jobs" / job_id / "result.json"

    try:
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        log.info(f"Saved result to {result_file}")
    except Exception as e:
        log.error(f"Failed to save result: {e}")
        raise


def run_wiki_generation(input_data: dict[str, Any]) -> dict[str, Any]:
    """
    Run the wiki generation process.

    Args:
        input_data: Input parameters including:
            - query: User prompt for generation
            - llm_settings: LLM configuration
            - embedding_model: Optional embedding model
            - repo_config: Multi-provider repository configuration
            - github_configuration/github_repository/github_base_branch: legacy GitHub fields
            - active_branch: Optional active branch override
            - force_rebuild_index: Whether to rebuild index
            - indexing_method: "filesystem" or "github"

    Returns:
        Result dict returned by HybridWikiToolkitWrapper.generate_wiki
    """
    from plugin_implementation.hybrid_wiki_toolkit_wrapper import HybridWikiToolkitWrapper
    from plugin_implementation.repo_providers import RepoProviderFactory
    from plugin_implementation.wiki_subprocess_worker import _build_llm_and_embeddings

    base_path = os.path.abspath(os.environ.get("WIKIS_BASE_PATH", "/data/wiki_builder"))
    query = input_data.get("query")
    if not query:
        raise ValueError("query is required")

    llm_settings = input_data.get("llm_settings") or {}
    embedding_model = input_data.get("embedding_model")

    repo_config = input_data.get("repo_config") or {}
    github_configuration = input_data.get("github_configuration") or {}

    if repo_config:
        provider_type = repo_config.get("provider_type", "github")
        provider_config = repo_config.get("provider_config", {})
        repository = repo_config.get("repository")
        branch = repo_config.get("branch", "main")
        project = repo_config.get("project")
        active_branch = input_data.get("active_branch") or branch
    else:
        provider_type = "github"
        provider_config = github_configuration
        repository = input_data.get("github_repository")
        branch = input_data.get("github_base_branch", "main")
        project = None
        active_branch = input_data.get("active_branch") or branch

    if not repository:
        raise ValueError("repository is required (either in repo_config or github_repository)")

    force_rebuild_index = bool(input_data.get("force_rebuild_index", True))
    indexing_method = input_data.get("indexing_method", "filesystem")

    log.info(f"Starting wiki generation for: {repository}")
    start_time = datetime.now(UTC)

    llm, embeddings, _ = _build_llm_and_embeddings(llm_settings, embedding_model)

    try:
        clone_config = RepoProviderFactory.from_toolkit_config(
            provider_type=provider_type,
            config=provider_config,
            repository=repository,
            branch=branch,
            project=project,
        )
    except Exception as e:
        log.warning(f"Failed to build clone config: {e}. Will fall back to direct credentials if available.")
        clone_config = None

    wrapper = HybridWikiToolkitWrapper(
        repo_config=repo_config,
        clone_config=clone_config,
        github_repository=repository,
        github_base_branch=branch,
        active_branch=active_branch,
        cache_dir=os.path.join(base_path, "cache"),
        force_rebuild_index=force_rebuild_index,
        llm=llm,
        embeddings=embeddings,
        indexing_method=indexing_method,
    )

    result = wrapper.generate_wiki(query=query)

    # Match subprocess worker post-processing for manifest metadata.
    if result and result.get("success"):
        try:
            # Use actual branch from clone result (handles master vs main, etc.)
            actual_branch = result.get("branch") or active_branch
            if actual_branch != active_branch:
                log.info("Using actual branch from clone: %s (requested: %s)", actual_branch, active_branch)
                active_branch = actual_branch

            repo_context = result.get("repository_context", "")
            commit_hash = result.get("commit_hash")

            # Use identifier format with commit hash for cache isolation
            if commit_hash:
                commit_short = commit_hash[:8]
                repo_identifier = f"{repository}:{active_branch}:{commit_short}"
            else:
                repo_identifier = f"{repository}:{active_branch}"

            wiki_version_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]

            if repo_context:
                log.info("Repository analysis available in unified DB (%s chars)", len(repo_context))
            else:
                log.warning("No repository_context in result")

            # Context7-style: all artifacts are prefixed with wiki_id folder
            from plugin_implementation.registry_manager import normalize_wiki_id

            wiki_id = normalize_wiki_id(repo_identifier)
            log.info("Wiki ID for folder structure: %s", wiki_id)

            # Prefix markdown artifact names with {wiki_id}/wiki_pages/
            pages = []
            try:
                for art in result.get("artifacts") or []:
                    name = art.get("name")
                    if art.get("type") == "text/markdown" and isinstance(name, str) and name.endswith(".md"):
                        # If name already starts with wiki_id, don't add prefix again
                        if name.startswith(f"{wiki_id}/"):
                            pages.append(name)
                        else:
                            prefixed_name = f"{wiki_id}/wiki_pages/{name}"
                            art["name"] = prefixed_name
                            pages.append(prefixed_name)
            except Exception:
                pages = []

            # Extract wiki_title from wiki_structure, build description from repository_context
            wiki_title = ""
            wiki_description = ""
            try:
                for art in result.get("artifacts") or []:
                    name = art.get("name", "")
                    if art.get("type") == "application/json" and "wiki_structure" in name:
                        data = art.get("data", "")
                        if isinstance(data, str):
                            ws = json.loads(data)
                            wiki_title = ws.get("wiki_title", "")
                        break
            except Exception as ws_err:
                log.warning("Failed to extract wiki title: %s", ws_err)

            # Build description from repository_context JSON for LLM resolution
            try:
                if repo_context and repo_context.strip().startswith("{"):
                    ctx_json = json.loads(repo_context)
                    exec_summary = ctx_json.get("executive_summary", "")
                    core_purpose = ctx_json.get("core_purpose", "")
                    if exec_summary or core_purpose:
                        wiki_description = f"{exec_summary} {core_purpose}".strip()
            except Exception as ctx_err:
                log.warning("Failed to parse repository_context for description: %s", ctx_err)

            # Also prefix JSON artifacts (wiki_structure) with wiki_id folder
            try:
                for art in result.get("artifacts") or []:
                    name = art.get("name")
                    if art.get("type") == "application/json" and isinstance(name, str) and "wiki_structure" in name:
                        if not name.startswith(wiki_id):
                            art["name"] = f"{wiki_id}/analysis/{name}"
            except Exception:  # noqa: S110
                pass

            manifest = {
                "schema_version": 2,  # Version 2 = folder structure
                "wiki_id": wiki_id,
                "wiki_title": wiki_title,
                "description": wiki_description,
                "wiki_version_id": wiki_version_id,
                "created_at": datetime.now(UTC).isoformat(),
                "canonical_repo_identifier": repo_identifier,
                "repository": repository,
                "branch": active_branch,
                "commit_hash": commit_hash,
                "pages": pages,
                "provider_type": provider_type,
            }

            # Attach cache keys for deterministic artifact resolution (vectorstore/graph/docstore).
            try:
                from plugin_implementation.repo_resolution import load_cache_index

                idx = load_cache_index(cache_dir)
                faiss_key = idx.get(repo_identifier) if isinstance(idx, dict) else None
                graphs_idx = idx.get("graphs", {}) if isinstance(idx.get("graphs", {}), dict) else {}
                docs_idx = idx.get("docs", {}) if isinstance(idx.get("docs", {}), dict) else {}
                bm25_idx = idx.get("bm25", {}) if isinstance(idx.get("bm25", {}), dict) else {}

                graph_key = graphs_idx.get(f"{repo_identifier}:combined")
                docstore_key = docs_idx.get(repo_identifier)
                bm25_key = bm25_idx.get(repo_identifier)

                if isinstance(faiss_key, str):
                    manifest["faiss_cache_key"] = faiss_key
                if isinstance(graph_key, str):
                    manifest["graph_cache_key"] = graph_key
                if isinstance(docstore_key, str):
                    manifest["docstore_cache_key"] = docstore_key
                    manifest["docstore_files"] = [
                        f"{docstore_key}.docstore.bin",
                        f"{docstore_key}.doc_index.json",
                    ]
                if isinstance(bm25_key, str):
                    manifest["bm25_cache_key"] = bm25_key
                    manifest["bm25_files"] = [
                        f"{bm25_key}.bm25.sqlite",
                    ]
            except Exception as mf_cache_err:
                log.warning("Failed to attach cache keys to manifest: %s", mf_cache_err)

            try:
                artifacts = result.get("artifacts")
                if not isinstance(artifacts, list):
                    artifacts = []
                    result["artifacts"] = artifacts

                # Store manifest at {wiki_id}/wiki_manifest_{version}.json
                artifacts.append(
                    {
                        "name": f"{wiki_id}/wiki_manifest_{wiki_version_id}.json",
                        "object_type": "wiki_manifest",
                        "type": "application/json",
                        "data": json.dumps(manifest, indent=2, ensure_ascii=False),
                    }
                )

                # Surface metadata at the top-level result for invoke.py registry
                result["wiki_version_id"] = wiki_version_id
                result["wiki_id"] = wiki_id
                result["canonical_repo_identifier"] = repo_identifier
                result["branch"] = active_branch
                result["commit_hash"] = commit_hash
                result["provider_type"] = provider_type
                result["wiki_title"] = wiki_title
                result["wiki_description"] = wiki_description
            except Exception as mf_err:
                log.warning("Failed to append wiki manifest artifact: %s", mf_err)
        except Exception as save_err:
            log.warning("Failed to build wiki manifest metadata: %s", save_err)

    end_time = datetime.now(UTC)
    duration = (end_time - start_time).total_seconds()
    log.info(f"Wiki generation completed in {duration:.1f}s")

    return result


def main():
    """Main entry point for the job worker."""
    parser = argparse.ArgumentParser(description="Wikis Job Worker")
    parser.add_argument("--job-id", required=True, help="Job ID to process")
    args = parser.parse_args()

    job_id = args.job_id
    log.info("=== Wikis Job Worker starting ===")
    log.info(f"Job ID: {job_id}")
    log.info(f"Pod: {os.environ.get('HOSTNAME', 'unknown')}")

    try:
        # Load input
        input_data = load_input(job_id)
        if input_data is None:
            error_msg = "Failed to load input data"
            write_termination_message(f"Wikis Job Error: {error_msg}")
            save_result(job_id, {"success": False, "error": error_msg, "error_category": "input_error"})
            sys.exit(1)

        repo_config = input_data.get("repo_config") or {}
        repo_label = repo_config.get("repository") or input_data.get("github_repository") or "unknown"
        log.info(f"Loaded input: repository={repo_label}")

        # Run wiki generation
        result = run_wiki_generation(input_data)

        # Save result
        save_result(job_id, result)

        # Check if result indicates failure
        if not result.get("success", True):
            error_msg = result.get("error", "Wiki generation failed")
            write_termination_message(f"Wikis Generation Error: {error_msg}")
            log.error(f"=== Job completed with failure: {error_msg} ===")
            sys.exit(1)

        log.info("=== Job completed successfully ===")
        sys.exit(0)

    except Exception as e:
        log.error(f"Job failed with error: {e}")
        tb = traceback.format_exc()
        log.error(tb)

        # Categorize the error for better UI messaging
        error_str = str(e)
        error_category = "unknown_error"
        if "OOM" in error_str or "memory" in error_str.lower():
            error_category = "out_of_memory"
        elif "timeout" in error_str.lower():
            error_category = "timeout"
        elif "rate limit" in error_str.lower() or "429" in error_str:
            error_category = "rate_limit"
        elif "authentication" in error_str.lower() or "401" in error_str or "403" in error_str:
            error_category = "authentication_error"
        elif "repository" in error_str.lower() and "not found" in error_str.lower():
            error_category = "repository_not_found"

        # Build structured error message for termination log
        termination_msg = "Wikis Job Failed\n"
        termination_msg += f"Error: {error_str}\n"
        termination_msg += f"Category: {error_category}\n"
        termination_msg += f"Job ID: {job_id}\n"
        if len(tb) < 2000:
            termination_msg += f"\nTraceback:\n{tb}"

        write_termination_message(termination_msg)

        # Save error result
        try:
            save_result(
                job_id, {"success": False, "error": error_str, "error_category": error_category, "traceback": tb}
            )
        except Exception:  # noqa: S110
            pass  # Best effort

        sys.exit(1)


if __name__ == "__main__":
    main()
