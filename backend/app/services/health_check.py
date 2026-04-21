"""Provider health checks — test LLM/embedding connectivity at startup."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.config import Settings

logger = logging.getLogger(__name__)

# How long a successful provider-health snapshot stays trusted before we
# bother re-pinging the LLM and embedding endpoints.  Docker-compose pings
# /health every 30s; with the previous 60s TTL every other probe burned a
# real LLM + embeddings round-trip and re-emitted the startup INFO logs,
# flooding the indexing logs and consuming API quota for no reason.
HEALTH_CHECK_TTL_SECONDS = 600


@dataclass
class ProviderHealth:
    """Health status for LLM and embedding providers."""

    llm: str = "unchecked"  # ok | unreachable: {reason} | unchecked
    embeddings: str = "unchecked"
    checked_at: float = field(default_factory=time.monotonic)

    @property
    def healthy(self) -> bool:
        """False only when a component explicitly reports an error."""
        return not (self.llm.startswith("unreachable:") or self.embeddings.startswith("unreachable:"))

    @property
    def stale(self) -> bool:
        return time.monotonic() - self.checked_at > HEALTH_CHECK_TTL_SECONDS


def validate_credentials(settings: Settings) -> str | None:
    """Check that required credentials are present for the configured LLM provider.

    Returns None if credentials look valid, or an error detail string if missing.
    Does NOT make any network calls — purely config inspection.
    """
    provider = settings.llm_provider

    if provider in ("openai", "anthropic", "custom", "copilot", "github"):
        api_key = settings.llm_api_key.get_secret_value()
        if not api_key:
            return "LLM_API_KEY is not set"
    elif provider == "gemini":
        if not settings.gemini_api_key:
            return "GEMINI_API_KEY is not set"
    elif provider == "bedrock":
        # Bedrock can use IAM roles — only validate explicit key pair consistency
        has_key_id = bool(settings.aws_access_key_id)
        has_secret = bool(settings.aws_secret_access_key)
        if has_key_id and not has_secret:
            return "AWS_ACCESS_KEY_ID is set but AWS_SECRET_ACCESS_KEY is missing"
        if has_secret and not has_key_id:
            return "AWS_SECRET_ACCESS_KEY is set but AWS_ACCESS_KEY_ID is missing"
    elif provider == "ollama":
        pass  # No credentials needed for local Ollama
    else:
        return f"Unknown provider '{provider}'"

    return None


async def check_providers(
    settings: Settings,
    previous: ProviderHealth | None = None,
) -> ProviderHealth:
    """Test LLM and embedding connectivity. Never raises — logs warnings.

    Pass ``previous`` for periodic re-checks so unchanged status only logs at
    DEBUG level (avoids flooding logs with the same startup banner every minute).
    """
    health = ProviderHealth()

    if not settings.health_check_on_startup:
        logger.info("Health check disabled (HEALTH_CHECK_ON_STARTUP=false)")
        return health

    def _log_status(component: str, new_status: str, prev_status: str | None) -> None:
        """INFO on first check or status change; DEBUG when status is unchanged."""
        ok = new_status == "ok"
        changed = prev_status != new_status
        if ok:
            msg = f"{component} health check passed ({settings.llm_provider}/{settings.llm_model})"
            if prev_status is None or changed:
                logger.info(msg)
            else:
                logger.debug(msg)
        else:
            # Failures always log at WARNING regardless of repetition
            logger.warning(f"{component} health check failed ({settings.llm_provider}): {new_status}")

    prev_llm = previous.llm if previous else None
    prev_emb = previous.embeddings if previous else None

    # Validate LLM credentials before attempting network calls
    cred_error = validate_credentials(settings)
    if cred_error:
        msg = f"LLM provider {settings.llm_provider} configured but credentials missing: {cred_error}"
        if prev_llm != f"unreachable: {cred_error}":
            logger.error(msg)
        health.llm = f"unreachable: {cred_error}"
    else:
        # Test LLM only when credentials are valid
        try:
            from app.services.llm_factory import create_llm

            llm = create_llm(settings)
            await llm.ainvoke("Say OK")
            health.llm = "ok"
            _log_status("LLM", health.llm, prev_llm)
        except Exception as e:
            health.llm = f"unreachable: {e}"
            _log_status("LLM", health.llm, prev_llm)

    # Test embeddings independently — they may use a different provider/key
    try:
        from app.services.llm_factory import create_embeddings

        emb = create_embeddings(settings)
        await emb.aembed_query("test")
        health.embeddings = "ok"
        _log_status("Embedding", health.embeddings, prev_emb)
    except Exception as e:
        health.embeddings = f"unreachable: {e}"
        _log_status("Embedding", health.embeddings, prev_emb)

    return health
