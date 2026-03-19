from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from collections.abc import Iterator

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

try:  # pragma: no cover - platform dependent
    import resource  # type: ignore
except Exception:  # pragma: no cover
    resource = None  # type: ignore


def _resource_logging_enabled() -> bool:
    return os.getenv("WIKIS_RESOURCE_LOG", "0") in {"1", "true", "yes", "on"}


def _min_duration_ms() -> int:
    try:
        return int(os.getenv("WIKIS_RESOURCE_LOG_MIN_DURATION_MS", "0"))
    except ValueError:
        return 0


def _format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"
    sign = "-" if num_bytes < 0 else ""
    value = abs(float(num_bytes))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024:
            return f"{sign}{value:.1f}{unit}"
        value /= 1024
    return f"{sign}{value:.1f}PB"


def _rss_bytes() -> int | None:
    if psutil is not None:
        try:
            return psutil.Process().memory_info().rss
        except Exception:
            return None
    if resource is None:
        return None
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On macOS ru_maxrss is bytes, on Linux it's KiB
        return int(rss if sys.platform == "darwin" else rss * 1024)
    except Exception:
        return None


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss if sys.platform == "darwin" else rss * 1024)
    except Exception:
        return None


@contextlib.contextmanager
def resource_monitor(phase: str, logger: logging.Logger | None = None) -> Iterator[None]:
    if not _resource_logging_enabled():
        yield
        return

    log = logger or logging.getLogger(__name__)
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    start_rss = _rss_bytes()
    start_peak = _peak_rss_bytes()

    try:
        yield
    finally:
        end_wall = time.perf_counter()
        end_cpu = time.process_time()
        end_rss = _rss_bytes()
        end_peak = _peak_rss_bytes()

        wall_s = max(0.0, end_wall - start_wall)
        cpu_s = max(0.0, end_cpu - start_cpu)
        min_ms = _min_duration_ms()
        if not (min_ms and wall_s * 1000 < min_ms):
            cpu_pct = (cpu_s / wall_s * 100.0) if wall_s > 0 else 0.0

            msg = f"[RESOURCE] {phase} wall={wall_s:.2f}s cpu={cpu_s:.2f}s cpu%={cpu_pct:.1f}%"
            if start_rss is not None and end_rss is not None:
                msg += f" rss={_format_bytes(end_rss)} delta={_format_bytes(end_rss - start_rss)}"
            if start_peak is not None and end_peak is not None:
                msg += f" peak={_format_bytes(end_peak)}"

            log.info(msg)
