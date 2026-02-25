"""
Pipeline timer utility.

Lightweight context manager that times individual pipeline stages
and emits both Prometheus histograms and a dict for QueryLog storage.

Usage:
    timer = PipelineTimer()

    with timer.stage("retrieval"):
        results = await search_pinecone(query)

    with timer.stage("generation"):
        answer = await generate_answer(results)

    # Save to QueryLog
    log.stage_timings = timer.as_dict()  # {"retrieval": 0.182, "generation": 1.34}

    # Prometheus is emitted automatically per stage
"""

import time
from contextlib import contextmanager
from typing import Optional

from api.utils.metrics import pipeline_stage_latency


class PipelineTimer:
    """Tracks per-stage latencies for a single pipeline run.

    Why a class and not raw decorators: we need to accumulate timings
    across multiple stages in a single request, then persist them as
    one JSON blob to QueryLog.
    """

    def __init__(self, department: str = "unknown") -> None:
        self._timings: dict[str, float] = {}
        self._department = department
        self._start: Optional[float] = None

    @contextmanager
    def stage(self, name: str):
        """Time a pipeline stage.

        Args:
            name: Stage identifier (e.g. "retrieval", "generation",
                  "reformulation", "verification").
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._timings[name] = round(elapsed, 4)
            # Emit to Prometheus
            pipeline_stage_latency.labels(
                stage=name, department=self._department
            ).observe(elapsed)

    def as_dict(self) -> dict[str, float]:
        """Return all stage timings as a flat dict for JSON storage."""
        return dict(self._timings)

    @property
    def total_ms(self) -> float:
        """Total time across all stages in milliseconds."""
        return round(sum(self._timings.values()) * 1000, 2)
