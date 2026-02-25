"""
Tests for PipelineTimer utility.

Run with: PYTHONPATH=. uv run pytest tests/test_pipeline_timer.py -v
"""

import pytest
import time

from api.utils.pipeline_timer import PipelineTimer


def test_pipeline_timer_records_stages():
    """Verify PipelineTimer captures per-stage timings."""
    timer = PipelineTimer(department="Finance")

    with timer.stage("retrieval"):
        time.sleep(0.05)

    with timer.stage("generation"):
        time.sleep(0.05)

    timings = timer.as_dict()

    assert "retrieval" in timings
    assert "generation" in timings
    assert timings["retrieval"] >= 0.04  # Allow some tolerance
    assert timings["generation"] >= 0.04


def test_pipeline_timer_total_ms():
    """Verify total_ms sums all stages correctly."""
    timer = PipelineTimer(department="HR")

    with timer.stage("step_a"):
        time.sleep(0.02)

    with timer.stage("step_b"):
        time.sleep(0.02)

    assert timer.total_ms >= 30  # >=30ms total (2 x ~20ms with tolerance)


def test_pipeline_timer_empty():
    """Verify empty timer returns zero."""
    timer = PipelineTimer()
    assert timer.as_dict() == {}
    assert timer.total_ms == 0.0
