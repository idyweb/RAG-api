"""
Tests for UpdatePermissionsRequest schema validation.

Covers edge cases in the Pydantic model to ensure bad input
is rejected before hitting the service layer.
"""

import pytest
from pydantic import ValidationError

from api.apps.documents.schemas import UpdatePermissionsRequest


def test_valid_departments():
    """Normal list of departments is accepted."""
    req = UpdatePermissionsRequest(allowed_departments=["HR", "Finance"])
    assert req.allowed_departments == ["HR", "Finance"]


def test_duplicates_are_removed():
    """Duplicate department names are deduplicated."""
    req = UpdatePermissionsRequest(allowed_departments=["HR", "HR", "Sales", "Sales"])
    assert req.allowed_departments == ["HR", "Sales"]


def test_whitespace_is_stripped():
    """Leading/trailing whitespace on department names is stripped."""
    req = UpdatePermissionsRequest(allowed_departments=["  HR  ", " Sales"])
    assert req.allowed_departments == ["HR", "Sales"]


def test_empty_strings_are_filtered():
    """Empty strings are removed; if all empty, validation fails."""
    with pytest.raises(ValidationError):
        UpdatePermissionsRequest(allowed_departments=["", "  ", ""])


def test_empty_list_rejected():
    """Empty list is rejected by min_length=1."""
    with pytest.raises(ValidationError):
        UpdatePermissionsRequest(allowed_departments=[])


def test_missing_field_rejected():
    """allowed_departments is required."""
    with pytest.raises(ValidationError):
        UpdatePermissionsRequest()


def test_single_department():
    """Single department is valid."""
    req = UpdatePermissionsRequest(allowed_departments=["IT"])
    assert req.allowed_departments == ["IT"]


def test_all_keyword():
    """'All' is a valid department value for company-wide access."""
    req = UpdatePermissionsRequest(allowed_departments=["All"])
    assert req.allowed_departments == ["All"]
