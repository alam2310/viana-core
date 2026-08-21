import pytest
from fastapi import HTTPException

from orchestrator.routes.profiles import _require_project_id


def test_require_project_id_valid() -> None:
    """_require_project_id accepts valid slug."""
    _require_project_id("valid-id-123")
    _require_project_id("123")
    _require_project_id("abc")
    _require_project_id("a_b-c")


def test_require_project_id_invalid() -> None:
    """_require_project_id rejects invalid string with HTTPException 400."""
    with pytest.raises(HTTPException) as excinfo:
        _require_project_id("invalid ID!")
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "project_id must match [a-z0-9][a-z0-9_-]*"

    with pytest.raises(HTTPException) as excinfo:
        _require_project_id("Capital")
    assert excinfo.value.status_code == 400

    with pytest.raises(HTTPException) as excinfo:
        _require_project_id("-starts-with-dash")
    assert excinfo.value.status_code == 400
