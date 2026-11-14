"""Тесты MCP-инструмента frac_hit_verify на детерминированных фикстурах."""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "mcp"))
sys.path.insert(0, str(ROOT / "scenario"))

from tool_core import frac_hit_verify  # noqa: E402
import make_fixtures  # noqa: E402


@pytest.fixture(scope="module")
def fixtures():
    make_fixtures.build()
    return ROOT / "scenario" / "fixtures"


def test_hit_detected(fixtures):
    v = frac_hit_verify(str(fixtures / "rec_hit.npz"))
    assert v["valid"]
    assert v["hit"]["estimated"]
    assert 80 <= v["hit"]["hit_time_min"] <= 105  # истинный hit ~90 мин


def test_nohit(fixtures):
    v = frac_hit_verify(str(fixtures / "rec_nohit.npz"))
    assert v["valid"]
    assert not v["hit"]["estimated"]


@pytest.mark.parametrize("name", ["bad_dt.npz", "bad_missing_key.npz", "bad_nan.npz"])
def test_invalid_flagged(fixtures, name):
    v = frac_hit_verify(str(fixtures / name))
    assert not v["valid"]
    assert v["errors"]
