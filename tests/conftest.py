from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def gold_csv() -> Path:
    """The gold set (Gold_second.csv, revision of 2026-09-28), copied as a test fixture."""
    return FIXTURES / "gold_second.csv"
