"""
conftest.py — shared pytest fixtures for the Eco-Scale test suite.

Makes the project root importable (so `from environment...` works no matter where
pytest is invoked) and provides the held-out test traces as a fixture.
"""

import os
import sys
import json
import glob

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def test_traces():
    """Absolute paths to the 5 held-out TEST traces (different data values)."""
    split = json.load(open(os.path.join(ROOT, "data", "split.json")))
    return [os.path.join(ROOT, p) for p in split["test"]]


@pytest.fixture(scope="session")
def all_traces():
    """Every trace on disk — used for broad data-value coverage."""
    return sorted(glob.glob(os.path.join(ROOT, "data", "traces", "trace_*.npy")))
