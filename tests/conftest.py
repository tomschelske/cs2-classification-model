"""Shared fixtures. The TestClient context manager triggers the app lifespan,
so the model + featured round are loaded exactly as in production."""
import pytest
from fastapi.testclient import TestClient

from cs2wp.api import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
