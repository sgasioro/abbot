"""Shared fixtures."""

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def ui_states(monkeypatch):
    """Every ``UIState`` the UI pages make in this test, newest last (needs nicegui)."""
    pytest.importorskip("nicegui")
    from abbot.ui import app

    states = []
    original = app.UIState.__init__

    def keep(self, *args, **kwargs):
        original(self, *args, **kwargs)
        states.append(self)

    monkeypatch.setattr(app.UIState, "__init__", keep)
    return states
