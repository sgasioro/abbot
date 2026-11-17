"""Shared fixtures."""

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    """``cuda`` tests need a GPU; without one (CI, a laptop) they're skipped."""
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs CUDA (torch.cuda.is_available() is False)")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _campaigns_dir(tmp_path, monkeypatch):
    """The UI saves campaigns here, never in the real ~/abbot-campaigns."""
    monkeypatch.setenv("ABBOT_UI_DIR", str(tmp_path / "campaigns"))


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
