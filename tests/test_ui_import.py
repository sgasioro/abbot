"""Smoke test: the UI module imports and its state helper builds a campaign.

Skipped when nicegui isn't installed, so the core test suite doesn't require the
[ui] extra. The spec-building / optimizer-selection logic is covered browser-free
in test_ui_builder.py; this just guards that app.py imports and UIState wires up.
"""

import pytest

pytest.importorskip("nicegui")


def test_app_imports():
    from abbot.ui import app  # noqa: F401
    from abbot.ui.app import UIState

    state = UIState()
    # Default rows form a valid single-variable, single-objective campaign.
    state.start_campaign()
    assert state.controller is not None
    assert state.spec.n_objectives == 1
    assert state.controller.can_auto is False  # manual UI session


def test_uistate_surfaces_mixed_space_error():
    from abbot.ui.app import UIState

    state = UIState()
    state.variable_rows = [
        {"kind": "continuous", "name": "x", "lower": 0.0, "upper": 1.0},
        {"kind": "enumerated", "name": "c", "values": "0, 1, 2"},
    ]
    with pytest.raises(ValueError, match="Mixed"):
        state.start_campaign()


def test_free_port_falls_back_when_busy():
    import socket

    from abbot.ui.app import _free_port

    assert _free_port("127.0.0.1", 8531) == 8531  # free -> as requested
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 8532))
    s.listen()
    try:
        alt = _free_port("127.0.0.1", 8532)  # busy -> different free port
        assert alt != 8532 and alt > 0
    finally:
        s.close()
