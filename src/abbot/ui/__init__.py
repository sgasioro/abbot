"""abbot web UI (NiceGUI). Optional: install with ``abbot[ui]``.

The UI is a thin client over :class:`abbot.session.Campaign` — it holds NO
optimization logic. ``RunController`` (framework-free, testable) is the auto/manual
run-state machine; ``app`` builds the NiceGUI screens over it. :func:`serve` opens
the UI on a campaign or experiment from your code.
"""

from abbot.ui.controller import RunController, RunMode, RunStatus

__all__ = ["RunController", "RunMode", "RunStatus", "serve"]


def __getattr__(name: str):
    if name == "serve":  # imported on use: it needs NiceGUI, the rest doesn't
        from abbot.ui.app import serve

        return serve
    raise AttributeError(f"module 'abbot.ui' has no attribute {name!r}")
