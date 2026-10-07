"""abbot web UI (NiceGUI). Optional: install with ``abbot[ui]``.

The UI is a thin client over :class:`abbot.session.Campaign` — it holds NO
optimization logic. ``RunController`` (framework-free, testable) is the auto/manual
run-state machine; ``app`` builds the NiceGUI screens over it.
"""

from abbot.ui.controller import RunController, RunMode, RunStatus

__all__ = ["RunController", "RunMode", "RunStatus"]
