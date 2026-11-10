"""The UI's look: colors, type, and a few small components the screens share.

Plain white panels, one blue accent, monospace for numbers and the expert panel.
Fonts are the system's (no web fonts), so the page works offline in a lab.
"""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

ACCENT = "#1d4ed8"
INK = "#111827"
MUTED = "#6b7280"
LINE = "#e5e7eb"
PANEL = "#f7f8fa"
GOOD = "#15803d"
WARN = "#b45309"
BAD = "#b91c1c"

CSS = f"""
:root {{
  --ab-accent: {ACCENT}; --ab-ink: {INK}; --ab-muted: {MUTED}; --ab-line: {LINE};
  --ab-panel: {PANEL};
}}
body {{ background: white; color: var(--ab-ink);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
.ab-mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
.ab-header {{ background: white; color: var(--ab-ink); border-bottom: 1px solid var(--ab-line); }}
.ab-brand {{ font-weight: 700; font-size: 1.25rem; letter-spacing: -0.01em; }}
.ab-chip {{ display: inline-flex; align-items: center; gap: .4rem; padding: .3rem .7rem;
  border-radius: 9999px; font-size: .9rem; color: var(--ab-ink); cursor: default; }}
.ab-chip .ab-dot {{ width: 1.3rem; height: 1.3rem; border-radius: 9999px; display: inline-flex;
  align-items: center; justify-content: center; font-size: .75rem;
  border: 1px solid #9ca3af; color: #4b5563; }}
.ab-chip.done .ab-dot {{ background: #dcfce7; border-color: #dcfce7; color: {GOOD}; }}
.ab-chip.current {{ background: #e8eefc; color: var(--ab-accent); font-weight: 600; }}
.ab-chip.current .ab-dot {{ background: var(--ab-accent); border-color: var(--ab-accent); color: white; }}
.ab-chip.reachable {{ cursor: pointer; }}
.ab-chip.reachable:hover {{ background: #f3f4f6; }}
.ab-title {{ font-size: 1.85rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.2; }}
.ab-sub {{ color: #4b5563; }}
.ab-card {{ border: 1px solid var(--ab-line); border-radius: 14px; background: white; }}
.ab-option {{ border: 1px solid var(--ab-line); border-radius: 12px; padding: 1rem 1.25rem;
  cursor: pointer; background: white; }}
.ab-option:hover {{ border-color: #9ca3af; }}
.ab-option.selected {{ border: 2px solid var(--ab-accent); background: #f5f8ff;
  padding: calc(1rem - 1px) calc(1.25rem - 1px); }}
.ab-option-title {{ font-weight: 600; font-size: 1.05rem; }}
.ab-section {{ font-size: .72rem; font-weight: 600; letter-spacing: .08em; color: #4b5563;
  text-transform: uppercase; }}
.ab-expert {{ background: var(--ab-panel); border-left: 1px solid var(--ab-line); }}
.ab-table .q-table {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .82rem; }}
.ab-table .q-table thead tr {{ background: #f3f4f6; }}
.ab-table .q-table th {{ color: #4b5563; font-weight: 500; }}
.ab-value-label {{ font-size: .85rem; color: #4b5563; }}
.ab-value {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 1.9rem; font-weight: 600; letter-spacing: -0.02em; }}
.ab-value.label {{ font-family: inherit; }}
.ab-link {{ color: var(--ab-accent); cursor: pointer; }}
.ab-link:hover {{ text-decoration: underline; }}
.ab-warn-link {{ color: {WARN}; cursor: pointer; }}
.ab-warn-link:hover {{ text-decoration: underline; }}
.ab-status {{ border-radius: 9999px; padding: .15rem .65rem; font-size: .8rem; font-weight: 600; }}
.ab-status.green {{ background: #dcfce7; color: {GOOD}; }}
.ab-status.amber {{ background: #fef3c7; color: {WARN}; }}
.ab-status.red {{ background: #fee2e2; color: {BAD}; }}
.ab-status.blue {{ background: #dbeafe; color: {ACCENT}; }}
.ab-status.grey {{ background: #f3f4f6; color: #4b5563; }}
.ab-error {{ color: {BAD}; font-size: .9rem; }}
.ab-seg .q-btn {{ text-transform: none; }}
@media (max-width: 639px) {{ .ab-chip.ab-not-current {{ display: none; }} }}
"""


def apply() -> None:
    """Install the theme on the current page."""
    ui.colors(primary=ACCENT, secondary=INK, positive=GOOD, warning=WARN, negative=BAD)
    ui.add_css(CSS)


def title(text: str, sub: str | None = None) -> None:
    ui.label(text).classes("ab-title")
    if sub:
        ui.label(sub).classes("ab-sub")


def section(text: str) -> ui.label:
    return ui.label(text).classes("ab-section")


def option(heading: str, body: str, selected: bool, on_click: Callable[[], None],
           marker: str | None = None) -> ui.element:
    """A big choice card; ``selected`` draws it chosen."""
    card = ui.element("div").classes("ab-option w-full" + (" selected" if selected else ""))
    card.on("click", lambda _: on_click())
    if marker:
        card.mark(marker)
    with card:
        ui.label(heading).classes("ab-option-title")
        ui.label(body).classes("ab-sub text-sm")
    return card


def segmented(options: dict, value, on_change: Callable, marker: str | None = None):
    """A row of joined buttons (Higher / Lower / Just watch)."""
    toggle = ui.toggle(options, value=value, on_change=lambda e: on_change(e.value)).props(
        "no-caps unelevated toggle-color=primary color=white text-color=grey-9 "
        "padding='4px 14px'"
    ).classes("ab-seg border rounded-lg overflow-hidden")
    if marker:
        toggle.mark(marker)
    return toggle


def primary(text: str, on_click: Callable, marker: str | None = None):
    b = ui.button(text, on_click=on_click).props("unelevated no-caps size=md").classes(
        "rounded-lg px-5 py-1 text-base font-semibold"
    )
    if marker:
        b.mark(marker)
    return b


def link(text: str, on_click: Callable, warn: bool = False, marker: str | None = None):
    lab = ui.label(text).classes("ab-warn-link" if warn else "ab-link")
    lab.on("click", lambda _: on_click())
    if marker:
        lab.mark(marker)
    return lab


def table(columns: list[str], rows: list[dict], key: str | None = None, **kwargs):
    """A dense monospace table: ``columns`` names, ``rows`` dicts keyed by them."""
    cols = [{"name": c, "label": c, "field": c, "align": "left"} for c in columns]
    t = ui.table(columns=cols, rows=rows, row_key=key or columns[0], **kwargs).props(
        "dense flat bordered separator=horizontal hide-bottom"
    ).classes("ab-table w-full")
    return t
