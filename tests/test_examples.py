"""Smoke-test the example notebooks: execute each top-to-bottom, fail on any error.

Keeps the notebooks from silently rotting as the API evolves. Marked ``slow`` (they
fit GPs) and skipped if the ``[examples]`` deps (nbclient/nbformat) aren't installed.
"""

from pathlib import Path

import pytest

pytest.importorskip("nbclient")
pytest.importorskip("nbformat")

import nbformat
from nbclient import NotebookClient

pytestmark = pytest.mark.slow

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("[0-9]*.ipynb"))


@pytest.mark.parametrize("nb_path", EXAMPLES, ids=lambda p: p.name)
def test_notebook_runs_without_error(nb_path):
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()  # raises CellExecutionError if any cell fails
