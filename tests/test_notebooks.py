# tests/test_notebooks.py
import sys
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

# Récupère tous les notebooks dans /notebooks
NOTEBOOKS = sorted(Path("notebooks").glob("[0-9]*.ipynb"))


@pytest.mark.parametrize("notebook_path", NOTEBOOKS)
def test_notebook_execution(notebook_path: Path) -> None:
    assert notebook_path.exists(), f"Notebook {notebook_path} introuvable."

    nb = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_kwargs={"executable": sys.executable},
        allow_errors=True,
        resources={"metadata": {"path": notebook_path.parent}},
    )

    try:
        client.execute()
    except Exception as e:
        raise AssertionError(f"Echec dans le notebook {notebook_path}: {e}")


if __name__ == "__main__":
    for notebook in NOTEBOOKS:
        # test_notebook_execution(notebook)
        nb = nbformat.read(notebook, as_version=4)
        client = NotebookClient(
            nb,
            timeout=600,
            kernel_kwargs={"executable": sys.executable},
            allow_errors=True,
            resources={"metadata": {"path": notebook.parent}},
        )

        client.execute()  # exécute toutes les cellules

        # Affiche les outputs
        for i, cell in enumerate(nb.cells):
            if "outputs" in cell:
                for output in cell.outputs:
                    if output.output_type == "stream":
                        print(f"Cell {i} stdout: {output.text}")
                    elif output.output_type == "error":
                        print(f"Cell {i} error: {output.ename} {output.evalue}")
