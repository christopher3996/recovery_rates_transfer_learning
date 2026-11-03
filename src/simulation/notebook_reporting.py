"""Console and figure presentation for the simulation notebook (no model settings)."""
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from unittest.mock import patch
import os

from IPython.display import FileLink, Image, display
from tqdm.std import tqdm

from simulation.revision import SCHEMAS, SHIFTS, run_cell


class _QuietEpochs(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **dict(kwargs, disable=True))

    def write(self, *args, **kwargs):
        pass  # Early-stop status is reported with the selected and actual epochs.


@contextmanager
def quiet_epochs():
    """Suppress only the FT epoch widget and its progress-bar messages."""
    with patch('models.ft_mdn_transformer.tqdm', _QuietEpochs):
        yield


def rebuild_panel(config, folder, cap):
    shifts = SHIFTS if cap == 20 else ('none', 'label')
    schemas = SCHEMAS if cap == 20 else ('equal', 'target_subset')
    cases = list(product(range(5), shifts, schemas, (100, 500)))
    with quiet_epochs():
        for i, (rep, shift, schema, budget) in enumerate(cases, 1):
            row = run_cell(config, folder, rep, shift, schema, budget, target_epochs=cap)
            print(f'[{i:03}/{len(cases)}] rep {rep} | {shift} | {schema} | n={budget} | '
                  f'FT selected epochs: source {row["source_epoch"]}, scratch {row["scratch"]["epoch"]}, '
                  f'transfer {row["transfer"]["epoch"]} | XGB scratch and residual scored', flush=True)


def show_figure(path):
    """Display the actual publication PDF as a linked file, with a portable optional preview."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    display(FileLink(os.path.relpath(path)))
    try:
        import fitz
    except ImportError:
        return
    with fitz.open(path) as pdf:
        page = pdf[0]
        scale = 850/page.rect.width
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False)
        display(Image(data=pixmap.tobytes("png")))
