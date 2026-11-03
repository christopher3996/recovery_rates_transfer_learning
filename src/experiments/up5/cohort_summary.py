"""Aggregate UP5 descriptive and tail summaries."""
from pathlib import Path

import numpy as np

from .distribution import _read


def tail_probabilities(run_dir):
    """Equal-weight marginal tail probabilities on the same fixed UP5 rows."""
    import pandas as pd
    from scipy.special import ndtr

    run = Path(run_dir)
    reference = pd.read_csv(run / 'FT-MDN-predictions.csv')
    with np.load(run / 'FT-MDN-distribution.npz') as saved:
        w, m, s = (saved[key].astype(float) for key in ('alpha', 'mu', 'sigma'))
    np.testing.assert_allclose((w * m).sum(1), reference.predicted, atol=1e-6)
    return pd.DataFrame({
        'Below zero (%)': [100 * (w * ndtr(-m / s)).sum(1).mean(), 100 * reference.observed.lt(0).mean()],
        'Above par (%)': [100 * (w * ndtr((m - 1) / s)).sum(1).mean(), 100 * reference.observed.gt(1).mean()],
    }, index=['Mean forecast probability', 'Observed frequency'])



def recovery_summary(data_dir):
    """UP5 counts and distribution moments before rounding for the paper."""
    import pandas as pd

    train, test = [pd.Series(_read(Path(data_dir) / f'{part}_labels2.xlsx')[1].ravel())
                   for part in ('train', 'test')]
    return pd.DataFrame({name: values.agg(['count', 'mean', 'std', 'skew'])
                         for name, values in [('Training', train), ('Evaluation', test),
                                              ('Combined', pd.concat([train, test]))]}).T
