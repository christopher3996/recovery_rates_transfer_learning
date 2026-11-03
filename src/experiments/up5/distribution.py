"""Score the saved UP5 mixtures and a conditional Gaussian ridge benchmark."""
import csv
import math
from pathlib import Path

import numpy as np


def _read(path):
    from openpyxl import load_workbook
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = iter(book.active.values)
        return next(rows), np.asarray(list(rows), dtype=float)
    finally:
        book.close()


def mixture_scores(y, w, m, s):
    def cdf(z):
        return (1 + np.vectorize(math.erf)(z / np.sqrt(2))) / 2

    def absolute(z, sd):
        return sd * np.sqrt(2 / np.pi) * np.exp(-0.5*(z/sd)**2) + z*(2*cdf(z/sd)-1)

    assert np.all(s > 0) and np.allclose(w.sum(1), 1)
    err = y[:, None] - m
    first = (w * absolute(err, s)).sum(1)
    second = (w[:, :, None]*w[:, None, :]*absolute(m[:, :, None]-m[:, None, :],
              np.hypot(s[:, :, None], s[:, None, :]))).sum((1, 2))
    logpdf = np.log(w) - np.log(s) - 0.5*np.log(2*np.pi) - 0.5*(err/s)**2
    pit = (w*cdf(err/s)).sum(1)
    def quantile(q):
        lo, hi = (m-10*s).min(1), (m+10*s).max(1)
        for _ in range(48):
            mid = (lo+hi)/2
            below = (w*cdf((mid[:, None]-m)/s)).sum(1) < q
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        return (lo+hi)/2
    return dict(CRPS=(first-second/2).mean(), NLL=-np.logaddexp.reduce(logpdf, axis=1).mean(),
                coverage80=((pit >= .1) & (pit <= .9)).mean(), width80=(quantile(.9)-quantile(.1)).mean())


def compare(data_dir, run_dir):
    data, run = Path(data_dir), Path(run_dir)
    h, x = _read(data/'train_features2.xlsx')
    ht, xt = _read(data/'test_features2.xlsx')
    assert h == ht
    y = _read(data/'train_labels2.xlsx')[1].ravel()
    yt = _read(data/'test_labels2.xlsx')[1].ravel()
    with (run/'FT-MDN-predictions.csv').open() as f:
        records = list(csv.DictReader(f))
    assert [int(r['excel_row']) for r in records] == list(range(2, len(yt)+2))
    assert np.allclose(yt, [float(r['observed']) for r in records])
    with np.load(run/'FT-MDN-distribution.npz') as dist:
        w, m, s = (dist[k].astype(float) for k in ('alpha','mu','sigma'))
    assert np.allclose((w*m).sum(1), [float(r['predicted']) for r in records])
    order = np.random.RandomState(42).permutation(len(y))
    val, fit = order[:259], order[259:]
    center, scale = x[fit].mean(0), x[fit].std(0)
    scale[scale == 0] = 1
    x, xt = (x-center)/scale, (xt-center)/scale
    u, d, v = np.linalg.svd(x[fit], full_matrices=False)
    base = y[fit].mean()
    projected = u.T @ (y[fit]-base)
    candidates = []
    for penalty in (1, 10, 100, 1000):
        beta = v.T @ (d/(d*d+penalty)*projected)
        rmse = np.mean((y[val]-base-x[val]@beta)**2)**.5
        candidates.append((rmse, penalty, beta))
    sd, penalty, beta = min(candidates, key=lambda row: row[0])
    mu = (base+xt@beta)[:, None]
    print('Gaussian ridge: validation-selected penalty', penalty, 'residual scale', sd)
    result = {'FT-MDN': mixture_scores(yt,w,m,s),
              'Gaussian ridge': mixture_scores(yt,np.ones_like(mu),mu,np.full_like(mu,sd))}
    t = y[fit]
    result['Training empirical'] = dict(CRPS=np.abs(yt[:,None]-t).mean()-.5*np.abs(t[:,None]-t).mean())
    return result
