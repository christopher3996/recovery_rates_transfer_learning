"""UP5 backbone comparison with fixed settings and saved FT assessment forecasts."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import load_dataset
from models.ft_mdn_transformer import MDNHead, mdn_gaussian_nll
from .distribution import mixture_scores


def compare_backbones(data_dir, run_dir):
    run = Path(run_dir)
    x, test, y, observed = load_dataset(data_dir, only_36=False)
    assert x.shape == (1293, 317) and test.shape == (432, 317)
    columns = sorted(x.columns)
    fit, val = train_test_split(np.arange(len(y)), test_size=.2, random_state=42)
    scaler = StandardScaler().fit(x.iloc[fit][columns])
    def inputs(frame):
        return torch.tensor(scaler.transform(frame[columns]), dtype=torch.float32)

    xfit, xval = inputs(x.iloc[fit]), inputs(x.iloc[val])
    labels = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    torch.manual_seed(42)
    # Same hidden widths, activation and dropout as the UP5 point MLP.
    trunk = nn.Sequential(nn.Linear(len(columns), 128), nn.ReLU(), nn.Dropout(.5),
                          nn.Linear(128, 64), nn.ReLU(), nn.Dropout(.5))
    head = MDNHead(64, n_components=2, hidden_size=24, temperature=2.5)
    model = nn.Sequential(trunk, head)
    optimizer = torch.optim.AdamW([{'params': trunk.parameters(), 'lr': .0003},
                                   {'params': head.parameters(), 'lr': .0009}], weight_decay=.00005)
    loader = DataLoader(TensorDataset(xfit, labels[fit]), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    best = -np.inf
    state = None
    for epoch in range(1, 81):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = mdn_gaussian_nll(*model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            a, m, s = model(xval)
            score = r2_score(y[val], (a*m).sum(1).numpy())
            if score > best + .0001:
                best, best_epoch = score, epoch
                validation_nll = mdn_gaussian_nll(a, m, s, labels[val]).item()
                state = deepcopy(model.state_dict())
        if state is None:
            raise ValueError('Nonfinite validation score')
        if epoch-best_epoch >= 15:
            break
    model.load_state_dict(state)
    with torch.no_grad():
        params = [t.numpy().astype(float) for t in model(inputs(test))]
    ft_rows = pd.read_csv(run/'FT-MDN-predictions.csv')
    assert np.array_equal(ft_rows.excel_row, test.index+2) and np.allclose(observed, ft_rows.observed)
    with np.load(run/'FT-MDN-distribution.npz') as saved:
        ft = [saved[k].astype(float) for k in ('alpha','mu','sigma')]
    assert np.allclose((ft[0]*ft[1]).sum(1), ft_rows.predicted)
    rows = {}
    for name, (a,m,s) in [('FT-MDN', ft), ('MLP-MDN', params)]:
        pred = (a*m).sum(1)
        rows[name] = dict(R2=r2_score(observed,pred), MAE=mean_absolute_error(observed,pred),
                          **mixture_scores(observed,a,m,s))
    table = pd.DataFrame(rows).T
    np.savez(run/'MLP-MDN-distribution.npz', alpha=params[0], mu=params[1], sigma=params[2],
             excel_row=test.index+2, best_epoch=best_epoch, val_r2=best, val_nll=validation_nll)
    table.to_csv(run/'mdn-backbones.csv')
    print(f'MLP checkpoint epoch {best_epoch}, validation R2 {best:.4f}, NLL {validation_nll:.4f}')
    return table
