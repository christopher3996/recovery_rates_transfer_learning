"""Fixed-schema native feature ablation for the UP5 revision notebook."""
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from .data import load_dataset
from models.ft_mdn_transformer import FTTransformerModel
from .distribution import mixture_scores


class PooledTokenMLP(nn.Module):
    """Replace the attention stack with a token-wise MLP and a masked mean."""
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width),
                                 nn.GELU(), nn.Dropout(.15), nn.Linear(2 * width, width))

    def forward(self, x, key_padding_mask):
        tokens = x[:, 1:]
        present = (~key_padding_mask[:, 1:]).unsqueeze(-1)
        pooled = ((tokens + self.net(tokens)) * present).sum(1) / present.sum(1).clamp_min(1)
        return torch.cat([pooled.unsqueeze(1), tokens], dim=1)


def native_frame(raw):
    fields = ('event_type_subcategory_sum', 'Industry_subgroup', 'Industry_sector',
              'Industry_group', 'domicile_country', 'exchange_country', 'seniorioty_adj',
              'currency', 'event_type')
    remaining = set(raw.columns)
    categories = {}
    audit = {}
    for field in fields:
        cols = sorted(c for c in remaining if c.startswith(field + '_'))
        if not cols:
            raise ValueError(f'Missing expected indicator family: {field}')
        remaining.difference_update(cols)
        indicators = raw[cols].to_numpy()
        if not np.isin(indicators, [0, 1]).all():
            raise ValueError(f'Nonbinary indicators in {field}')
        counts = indicators.sum(axis=1)
        # Treat the exact observed pattern as a category, including all-zero and
        # multi-active patterns. No assumption about a dropped reference level.
        categories[field] = ['|'.join(str(j) for j in np.flatnonzero(row)) or 'none'
                             for row in indicators]
        audit[field] = (int((counts == 0).sum()), int((counts > 1).sum()))
    result = raw[sorted(remaining)].astype(float).copy()
    for field, values in categories.items():
        result[field] = values
    return result, audit


def compare_native(data_dir, output_root):
    raw, test_raw, y, truth = load_dataset(data_dir, only_36=False)
    assert raw.shape == (1293, 317) and test_raw.shape == (432, 317)
    x, audit = native_frame(raw)
    test, test_audit = native_frame(test_raw)
    print('All-zero / multi-active counts by family:', audit, test_audit)
    fit, val = train_test_split(np.arange(len(y)), test_size=.2, random_state=42)
    numeric = x.select_dtypes(include=np.number).columns
    scaler = StandardScaler().fit(x.iloc[fit][numeric])
    x[numeric] = scaler.transform(x[numeric])
    test[numeric] = scaler.transform(test[numeric])
    print('Native token count:', x.shape[1], 'of which numeric:', len(numeric))
    folder = Path(output_root) / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    folder.mkdir(parents=True)
    rows = []
    for name in ('FT-Transformer-MDN', 'FT-pooled-MLP-MDN'):
        start = perf_counter()
        model = FTTransformerModel(x.iloc[fit], seed=42, epochs=80, d_token=56,
            n_heads=4, n_blocks=4, ff_factor=8, dropout=.15, learning_rate=.0003,
            weight_decay=.00005, mdn_n_components=2, mdn_temperature=2.5,
            mdn_hidden_size=24, batch_size=32, use_interaction=True, use_scheduler=False,
            verbose=False)
        if name == 'FT-pooled-MLP-MDN':
            model.model.blocks = nn.ModuleList([PooledTokenMLP(56)]).to(model.device)
            model._build_optimizer(head_lr_mult=3.0)
        model.train(x.iloc[fit], y[fit], x.iloc[val], y[val])
        model.select_best_model()
        dist = model.predict_distribution(test)
        a, m, s = (dist[k].astype(float) for k in ('alpha', 'mu', 'sigma'))
        pred = dist['mean']
        np.savez(folder / (name + '.npz'), **dist, excel_row=test.index+2)
        rows.append(dict(model=name, parameters=sum(p.numel() for p in model.model.parameters()),
                         epoch=model.best_epoch, val_R2=model.best_val_r2,
                         R2=r2_score(truth,pred), MAE=mean_absolute_error(truth,pred),
                         **mixture_scores(truth,a,m,s), seconds=perf_counter()-start))
        pd.DataFrame(rows).to_csv(folder/'comparison.csv', index=False)
    print('Saved:', folder/'comparison.csv')
    return pd.DataFrame(rows).set_index('model')
