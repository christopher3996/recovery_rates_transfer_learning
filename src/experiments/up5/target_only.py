"""Target-only UP5 comparison. Run from the revision cells in the UP5 notebook."""
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import load_dataset


def run_benchmark(data_dir, output_dir, tabpfn_kwargs=None):
    from models.ft_mdn_transformer import FTTransformerModel
    from models.pytorch_model import PyTorchModel
    from xgboost import XGBRegressor

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test, y, truth = load_dataset(data_dir, only_36=False)
    assert train.shape == (1293, 317) and test.shape == (432, 317)
    columns = sorted(train.columns)
    train, test = train[columns].astype(float), test[columns].astype(float)
    fit, val = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(train.iloc[fit])
    def scaled(frame):
        return pd.DataFrame(scaler.transform(frame), columns=columns, index=frame.index)

    scores = []
    for name in ['Training mean', 'XGBoost', 'MLP', 'FT-MDN', 'TabPFN']:
        start = perf_counter()
        if name == 'Training mean':
            pred = np.repeat(y[fit].mean(), len(test))
        elif name == 'XGBoost':
            model = XGBRegressor(random_state=42)
            model.fit(train.iloc[fit], y[fit])
            pred = model.predict(test)
        elif name == 'TabPFN':
            from models.baselines.tabpfn_model import TabPFNRegressorModel
            model = TabPFNRegressorModel(random_state=42, **(tabpfn_kwargs or {}))
            model.train(train.iloc[fit], y[fit])
            pred = model.predict(test)
            print('TabPFN package:', version('tabpfn'), 'configuration:', model.model.get_params())
        else:
            xfit, xval = scaled(train.iloc[fit]), scaled(train.iloc[val])
            if name == 'MLP':
                model = PyTorchModel(hidden_layers=[128, 64], epochs=40, seed=42)
            else:
                model = FTTransformerModel(sample_df=xfit, seed=42, epochs=80, d_token=56,
                    n_heads=4, n_blocks=4, ff_factor=8, dropout=0.15, learning_rate=3e-4,
                    weight_decay=5e-5, mdn_n_components=2, mdn_temperature=2.5,
                    mdn_hidden_size=24, batch_size=32, use_interaction=True, use_scheduler=False)
            model.train(xfit, pd.Series(y[fit]), xval, pd.Series(y[val]))
            if name == 'FT-MDN':
                model.select_best_model()
                dist = model.predict_distribution(scaled(test))
                np.savez(output_dir / 'FT-MDN-distribution.npz', **dist)
                pred = dist['mean']
            else:
                pred = model.predict(scaled(test))
        pred = np.asarray(pred).reshape(-1)
        scores.append(dict(model=name, r2=r2_score(truth, pred), mae=mean_absolute_error(truth, pred),
                           seconds=perf_counter()-start, n_fit=len(fit), n_validation=len(val)))
        pd.DataFrame(dict(excel_row=test.index+2, observed=truth, predicted=pred)).to_csv(
            output_dir / (name + '-predictions.csv'), index=False)
        pd.DataFrame(scores).to_csv(output_dir / 'scores.csv', index=False)
    return pd.DataFrame(scores)
