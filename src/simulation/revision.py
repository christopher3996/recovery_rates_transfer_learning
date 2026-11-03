"""Paired recovery-transfer experiment. Legacy generator and runner remain intact."""
import hashlib
import json
import platform
from importlib.metadata import version
from tempfile import NamedTemporaryFile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, r2_score

from data_generation.source_data import SimulationDataGeneration
from models.ft_mdn_transformer import FTTransformerModel

SHIFTS = ('none', 'covariate', 'conditional', 'label')
SCHEMAS = ('equal', 'source_subset', 'target_subset')
EXTRA = ('Feature_3', 'Feature_5', 'Feature_7')


def _stream(rep, role, batch=0):
    return int(np.random.SeedSequence([20261101, rep, role, batch]).generate_state(1)[0])


def _draw(config, count, rep, role, shift='none', intensity=.5):
    """Accept independent base rows using a bounded weight or transform their label."""
    rng = np.random.default_rng(_stream(rep, role+10))
    pieces = []
    remaining = count
    batch = 0
    while remaining:
        size = remaining if shift in ('none', 'conditional') else 3*remaining
        # Legacy generator uses global NumPy, including the configured custom feature.
        state = np.random.get_state()
        try:
            cfg = dict(config, n_samples=size, seed=_stream(rep, role, batch))
            frame = SimulationDataGeneration(cfg).generate_data().drop(columns='secured')
        finally:
            np.random.set_state(state)
        if shift in ('covariate', 'label'):
            key = frame.Feature_1 if shift == 'covariate' else frame.recovery_rate
            keep = rng.random(size) < np.where(key >= .4, 1., 1.-intensity)
            frame = frame.loc[keep]
        elif shift == 'conditional':
            # Bounded upward shift, positive on interior rows, leaves X intact.
            r = frame.recovery_rate
            frame['recovery_rate'] = r + intensity*(1-r)*.5
        pieces.append(frame.iloc[:remaining])
        remaining -= len(pieces[-1])
        batch += 1
    return pd.concat(pieces, ignore_index=True)


def _split(frame, drop):
    return frame.drop(columns=['recovery_rate', *drop]), frame.recovery_rate


def _adapt(source, target):
    """Copy only named shared tokenizers plus the source representation and MDN."""
    for attr in ('blocks','interaction','head'):
        getattr(target.model,attr).load_state_dict(getattr(source.model,attr).state_dict())
    with torch.no_grad():
        target.model.cls_token.copy_(source.model.cls_token)
        assert source.shared_num_cols == target.shared_num_cols
        target.model.shared_num_emb.load_state_dict(source.model.shared_num_emb.state_dict())
        for i, col in enumerate(source.shared_cat_cols):
            for value, new_id in target.cat_maps[col].items():
                old_id = source.cat_maps[col].get(value)
                if old_id is not None:
                    target.model.shared_cat_emb[i].weight[new_id].copy_(
                        source.model.shared_cat_emb[i].weight[old_id])


def run_cell(config_path, output, rep=0, shift='none', schema='equal', budget=100,
             source_n=1000, test_n=1000, epochs=20, intensity=.5, target_epochs=None):
    if shift not in SHIFTS or schema not in SCHEMAS or not 0 <= intensity < 1:
        raise ValueError('Unsupported shift, schema or intensity')
    if rep < 0 or min(budget, source_n) < 10 or test_n < 2 or epochs < 1:
        raise ValueError('Invalid replication or sample budget')
    target_epochs = epochs if target_epochs is None else target_epochs
    if target_epochs < 1:
        raise ValueError("Invalid target epoch cap")
    config_path = Path(config_path)
    spec = dict(rep=rep, shift=shift, schema=schema, budget=budget, source_n=source_n,
                test_n=test_n, epochs=epochs, intensity=intensity,
                target_epochs=target_epochs, environment=dict(python=platform.python_version(),
                    device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                    threads=torch.get_num_threads(), packages={name: version(name) for name in
                    ("numpy","pandas","scipy","scikit-learn","torch","xgboost","PyYAML")}),
                dependencies={file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in
                    (Path(__file__).resolve().parents[1]/"models"/"ft_mdn_transformer.py",
                     Path(__file__).resolve().parents[1]/"data_generation"/"source_data.py")},
                config_sha=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                runner_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    key = hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:20]
    folder = Path(output)
    folder.mkdir(parents=True,exist_ok=True)
    path = folder / (key+'.json')
    if path.exists():
        return json.loads(path.read_text())
    # Use only the repository-owned YAML: it contains two executable custom functions.
    config = yaml.load(config_path.read_text(), Loader=yaml.UnsafeLoader)
    ns, nt = source_n//5, budget//5
    frames = [_draw(config, source_n-ns, rep, 0), _draw(config, ns, rep, 1),
              _draw(config, budget-nt, rep, 2, shift, intensity),
              _draw(config, nt, rep, 3, shift, intensity),
              _draw(config, test_n, rep, 4, shift, intensity)]
    hashes = [hashlib.sha256(pd.util.hash_pandas_object(f,index=False).values.tobytes()).hexdigest()
              for f in frames]
    source_drop = EXTRA if schema=='source_subset' else ()
    target_drop = EXTRA if schema=='target_subset' else ()
    (xs,ys),(xv,yv),(xt,yt),(xw,yw),(xe,ye) = [
        _split(f,source_drop if i<2 else target_drop) for i,f in enumerate(frames)]
    shared = [c for c in xs if c in xt]
    cat = [c for c in shared if not pd.api.types.is_numeric_dtype(xs[c])]
    settings = dict(shared_num_cols=[c for c in shared if c not in cat], shared_cat_cols=cat,
                    seed=_stream(rep, 5), epochs=epochs, d_token=32, n_heads=4, n_blocks=2,
                    ff_factor=4, dropout=0., learning_rate=.0001, weight_decay=0.,
                    mdn_n_components=2, mdn_temperature=2.5, mdn_hidden_size=24,
                    batch_size=64, use_mdn=True, use_interaction=True, use_scheduler=False,
                    early_stopping=True, verbose=False)
    source = FTTransformerModel(xs, **settings)
    source.train(xs,ys,xv,yv)
    source.select_best_model()
    result = dict(spec=spec, data_hashes=hashes, source_epoch=source.best_epoch,
                  shared=shared, n_fit=len(yt), n_validation=len(yw))
    for mode in ('scratch','transfer'):
        target = FTTransformerModel(xt, **dict(settings, epochs=target_epochs))
        if mode=='transfer':
            _adapt(source,target)
        target.train(xt,yt,xw,yw)
        target.select_best_model()
        pred = target.predict(xe)
        result[mode] = dict(r2=float(r2_score(ye,pred)), mae=float(mean_absolute_error(ye,pred)),
                            epoch=target.best_epoch)
    # Tree baseline has identical target label access and can use target-only columns.
    from xgboost import XGBRegressor
    def design(train, other):
        encoded = pd.get_dummies(train, dtype=float)
        return encoded, pd.get_dummies(other, dtype=float).reindex(columns=encoded.columns, fill_value=0)
    def tree():
        return XGBRegressor(n_estimators=50, max_depth=6, learning_rate=.05,
                            n_jobs=1, random_state=_stream(rep, 5))
    xfit, xtest = design(xt, xe)
    source_fit, target_shared = design(xs[shared], pd.concat([xt[shared], xe[shared]]))
    base = tree().fit(source_fit, ys).predict(target_shared)
    tree_preds = {
        'xgb_scratch': tree().fit(xfit, yt).predict(xtest),
        'xgb_residual': base[len(yt):] + tree().fit(xfit, yt.to_numpy()-base[:len(yt)]).predict(xtest)
    }
    for name, values in tree_preds.items():
        result[name] = dict(r2=float(r2_score(ye, values)), mae=float(mean_absolute_error(ye, values)))
    result['gain_r2'] = result['transfer']['r2']-result['scratch']['r2']
    result['gain_mae'] = result['scratch']['mae']-result['transfer']['mae']
    # Per-cell cache for sequential notebook runs; incomplete fits are never counted.
    with NamedTemporaryFile(mode='w', dir=folder, suffix='.tmp', delete=False) as temp:
        temp.write(json.dumps(result,allow_nan=False))
    Path(temp.name).replace(path)
    return result


def run_panel(config_path, output, reps=5):
    """Fixed primary panel. Raise on a failed fit; rerun to resume completed cells."""
    if reps < 2:
        raise ValueError('At least two independent replications required for MCSE')
    rows = []
    for rep in range(reps):
        for shift in SHIFTS:
            for schema in SCHEMAS:
                for budget in (100,500):
                    r = run_cell(config_path, output, rep, shift, schema, budget)
                    rows.append(dict(rep=rep, shift=shift, schema=schema, budget=budget,
                                     gain_r2=r['gain_r2'], gain_mae=r['gain_mae'],
                                     scratch_r2=r['scratch']['r2'], transfer_r2=r['transfer']['r2'],
                                     xgb_r2=r['xgb_scratch']['r2'], residual_r2=r['xgb_residual']['r2']))
    raw = pd.DataFrame(rows)
    summary = raw.groupby(['shift','schema','budget']).agg(
        n=('rep','size'), mean_gain_r2=('gain_r2','mean'), mcse_r2=('gain_r2','sem'),
        mean_gain_mae=('gain_mae','mean'), mcse_mae=('gain_mae','sem'),
        ft_scratch=('scratch_r2','mean'), ft_transfer=('transfer_r2','mean'),
        xgb_scratch=('xgb_r2','mean'), xgb_residual=('residual_r2','mean')).reset_index()
    raw.to_csv(Path(output)/'paired_runs.csv', index=False)
    summary.to_csv(Path(output)/'paired_summary.csv', index=False)
    return summary
