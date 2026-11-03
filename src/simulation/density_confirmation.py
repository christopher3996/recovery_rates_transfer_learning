"""Fixed fresh-seed follow-up to the 80-epoch recovery-transfer panel."""
import hashlib
import json
import platform
import time
from importlib.metadata import version
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from experiments.up5.distribution import mixture_scores
from .revision import EXTRA, FTTransformerModel, _adapt, _draw, _split, _stream

REPS = tuple(range(5, 10))
SHIFTS = ('none', 'label')
SCHEMAS = ('equal', 'target_subset')
BUDGETS = (100, 500)
CAP = 160


def _model(x, shared, rep, cap):
    cat = [c for c in shared if not pd.api.types.is_numeric_dtype(x[c])]
    return FTTransformerModel(x, shared_num_cols=[c for c in shared if c not in cat],
        shared_cat_cols=cat, seed=_stream(rep, 5), epochs=cap, d_token=32, n_heads=4,
        n_blocks=2, ff_factor=4, dropout=0., learning_rate=.0001, weight_decay=0.,
        mdn_n_components=2, mdn_temperature=2.5, mdn_hidden_size=24, batch_size=64,
        use_mdn=True, use_interaction=True, use_scheduler=False, early_stopping=True,
        verbose=False)


def _ridge(fit, val, test):
    def encode(x):
        return pd.get_dummies(x.astype({c: object for c in x.select_dtypes('category')}), dtype=float)
    training = encode(fit[0])
    scale = StandardScaler().fit(training)
    def design(frame):
        return scale.transform(encode(frame).reindex(columns=training.columns, fill_value=0))
    x, v, e = scale.transform(training), design(val[0]), design(test)
    candidates = []
    for penalty in (1., 10., 100., 1000.):
        model = Ridge(alpha=penalty).fit(x, fit[1])
        rmse = np.sqrt(np.mean((model.predict(v)-val[1].to_numpy())**2))
        candidates.append((rmse, penalty, model))
    sd, penalty, model = min(candidates, key=lambda item: item[:2])
    mu = model.predict(e)[:, None]
    return dict(alpha=np.ones_like(mu), mu=mu, sigma=np.full_like(mu, max(sd, 1e-8))), penalty


def _folder(config, output, budgets=BUDGETS):
    src = Path(__file__).resolve().parents[1]
    files = [Path(__file__), Path(config), src/'simulation/revision.py',
             src/'experiments/up5/distribution.py', src/'models/ft_mdn_transformer.py',
             src/'data_generation/source_data.py']
    spec = dict(reps=REPS, shifts=SHIFTS, schemas=SCHEMAS, budgets=budgets,
                epochs=CAP, source_epochs=20, target_validation_fraction=.2,
                source_n=1000, assessment_n=1000, intensity=.5,
                python=platform.python_version(), threads=torch.get_num_threads(),
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                packages={p:version(p) for p in ('torch','numpy','pandas','scipy','scikit-learn','PyYAML')},
                sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    manifest = json.dumps(spec, indent=2, sort_keys=True)
    dest = Path(output)/hashlib.sha256(manifest.encode()).hexdigest()[:12]
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'protocol.json').write_text(manifest)
    return dest


def run(config_path, output, budgets=BUDGETS):
    """Cache one atomic forecast file per cell. Train each source once per rep/schema."""
    dest = _folder(config_path, output, budgets)
    config = yaml.load(Path(config_path).read_text(), Loader=yaml.UnsafeLoader)
    paths = [dest/f'{r}-{s}-{h}-{b}.npz' for r,s,h,b in product(REPS,SCHEMAS,SHIFTS,budgets)]
    cached = sum(p.is_file() for p in paths)
    print(f'Confirmation | {cached}/{len(paths)} paired cases cached | output {dest}', flush=True)
    completed = cached
    for rep, schema in product(REPS, SCHEMAS):
        source = None
        drop = EXTRA if schema == 'target_subset' else ()
        for shift, budget in product(SHIFTS, budgets):
            path = dest/f'{rep}-{schema}-{shift}-{budget}.npz'
            if path.exists():
                continue
            start = time.monotonic()
            label = f'[{completed+1:02}/{len(paths)}] rep {rep} | {shift} | {schema} | n={budget}'
            print(f'{label} | fitting FT scratch + transfer and Gaussian ridge', flush=True)
            fit, val, (xe, ye) = [_split(_draw(config,n,rep,role,shift),drop)
                for n,role in ((budget-budget//5,2),(budget//5,3),(1000,4))]
            shared = list(fit[0].columns)
            if source is None:
                xs, ys = _split(_draw(config,800,rep,0),())
                xv, yv = _split(_draw(config,200,rep,1),())
                source = _model(xs,shared,rep,20)
                source.train(xs,ys,xv,yv)
                source.select_best_model()
            arrays = dict(y=ye.to_numpy(), source_epoch=source.best_epoch)
            for arm in ('scratch','transfer'):
                target = _model(fit[0],shared,rep,CAP)
                if arm == 'transfer':
                    _adapt(source,target)
                target.train(*fit,*val)
                target.select_best_model()
                dist = target.predict_distribution(xe)
                np.testing.assert_allclose((dist['alpha']*dist['mu']).sum(1),dist['mean'],atol=1e-6)
                arrays.update({arm+'_'+k:dist[k] for k in ('alpha','mu','sigma')})
                arrays[arm+'_epoch'] = target.best_epoch
                arrays[arm+'_trained'] = target.max_train_epoch
            ridge, arrays['ridge_penalty'] = _ridge(fit,val,xe)
            arrays.update({'ridge_'+k:v for k,v in ridge.items()})
            with NamedTemporaryFile(dir=dest,suffix='.tmp',delete=False) as tmp:
                np.savez_compressed(tmp,**arrays)
            Path(tmp.name).replace(path)
            completed += 1
            print(f'{label} | saved | FT epochs selected/trained: '
                  f'scratch {arrays["scratch_epoch"]}/{arrays["scratch_trained"]}, '
                  f'transfer {arrays["transfer_epoch"]}/{arrays["transfer_trained"]} | '
                  f'source selected {source.best_epoch} | ridge penalty {arrays["ridge_penalty"]:g} | '
                  f'{time.monotonic()-start:.0f}s', flush=True)
    return dest


def score_panel(dest, budgets=BUDGETS):
    """Recompute absolute scores and paired gains for the specified budgets."""
    dest = Path(dest)
    rows = []
    for rep, schema, shift, budget in product(REPS, SCHEMAS, SHIFTS, budgets):
        with np.load(dest/f'{rep}-{schema}-{shift}-{budget}.npz') as z:
            y = z['y']
            for arm in ('scratch','transfer','ridge'):
                w,m,s = [z[arm+'_'+k].astype(float) for k in ('alpha','mu','sigma')]
                scores = mixture_scores(y,w,m,s)
                if not np.isfinite(list(scores.values())).all():
                    raise ValueError(f'Invalid scores in {rep}/{schema}/{shift}/{budget}/{arm}')
                pred = (w*m).sum(1)
                rows.append(dict(rep=rep,schema=schema,shift=shift,budget=budget,arm=arm,
                    R2=r2_score(y,pred), MAE=mean_absolute_error(y,pred), **scores,
                    selected_epoch=int(z[arm+'_epoch']) if arm!='ridge' else None,
                    epochs_run=int(z[arm+'_trained']) if arm!='ridge' else None))
    raw = pd.DataFrame(rows)
    raw.to_csv(dest/'absolute_scores.csv',index=False)
    metrics = ['CRPS','R2','MAE','NLL']
    indexed = raw.set_index(['rep','schema','shift','budget','arm'])[metrics]
    gain = indexed.xs('scratch',level='arm')-indexed.xs('transfer',level='arm')
    gain['R2'] *= -1
    gain.to_csv(dest/'paired_gains.csv')
    return gain


def summarize(dest):
    """Contrast the two original budgets after averaging settings within replication."""
    dest = Path(dest)
    gain = score_panel(dest)
    by_rep = gain.groupby(['rep','budget']).mean()
    contrast = by_rep.xs(100,level='budget')-by_rep.xs(500,level='budget')
    contrast.to_csv(dest/'scarcity_contrast_by_rep.csv')
    summary = pd.DataFrame({'mean':contrast.mean(),'MCSE':contrast.sem()})
    summary.to_csv(dest/'scarcity_contrast_summary.csv')
    return summary

def midpoint(config, original, destination, fit=False):
    """Add 250 labels to the archived confirmation protocol without refitting endpoints."""
    original, destination = Path(original), Path(destination)
    if fit:
        folder = _folder(config, destination, (250,))
    else:
        folders = list(destination.glob("*/protocol.json"))
        if not folders:
            print("250-label extension pending.")
            return
        if len(folders) != 1:
            raise ValueError("Multiple midpoint protocols: select a single archived run.")
        folder = folders[0].parent

    def comparable(path):
        protocol = json.loads((path/"protocol.json").read_text())
        protocol.pop("budgets")
        # The runner was generalized to accept the intermediate budget.
        protocol["sha256"].pop("density_confirmation.py")
        return protocol
    if comparable(original) != comparable(folder):
        raise ValueError("Use the same kernel, configuration and model as the endpoint panel.")
    if fit:
        run(config, destination, budgets=(250,))
    expected = list(product(REPS, SCHEMAS, SHIFTS))
    for r,s,h in expected:
        mid = folder/f"{r}-{s}-{h}-250.npz"
        if not mid.exists() and not fit:
            print("250-label extension incomplete. Resume the extension cell to finish.")
            return
        with np.load(mid) as middle:
            for budget in BUDGETS:
                with np.load(original/f"{r}-{s}-{h}-{budget}.npz") as endpoint:
                    np.testing.assert_array_equal(endpoint["y"], middle["y"])
                    np.testing.assert_equal(endpoint["source_epoch"], middle["source_epoch"])
    gains = pd.concat([score_panel(original), score_panel(folder,(250,))])
    per_rep = gains.groupby(["budget","rep"]).mean()
    summary = per_rep.groupby("budget").agg(["mean","sem"])
    summary.to_csv(folder/"three_budget_summary.csv")
    per_rep.to_csv(folder/"three_budget_by_rep.csv")
    absolute = pd.concat([pd.read_csv(f/"absolute_scores.csv") for f in (original,folder)])
    absolute.groupby(["budget","arm"])[["R2","MAE","CRPS","NLL"]].mean().to_csv(
        folder/"three_budget_absolute.csv")
    return summary
