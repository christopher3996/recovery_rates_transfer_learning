"""Within-UP5 schema expansion using the current tokenizer and transfer routine."""
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import get_36_columns, load_dataset
from .native_backbones import native_frame
from models.ft_mdn_transformer import FTTransformerModel
from simulation.revision import _adapt
from simulation.notebook_reporting import quiet_epochs

SETTINGS = dict(d_token=56, n_heads=4, n_blocks=4, ff_factor=8, dropout=.15,
                learning_rate=.0003, weight_decay=.00005, mdn_n_components=2,
                mdn_temperature=2.5, mdn_hidden_size=24, batch_size=32,
                use_interaction=True, use_scheduler=False, verbose=False,
                early_stopping=True)


def prepare(data_dir):
    raw, test_raw, y, truth = load_dataset(data_dir, only_36=False)
    x, audit = native_frame(raw)
    test, test_audit = native_frame(test_raw)
    shared = [name.removesuffix("_") for name in get_36_columns()]
    if len(shared) != 36 or not set(shared).issubset(x.columns):
        raise ValueError("The declared 36 shared predictors must be present in UP5.")
    if list(x.columns) != list(test.columns) or x.shape[1] != 164:
        raise ValueError("Expected matching schemas with 164 native predictors.")
    fit, val = train_test_split(np.arange(len(y)), test_size=.2, random_state=42)
    numeric = list(x.select_dtypes(include=np.number))
    scaler = StandardScaler().fit(x.iloc[fit][numeric])
    x[numeric] = scaler.transform(x[numeric])
    test[numeric] = scaler.transform(test[numeric])
    return x, test, y, truth, fit, val, shared, dict(
        training=audit, evaluation=test_audit, numeric=numeric,
        center=scaler.mean_.tolist(), scale=scaler.scale_.tolist())


def shared_prediction(model, frame):
    """Check copied weights with added tokens absent and masked."""
    encoded = model._encode_categories(frame)
    previous = model.model.mask_task
    model.model.mask_task = True
    model.model.eval()
    try:
        with torch.no_grad():
            pred, _ = model.model(
                model._to_tensor_cols(encoded, model.shared_num_cols, torch.float32),
                None,
                model._to_tensor_cols(encoded, model.shared_cat_cols, torch.int64),
                None)
        return pred.cpu().numpy().ravel()
    finally:
        model.model.mask_task = previous


def transfer_checked(source, target, frame):
    """Verify schema ordering, copied weights and preservation of the shared prediction."""
    if source.shared_num_cols != target.shared_num_cols or source.shared_cat_cols != target.shared_cat_cols:
        raise ValueError("Source and target shared-feature ordering differs.")
    _adapt(source, target)
    for name in ("blocks", "interaction", "head", "shared_num_emb"):
        before = getattr(source.model, name).state_dict()
        after = getattr(target.model, name).state_dict()
        if any(not torch.equal(value, after[key]) for key, value in before.items()):
            raise AssertionError(f"Transfer did not preserve {name}.")
    for i, col in enumerate(source.shared_cat_cols):
        for category, target_id in target.cat_maps[col].items():
            source_id = source.cat_maps[col].get(category)
            if source_id is not None:
                torch.testing.assert_close(target.model.shared_cat_emb[i].weight[target_id],
                                           source.model.shared_cat_emb[i].weight[source_id],
                                           rtol=0, atol=0)
    before = source.predict(frame[source.num_cols + source.cat_cols])
    after = shared_prediction(target, frame)
    np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-6)
    return float(np.max(np.abs(before - after)))


def _save(path, content):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(content, indent=2, allow_nan=False))
    tmp.replace(path)


def run(data_dir, output_root, seeds=tuple(range(10)), epochs=80):
    """Resume matching completed seeds. Fixed phase lengths, training-only validation."""
    seeds = tuple(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or epochs < 1:
        raise ValueError("Use at least two distinct seeds and a positive epoch budget.")
    data_dir, output_root = Path(data_dir), Path(output_root)
    repo = Path(__file__).resolve().parents[3]
    files = [Path(__file__), repo/"src/models/ft_mdn_transformer.py",
             repo/"src/simulation/revision.py", repo/"src/experiments/up5/native_backbones.py",
             repo/"src/experiments/up5/data.py"]
    files += [data_dir/f"{part}_{kind}2.xlsx" for part in ("train", "test") for kind in ("features", "labels")]
    protocol = dict(settings=SETTINGS, epochs=epochs, seeds=list(seeds), split_seed=42,
                    threads=1, python=platform.python_version(), numpy=np.__version__,
                    pandas=pd.__version__, torch=torch.__version__,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    selection="validation R2", freezing="none", early_stopping="disabled by patience > epochs")
    folder = output_root / hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    _save(folder/"protocol.json", protocol)
    x, test, y, truth, fit, val, shared, processing = prepare(data_dir)
    shared_num = [c for c in shared if pd.api.types.is_numeric_dtype(x[c])]
    shared_cat = [c for c in shared if c not in shared_num]
    _save(folder/"inputs.json", dict(shared=shared, full=list(x.columns), fit_rows=fit.tolist(),
                                    validation_rows=val.tolist(), preprocessing=processing))
    print(f"Schema expansion | {len(shared)} → {x.shape[1]} native predictors | {len(seeds)} seeds | "
          f"{epochs} epochs per phase | fit {len(fit)}, validation {len(val)}, evaluation {len(test)}", flush=True)
    print(f"Results: {folder}", flush=True)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    rows = []
    try:
        for pos, seed in enumerate(seeds, 1):
            path = folder/f"seed_{seed}.json"
            if path.exists():
                row = json.loads(path.read_text())
                print(f"[{pos}/{len(seeds)}] seed {seed} | saved run loaded", flush=True)
            else:
                start = perf_counter()
                settings = dict(SETTINGS, seed=seed, epochs=epochs,
                                shared_num_cols=shared_num, shared_cat_cols=shared_cat)
                source = FTTransformerModel(x.iloc[fit][shared], **settings)
                source.patience = epochs + 1  # Record full curves, still retain validation-best weights.
                pre_start = r2_score(y[val], source.predict(x.iloc[val][shared]))
                print(f"[{pos}/{len(seeds)}] seed {seed} | pretraining | initial validation R² {pre_start:.3f}", flush=True)
                with quiet_epochs():
                    source.train(x.iloc[fit][shared], y[fit], x.iloc[val][shared], y[val])
                pre_curve = list(source.val_r2)
                source.select_best_model()
                pre_selected = r2_score(y[val], source.predict(x.iloc[val][shared]))
                print(f"[{pos}/{len(seeds)}] pretraining complete | last {pre_curve[-1]:.3f} | "
                      f"selected {pre_selected:.3f} at epoch {source.best_epoch}", flush=True)
                target = FTTransformerModel(x.iloc[fit], **settings)
                target.patience = epochs + 1
                error = transfer_checked(source, target, x.iloc[val])
                fine_start = r2_score(y[val], target.predict(x.iloc[val]))
                print(f"[{pos}/{len(seeds)}] fine-tuning | shared prediction max difference {error:.2g} | "
                      f"all features, before training R² {fine_start:.3f}", flush=True)
                with quiet_epochs():
                    target.train(x.iloc[fit], y[fit], x.iloc[val], y[val])
                fine_curve = list(target.val_r2)
                target.select_best_model()
                row = dict(seed=seed, pre_start=pre_start, pre_last=pre_curve[-1],
                           pre_selected=pre_selected, pre_epoch=source.best_epoch,
                           fine_start=fine_start, fine_first=fine_curve[0], fine_last=fine_curve[-1],
                           fine_selected=float(r2_score(y[val], target.predict(x.iloc[val]))),
                           fine_epoch=target.best_epoch, shared_prediction_max_error=error,
                           pre_assessment=float(r2_score(truth, source.predict(test[shared]))),
                           fine_assessment=float(r2_score(truth, target.predict(test))),
                           pre_curve=pre_curve, fine_curve=fine_curve, seconds=perf_counter()-start)
                if len(pre_curve) != epochs or len(fine_curve) != epochs:
                    raise AssertionError("Both phases must contain the specified number of epochs.")
                _save(path, row)
                print(f"[{pos}/{len(seeds)}] finished | fine first {row['fine_first']:.3f}, "
                      f"last {row['fine_last']:.3f}, selected {row['fine_selected']:.3f} | "
                      f"{row['seconds']/60:.1f} min", flush=True)
            rows.append({k: v for k, v in row.items() if not k.endswith("_curve")})
            pd.DataFrame(rows).to_csv(folder/"summary.csv", index=False)
    finally:
        torch.set_num_threads(previous_threads)
    (folder/"COMPLETE").touch()
    return folder


def plot(folder):
    """Plot epochs 1 onward for every seed; pre-update diagnostics remain in the tables."""
    import matplotlib.pyplot as plt
    from simulation.paper_results import BLUE, ORANGE, PAPER_STYLE
    folder = Path(folder)
    protocol = json.loads((folder/"protocol.json").read_text())
    rows = [json.loads((folder/f"seed_{seed}.json").read_text()) for seed in protocol["seeds"]]
    epochs = np.arange(1, protocol["epochs"] + 1)
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        for phase, color, label in [
            ("pre", BLUE, "Pretraining (shared predictors)"),
            ("fine", ORANGE, "Fine-tuning (expanded schema)")]:
            values = np.asarray([r[f"{phase}_curve"] for r in rows])
            ax.plot(epochs, values.mean(0), color=color, lw=2, label=label)
            ax.fill_between(epochs, *np.quantile(values, [.25, .75], axis=0), color=color, alpha=.2)
        ax.set(xlabel="Epoch within training phase",
               ylabel="Validation R²", title=f"FT–MDN–T: schema expansion within UP5 ({len(rows)} seeds)")
        ax.legend(loc="lower right")
        filename = folder/"ft_mdn_transformer_schema_expansion.pdf"
        fig.savefig(filename, bbox_inches="tight", metadata={"CreationDate": None})
        plt.close(fig)
    return filename
