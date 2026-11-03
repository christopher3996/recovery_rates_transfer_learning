"""Reconstruct the UP5-only panels from the archived analysis. Run in a subprocess."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulation.notebook_reporting import quiet_epochs
from simulation.paper_results import PAPER_STYLE


def descriptive(x, y, shared, folder):
    """Reconstruct the manuscript's training-set UP5 descriptive panels."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(y, bins=15, color="#f6d3ac", edgecolor="black", linewidth=0.8)
    ax.grid(False)
    ax.set(xlabel="Recovery rate", ylabel="Count")
    fig.savefig(folder / "rr_dist_up5.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    cols = [col for col in shared if col in x and pd.api.types.is_numeric_dtype(x[col])]
    if len(cols) != 31:
        raise ValueError(f"Expected 31 shared numeric columns, found {len(cols)}")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.grid(False)
    im = ax.imshow(x[cols].corr(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols)), cols, fontsize=7)
    ax.set_title("Correlation heatmap of shared UP5 predictors")
    fig.colorbar(im, ax=ax)
    fig.savefig(folder / "correlation_heatmap_up5.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def replay(repo, folder):
    """Only selected historical cells are used. The revision benchmark is separate."""
    repo, folder = Path(repo).resolve(), Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    archive = repo/"notebooks/archive"
    sys.path.insert(0, str(archive))
    import up5_helper
    from models.ft_mdn_transformer import FTTransformerModel
    cells = json.loads((archive/"up5_mdn_ablations.ipynb").read_text())["cells"]
    scripts = ["".join(c["source"]) for c in cells]
    x, test, y, truth = up5_helper.load_dataset(repo/"data/UP5", only_36=False)
    native, test_native = map(up5_helper.reconstruct_categories, (x, test))
    shared = [key.rstrip("_") for key in up5_helper.get_36_columns()]
    descriptive(native, y, shared, folder)
    ns = dict(np=np, pd=pd, plt=plt, FTTransformerModel=FTTransformerModel, folder=folder,
              up5_helper=up5_helper, X_train=x, X_test=test, y_train=y, y_test=truth,
              X_train_reconstructed=native, X_test_reconstructed=test_native,
              shared_columns=shared)
    def execute(i, replacements=()):
        source = scripts[i]
        for old, new in replacements:
            if old not in source:
                raise ValueError(f"Archived cell {i} no longer contains the expected expression: {old}")
            source = source.replace(old, new)
        exec(compile(source, f"{archive.name}/up5_mdn_ablations:cell{i}", "exec"), ns)

    # Historical density illustrations used native columns and validation checkpoints.
    execute(8, [('sample_df=X_train','sample_df=X_train_reconstructed'),
                ('X_train[all_cols]', 'X_train_reconstructed'),
                ('X_test[all_cols]', 'X_test_reconstructed')])
    execute(10)
    ns["plot_portfolio_density_side_by_side"](ns["model"],native,y,test_native,truth,
        y_grid=np.linspace(0,1.1,600),bins=30,title="FT–MDN–T: recovery distributions across UP5 bonds",
        savepath=folder/"up5_ft_mdn_density_side_by_side.pdf")
    execute(16, [('mixture+0.04', 'mixture'),
                 ('f"../figures/mdn_dist_sample_{idx}.pdf"', 'folder / f"mdn_dist_sample_{panel}.pdf"')])
    for panel, idx in enumerate((57,58,63,89), 1):
        ns["panel"] = panel
        ns["plot_mdn_for_sample_styled"](ns["model"],test_native,truth,idx,
                                         title=f"FT–MDN–T: conditional recovery density (UP5 bond {idx})")
        plt.close("all")

    execute(48)  # archived parameter dictionaries
    # Save aggregate histories, so plotting again does not repeat the 15-seed fits.
    for cell, variable in [(69,"results_ft_transfer"),(95,"results_with_dummies"),(96,"results_categorical")]:
        cache = folder/(variable+".json")
        if cache.is_file():
            ns[variable] = json.loads(cache.read_text())
        else:
            print(f"Historical UP5 | {variable} | 15 seeds", flush=True)
            execute(cell)
            histories = [{k: v[k] for k in ("val_r2","max_train_epoch","max_finetune_epoch")}
                         for v in ns[variable]]
            temp = cache.with_suffix(".tmp")
            temp.write_text(json.dumps(histories,default=lambda v: v.item(),allow_nan=False))
            temp.replace(cache)
    execute(70, [('f"ft_mdn_transformer_transfer_up5_only_heterogeneous_{up5_helper.timestamp()}.pdf"',
                  'folder/"ft_mdn_transformer_transfer_up5_only_heterogeneous.pdf"'),
                 ('Pretrain (36 Features of UP5)','Shared-input pretraining'),
                 ('Finetune (317 Features of UP5)','Full-input fine-tuning'),
                 ('FT-MDNTransformer: Heterogeneous Feature Transfer Learning with UP5 only',
                  'FT–MDN–T: schema expansion within UP5')])
    for name in ("with_dummies", "categorical"):
        ns["resulting_values_"+name] = [v["val_r2"] for v in ns["results_"+name]]
    execute(100, [('f"../figures/up5_dummy_vs_category_ft_mdn_comparison_{up5_helper.timestamp()}.pdf"',
                   'folder/"ft_mdn_transformer_cat_vs_dummy.pdf"'),
                  ('pd.DataFrame(stopping_points)','pd.DataFrame(stopping_points, columns=["epoch","r2","model"])'),
                  ('FT-MDNTransformer: Dummy vs. Categorical Features with IQR and Early Stopping (15 runs each)',
                   'FT–MDN–T: dummy encoding and categorical embeddings (15 runs each)')])
    plt.close("all")


if __name__ == "__main__":
    with quiet_epochs(), plt.rc_context(PAPER_STYLE):
        replay(*sys.argv[1:])
