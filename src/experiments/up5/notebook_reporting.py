"""Notebook presentation and subprocess entry points for the UP5 experiments."""
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import os
import subprocess
import sys

import pandas as pd
from simulation.table_reporting import present_table
from simulation.notebook_reporting import quiet_epochs, show_figure


def fit_step(tables, label, function, *args):
    print(f"{label} | running", flush=True)
    with (tables / "training.log").open("a") as log, redirect_stdout(log), redirect_stderr(log), quiet_epochs():
        result = function(*args)
    print(f"{label} | complete", flush=True)
    return result


def show_table(tables, filename):
    table = pd.read_csv(tables / filename, index_col=0)
    table.index = table.index.to_series().replace({
        "FT-MDN": "FT-MDN-T", "FT-Transformer-MDN": "FT-MDN-T",
        "FT-pooled-MLP-MDN": "FT-pooled-MLP-MDN"
    })
    present_table(table)


def build_figures(repo, folder, selection):
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / f"{selection}_training.log"
    print(f"UP5 {selection} | progress log: {log_path.relative_to(repo)}", flush=True)
    env = dict(os.environ, PYTHONPATH=str(repo / "src"), MPLBACKEND="Agg", PYTHONHASHSEED="0")
    with log_path.open("w") as log:
        subprocess.run([sys.executable, "-u", "-m", "experiments.up5.historical_figures",
                        str(repo), str(folder), selection], cwd=repo, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def show_figures(repo, folder, filenames, mode):
    for filename in filenames:
        path = folder / filename
        if not path.is_file() and mode == "review":
            path = repo / "results/up5/paper_figures" / filename
            if path.is_file():
                print(f"Saved manuscript figure: {filename}")
        if path.is_file():
            show_figure(path)
        else:
            print(f"Figure unavailable: {filename}. Rebuild calculates it.")


def complete_run(tables, folder, schema_run):
    required = ("scores.csv", "density-scores.csv", "tails.csv", "mdn-backbones.csv",
                "native-backbones.csv", "recovery-summary.csv")
    figures = ("rr_dist_up5.png", "correlation_heatmap_up5.png", "ft_mdn_transformer_cat_vs_dummy.pdf",
               "up5_ft_mdn_density_side_by_side.pdf", *[f"mdn_dist_sample_{i}.pdf" for i in range(1, 5)])
    missing = [str(tables / name) for name in required if not (tables / name).is_file()]
    missing += [str(folder / name) for name in figures if not (folder / name).is_file()]
    if schema_run is None or not (schema_run / "COMPLETE").is_file():
        missing.append("completed schema-expansion run")
    if missing:
        raise RuntimeError(f"Run all experiment sections before completing this rebuild. Missing: {missing}")
    (tables / "COMPLETE").touch()
