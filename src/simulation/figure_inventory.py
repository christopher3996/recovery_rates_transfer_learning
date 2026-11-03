"""Plot-only entry points for the simulation notebooks. No model fitting."""
import ast
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml

from simulation.paper_results import BLUE, ORANGE, FT_HIGHLIGHT, PAPER_STYLE, SCHEMA_NAMES, SHIFT_LABEL
from simulation.revision import _draw, SHIFTS, SCHEMAS


@mpl.rc_context(rc=PAPER_STYLE)
def plot_generator(config_path, destination):
    """Illustrative independent domain draws, separate from assessment outcomes."""
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    cfg = yaml.load(Path(config_path).read_text(), Loader=yaml.UnsafeLoader)
    source = _draw(cfg, 1000, 0, 0)
    fig, axes = plt.subplots(3,4,figsize=(12,7.5),constrained_layout=True,sharey='row')
    for j, shift in enumerate(SHIFTS):
        target = _draw(cfg, 1000, 0, 4, shift)
        for i,col in enumerate(('recovery_rate','Feature_1')):
            bins = np.histogram_bin_edges(pd.concat([source[col],target[col]]),bins=24)
            for df,color,label in [(source,BLUE,'Source'),(target,ORANGE,'Target')]:
                axes[i,j].hist(df[col],bins=bins,density=True,histtype='step',lw=1.5,color=color,label=label)
            axes[i,j].set_xlabel('Recovery rate' if i==0 else 'Shared feature 1')
        nums = [c for c in source.select_dtypes('number') if c!='recovery_rate']
        pos=np.arange(len(nums))
        for df,color,dx in [(source,BLUE,-.2),(target,ORANGE,.2)]:
            axes[2,j].bar(pos+dx,df[nums].corrwith(df.recovery_rate),.4,color=color)
        axes[2,j].set_xticks(pos,[str(i+1) for i in pos[:-1]]+['Int.'])
        axes[2,j].set_xlabel('Numeric predictor')
        axes[0,j].set_title(SHIFT_LABEL[shift])
    axes[0,0].set_ylabel('Density')
    axes[1,0].set_ylabel('Density')
    axes[2,0].set_ylabel('Correlation with recovery')
    axes[0,-1].legend()
    fig.savefig(dest/'generator_diagnostics.pdf',metadata={'CreationDate':None})
    plt.close(fig)


@mpl.rc_context(rc=PAPER_STYLE)
def plot_augmentation(final):
    final = Path(final)
    augmented = pd.read_csv(final/'augmentation_runs.csv')
    primary = pd.read_csv(final/'replications.csv')
    keys=['rep','shift','schema','budget']
    raw=primary.merge(augmented,on=keys,validate='one_to_one')
    if len(raw)!=120:
        raise ValueError('Expected all 120 matched augmentation and primary runs')
    fig,axes=plt.subplots(2,3,figsize=(10,5),sharey=True,constrained_layout=True)
    for row,budget in enumerate((100,500)):
        for ax,schema,title in zip(axes[row],SCHEMAS,SCHEMA_NAMES):
            g=raw.query('budget==@budget and schema==@schema').groupby('shift')
            for dx,col,color,label in [(-.09,'transfer_r2',BLUE,'FT–MDN–T (pretrained)'),(.09,'augmented_r2',ORANGE,'Augmented ridge')]:
                series=g[col]
                ax.errorbar(np.arange(4)+dx,series.mean().loc[list(SHIFTS)],yerr=series.sem().loc[list(SHIFTS)],
                            fmt='o',color=color,capsize=3,label=label)
            ax.set_xticks(range(4),[SHIFT_LABEL[s] for s in SHIFTS],rotation=30,ha='right')
            ax.set_title(f'{title}\n{budget} target labels')
        axes[row,0].set_ylabel('Mean assessment $R^2$')
    axes[0,-1].legend()
    fig.savefig(final/'augmentation_r2.pdf',bbox_inches='tight', metadata={'CreationDate':None})
    plt.close(fig)


def _legacy_functions(notebook):
    """Reuse the repository-owned plotting definitions without executing sweep or example cells."""
    functions = {}
    namespace = dict(np=np,pd=pd,mpl=mpl,plt=plt,Line2D=Line2D,mpatches=mpatches,
                     savefig_pdf=lambda path: plt.savefig(path,bbox_inches='tight'))
    starts=('def plot_transfer_vs_combined_score','def plot_transfer_vs_shift_by_type',
            'import numpy as np','def _paper_rc')
    for cell in json.loads(Path(notebook).read_text())['cells']:
        source=''.join(cell['source'])
        if not any(s in source for s in starts) or not any(s in source for s in
                ('def plot_transfer_vs_', 'def plot_hetero_transfer_highlight','def plot_ft_mdn_heatmap')):
            continue
        defs=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)]
        exec(compile(ast.Module(body=defs,type_ignores=[]),str(notebook),'exec'),namespace)
        functions.update({node.name:namespace[node.name] for node in defs})
    return functions


@mpl.rc_context(rc=PAPER_STYLE)
def plot_historical(repo, raw_root, destination):
    """Historical diagnostics only. The original sample-size plotting function is missing."""
    repo,root,dest=map(Path,(repo,raw_root,destination))
    index=pd.read_csv(root/'results/index.csv')
    tables=[]
    for record in index.itertuples():
        part=pd.read_csv(root/record.raw)
        for key in ('shift_type','hetero','target_n'):
            part[key]=getattr(record,key)
        tables.append(part)
    raw=pd.concat(tables,ignore_index=True)
    if 'model_name' not in raw and 'model_index' in raw:
        raw=raw.rename(columns={'model_index':'model_name'})
    dest.mkdir(parents=True,exist_ok=True)
    keys=['shift_type','hetero','target_n','model_name']
    if 'simulation_id' in raw and raw.duplicated(keys+['simulation_id']).any():
        raise ValueError('Index contains duplicated historical model/replication rows')
    metrics=['r2_target_only','r2_transfer_learning','mae_target_only','mae_transfer_learning']
    raw.groupby(keys)[metrics].agg(['size','count','mean','sem']).to_csv(dest/'historical_condition_summary.csv')
    f=_legacy_functions(repo/'notebooks/archive/simulation_final.ipynb')
    model_labels = {'FT-MDN': 'FT–MDN–T', 'FT-Reg': 'FT–Reg', 'XGB': 'XGBoost', 'MLP': 'MLP'}
    display_rows = raw.assign(model_name=raw.model_name.replace(model_labels))

    def draw(name, filename, **kwargs):
        # The archived heatmap selects the mixture model by its stored identifier.
        if name == 'plot_ft_mdn_heatmap':
            fig, _ = f[name](raw, save_path=None, **kwargs)
            for title in fig.texts:
                title.set_text(title.get_text().replace('FT-MDN', model_labels['FT-MDN']))
        else:
            fig, _ = f[name](display_rows, save_path=None, **kwargs)
        fig.savefig(dest/filename, bbox_inches='tight')
        plt.close(fig)
    draw('plot_transfer_vs_combined_score','r2_transfer_vs_combinedKL_allTargets.pdf')
    for n in (100,500,1000):
        if n in raw.target_n.values:
            draw('plot_transfer_vs_shift_by_type',f'r2_transfer_by_shift_target{n}.pdf',target_n=n,panel_size=5)
    draw('plot_hetero_transfer_highlight','hetero_transfer_violin_highlight.pdf',
         models_order=list(model_labels.values()),single_color=FT_HIGHLIGHT,panel_size=5,show_delta=True)
    draw('plot_ft_mdn_heatmap','FTMDN_heatmap_headers.pdf',n_shift_bins=4,cmap='viridis',
         smooth_linear_iters=2,smooth_neighbor_iters=1)
    # Reconstructed, not the unavailable figure-producing function. Keep negative scores visible.
    fig,axes=plt.subplots(1,3,sharey=True,figsize=(10,3.3),constrained_layout=True)
    for ax,schema,title in zip(axes,SCHEMAS,SCHEMA_NAMES):
        for name,color in [('FT-MDN','#1f77b4'),('FT-Reg','#ff7f0e'),('XGB','#2ca02c'),('MLP','#d62728')]:
            group=raw[(raw.hetero==schema)&(raw.model_name==name)].groupby('target_n')
            for col,fill in [('r2_target_only','white'),('r2_transfer_learning',color)]:
                ax.plot(group[col].mean(),marker='o',mfc=fill,color=color,
                        linestyle='none' if fill=='white' else '-',label=model_labels[name] if fill!='white' else None)
        ax.set(xscale='log',title=title,xlabel='Target labels')
    axes[0].set_ylabel('Historical validation $R^2$')
    axes[-1].legend()
    fig.suptitle('Target-only fits (open markers) and transfer fits (connected markers)')
    for ax in axes:
        ax.set_xticks(sorted(raw.target_n.unique()), [str(n) for n in sorted(raw.target_n.unique())])
        ax.minorticks_off()
    fig.savefig(dest/'transfer_vs_target_size_reconstruction.pdf', bbox_inches='tight')
    plt.close(fig)
    return sorted(p.name for p in dest.glob('*.pdf'))
