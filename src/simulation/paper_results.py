"""Validated exports for the recovery-transfer study. No fitting during export."""
import hashlib
import json
from itertools import product
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

from simulation.revision import SCHEMAS, SHIFTS, run_cell

KEY = ['shift', 'schema', 'budget', 'rep']
MODELS = ('scratch', 'transfer', 'xgb_scratch', 'xgb_residual')
NAMES = ('FT-MDN-T (scratch)', 'FT-MDN-T (pretrained)', 'XGBoost (target only)', 'XGBoost (residual)')
SCHEMA_LABEL = {'equal': 'Same predictors', 'source_subset': 'Fewer source predictors',
                'target_subset': 'Fewer target predictors'}
SCHEMA_NAMES = tuple(SCHEMA_LABEL[schema] for schema in SCHEMAS)
SHIFT_LABEL = {'none': 'No shift', 'covariate': 'Covariate shift',
              'conditional': 'Conditional shift', 'label': 'Label shift'}
SENS_SHIFTS = ('none', 'label')
SENS_SCHEMAS = ('equal', 'target_subset')


def read_panel(folder, config, longer=False):
    shifts, schemas = (SENS_SHIFTS, SENS_SCHEMAS) if longer else (SHIFTS, SCHEMAS)
    expected = set(product(shifts, schemas, (100, 500), range(5)))
    digest = hashlib.sha256(Path(config).read_bytes()).hexdigest()
    rows, files, protocols = [], {}, set()
    for path in sorted(Path(folder).glob('*.json')):
        if len(path.stem) != 20:
            continue
        obj = json.loads(path.read_text())
        spec = obj['spec']
        encoded = json.dumps(spec, sort_keys=True).encode()
        if hashlib.sha256(encoded).hexdigest()[:20] != path.stem:
            raise ValueError(f'Cache key mismatch: {path}')
        cap = 80 if longer else 20
        if (spec['config_sha'], spec['epochs'], spec.get('target_epochs',20), spec['source_n'],
            spec['test_n'], spec['intensity']) != (digest,20,cap,1000,1000,.5):
            raise ValueError(f'Unexpected protocol in {path}; use a separate output folder')
        protocols.add(json.dumps({k:v for k,v in spec.items() if k not in KEY},sort_keys=True))
        row = {k:spec[k] for k in KEY}
        row.update(file=path.name, hashes=tuple(obj['data_hashes']), source_epoch=obj['source_epoch'])
        assert len(set(row['hashes'])) == 5
        for mode in MODELS:
            for metric in ('r2','mae'):
                row[f'{mode}_{metric}'] = obj[mode][metric]
        for mode in MODELS[:2]:
            row[f'{mode}_epoch'] = obj[mode]['epoch']
            if not 1 <= row[f'{mode}_epoch'] <= cap:
                raise ValueError(f'Bad selected epoch: {path}')
        for metric, sign in [('r2',1),('mae',-1)]:
            gain = sign*(row[f'transfer_{metric}']-row[f'scratch_{metric}'])
            if not np.isclose(gain,obj[f'gain_{metric}']):
                raise ValueError(f'Inconsistent paired gain: {path}')
            row[f'gain_{metric}'] = gain
        rows.append(row)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    keys = {tuple(row[k] for k in KEY) for row in rows}
    if keys != expected or len(rows) != len(expected) or len(protocols) != 1:
        raise ValueError(f'Incomplete, duplicated or mixed panel in {folder}: {len(rows)}/{len(expected)} cells')
    raw = pd.DataFrame(rows).sort_values(KEY)
    if not np.isfinite(raw.select_dtypes('number')).all().all():
        raise ValueError('Nonfinite scores')
    return raw, dict(protocol=json.loads(protocols.pop()), files=files)


def run_sensitivity(config, primary, destination):
    """Same source budget and source checkpoint rule; both target arms get up to 80 epochs."""
    read_panel(primary, config)
    for shift, schema, budget, rep in product(SENS_SHIFTS,SENS_SCHEMAS,(100,500),range(5)):
        run_cell(config, destination, rep, shift, schema, budget, target_epochs=80)


def summarize(raw):
    by = raw.groupby(KEY[:-1])
    metrics = [f'{mode}_{metric}' for mode in MODELS for metric in ('r2','mae')]
    metrics += ['gain_r2','gain_mae']
    table = by[metrics].mean()
    for metric in ('gain_r2','gain_mae'):
        table[metric+'_mcse'] = by[metric].sem()
    table['n'] = by.size()
    return table.reset_index()


# Model comparisons use paper blue and orange. Open markers
# denote scratch fits; filled markers denote transfer or residual fits.
MODEL_BLUE, GREEN = '#00376c', '#EF7C00'  # FT-MDN-T and XGBoost
BLUE, ORANGE = '#00376c', '#EF7C00'       # source / target palette
FT_LIGHT, FT_HIGHLIGHT = '#5b89b0', '#00376c'
PAPER_STYLE = {'font.family':'serif', 'font.serif':['Times New Roman','Times','DejaVu Serif'],
               'mathtext.fontset':'stix', 'pdf.fonttype':42, 'ps.fonttype':42,
               'axes.grid':True, 'axes.axisbelow':True, 'grid.linestyle':':',
               'grid.alpha':.25, 'legend.frameon':False, 'font.size':10}


def _original_style_panels(raw, table, dest):
    import matplotlib.pyplot as plt
    # Facet by shift rather than pooling four different target laws.
    for shift in SHIFTS:
        fig, axes = plt.subplots(1,3,sharey=True,figsize=(10,3.3),constrained_layout=True)
        for ax,schema,title in zip(axes,SCHEMAS,SCHEMA_NAMES):
            group = raw.query('shift == @shift and schema == @schema').groupby('budget')
            for j,(mode,name) in enumerate(zip(MODELS,NAMES)):
                series = group[mode+'_r2']
                mean, se = series.mean(), series.sem()
                color = MODEL_BLUE if j<2 else GREEN
                ax.errorbar(mean.index,mean,yerr=se,color=color,marker='o',
                            markerfacecolor=color if j%2 else 'white',
                            linestyle='-' if j%2 else '--',capsize=3,label=name)
            ax.set(title=title,xlabel='Target labels',xticks=[100,500])
        axes[0].set_ylabel('Assessment $R^2$')
        axes[-1].legend(fontsize=8)
        fig.suptitle(SHIFT_LABEL[shift])
        fig.savefig(dest/f'sample_size_{shift}.pdf',bbox_inches='tight', metadata={'CreationDate':None})
        plt.close(fig)
    # Five paired observations per condition: show actual draws instead of a smoothed violin.
    fig,axes=plt.subplots(4,3,sharey=True,figsize=(9,8),constrained_layout=True)
    for row,shift in enumerate(SHIFTS):
        for ax,schema,title in zip(axes[row],SCHEMAS,SCHEMA_NAMES):
            for xpos,budget in enumerate((100,500)):
                subset=raw.query('shift == @shift and schema == @schema and budget == @budget')
                for r in subset.itertuples():
                    a,b=xpos-.09,xpos+.09
                    ax.plot([a,b],[r.scratch_r2,r.transfer_r2],color=FT_LIGHT,lw=.8,alpha=.7)
                    ax.scatter(a,r.scratch_r2,facecolor='white',edgecolor=FT_HIGHLIGHT,s=18)
                    ax.scatter(b,r.transfer_r2,color=FT_HIGHLIGHT,s=18)
            ax.set_xticks([0,1],['100','500'])
            ax.set_xlim(-.35,1.35)
            if row==0: ax.set_title(title)
            if row==3: ax.set_xlabel('Target labels')
        axes[row,0].set_ylabel(SHIFT_LABEL[shift]+'\nAssessment $R^2$')
    fig.suptitle('FT-MDN-T: scratch (open) and pretrained (filled)')
    fig.savefig(dest/'paired_replications.pdf',metadata={'CreationDate':None})
    plt.close(fig)
    # Discrete measured cells only: no interpolation or intensity quartiles.
    fig,axes = plt.subplots(1,3,sharey=True,figsize=(8,3.4),constrained_layout=True)
    low,high = table.transfer_r2.min(),table.transfer_r2.max()
    for ax,schema,title in zip(axes,SCHEMAS,SCHEMA_NAMES):
        values = table.query('schema == @schema').pivot(index='shift',columns='budget',values='transfer_r2').loc[list(SHIFTS)]
        ax.grid(False)
        im=ax.imshow(values,interpolation='nearest',cmap='viridis',vmin=low,vmax=high,aspect='equal')
        for (i,j),v in np.ndenumerate(values.to_numpy()):
            ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='black' if v>(low+high)/2 else 'white')
        ax.set(title=title,xlabel='Target labels',xticks=[0,1],xticklabels=values.columns,
               yticks=range(4),yticklabels=[SHIFT_LABEL[x] for x in SHIFTS])
    fig.colorbar(im,ax=axes,label='Pretrained FT-MDN-T: mean assessment $R^2$')
    fig.savefig(dest/'transfer_heatmap.pdf',metadata={'CreationDate':None})
    plt.close(fig)


def _tables_and_figures(table, dest):
    import matplotlib.pyplot as plt
    # Each interval uses the exact same five paired differences as its center.
    for metric, label in [('r2','Transfer gain in $R^2$'), ('mae','Reduction in MAE')]:
        fig, axes = plt.subplots(1,3,sharey=True,figsize=(9,3.3),constrained_layout=True)
        for ax,schema,title in zip(axes,SCHEMAS,SCHEMA_NAMES):
            for budget,offset,color,marker in [(100,-.09,BLUE,'o'),(500,.09,ORANGE,'s')]:
                t = table.query('schema == @schema and budget == @budget').set_index('shift').loc[list(SHIFTS)]
                ax.errorbar(np.arange(4)+offset,t['gain_'+metric],yerr=t['gain_'+metric+'_mcse'],
                            fmt=marker,color=color,capsize=3,label=f'{budget} target labels')
            ax.axhline(0,color='0.5',linewidth=.7)
            ax.set_xticks(range(4),[SHIFT_LABEL[x] for x in SHIFTS],rotation=30,ha='right')
            ax.set_title(title)
        axes[0].set_ylabel(label)
        fig.suptitle('FT-MDN-T: effect of source pretraining')
        axes[-1].legend(fontsize=8)
        fig.savefig(dest/f'paired_gain_{metric}.pdf',metadata={'CreationDate':None})
        plt.close(fig)
    fig,axes = plt.subplots(2,3,sharey=True,figsize=(10,5),constrained_layout=True)
    for i,b in enumerate((100,500)):
        for ax,schema,title in zip(axes[i],SCHEMAS,SCHEMA_NAMES):
            t = table.query('schema == @schema and budget == @b').set_index('shift').loc[list(SHIFTS)]
            for j,(mode,name) in enumerate(zip(MODELS,NAMES)):
                ax.plot(np.arange(4)+(j-1.5)*.10,t[mode+'_r2'],marker='o',color=MODEL_BLUE if j<2 else GREEN,markerfacecolor='white' if j%2==0 else (MODEL_BLUE if j<2 else GREEN),ls='',label=name)
            ax.set_xticks(range(4),[SHIFT_LABEL[x] for x in SHIFTS],rotation=30,ha='right')
            ax.set_title(f'{title}\n{b} target labels')
        axes[i,0].set_ylabel('Mean assessment $R^2$')
    axes[0,-1].legend(fontsize=7)
    fig.savefig(dest/'absolute_r2.pdf',bbox_inches='tight', metadata={'CreationDate':None})
    plt.close(fig)


@mpl.rc_context(rc=PAPER_STYLE)
def export(config, primary, destination, sensitivity=None):
    dest = Path(destination)
    dest.mkdir(parents=True,exist_ok=True)
    raw, provenance = read_panel(primary,config)
    table = summarize(raw)
    manifest = {'primary':provenance}
    table.to_csv(dest/'condition_means.csv',index=False)
    table[['shift','schema','budget','n','gain_r2','gain_r2_mcse','gain_mae','gain_mae_mcse']].to_latex(
        dest/'paired_gains.tex',index=False,float_format='%.3f',escape=True,
        header=['Shift','Schema','Labels','Runs','Gain R2','MCSE','MAE reduction','MCSE'])
    raw.drop(columns='hashes').to_csv(dest/'replications.csv',index=False)
    for metric in ('r2','mae'):
        columns = ['shift','schema','budget'] + [f'{mode}_{metric}' for mode in MODELS]
        table[columns].to_latex(dest/f'absolute_{metric}.tex',index=False,float_format='%.3f',
                                escape=True,header=['Shift','Schema','Labels',*NAMES])
    _tables_and_figures(table,dest)
    _original_style_panels(raw,table,dest)
    if sensitivity is not None:
        long, manifest['sensitivity'] = read_panel(sensitivity,config,longer=True)
        paired = long.merge(raw,on=KEY,validate='one_to_one',suffixes=('_80','_20'))
        if len(paired)!=40 or not (paired.hashes_80==paired.hashes_20).all():
            raise ValueError('Long and short fits must use identical data partitions')
        if not (paired.source_epoch_80==paired.source_epoch_20).all():
            raise ValueError('Source selection changed; check runtime before comparing')
        for mode in MODELS[:2]:
            for cap in (20,80):
                paired[f'{mode}_at_cap_{cap}'] = paired[f'{mode}_epoch_{cap}']==cap
        paired['change_gain_r2'] = paired.gain_r2_80-paired.gain_r2_20
        groups = paired.groupby(KEY[:-1])
        cols = ['gain_r2_20','gain_r2_80','change_gain_r2','gain_mae_20','gain_mae_80',
                'scratch_at_cap_20','scratch_at_cap_80','transfer_at_cap_80']
        stability = groups[cols].mean()
        stability['mcse_change'] = groups.change_gain_r2.sem()
        stability.reset_index().to_csv(dest/'sensitivity.csv',index=False)
        stability[['gain_r2_20','gain_r2_80','change_gain_r2','mcse_change']].reset_index().to_latex(
            dest/'sensitivity.tex', index=False, float_format='%.3f', escape=True,
            header=['Shift','Schema','Labels','20 epochs','80 epochs','Change','MCSE'])
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,2,sharey=True,figsize=(9.2,3.2),constrained_layout=True)
        for ax, schema in zip(axes,SENS_SCHEMAS):
            for shift in SENS_SHIFTS:
                for budget in (100,500):
                    subset = paired[(paired.schema==schema)&(paired['shift']==shift)&(paired.budget==budget)]
                    if len(subset) != 5:
                        raise ValueError('Sensitivity plot requires five paired replications per condition')
                    gains = subset[['gain_r2_20','gain_r2_80']]
                    ax.errorbar([0,1],gains.mean(),yerr=gains.sem(),marker='o' if budget==100 else 's',color=BLUE if shift=='none' else ORANGE,
                                linestyle='-' if shift=='none' else '--',capsize=3,label=f'{SHIFT_LABEL[shift]}, {budget} target labels')
            ax.axhline(0,color='0.5',linewidth=.7)
            ax.set_xticks([0,1],['20','80'])
            ax.set_xlabel('Maximum target training epochs')
            ax.set_title(SCHEMA_LABEL[schema])
        axes[0].set_ylabel('Transfer gain in $R^2$')
        fig.suptitle('FT-MDN-T: effect of source pretraining')
        axes[-1].legend(loc='center left', bbox_to_anchor=(1.01, .5), fontsize=8)
        fig.savefig(dest/'training_sensitivity.pdf',bbox_inches='tight', metadata={'CreationDate':None})
        plt.close(fig)
        paired.drop(columns=['hashes_20','hashes_80']).to_csv(dest/'sensitivity_pairs.csv',index=False)
    (dest/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return table
