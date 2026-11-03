"""Daumé-style feature augmentation with ridge regression on aligned column names."""
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from simulation.paper_results import KEY, read_panel
from simulation.revision import EXTRA, _draw, _split
from pathlib import Path
import hashlib


def augmentation_panel(config_path, primary, destination):
    reference, _ = read_panel(primary, config_path)
    config = yaml.load(Path(config_path).read_text(), Loader=yaml.UnsafeLoader)
    rows = []
    # All schema projections within this replication use the same full draws.
    for (rep, shift, budget), block in reference.groupby(['rep','shift','budget']):
        nv = budget//5
        frames = [_draw(config,800,rep,0), _draw(config,200,rep,1),
                  _draw(config,budget-nv,rep,2,shift), _draw(config,nv,rep,3,shift),
                  _draw(config,1000,rep,4,shift)]
        hashes = tuple(hashlib.sha256(pd.util.hash_pandas_object(f,index=False).values.tobytes()).hexdigest()
                       for f in frames)
        for ref in block.itertuples():
            if hashes != ref.hashes:
                raise ValueError('Regenerated data differ from the primary panel')
            (xs,ys),(xt,yt),(xv,yv),(xe,ye) = [
                _split(frames[i],EXTRA if ((i==0 and ref.schema=='source_subset') or
                                          (i!=0 and ref.schema=='target_subset')) else ())
                for i in (0,2,3,4)]
            # Category vocabulary and dimensionality are fitted only on fit rows.
            def encoded(df):
                copy = df.copy()
                for col in copy.select_dtypes(include='category'):
                    copy[col] = copy[col].astype(object)
                return pd.get_dummies(copy, dtype=float)
            source, target = encoded(xs), encoded(xt)
            cols = source.columns.union(target.columns)
            def align(df):
                return df.reindex(columns=cols,fill_value=0).to_numpy(dtype=float)
            s,t,v,e = map(align,[source,target,encoded(xv),encoded(xe)])
            # Preserve structural zeros for domain-exclusive predictors.
            scale = StandardScaler(with_mean=False).fit(np.vstack([s,t]))
            s,t,v,e = map(scale.transform,[s,t,v,e])
            def augmented(x, is_source):
                zero = np.zeros_like(x)
                return np.hstack([x, x if is_source else zero, zero if is_source else x,
                                  np.full((len(x),1),float(is_source))])
            X = np.vstack([augmented(s,True),augmented(t,False)])
            Y = np.concatenate([ys,yt])
            row = {k:getattr(ref,k) for k in KEY}
            for name,train,labels,val,test in [
                ('ridge',t,yt,v,e),('augmented',X,Y,augmented(v,False),augmented(e,False))]:
                best_loss = np.inf
                for alpha in (1.,10.,100.,1000.):
                    model = Ridge(alpha=alpha).fit(train,labels)
                    loss = np.mean((model.predict(val)-yv.to_numpy())**2)
                    if loss < best_loss:
                        best_loss,best,selected = loss,model,alpha
                pred = best.predict(test)
                row.update({name+'_r2':r2_score(ye,pred),name+'_mae':mean_absolute_error(ye,pred),
                            name+'_alpha':selected})
            rows.append(row)
    table = pd.DataFrame(rows).sort_values(KEY)
    dest = Path(destination)
    dest.mkdir(parents=True,exist_ok=True)
    table.to_csv(dest/'augmentation_runs.csv',index=False)
    summary = table.groupby(KEY[:-1])[['ridge_r2','augmented_r2','ridge_mae','augmented_mae']].mean()
    summary.to_csv(dest/'augmentation_summary.csv')
    summary.reset_index().to_latex(dest/'augmentation.tex',index=False,float_format='%.3f',escape=True)
    return summary
