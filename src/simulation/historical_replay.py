"""Run the historical notebook's setup and sweep in a dedicated output directory."""
import hashlib
import json
import os
from pathlib import Path
import sys
import platform
from importlib.metadata import version


def main(repo, destination=None):
    import torch
    repo = Path(repo).resolve()
    torch.set_num_threads(1)
    notebook = repo/'notebooks/archive/simulation_final.ipynb'
    source = json.loads(notebook.read_text())['cells']
    stop = next(i for i,c in enumerate(source) if ''.join(c['source']).startswith('## 6) Aggregate'))
    cells = [''.join(c['source']) for c in source[:stop] if c['cell_type']=='code']
    folder = Path(destination).resolve() if destination else repo/'outputs/historical_replay'
    folder.mkdir(parents=True, exist_ok=True)
    provenance = dict(hash_seed=os.environ['PYTHONHASHSEED'], python=platform.python_version(),
                      threads=torch.get_num_threads(),device='cuda' if torch.cuda.is_available() else 'cpu',
                      packages={k:version(k) for k in ('numpy','pandas','torch','scipy','xgboost','scikit-learn')},
                      sweep_sha256=hashlib.sha256('\n'.join(cells).encode()).hexdigest(),
                      files={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for base in ('src/models','src/utils','src/data_generation','configs/simulation')
                             for p in sorted((repo/base).glob('*')) if p.suffix in ('.py','.yaml')})
    runner=repo/'src/simulation/monte_carlo_simulation.py'
    provenance['runner_sha256']=hashlib.sha256(runner.read_bytes()).hexdigest()
    manifest=folder/'replay_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=provenance:
        raise RuntimeError('Historical environment or code changed; archive the previous replay directory first.')
    manifest.write_text(json.dumps(provenance,indent=2))
    os.chdir(repo/'notebooks')
    namespace = dict(__name__='__main__', display=print)
    for code in cells:
        code=code.replace('Path("../outputs/sim_runs")',f'Path({str(folder)!r})')
        exec(compile(code,str(notebook),'exec'),namespace)


if __name__=='__main__':
    main(*sys.argv[1:])
