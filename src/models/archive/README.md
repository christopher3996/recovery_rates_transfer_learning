# Model archive

Earlier model variants are retained for provenance. Current paper entry points are
[the simulation and UP5 notebooks](../../../notebooks).

| Files | Role |
|---|---|
| `ft_*`, `wide_deep_regression.py`, `xgb_*` | Earlier architectures and transfer implementations; not the default paper models. |
| `tabpfn_helpers/` | Earlier vendored TabPFN source; not imported by the active baseline wrapper. |
| `up5_giw.py`, `giw_helpers/` | Earlier importance-weighting experiments. |

The deprecated `../DEPRECATED_ft_transformer_mdn_transfer.py` remains at its original
path because the validation-sweep manifest fingerprints all top-level model files.
Moving it would invalidate compatibility with saved run manifests. It is not a
recommended entry point. No files here should be removed based on model performance.

The active TabPFN wrapper is now in [../baselines/](../baselines/README.md).
The final paper pipeline does not import models from this archive.
