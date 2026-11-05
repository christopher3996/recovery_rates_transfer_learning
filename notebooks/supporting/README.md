# Supporting experiment sources

These files are **active dependencies**, not archived experiments. Run the two
paper notebooks one directory above; their helpers load selected cells and functions
from these sources. Do not run these supporting notebooks end-to-end.

| Source | Used by | Purpose |
|---|---|---|
| `simulation_final.ipynb` | `src/simulation/historical_replay.py`, `figure_inventory.py` | Validation sweep setup and plotting definitions |
| `up5_mdn_ablations.ipynb` | `src/experiments/up5/historical_figures.py` | Density illustrations and encoding comparison |
| `up5_helper.py` | UP5 figure generator | Loading and categorical reconstruction |

The source notebooks include older exploratory cells that are not executed by the
paper pipeline. Keep cell order and contents intact: the loaders select specific
cells and the validation manifest hashes their source text. Relocation from the
archive preserved all three files byte-for-byte. Extracting the used definitions
into ordinary Python modules can be a separate refactor with equivalence checks.
