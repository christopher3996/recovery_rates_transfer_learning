# Transfer Learning for Loan Recovery Prediction under Distribution Shifts with Heterogeneous Feature Spaces

Christopher Gerling, Hanqiu Peng, Ying Chen and Stefan Lessmann

[Start](#start-here) · [Paper results](#where-to-find-each-paper-result) · [Execution](#notebook-execution-order) · [Environment](#environment) · [Data and outputs](#data-and-outputs) · [Archive](notebooks/archive/README.md)

## Start here

Open either notebook, leave the first code cell in `review` mode, and select **Restart Kernel and Run All Cells**. Run Jupyter from the repository root. The default paths use the distributed results without model training or private data.

| Notebook | What it reproduces | When to change the mode |
|---|---|---|
| [Simulation](notebooks/simulation_paper.ipynb) | Controlled shifts and feature schemas, training-budget sensitivity, paired density forecasts at 100/250/500 labels, augmented ridge, and the validation curves across target sample sizes. | `rebuild` regenerates the experiment inputs and results in a separate output tree. Allow several hours on CPU. |
| [UP5](notebooks/up5_paper.ipynb) | Point benchmarks, probabilistic scores and tail probabilities, encoded and native-token backbone comparisons. | `rebuild` fits the UP5 models and generates their figures. |

FT-MDN-T denotes the FT-MDN-Transformer. In simulation figures, **scratch** means training on target data alone; **pretrained** means source pretraining followed by target fine-tuning. In the UP5 schema-expansion figure, **pretraining** and **fine-tuning** are consecutive stages with shared and expanded predictors, respectively. The pretraining curve is not a target-only scratch comparison. The residual XGBoost and augmented ridge baselines have distinct transfer procedures described in the simulation notebook.

## Where to find each paper result

The figure and table numbers below follow the revised manuscript. The **online appendix has its own numbering**. Section numbers in the notebook column refer to headings inside the linked notebook.

### Main manuscript

| Paper item | Notebook or source | What reviewers can inspect or regenerate |
|---|---|---|
| Figures 1–3 · Architecture, two-stage schema adaptation and evaluation scenarios | Manuscript §3 | Figure 1 is the architecture illustration. Figures 2 and 3 are drawn in LaTeX. |
| Table 1 · Core sample characteristics | [UP5](notebooks/up5_paper.ipynb) §§1, 4 | The final UP5 extract contains 1,725 bonds. The native-token comparison uses 155 numeric and nine categorical fields. Raw cohort counts, observation windows and GCD characteristics come from the study documentation. |
| Table 2 · Distributional gaps | [UP5 cohort summary](results/up5/recovery-summary.csv) | The **Combined** row gives the UP5 mean, standard deviation and skewness. The GCD column and cross-portfolio differences require GCD. |
| Figure 4 · Recovery distributions and feature overlap | [UP5](notebooks/up5_paper.ipynb) §1 | Review displays the saved UP5 panel; rebuild reconstructs the UP5 training-set histogram. The overlap diagram is drawn in LaTeX. The GCD histogram requires restricted data. |
| Figure 5 · Correlation heatmaps | UP5 §1 | Review displays the saved UP5 panel; rebuild reconstructs the UP5 panel for the 31 shared numerical predictors. The GCD panel requires restricted data. |
| §4.3, Step 1 · Target-only benchmarks reported in the text | UP5 §§2–4, `review` | [Point forecasts](results/up5/scores.csv), [distribution and interval scores](results/up5/density-scores.csv), [tail probabilities](results/up5/tails.csv), [encoded backbone comparison](results/up5/mdn-backbones.csv) and [native-token backbone comparison](results/up5/native-backbones.csv). |
| Figure 6 · Single-loan recovery distributions (UP5 bonds) | UP5 §3 | Review displays saved panels; rebuild fits the illustration model. Four conditional mixture densities with component contributions, predicted means and observed recoveries. Evaluation row positions 57, 58, 63 and 89 (zero-based) correspond to panels 1–4. |
| Figure 7 · UP5 portfolio-level recovery distributions | UP5 §3 | Review displays the saved figure; rebuild averages the individual predictive densities within each partition and overlays them on the training and evaluation histograms. |
| Figure 8 · Categorical embeddings versus dummy encoding | UP5 §5 | Review displays the saved figure; rebuild fits 15 seeds per encoding and plots validation R² by epoch, with interquartile bands and early-stopping markers. The backbone comparison in §4.3 of the paper is a separate experiment. |
| Figure 9 · Within-UP5 schema expansion | UP5 §6 | Revised experiment with explicit shared-feature transfer, ten seeds and 80 epochs per phase. Use `review` to read the selected completed run and `rebuild` to calculate it. Saved runs produce `ft_mdn_transformer_schema_expansion.pdf`. |
| Figure 10 · GCD-to-UP5 transfer | Manuscript §4.6 | The original R² and MAE panels are in the manuscript. Regeneration requires restricted GCD data and matching run records. |
| Table 3 · Feature relationships for synthetic data generation | [Source generator configuration](configs/simulation/source_data_config_categories.yaml) and [data generators](src/data_generation) | Specifies the synthetic predictors and their relationships. |
| Figure 11 · Source pretraining across shifts and schemas | [Simulation](notebooks/simulation_paper.ipynb) §1, `review` | Paired assessment R² gains at the 20-epoch target cap. Output: `paired_gain_r2.pdf`. |
| Figure 12 · Transfer gains under longer target training | Simulation §2, `review` | Training-budget sensitivity. Output: `training_sensitivity.pdf`. |
| Figure 13 · Point and density forecasts at three target budgets | Simulation §3, `review` | R² and CRPS at 100, 250 and 500 target labels. Output: `three_budget_forecasts.pdf`. |
| Appendix Table A.4 · Baseline model specifications and transfer mechanisms | Manuscript Appendix A and [model implementations](src/models) | Reference table describing the candidate models and their transfer procedures. |

**Evaluation protocols.** Figures 6–8 use the supplied UP5 evaluation partition for validation. The benchmarks and schema-expansion experiment reserve validation observations within the training partition. The Figure 8 tokenizer processes overlapping event-type prefixes in their original order. The native-token backbone and schema-expansion experiments explicitly separate these fields. Figure 9 uses 36 shared predictors, 164 expanded native predictors, ten seeds and 80 epochs per phase. Its saved results require alignment with the manuscript, which still describes 37 shared predictors and 100 pretraining epochs. The two-phase experiment measures performance after schema expansion. A full-schema scratch control would be needed to isolate the contribution of pretraining.

**Appendix validation study.** The five-size sweep has 222 completed settings out of 225. The sample-size curve was reconstructed from available runs through 1,000 labels. The original plotting helper and the 2,000-label run were unavailable.

### Online appendix

All five items below are available in the [simulation notebook](notebooks/simulation_paper.ipynb) in `review` mode.

| Online appendix item | Notebook section | Output |
|---|---|---|
| Figure 1 · Synthetic source and target domains | §1 | `generator_diagnostics.pdf` |
| Figure 2 · Absolute model performance | §4 | `absolute_r2.pdf` |
| Figure 3 · Feature augmentation comparison | §4 | `augmentation_r2.pdf` |
| Table 1 · Simulation point and density scores | §3 | Mean R², MAE, CRPS, NLL, coverage and interval width for both FT-MDN-T arms and Gaussian ridge at all three target budgets. |
| Figure 4 · Validation results across target sample sizes | §5 | `transfer_vs_target_size_reconstruction.pdf`, using available runs from 100 through 1,000 target observations. |

## Notebook execution order

Each section contains a calculation cell and its result cells. In `review`, calculation cells load or summarize saved runs and the result cells display tables or figures with file links. In `rebuild`, run the sections in order so dependent analyses use the newly calculated forecasts.

| Notebook | Section order |
|---|---|
| UP5 | 1. Cohort → 2. Point forecasts → 3. Distribution scores and densities → 4. Backbone comparisons → 5. Encoding → 6. Schema expansion → 7. Result files |
| Simulation | 1. Main transfer experiment → 2. Training sensitivity → 3. Three target sample sizes → 4. Baselines → 5. Validation sample-size study → 6. Supplementary figures |

UP5 subprocess handling, table formatting and file selection are in `src/experiments/up5/notebook_reporting.py`. The experiment modules retain the fitting and scoring procedures. Generated files and saved manuscript figures are identified when displayed. Select the schema-expansion run explicitly in UP5 Section 6.

### Presentation conventions

New figures use Times New Roman when installed, with Times/DejaVu Serif as fallbacks. The main colors are blue (`#00376c`) and orange (`#EF7C00`); panels with more than two models use additional colors. Existing manuscript image assets are unchanged. Notebook tables display four significant digits and a dash for missing values; CSV exports retain their original precision and identifiers.

## Environment

The reference Jupyter kernel used Python 3.9.19. Install the direct package versions in a fresh environment, then launch Jupyter with that environment active.

```sh
python -m pip install -r requirements.txt
python -m ipykernel install --user --name recovery-transfer --display-name "Recovery transfer"
jupyter notebook
```

Select **Recovery transfer** as the kernel. The pins follow the saved simulation manifest. The local Python environment also contains overlapping installations: a direct import returned NumPy 1.21.5 and scikit-learn 1.6.1, while its package metadata and the run manifest report 1.24.4 and 1.6.0. The author’s TabPFN installation was an editable checkout reporting version 2.0.3 at commit `a05404514dee58ce86ce01f9b0f315d32a59d8ff`. Hardware and numerical libraries can affect fitted values and elapsed time.

## Data and outputs

- **Simulation.** The simulation notebook reads synthetic run records and saved forecasts from `outputs/`, regenerates summaries and styled figures, and links to the generated files. Keep the distributed inputs when cleaning local outputs. The [supporting sources](notebooks/supporting/README.md) contain the source cells used by the validation sweep and UP5 figure generation. They are active dependencies.
- **UP5.** [results/up5](results/up5) holds aggregate tables. Rebuilding requires four local workbooks in `data/UP5/` named `train_features2.xlsx`, `train_labels2.xlsx`, `test_features2.xlsx` and `test_labels2.xlsx`. TabPFN also requires access to its pretrained weights. Individual forecasts, workbooks and training logs stay in ignored local directories. New revision fits write to `outputs/up5_rebuild/<run>/`, independently of the distributed tables.
- **UP5 figures.** Review uses the [saved manuscript figures](results/up5/paper_figures/README.md) for Figures 4–8 when local figure outputs are unavailable. The schema-expansion runs are saved under `outputs/up5_schema_expansion/<protocol>/` when run separately, or in the rebuild’s `schema_expansion/` directory. Repeating the same protocol resumes completed seeds. Figures 6–8 continue to use the earlier validation workflow. Review reads the latest completed local benchmark run. Without a completed rebuild, it displays the distributed aggregate tables and saved manuscript figures. Existing output directory names are retained for compatibility; they do not denote another execution mode. Section 6 selects schema-expansion results independently of the benchmark tables.
- **GCD.** The loan-level data and matching training records are restricted. The original GCD panels and GCD-to-UP5 transfer results cannot be regenerated from this public checkout.

The numerical implementation is in [src/models](src/models), [src/simulation](src/simulation) and [src/experiments/up5](src/experiments/up5). Simulation settings live in [configs/simulation](configs/simulation). The [supporting sources](notebooks/supporting/README.md) remain part of the active pipeline; the [notebook archive](notebooks/archive) contains earlier analyses that the paper notebooks do not execute. The active TabPFN wrapper lives in [src/models/baselines](src/models/baselines).

## Citation, use and contact

Please cite the paper by Christopher Gerling, Hanqiu Peng, Ying Chen and Stefan Lessmann when using this work. The repository is provided for academic and research purposes. Contact Christopher Gerling at Humboldt-Universität zu Berlin for code or commercial licensing questions. Parts of the work were conducted during a research stay at the National University of Singapore, through which access to GCD was provided under its data usage policies.
