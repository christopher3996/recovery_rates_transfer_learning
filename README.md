# Transfer Learning for Loan Recovery Prediction under Distribution Shifts with Heterogeneous Feature Spaces

This repository contains the code accompanying the paper:

> **Transfer Learning for Loan Recovery Prediction under Distribution Shifts with Heterogeneous Feature Spaces**  
> Christopher Gerling, Hanqiu Peng, Ying Chen, Stefan Lessmann  

The code implements the FT–MDN–Transformer architecture, the Monte Carlo simulation framework for controlled distribution shifts, and the experimental pipelines used for the empirical analyses.

---

## Overview

The repository supports two main experimental settings:

1. **Monte Carlo Simulation**  
   Controlled experiments to study covariate, conditional, and label shift under heterogeneous feature spaces.

2. **Real-World Transfer Learning**  
   Transfer from a loans dataset (Global Credit Data, GCD) to a bonds dataset (UP5), evaluating model performance under realistic feature heterogeneity and data scarcity.

The implementation focuses on reproducibility and modularity, with YAML-based configuration files and a unified experiment runner.

---

## Repository Structure

```
recovery_rates_transfer_learning-main/
├── configs/
│   └── simulation/
│       ├── monte_carlo_config.yaml
│       └── source_data_config_categories.yaml
├── data/
│   └── place_data_here
├── notebooks/
│   ├── simulation_final.ipynb
│   ├── up5_mdn_ablations.ipynb
│   └── up5_helper.py
├── outputs/
│   └── place_for_outputs
├── src/
│   ├── data_generation/
│   │   └── shifted_data.py
│   ├── models/
│   │   ├── ft_mdn_transformer.py
│   │   ├── baselines.py
│   │   └── archive/
│   └── utils/
└── README.md
```

Key components:

- `src/data_generation/shifted_data.py` — Monte Carlo data generation with shift mechanisms  
- `src/models/ft_mdn_transformer.py` — FT–MDN–Transformer implementation  
- `configs/` — YAML configuration files for simulation and experiments  
- `notebooks/` — Analysis and ablation notebooks used for paper figures and tables

---

## Requirements

The code is written in Python 3.9+ and relies on standard scientific and deep learning libraries. Typical dependencies include:

- numpy
- pandas
- scikit-learn
- torch
- pyyaml
- matplotlib / seaborn
- jupyter (for notebooks)

We recommend using a virtual environment (e.g., `venv` or `conda`).

Example:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

*(If no `requirements.txt` is provided, please install the above packages manually.)*

---

## Data Availability

### Global Credit Data (GCD)
The GCD dataset is proprietary and **cannot be redistributed** through this repository. Access must be obtained directly from Global Credit Data.

### UP5 Bonds Dataset
The UP5 dataset used in the real-world transfer experiments is proprietary and cannot be shared publicly.

Accordingly, the `data/` directory only contains placeholders. To reproduce the real-world experiments, users must supply their own licensed copies of the respective datasets and adapt the data loading utilities as needed.

The Monte Carlo simulation experiments can be fully reproduced without proprietary data.

---

## Reproducing Results

### Monte Carlo Simulation

1. Configure the experiment in:
   ```
   configs/simulation/monte_carlo_config.yaml
   ```

2. Run the following notebook:
   ```
   notebooks/simulation_final.ipynb
   ```

3. Outputs (metrics, logs, plots) will be written to:
   ```
   outputs/
   ```

### Real-World Transfer Learning

Due to data restrictions, full reproduction requires access to GCD and UP5. After placing the datasets in the appropriate locations and updating data paths, the real-world experiments can be run analogously via `src/main.py` or through the provided notebooks.

Relevant notebooks:

- `notebooks/up5_mdn_ablations.ipynb`
- `notebooks/simulation_final.ipynb`

---

## Configuration

Experiments are controlled via YAML files in `configs/`. These define:

- Sample sizes and train/validation/test splits
- Types and strengths of distribution shifts
- Feature overlap and heterogeneity
- Model hyperparameters (Transformer size, MDN components, learning rates, etc.)

This design allows systematic variation of experimental conditions as described in the paper.

---

## Citation

If you use this code in your research, please cite:

```
@article{Gerling2026TLRecovery,
  title   = {Transfer Learning for Loan Recovery Prediction under Distribution Shifts with Heterogeneous Feature Spaces},
  author  = {Gerling, Christopher and Peng, Hanqiu and Chen, Ying and Lessmann, Stefan},
  year    = {2026},
}
```

(Please update once the paper is accepted and assigned final bibliographic details.)

---

## License

This repository is provided for academic and research purposes.  
Please contact the authors for licensing questions related to commercial use.

---

## Contact

For questions regarding the code or experiments:

- Christopher Gerling — Humboldt-Universität zu Berlin  

---

## Notes

- Parts of this work were conducted during a research stay of the first author at the National University of Singapore.
- Access to GCD was provided through this affiliation, in accordance with GCD’s data usage policies.
- The repository reflects the experimental code used for the paper and may contain archived or exploratory modules not used in the final analysis.
