import os
import random
import pandas as pd
import numpy as np
from copy import deepcopy
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from data_generation.source_data import SimulationDataGeneration
from data_generation.shifted_data import ShiftedSimulationDataGeneration
from utils.seed import set_seed


class MonteCarloSimulation:
    """
    The MonteCarloSimulation class runs multiple simulations under three scenarios:
      1. SourceOnly: Model is trained on source data.
      2. TargetOnly: Model is trained on target (shifted) data.
      3. TransferLearning: Model is first pretrained on source data, then fine-tuned on target data.

    The evaluation sets depend on the `eval_all_on_target` parameter:
      - If eval_all_on_target=True, SourceOnly and TransferLearning pretrain phases are evaluated on the target test set.
      - If eval_all_on_target=False, those phases are evaluated on the source test set.
      - TargetOnly and TransferLearning fine-tune phase are always trained and evaluated on target train/test sets.

    This class performs a train-test split for both source and target domains to ensure proper
    separation of training and evaluation data.
    """

    def __init__(self,
                source_config: dict,
                shift_param_ranges: dict,
                shift_type: str,  # now one of "covariate", "conditional", "label", or "all"
                num_simulations: int,
                model_class,
                model_names=None,
                model_params=None,
                scenarios=None,
                hetero_feature_mode="equal",    #  equal | source_subset | target_subset
                drop_fraction=0.0,        #  share of cols to drop 
                store_datasets=False,
                verbose=1,
                output_folder: str = "../results"
                ):
        """
        Initialize the MonteCarloSimulation class.

        Args:
            source_config (dict): Configuration dictionary for source data generation.
            shift_param_ranges (dict): Configuration dictionary for shift parameter ranges.
            num_simulations (int): Number of Monte Carlo simulations to run.
            model_class (class or list): The model class(es) to use.
                                         Could be a single class or a list of classes.
            model_names (list): Optional list of model names for labeling.
            model_params (dict or list): Parameters to initialize model instance(s).
                                         Could be a single dict or a list of dicts.
            scenarios (list of list): Each element is a list of strings specifying 
                                      which scenarios to run for that model.
                                      E.g. [["full"], ["target_only"]] 
                                      means first model runs all, second model only target.
            store_datasets (bool): Whether to save intermediate datasets to disk.
        """
        self.source_config = source_config
        self.shift_param_ranges = shift_param_ranges
        self.shift_type = shift_type
        self.num_simulations = num_simulations

        self.hetero_feature_mode = hetero_feature_mode
        self.drop_fraction = drop_fraction
        self.store_datasets = store_datasets
        self.verbose = verbose

        self.output_folder = output_folder
        self.results_folder = os.path.join(self.output_folder, "results")
        self.figures_folder = os.path.join(self.output_folder, "figures")
        self.data_folder    = os.path.join(self.output_folder, "data")
        os.makedirs(self.results_folder, exist_ok=True)
        os.makedirs(self.figures_folder, exist_ok=True)
        os.makedirs(self.data_folder, exist_ok=True)
        
        set_seed(seed=0)

        # Handle single vs. list of model classes
        if not isinstance(model_class, list):
            self.model_classes = [model_class]
        else:
            self.model_classes = model_class

        # Handle single vs. list of model params
        if model_params is None:
            # If nothing is given, use empty dict for each model
            self.model_params_list = [{} for _ in self.model_classes]
        elif not isinstance(model_params, list):
            # If a single dict is given, wrap in list
            self.model_params_list = [model_params]
        else:
            self.model_params_list = model_params
            
        self.model_names=model_names

        # Ensure lengths match if both are lists
        if len(self.model_classes) != len(self.model_params_list):
            raise ValueError("Lengths of model_class list and model_params list must match.")

        if self.model_names and (len(self.model_classes) != len(self.model_names)):
            raise ValueError("Lengths of model_class list and model_names list must match.")
            
        # Handle scenarios per model
        if scenarios is None:
            # If user doesn't specify, default to 'full' for each model
            self.scenarios = [["full"] for _ in self.model_classes]
        else:
            if len(scenarios) != len(self.model_classes):
                raise ValueError("Length of 'scenarios' list must match number of models.")
            self.scenarios = scenarios

        self.features = list(self.source_config['features'].keys())

        # We store ALL results in this list (one row per simulation per model)
        self.results = []

        # We also keep scenario-based losses in dictionaries keyed by model_index or name
        # Example: self.source_only_losses[model_index] = list-of-lists of losses
        self.source_only_losses = {}
        self.target_only_losses = {}
        self.transfer_learning_pretrain_losses = {}
        self.transfer_learning_finetune_losses = {}

        for i in range(len(self.model_classes)):
            self.source_only_losses[i] = []
            self.target_only_losses[i] = []
            self.transfer_learning_pretrain_losses[i] = []
            self.transfer_learning_finetune_losses[i] = []

        self.eval_all_on_target = False
        
        # Set Times New Roman as the default font
        plt.rcParams["font.family"] = "Times New Roman"

    def _timestamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M")

    def _drop_columns(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Return a copy with the requested columns removed (ignore missing)."""
        cols = [c for c in cols if c in df.columns]
        return df.drop(columns=cols)

    def _shared_numeric_cols(self, X1: pd.DataFrame, X2: pd.DataFrame):
        shared = X1.columns.intersection(X2.columns)
        # keep only numeric (KL/corr use numeric)
        return [c for c in shared if pd.api.types.is_numeric_dtype(X1[c])]


    def generate_random_shift_config(self):
        """
        Generate a random shift configuration within specified parameter ranges,
        based on the selected shift type.
        
        For "covariate", only the intercept is shifted (p(x) changes, but p(y|x) stays constant).
        For "conditional", only beta and skewness are shifted (changing p(y|x)).
        For "label", only label parameters are shifted.
        For "all", all three shifts are applied.
        """
        shift_config = {}
        
        # For feature shifts: only include if shift_type is "covariate", "conditional", or "all"
        if self.shift_type in ["covariate", "conditional", "all"]:
            feature_shifts = []
            for feature_name in self.features:
                feature_cfg = self.source_config.get("features", {}).get(feature_name, {})
                if not feature_cfg.get("shiftable", True):
                    continue  # Skip non-shiftable features.
                params = {}

                if self.shift_type == "covariate":
                    # Only shift the intercept and the noise standard scale.
                    intercept_range = self.shift_param_ranges['feature_shift_params']['intercept_shift']
                    params['intercept_shift'] = random.uniform(intercept_range['min'], intercept_range['max'])
                    # Make sure that the correlations are not distorted 
                    # Therefore we need equal delta differences from 1
                    std_range = self.shift_param_ranges['feature_shift_params']['noise_std_scale']
                    delta = (std_range['max'] - std_range['min']) / 2.0
                    params['noise_std_scale'] = random.uniform(1 - delta, 1 + delta)

                # For conditional shift, shift beta and skewness.
                elif self.shift_type == "conditional":
                    beta_range = self.shift_param_ranges['feature_shift_params']['beta_shift']
                    params['beta_shift'] = random.uniform(beta_range['min'], beta_range['max'])
                    skew_range = self.shift_param_ranges['feature_shift_params']['skewness_shift']
                    params['skewness_shift'] = random.uniform(skew_range['min'], skew_range['max'])
                    # Additionally unchanged std_noise from the config
                    noise_range = self.shift_param_ranges['feature_shift_params']['noise_std_scale']
                    params['noise_std_scale'] = random.uniform(noise_range['min'], noise_range['max'])
                
                # For "all", we choose to apply both the covariate and conditional components.
                elif self.shift_type == "all":
                    # Apply full shift: all parameters are shifted.
                    beta_range = self.shift_param_ranges['feature_shift_params']['beta_shift']
                    params['beta_shift'] = random.uniform(beta_range['min'], beta_range['max'])
                    intercept_range = self.shift_param_ranges['feature_shift_params']['intercept_shift']
                    params['intercept_shift'] = random.uniform(intercept_range['min'], intercept_range['max'])
                    noise_range = self.shift_param_ranges['feature_shift_params']['noise_std_scale']
                    params['noise_std_scale'] = random.uniform(noise_range['min'], noise_range['max'])
                    skew_range = self.shift_param_ranges['feature_shift_params']['skewness_shift']
                    params['skewness_shift'] = random.uniform(skew_range['min'], skew_range['max'])
                
                # Only add if any parameters were chosen.
                if params:
                    shift_params = {
                        'feature_name': feature_name,
                        'parameters': params
                    }
                    feature_shifts.append(shift_params)
            if feature_shifts:
                shift_config['feature_shifts'] = feature_shifts
        
        # For label shifts: include if shift_type is "label" or "all"
        if self.shift_type in ["label", "all"]:
            rr_params = self.shift_param_ranges.get('recovery_rate_shift_params', {})
            if rr_params:
                recovery_rate_parameters = {}
                for param_name, param_range in rr_params.items():
                    if 'min' in param_range and 'max' in param_range:
                        recovery_rate_parameters[param_name] = random.uniform(param_range['min'], param_range['max'])
                    else:
                        recovery_rate_parameters[param_name] = {}
                        for sub_param, sub_range in param_range.items():
                            recovery_rate_parameters[param_name][sub_param] = random.uniform(sub_range['min'], sub_range['max'])
                shift_config['recovery_rate_shifts'] = {'parameters': recovery_rate_parameters}
        
        return shift_config
        

    def _kl_divergence(self, p_data, q_data, n_bins=30, epsilon=1e-9):
        """
        Compute the KL divergence between two distributions using histogram binning.
        """
        p_counts, bin_edges = np.histogram(p_data, bins=n_bins, density=True)
        q_counts, _ = np.histogram(q_data, bins=bin_edges, density=True)
        
        p = p_counts + epsilon
        q = q_counts + epsilon
        p /= p.sum()
        q /= q.sum()
        
        return float(np.sum(p * np.log(p / q)))

    # def _kl_divergence_categorical(self, p_data, q_data, epsilon=1e-10):
    #     categories = sorted(set(p_data.unique()) | set(q_data.unique()))
    #     p = p_data.value_counts(normalize=True).reindex(categories, fill_value=0).values
    #     q = q_data.value_counts(normalize=True).reindex(categories, fill_value=0).values
    #     p = np.clip(p, epsilon, 1)
    #     q = np.clip(q, epsilon, 1)
    #     return np.sum(p * np.log(p / q))
        
    def measure_domain_shift(
            self, 
            X_source: pd.DataFrame, 
            y_source: pd.Series, 
            X_target: pd.DataFrame, 
            y_target: pd.Series, 
            n_bins=30, 
            epsilon=1e-9,
            kl_metric: str = "average",
            weighted: bool = False
        ) -> dict:
        """
        Compute multiple domain shift scores in a single pass:
          - feature_shift_score: KL divergence (averaged) across all X columns
          - label_shift_score: KL divergence between the distribution of y_source and y_target
          - combined_shift_score: sum or aggregation of the above
    
        Returns:
            dict with keys: {
              'feature_shift_score': float,
              'label_shift_score': float,
              'combined_shift_score': float
            }
        """
        feature_kl_values = []
        weights = []


        # for col in X_source.columns:
        #     if pd.api.types.is_numeric_dtype(X_source[col]):
        #         kl_value = self._kl_divergence(X_source[col], X_target[col], n_bins=n_bins, epsilon=epsilon)
        #     else:
        #         kl_value = self._kl_divergence_categorical(X_source[col], X_target[col], epsilon=epsilon)
        #     feature_kl_values.append(kl_value)

        for col in X_source.columns:
            if not pd.api.types.is_numeric_dtype(X_source[col]):
                continue  # skip categorical columns
                
            kl_value = self._kl_divergence(X_source[col], X_target[col], n_bins=n_bins, epsilon=epsilon)
            feature_kl_values.append(kl_value)
            
            if weighted:
                corr = np.abs(np.corrcoef(X_source[col], y_source)[0, 1])
                weights.append(corr)
            else:
                weights.append(1.0)
    
        if kl_metric == "max":
            feature_shift_score = float(np.max(feature_kl_values))
        else:
            if weighted:
                feature_shift_score = float(np.average(feature_kl_values, weights=weights))
            else:
                feature_shift_score = float(np.mean(feature_kl_values))
    
        label_shift_score = self._kl_divergence(y_source, y_target, n_bins=n_bins, epsilon=epsilon)
    
        combined_shift_score = (
                feature_shift_score * len(X_source.columns) 
                + label_shift_score
            ) / (len(X_source.columns) + 1)
    
        return {
            'feature_shift_score': feature_shift_score,
            'label_shift_score': label_shift_score,
            'combined_shift_score': combined_shift_score
        }

    def run_simulation(self, simulation_id: int, target_data_size=None):

        """
        Run a single simulation with SourceOnly, TargetOnly, and TransferLearning scenarios.

        Performs train-test splits for both source and target data:
          - Source data: split into source_train/source_test
          - Target (shifted) data: split into target_train/target_test

        Evaluation sets depend on eval_all_on_target:
          - If True: SourceOnly and TransferLearning pretrain are evaluated on target_test.
          - If False: SourceOnly and TransferLearning pretrain are evaluated on source_test.
          - TargetOnly and TransferLearning fine-tune always evaluated on target_test.

        Args:
            simulation_id (int): The ID/index of the simulation run.
            target_data_size (int or None): If specified, overrides the target domain size.
        """
        # 1) Generate shift config
        shift_config = self.generate_random_shift_config()
        source_config = deepcopy(self.source_config)
        
        MIN_TEST_SAMPLES = 1000

        # 2) Generate source data
        data_generator = SimulationDataGeneration(source_config)
        df_source = data_generator.generate_data()
        X_source = df_source.drop(columns=['recovery_rate', 'secured'])
        y_source = df_source['recovery_rate']
    
        
        # 3) Generate shifted target data from 'source_config' override
        # If user specified a target_data_size => total_target_needed = target_data_size + 1000
        if target_data_size is not None:
            total_target_needed = target_data_size + MIN_TEST_SAMPLES
            if self.verbose>0:
                print(f"[run_simulation] Requested {target_data_size} for target train => generating "
                  f"{total_target_needed} total target samples (so leftover ≥ {MIN_TEST_SAMPLES} for test).")
            source_config['n_samples'] = total_target_needed
        shifted_data_generator = ShiftedSimulationDataGeneration(source_config, shift_config)
        df_shifted = shifted_data_generator.generate_data()
        X_shifted = df_shifted.drop(columns=['recovery_rate', 'secured'])
        y_shifted = df_shifted['recovery_rate']

        # ───────────────────────────────────────────────────────────── hetero-features
        # choose mode
        mode = self.hetero_feature_mode

        # choose which columns may be dropped (only those declared "shiftable")
        eligible = [
            f for f, cfg in self.source_config["features"].items()
            if cfg.get("shiftable", True)
        ]
        k = math.ceil(self.drop_fraction * len(eligible))
        cols_to_drop = random.sample(eligible, k) if k else []
        
        # apply
        if mode == "source_subset" and cols_to_drop:
            X_source  = self._drop_columns(X_source,  cols_to_drop)
        elif mode == "target_subset" and cols_to_drop:
            X_shifted = self._drop_columns(X_shifted, cols_to_drop)
        # "equal"  → keep all columns

        print(f"X_source cols: {X_source.columns}")
        print(f"X_shifted cols: {X_shifted.columns}")
        
        # finally keep only the intersection so every later loop is safe
        shared_cols = X_source.columns.intersection(X_shifted.columns)
        shared_num = self._shared_numeric_cols(X_source, X_shifted)
        self.shared_cols = shared_cols
        self.shared_numeric_cols = shared_num


        
        #X_source   = X_source[common_cols]
        #X_shifted  = X_shifted[common_cols]
        # ─────────────────────────────────────────────────────────────────────────────

        # Create histograms:
        self.plot_source_target_distributions(X_source, y_source, X_shifted, y_shifted, simulation_id)
        
        # 4a) Measure domain shift
        shift_scores = self.measure_domain_shift(
            X_source[shared_num], y_source,
            X_shifted[shared_num], y_shifted,
            weighted=True
        )
        domain_shift_score = shift_scores["combined_shift_score"]

        # 4b) Compute the average absolute correlation for source and target
        source_corrs, target_corrs = [], []
        for col in shared_num:
            xs = X_source[col]; xt = X_shifted[col]
            # guard against degenerate variance
            if xs.nunique(dropna=True) < 2 or xt.nunique(dropna=True) < 2:
                continue
            source_corrs.append(abs(np.corrcoef(xs, y_source)[0, 1]))
            target_corrs.append(abs(np.corrcoef(xt, y_shifted)[0, 1]))
        
        if source_corrs:
            mean_source_corr = float(np.mean(source_corrs))
            std_source_corr = float(np.std(source_corrs))
        else:
            mean_source_corr = std_source_corr = np.nan
        
        if target_corrs:
            mean_target_corr = float(np.mean(target_corrs))
            std_target_corr = float(np.std(target_corrs))
        else:
            mean_target_corr = std_target_corr = np.nan
        

        if self.verbose>1:
            print(f"Mean Source Correlation: {mean_source_corr:.3f} ± {std_source_corr:.3f}")
            print(f"Mean Target Correlation: {mean_target_corr:.3f} ± {std_target_corr:.3f}")
        
        # Prepare extra info to be stored with this simulation run:
        simulation_correlations = {
            'mean_source_corr': mean_source_corr,
            'std_source_corr': std_source_corr,
            'mean_target_corr': mean_target_corr,
            'std_target_corr': std_target_corr
        }

        # 4c) Plot correlations
        self.plot_label_feature_correlation(X_source, y_source, X_shifted, y_shifted, simulation_id)
        
        # 5) Manually partition target data if user specified target_data_size
        if target_data_size is not None:
            # Combine X,y => randomize => slice
            target_df = X_shifted.copy()
            target_df["y"] = y_shifted.values
            target_df = target_df.sample(frac=1, random_state=simulation_id).reset_index(drop=True)

            X_target_train = target_df.iloc[:target_data_size].drop(columns="y")
            y_target_train = target_df.iloc[:target_data_size]["y"]
            X_target_test  = target_df.iloc[target_data_size:].drop(columns="y")
            y_target_test  = target_df.iloc[target_data_size:]["y"]

            if len(X_target_test) < MIN_TEST_SAMPLES:
                raise ValueError(f"[run_simulation] leftover test rows={len(X_target_test)} < {MIN_TEST_SAMPLES}!")
        else:
            # Standard approach if target_data_size not specified
            X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
                X_shifted, y_shifted, test_size=0.25, random_state=simulation_id
            )
        
        # 6) Perform train-test split for source data
        X_source_train, X_source_test, y_source_train, y_source_test = train_test_split(
            X_source, y_source, test_size=0.25, random_state=simulation_id
        )

        if self.verbose > 0:
            bold = "\033[1m"
            reset = "\033[0m"
            print(f"{simulation_id}: {bold}Source{reset} - Train: {X_source_train.shape}, {y_source_train.shape} | Test: {X_source_test.shape}, {y_source_test.shape} || {bold}Target{reset} - Train: {X_target_train.shape}, {y_target_train.shape} | Test: {X_target_test.shape}, {y_target_test.shape}")


        # 7) Store datasets for reproducibility
        if self.store_datasets:
            
            timestamp = self._timestamp()
            path = os.path.join(self.data_folder, f"run_{timestamp}_{simulation_id}")

            os.makedirs(path, exist_ok=True)
            
            # Save datasets to CSV files with the timestamp
            X_source_train.to_csv(os.path.join(path, f"X_source_train_sim_{timestamp}_{simulation_id}.csv"), index=False)
            y_source_train.to_csv(os.path.join(path, f"y_source_train_sim_{timestamp}_{simulation_id}.csv"), index=False)
            X_source_test.to_csv(os.path.join(path, f"X_source_test_sim_{timestamp}_{simulation_id}.csv"), index=False)
            y_source_test.to_csv(os.path.join(path, f"y_source_test_sim_{timestamp}_{simulation_id}.csv"), index=False)

            X_target_train.to_csv(os.path.join(path, f"X_target_train_sim_{timestamp}_{simulation_id}.csv"), index=False)
            y_target_train.to_csv(os.path.join(path, f"y_target_train_sim_{timestamp}_{simulation_id}.csv"), index=False)
            X_target_test.to_csv(os.path.join(path, f"X_target_test_sim_{timestamp}_{simulation_id}.csv"), index=False)
            y_target_test.to_csv(os.path.join(path, f"y_target_test_sim_{timestamp}_{simulation_id}.csv"), index=False)

            print(f"[run_simulation] Datasets for simulation {simulation_id} saved to {path} with timestamp {timestamp}.")

        # 8) For each model in the lists, run all scenarios
        for model_idx, (m_class, m_params) in enumerate(zip(self.model_classes, self.model_params_list)):

            mse_source_only = r2_source_only = mae_source_only = np.nan
            mse_target_only = r2_target_only = mae_target_only = np.nan
            mse_transfer_learning = r2_transfer_learning = mae_transfer_learning = np.nan
            source_only_metrics = target_only_metrics = transfer_learning_metrics = np.nan

            source_params = deepcopy(m_params)      
            target_params  = deepcopy(m_params)           
            transfer_params  = deepcopy(m_params)
            if 'epochs' in target_params:                 
                target_params['epochs'] = int(target_params['epochs'] / 2)              
            transfer_params['additional_epochs'] = int(transfer_params['epochs'] / 2)
            
            # If this is the TabPFN model, only train on every nth simulation (e.g. every 3rd)
            is_tabpfn = (self.model_names[model_idx] == "TabPFN")
            if is_tabpfn and (simulation_id % 3 != 0):
                continue
                
            # Get running scenario for this model
            scenarios_for_this_model = self.scenarios[model_idx]
            
            # =============================
            # 7a) SourceOnly scenario
            # =============================
            if "full" in scenarios_for_this_model or "source_only" in scenarios_for_this_model:
                if self.verbose>1:
                    print(f"[run_simulation] Running {self.model_names[model_idx]} in source_only mode.")
                model_source_only = m_class(**source_params, seed=simulation_id, sample_df=X_source_train, verbose=False)

                if self.eval_all_on_target:
                    # Train on source_train, evaluate on target_test
                    model_source_only.train(X_source_train, y_source_train, 
                                            X_target_test, y_target_test)
                    y_pred_source_only = model_source_only.predict(X_target_test)
                    mse_source_only = mean_squared_error(y_target_test, y_pred_source_only)
                    r2_source_only = r2_score(y_target_test, y_pred_source_only)
                    mae_source_only = mean_absolute_error(y_target_test, y_pred_source_only)
                else:
                    # Train on source_train, evaluate on source_test
                    model_source_only.train(X_source_train, y_source_train, 
                                            X_source_test, y_source_test)
                    y_pred_source_only = model_source_only.predict(X_source_test)
                    mse_source_only = mean_squared_error(y_source_test, y_pred_source_only)
                    r2_source_only = r2_score(y_source_test, y_pred_source_only)
                    mae_source_only = mean_absolute_error(y_source_test, y_pred_source_only)

                # Store the training losses (if present)
                source_only_metrics = model_source_only.get_epoch_metrics()
                self.source_only_losses[model_idx].append(
                    source_only_metrics.get('loss_pretrain', [])
                )
            
            # =============================
            # 7b) TargetOnly scenario
            # =============================
            if "full" in scenarios_for_this_model or "target_only" in scenarios_for_this_model:
                if self.verbose>1:
                    print(f"[run_simulation] Running {self.model_names[model_idx]} in target_only mode.")
                model_target_only = m_class(**target_params, seed=simulation_id, sample_df=X_target_train, verbose=False)
                model_target_only.train(X_target_train, y_target_train, 
                                        X_target_test, y_target_test)
                y_pred_target_only = model_target_only.predict(X_target_test)
                mse_target_only = mean_squared_error(y_target_test, y_pred_target_only)
                r2_target_only = r2_score(y_target_test, y_pred_target_only)
                mae_target_only = mean_absolute_error(y_target_test, y_pred_target_only)
                target_only_metrics = model_target_only.get_epoch_metrics()
                self.target_only_losses[model_idx].append(
                    target_only_metrics.get('loss_pretrain', [])
                )

            # =============================
            # 7c) TransferLearning scenario
            # =============================
            if "full" in scenarios_for_this_model or "transfer" in scenarios_for_this_model:
                if self.verbose>1:
                    print(f"[run_simulation] Running {self.model_names[model_idx]} in transfer_learning mode.")

                try:
                    
                    model_transfer_learning = m_class(**transfer_params, seed=simulation_id, sample_df=X_source_train, verbose=False)
                    if hasattr(model_transfer_learning, "fine_tune"):
                        # ---- TWO-STEP TRANSFER ----
                        # Pretrain on source
                        if self.eval_all_on_target:
                            # Evaluate pretraining phase on target_test
                            model_transfer_learning.train(X_source_train, y_source_train, X_target_test, y_target_test)
                        else:
                            # Evaluate pretraining phase on source_test
                            model_transfer_learning.train(X_source_train, y_source_train, X_source_test, y_source_test)
    
                        # Fine-tune on target
                        #model_transfer_learning.verbose = True
                        model_transfer_learning.fine_tune(X_target_train, y_target_train, X_target_test, y_target_test, **transfer_params)
                        y_pred_transfer_learning = model_transfer_learning.predict(X_target_test)
                        mse_transfer_learning = mean_squared_error(y_target_test, y_pred_transfer_learning)
                        r2_transfer_learning = r2_score(y_target_test, y_pred_transfer_learning)
                        mae_transfer_learning = mean_absolute_error(y_target_test, y_pred_transfer_learning)
    
                        # Store epoch-wise metrics
                        transfer_learning_metrics = model_transfer_learning.get_epoch_metrics()
                        self.transfer_learning_pretrain_losses[model_idx].append(
                            transfer_learning_metrics.get('loss_pretrain', [])
                        )
                        self.transfer_learning_finetune_losses[model_idx].append(
                            transfer_learning_metrics.get('loss_finetune', [])
                        )
    
                    else:
                        # ---- DIRECT TRANSFER ----
                        # Possibly do a direct/single-step training that includes both source + target data
    
                        model_transfer_learning.train(
                            X_source_train, y_source_train, 
                            X_target_test, y_target_test, 
                            X_target_train, y_target_train
                        )
    
                        y_pred_transfer_learning = model_transfer_learning.predict(X_target_test)
                        mse_transfer_learning = mean_squared_error(y_target_test, y_pred_transfer_learning)
                        r2_transfer_learning = r2_score(y_target_test, y_pred_transfer_learning)
                        mae_transfer_learning = mean_absolute_error(y_target_test, y_pred_transfer_learning)
    
                        # For a one-step model, there's no separate "pretrain vs. finetune" metrics
                        transfer_learning_metrics = model_transfer_learning.get_epoch_metrics()
                        # We might store in finetune losses or keep them empty. 
                        self.transfer_learning_pretrain_losses[model_idx].append(
                            transfer_learning_metrics.get('loss_pretrain', [])
                        )
                        self.transfer_learning_finetune_losses[model_idx].append(
                            transfer_learning_metrics.get('loss_finetune', [])
                        )
                        
                except Exception as e:                       
                    if self.verbose:
                        print(f"\033[93m[run_simulation] {self.model_names[model_idx]} "
                              f"TRANSFER failed: {e} – skipping this scenario.\033[0m")

            
            # =============================
            # 7d) Append row to self.results
            # =============================
            self.results.append({
                'simulation_id': simulation_id,
                'model_index': model_idx,
                'model_name': self.model_names[model_idx] if self.model_names else str(m_class.__name__)+"_"+str(model_idx),
                'feature_shift_score': shift_scores['feature_shift_score'],
                'label_shift_score': shift_scores['label_shift_score'],
                'combined_shift_score': shift_scores['combined_shift_score'],
                'mse_source_only': mse_source_only,
                'r2_source_only': r2_source_only,
                'mae_source_only': mae_source_only,
                'mse_target_only': mse_target_only,
                'r2_target_only': r2_target_only,
                'mae_target_only': mae_target_only,
                'mse_transfer_learning': mse_transfer_learning,
                'r2_transfer_learning': r2_transfer_learning,
                'mae_transfer_learning': mae_transfer_learning,
                'metrics_source_only': source_only_metrics,
                'metrics_target_only': target_only_metrics,
                'metrics_transfer_learning': transfer_learning_metrics,
                **simulation_correlations
            })

    def run(self, eval_all_on_target=False, target_data_size=None, progress_counter=None, counter_lock=None):
        """
        Run all simulations.
    
        Args:
            eval_all_on_target (bool): If True, SourceOnly and TransferLearning pretrain phases
                are evaluated on target test set instead of source test set.
            target_data_size (int or None): If specified, overrides the target domain size.
            progress_counter (multiprocessing.Value): Shared counter for progress tracking.
        """
        self.eval_all_on_target = eval_all_on_target
        # Use tqdm only if not using the shared counter (or disable internal progress if desired)
        for i in range(self.num_simulations):
            try:
                self.run_simulation(i, target_data_size)
            except Exception as e:
                print(f"\033[91m[run] Simulation {i} failed with error: {e}\033[0m")
                continue
            # Update the shared progress counter (if provided)
            if progress_counter is not None and counter_lock is not None:
                with counter_lock:
                    progress_counter.value += 1

    
        self.results_df = pd.DataFrame(self.results)
        # Save results
        filename = os.path.join(self.results_folder, f"raw_simulation_results_{self._timestamp()}.csv")
        self.results_df.to_csv(filename, index=False)

        if self.verbose>0:
            print(f"[run] Raw simulation results saved to '{filename}'")
        
    def run_varying_data_sizes(self, data_sizes, eval_all_on_target=False, epochs=None):
        """
        Run multiple simulations, each with a different training data size.
        Optional: Overwrites the 'epochs' entry in each model's param dict if desired.
        """

        # Update epochs for each model's param dict
        if epochs:
            for m_params in self.model_params_list:
                if 'epochs' in m_params:
                    m_params['epochs'] = epochs

        all_results = []  # Collect all results for concatenation
        
        def choose_num_simulations(data_size: int) -> int:
            if data_size <= 100:
                return min(self.num_simulations, 25)
            elif data_size <= 500:
                return min(self.num_simulations, 20)
            elif data_size <= 1000:
                return min(self.num_simulations, 15)
            else:
                return min(self.num_simulations, 10)  

        for ds in data_sizes:
            # 1) Adjust num_simulations based on the data size
            self.num_simulations = choose_num_simulations(ds)
            print(f"\n--- Running {self.num_simulations} simulations with source_data size={self.source_config['n_samples']} and target_data_size={ds} ---")

            # Clear previous results/loss arrays before each run
            self.results = []
            for i in range(len(self.model_classes)):
                self.source_only_losses[i] = []
                self.target_only_losses[i] = []
                self.transfer_learning_pretrain_losses[i] = []
                self.transfer_learning_finetune_losses[i] = []

            # Run the usual Monte Carlo loop
            self.run(eval_all_on_target=eval_all_on_target, target_data_size=ds)

            # Tag the results with data_size
            self.results_df['data_size'] = ds
            all_results.append(self.results_df.copy())
            
            print(self.summarize_results()[['model_index','r2_source_only_mean','r2_target_only_mean','r2_transfer_learning_mean']])

        # Combine into one DataFrame
        self.results_df = pd.concat(all_results, ignore_index=True)

        # Save results
        filename = os.path.join(self.results_folder, f"raw_simulation_results_varying_data_sizes_{self._timestamp()}.csv")
        self.results_df.to_csv(filename, index=False)
        if self.verbose>0:
            print(f"[run_varying_data_sizes] Raw simulation results for varying data sizes saved to '{filename}'")
        
        return self.results_df

    
    def summarize_results(self):
        """
        Mean ± std of MSE/R²/MAE per scenario, grouped by model name.
        Returns empty DataFrame if no results (e.g., all sims failed early).
        """
        import os
        import pandas as pd
    
        if not hasattr(self, "results_df") or self.results_df is None or self.results_df.empty:
            print("[summarize_results] No results to summarize (results_df is empty).")
            return pd.DataFrame()
    
        metrics = ['mse', 'r2', 'mae']
        scenarios = ['source_only', 'target_only', 'transfer_learning']
        group_cols = ['model_name'] if ('model_name' in self.results_df.columns) else ['model_index']
    
        summaries = []
        for name, group_df in self.results_df.groupby(group_cols):
            summary = {'model_index': name}
            for scenario in scenarios:
                for metric in metrics:
                    col = f'{metric}_{scenario}'
                    summary[f'{col}_mean'] = group_df[col].mean()
                    summary[f'{col}_std'] = group_df[col].std()
            summaries.append(summary)
    
        summary_df = pd.DataFrame(summaries)
        filename = os.path.join(self.results_folder, f"summarized_results_sim{self.num_simulations}_{self._timestamp()}.csv")
        summary_df.to_csv(filename, index=True)
        if self.verbose > 0:
            print(f"[summarize_results] Summarized results saved to {filename}")
        return summary_df

    def remove_outlier_simulations(self, column='feature_shift_score', lower_quantile=0.05, upper_quantile=0.95):
        """
        Remove simulation runs where the specified column value is outside the defined quantile range.
        This method updates self.results_df by filtering out outliers.
        
        Args:
            column (str): The column name to check for outliers (e.g., 'feature_shift_score').
            lower_quantile (float): The lower quantile cutoff (default 0.05).
            upper_quantile (float): The upper quantile cutoff (default 0.95).
        
        Returns:
            pd.DataFrame: The filtered results DataFrame.
        """
        if column not in self.results_df.columns:
            print(f"Column '{column}' not found in results_df.")
            return self.results_df
        
        lower_bound = self.results_df[column].quantile(lower_quantile)
        upper_bound = self.results_df[column].quantile(upper_quantile)
        
        filtered_df = self.results_df[(self.results_df[column] >= lower_bound) & (self.results_df[column] <= upper_bound)]
        print(f"Filtered out {len(self.results_df) - len(filtered_df)} outlier simulation runs based on '{column}'.")
        
        self.results_df = filtered_df.copy()
        return self.results_df


    def plot_results(self, metric=None, show_source_only=True): 
        """
        Plot boxplots of MSE, R², and MAE across simulations for each scenario,
        coloring/hue by the model_name instead of model_index.
        """
        if metric is None:
            metrics = ['mse', 'r2', 'mae']
        else:
            metrics = [metric]
        if show_source_only:
            scenarios = ['source_only', 'target_only', 'transfer_learning']
        else:
            scenarios = ['target_only', 'transfer_learning']

        # Melt the DataFrame, including 'model_name' rather than 'model_index'
        df_melted = self.results_df.melt(
            id_vars=['simulation_id', 'model_name'],  # <- Use model_name here
            value_vars=[f'{metric}_{scenario}' for scenario in scenarios for metric in metrics],
            var_name='metric_scenario',
            value_name='value'
        )

        # Extract columns 'metric' (mse, r2, mae) and 'scenario' (source_only, target_only, etc.)
        df_melted[['metric', 'scenario']] = df_melted['metric_scenario'].str.extract(r'^(mse|r2|mae)_(.+)$')

        # One figure per metric
        for metric in metrics:
            filtered_data = df_melted[df_melted['metric'] == metric]
            if filtered_data.empty:
                print(f"No data available for metric: {metric}")
                continue

            plt.figure(figsize=(10, 6))
            sns.boxplot(
                x='scenario', 
                y='value', 
                hue='model_name',   # <-- hue by model_name
                data=filtered_data
            )
            plt.title(f'Distribution of {metric.upper()} across Scenarios ({self.shift_type})')
            plt.ylabel(metric.upper())
            plt.xlabel('Scenario')
            plt.legend(title='Model', loc='best')  # Legend title = 'Model'
            plt.tight_layout()
            
            # Save figures
            filename = os.path.join(self.figures_folder, f"{metric}_boxplot_{self._timestamp()}.pdf")

            plt.savefig(filename, dpi=300, bbox_inches='tight')
            if self.verbose>0:
                print(f"[plot_results] Boxplot for '{metric}' saved to {filename}")

            plt.show()

            
    def plot_epoch_progress(self):
        """
        Plot average training loss over epochs for each model, for each scenario:
        - SourceOnly
        - TargetOnly
        - TransferLearning Fine-tune
        """

        def average_loss_curves(loss_list_of_lists):
            """Given a list of loss curves, average them into a single curve."""
            if not loss_list_of_lists:
                return None
            # Filter out empty
            filtered = [lst for lst in loss_list_of_lists if lst]
            if not filtered:
                return None
            max_len = max(len(lst) for lst in filtered)
            padded = [lst + [lst[-1]]*(max_len - len(lst)) for lst in filtered]
            avg = [sum(x)/len(x) for x in zip(*padded)]
            return avg

        # We'll store references to each scenario's losses plus the line style
        scenario_dict = {
            'SourceOnly':              (self.source_only_losses,            '--'),
            'TargetOnly':              (self.target_only_losses,            '-.'),
            'TransferLearning Fine-tune': (self.transfer_learning_finetune_losses, '-'),
        }

        # Define a color palette for the models
        model_colors = [
            "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
            "tab:brown", "tab:pink",  "tab:gray",  "tab:olive", "tab:cyan"
        ]

        plt.figure(figsize=(10, 6))

        # For each model index, plot one line per scenario
        for model_idx in self.source_only_losses:  
            # Determine the color for this model
            color = model_colors[model_idx % len(model_colors)]
            # Use model name if available, else fallback
            model_name = (self.model_names[model_idx] 
                          if getattr(self, 'model_names', None) 
                          else f"Model {model_idx}")

            # Loop over each scenario
            for scenario_name, (loss_dict, line_style) in scenario_dict.items():
                # Average the loss curves across simulations for this model & scenario
                avg_loss = average_loss_curves(loss_dict[model_idx])
                if avg_loss is not None:
                    plt.plot(
                        avg_loss,
                        label=f"{model_name} - {scenario_name}",
                        color=color,
                        linestyle=line_style
                    )

        plt.title(f'Average Training Loss over Epochs ({self.shift_type})')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.legend()
        plt.tight_layout()

        # Save figure
        filename = os.path.join(self.figures_folder, f"loss_over_epochs_sim{self.num_simulations}_{self._timestamp()}.pdf")

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        if self.verbose>0:
            print(f"[plot_epoch_progress] Figure saved to '{filename}'")
        
        plt.show()


    def plot_epoch_metric(self, metric='r2', show_source_models=True, ylim_cut=-0.2, save_path=None):
        """
        Plot epoch-wise metrics across scenarios, per model.

        - If metric includes '_pretrain' or '_finetune', plot only that phase for TransferLearning.
        - Otherwise, plot both phases if available.
        - SourceOnly/TargetOnly have just one training phase stored under 'pretrain'.
        - If show_source_models=False, skip SourceOnly and TransferLearning Pretrain lines.
        - Each model has a unique color; the scenario is indicated by line style.
        - Legend shows model_name (or 'Model {idx}') + scenario label.
        """

        # Define how to get scenario names from internal keys
        base_scenarios = {
            'metrics_source_only':       'SourceOnly',
            'metrics_target_only':       'TargetOnly',
            'metrics_transfer_learning': 'TransferLearning'
        }

        # Predefine line styles for each scenario variation
        # e.g. TransferLearning pretrain vs. fine-tune
        scenario_line_styles = {
            'SourceOnly':         '--',
            'TargetOnly':         '-.',
            'TL Pretrain':        ':',
            'TL Fine-tune':       '-',
        }

        # Store epoch metrics in a nested dict:
        scenario_data = {}
        for model_idx in range(len(self.model_classes)):
            scenario_data[model_idx] = {}
            for s in base_scenarios:
                scenario_data[model_idx][s] = {'pretrain': [], 'finetune': []}

        # Populate scenario_data with metric lists
        for result in self.results:
            m_idx = result['model_index']

            # For each scenario (source_only, target_only, transfer_learning)
            for s in base_scenarios:
                metric_dict = result[s]
                if not metric_dict:
                    continue

                # Check if the user requested a specialized metric, e.g. 'r2_pretrain'
                is_pretrain = ('_pretrain' in metric)
                is_finetune = ('_finetune' in metric)

                if is_pretrain or is_finetune:
                    # The metric is exactly "xxx_pretrain" or "xxx_finetune"
                    val = metric_dict.get(metric, None)
                    if val is not None:
                        if is_pretrain:
                            scenario_data[m_idx][s]['pretrain'].append(val)
                        else:
                            scenario_data[m_idx][s]['finetune'].append(val)
                else:
                    # The metric is "xxx", so look for "xxx_pretrain" and "xxx_finetune"
                    val_pre = metric_dict.get(metric + '_pretrain', None)
                    val_fin = metric_dict.get(metric + '_finetune', None)
                    if val_pre is not None:
                        scenario_data[m_idx][s]['pretrain'].append(val_pre)
                    if val_fin is not None:
                        scenario_data[m_idx][s]['finetune'].append(val_fin)

        # Helper to average multiple runs (lists) into one curve
        def average_curves(list_of_lists):
            if not list_of_lists:
                return None
            filtered = [lst for lst in list_of_lists if lst]
            if not filtered:
                return None
            max_len = max(len(lst) for lst in filtered)
            # Pad shorter runs so we can average them
            padded = [lst + [lst[-1]]*(max_len - len(lst)) for lst in filtered]
            avg = [sum(x)/len(x) for x in zip(*padded)]
            return avg

        # Predefine a color palette for models (one color per model)
        # You can add more if you have more than 10 models
        model_colors = [
            "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
            "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"
        ]

        plt.figure(figsize=(10, 6))

        # Plot lines for each model
        for model_idx in scenario_data:
            # Pick color for this model
            color = model_colors[model_idx % len(model_colors)]
            # Retrieve the model name (or fallback)
            model_name = (self.model_names[model_idx]
                          if self.model_names
                          else f"Model {model_idx}")

            # ----- SourceOnly scenario -----
            so_pre = average_curves(scenario_data[model_idx]['metrics_source_only']['pretrain'])
            # There's no finetune for SourceOnly
            if show_source_models and so_pre is not None:
                plt.plot(so_pre,
                         label=f"{model_name} - SourceOnly",
                         linestyle=scenario_line_styles['SourceOnly'],
                         color=color)

            # ----- TargetOnly scenario -----
            to_pre = average_curves(scenario_data[model_idx]['metrics_target_only']['pretrain'])
            if to_pre is not None:
                plt.plot(to_pre,
                         label=f"{model_name} - TargetOnly",
                         linestyle=scenario_line_styles['TargetOnly'],
                         color=color)

            # ----- TransferLearning scenario -----
            tl_pre = average_curves(scenario_data[model_idx]['metrics_transfer_learning']['pretrain'])
            tl_fin = average_curves(scenario_data[model_idx]['metrics_transfer_learning']['finetune'])

            # If user specifically asked for _pretrain or _finetune, we only plot that
            # If neither, we plot both if available
            if '_pretrain' in metric:
                if show_source_models and tl_pre is not None:
                    plt.plot(tl_pre,
                             label=f"{model_name} - TL Pretrain",
                             linestyle=scenario_line_styles['TL Pretrain'],
                             color=color)
            elif '_finetune' in metric:
                if tl_fin is not None:
                    plt.plot(tl_fin,
                             label=f"{model_name} - TL Fine-tune",
                             linestyle=scenario_line_styles['TL Fine-tune'],
                             color=color)
            else:
                # Generic metric => plot both phases (if they exist)
                if show_source_models and tl_pre is not None:
                    plt.plot(tl_pre,
                             label=f"{model_name} - TL Pretrain",
                             linestyle=scenario_line_styles['TL Pretrain'],
                             color=color)
                if tl_fin is not None:
                    plt.plot(tl_fin,
                             label=f"{model_name} - TL Fine-tune",
                             linestyle=scenario_line_styles['TL Fine-tune'],
                             color=color)

        # Final styling
        plt.ylim(bottom=ylim_cut)
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'Average {metric} over Epochs (Per Model)')
        plt.legend()
        plt.tight_layout()

        # Save figure
        filename = os.path.join(self.figures_folder, f"avg_r2_over_epochs_sim{self.num_simulations}_{self._timestamp()}.pdf")

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        if self.verbose>0:
            print(f"[plot_epoch_metric] Figure saved to {filename}")

        plt.show()

    
    def plot_shift_vs_r2_correlation(self,
                                     shift_cols=None,
                                     r2_cols=None,
                                     dropna=True):
        """
        Create a small correlation table/heatmap that shows how each shift metric 
        (rows) correlates with each R² metric (columns). 
        This avoids the redundancy of a large NxN correlation matrix.
    
        Args:
            shift_cols (list of str or None): Which shift columns to include. 
                If None, uses ['feature_shift_score', 'label_shift_score', 'combined_shift_score'] 
                (only if they exist in results_df).
            r2_cols (list of str or None): Which R² columns to include. 
                If None, uses ['r2_source_only', 'r2_target_only', 'r2_transfer_learning'] 
                (only if they exist).
            dropna (bool): Whether to drop rows with any NaN in those columns (default True).
        """

        df = self.results_df.copy()
    
        # 1) Default columns if not specified
        if shift_cols is None:
            shift_cols = ["feature_shift_score", "label_shift_score", "combined_shift_score"]
        if r2_cols is None:
            r2_cols = ["r2_source_only", "r2_target_only", "r2_transfer_learning"]
    
        # Filter out only columns that exist
        shift_cols = [c for c in shift_cols if c in df.columns]
        r2_cols    = [c for c in r2_cols if c in df.columns]
    
        # If no valid columns, bail
        if not shift_cols or not r2_cols:
            print("[plot_shift_vs_r2_correlation] No valid shift or R² columns found.")
            return
    
        # 2) Possibly drop rows with NaN in any of these columns
        all_cols = shift_cols + r2_cols
        if dropna:
            df = df.dropna(subset=all_cols)
        if df.empty:
            print("[plot_shift_vs_r2_correlation] DataFrame is empty after dropping NaNs.")
            return
    
        # 3) Build a small correlation table: shape = (len(shift_cols), len(r2_cols))
        corrs = pd.DataFrame(index=shift_cols, columns=r2_cols, dtype=float)
    
        for shift_col in shift_cols:
            for metric_col in r2_cols:
                # Compute correlation
                corr_val = df[[shift_col, metric_col]].corr().iloc[0, 1]  # correlation between the two columns
                corrs.loc[shift_col, metric_col] = corr_val
    
        # 4) Plot it as a heatmap
        plt.figure(figsize=(1.5 * len(r2_cols), 1.2 * len(shift_cols)))
        sns.heatmap(
            corrs, 
            annot=True, 
            cmap="coolwarm", 
            fmt=".2f", 
            vmin=-1, vmax=1,
            cbar=True
        )
        plt.title(f"Correlation: Shift vs. R² ({self.shift_type})")
        plt.tight_layout()
        
        # Save figure
        filename = os.path.join(self.figures_folder, f"corr_r2_vs_shift_sim{self.num_simulations}_{self._timestamp()}.pdf")

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        if self.verbose>0:
            print(f"[plot_shift_vs_r2_correlation] Figure saved to {filename}")
            plt.show()
        else:
            plt.close()


    def plot_source_target_distributions(self,
                                     X_source: pd.DataFrame,
                                     y_source: pd.Series,
                                     X_target: pd.DataFrame,
                                     y_target: pd.Series,
                                     simulation_id):
        """
        Histogram plots comparing source vs target, but only for columns present in both.
        Saves a PDF; suppresses display unless verbose>2.
        """
        import math
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        import os
    
        # Only plot shared feature columns
        shared = list(X_source.columns.intersection(X_target.columns))
        if not shared:
            print("[plot_source_target_distributions] No shared feature columns; skipping.")
            return
    
        columns = shared + ["recovery_rate"]  # include label
        n_cols = 4
        n_feats = len(columns)
        n_rows = math.ceil(n_feats / n_cols)
    
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = axes.flatten()
    
        for i, col in enumerate(columns):
            ax = axes[i]
            if col == "recovery_rate":
                src = pd.Series(y_source)
                tgt = pd.Series(y_target)
            else:
                src = X_source[col]
                tgt = X_target[col]
    
            # skip empty/NaN-only
            if src.dropna().empty and tgt.dropna().empty:
                ax.set_visible(False)
                continue
    
            sns.histplot(src, bins=30, color="#00376c", alpha=0.5, label="Source",
                         ax=ax, stat="density", kde=False)
            sns.histplot(tgt, bins=30, color="#EF7C00", alpha=0.5, label="Target",
                         ax=ax, stat="density", kde=False)
            ax.set_title(col)
            ax.legend()
    
        # Hide any unused subplot axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
    
        fig.suptitle(f"Source vs. Target Distributions ({self.shift_type})", y=1.02)
        plt.tight_layout()
        filename = os.path.join(self.figures_folder, f"source_target_distributions_{simulation_id}.pdf")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        if self.verbose > 2:
            plt.show()
        else:
            plt.close()


        
    def plot_r2_vs_data_size(self, ylim_cut=-0.2):
        """
        Plot R² vs. data size, grouping by scenario (TargetOnly, TransferLearning),
        and distinguishing each model by name with different color lines and scenario line styles.
        Each model gets a unique color from the default Matplotlib palette.
        """
        
        data_size_col='data_size'
        if data_size_col not in self.results_df.columns:
            print(f"Column '{data_size_col}' not found in results_df. Please ensure it exists.")
            return

        # R² for these two scenarios:
        metric = 'r2'
        scenarios = ['transfer_learning', 'target_only']  # In this order
        metric_cols = [f'{metric}_{s}' for s in scenarios]

        # 1) Reshape data so we have columns: (simulation_id, model_name, data_size_col, scenario, value)
        df_melted = self.results_df.melt(
            id_vars=['simulation_id', 'model_name', data_size_col],
            value_vars=metric_cols,
            var_name='metric_scenario',
            value_name='value'
        )
        df_melted = df_melted.dropna()

        # 2) Extract scenario from metric_scenario (e.g. 'r2_transfer_learning' -> 'transfer_learning')
        df_melted[['dummy', 'scenario']] = df_melted['metric_scenario'].str.extract(r'^(r2)_(.+)$')
        df_melted.drop(columns=['dummy'], inplace=True)

        # 3) Group by (model_name, scenario, data_size_col) to compute mean R²
        grouped = df_melted.groupby(['model_name', 'scenario', data_size_col])['value'].mean().reset_index()

        # 4) Plot setup
        fig, ax = plt.subplots(figsize=(10, 6))

        # Distinguish line style by scenario
        scenario_styles = {
            'transfer_learning': '-',
            'target_only': '--',
        }

        # 5) Let matplotlib choose one color per model from the default palette
        # Pull a color from the axis prop cycle for each model in the order they appear
        unique_model_names = grouped['model_name'].unique()
        color_map = {}
        for model_name in unique_model_names:
            color_map[model_name] = next(ax._get_lines.prop_cycler)['color']

        # 6) Plot each model-scenario line
        for model_name in unique_model_names:
            model_color = color_map[model_name]

            for scenario in scenarios:
                sub_df = grouped[(grouped['model_name'] == model_name) & (grouped['scenario'] == scenario)]
                if sub_df.empty:
                    continue

                # Decide a nice label
                scenario_label = "Transfer Learning Fine-tune" if scenario == "transfer_learning" else "Target Only"

                ax.plot(
                    sub_df[data_size_col],
                    sub_df['value'],
                    linestyle=scenario_styles[scenario],
                    color=model_color,
                    linewidth=2,
                    label=f"{model_name} - {scenario_label}"
                )

        ax.set_title(f'R² vs. Target Data Size ({self.shift_type})')
        ax.set_xlabel('Target Data Size')
        ax.set_ylabel('R²')
        ax.set_ylim(bottom=ylim_cut)
        ax.legend(title='Model & Scenario', loc='best')
        plt.tight_layout()

        # 7) Save figure
        filename = os.path.join(self.figures_folder, f"r2_vs_target_data_size_sim{self.num_simulations}_{self._timestamp()}.pdf")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[plot_r2_vs_data_size] Figure saved to '{filename}'")

        plt.show()


    def plot_simulation_avg_correlations(self):
        """
        Plot the distribution (as a boxplot and histogram) of the average feature-label correlations
        (for source and target) computed in each simulation, over all simulations.
        """
        # Check if the required columns exist
        if 'mean_source_corr' not in self.results_df.columns or 'mean_target_corr' not in self.results_df.columns:
            print("Average correlation columns not found in results_df. Ensure they're stored during each simulation.")
            return
        
        # Extract unique simulation-level averages. Assuming each simulation produces one row.
        df_avg = self.results_df[['simulation_id', 'mean_source_corr', 'mean_target_corr']].drop_duplicates()
        
        # Melt the DataFrame into long format for plotting.
        df_long = df_avg.melt(id_vars='simulation_id', value_vars=['mean_source_corr', 'mean_target_corr'],
                              var_name='Domain', value_name='Average Correlation')
        # Rename for clarity
        df_long['Domain'] = df_long['Domain'].map({
            'mean_source_corr': 'Source',
            'mean_target_corr': 'Target'
        })
        
        # --- Boxplot ---
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Domain', y='Average Correlation', data=df_long, palette=['#00376c', '#EF7C00'])
        plt.title(f'Distribution of Average Feature-Label Correlations\n(Across All Simulations) ({self.shift_type})')
        plt.tight_layout()

        # Save figure
        filename = os.path.join(self.figures_folder, f"simulation_avg_correlations_bp_{self.num_simulations}_{self._timestamp()}.pdf")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[plot_simulation_avg_correlations] Figure saved to '{filename}'")
        
        plt.show()
        
        # --- Histogram ---
        # Get min and max values from the data
        min_val = df_long['Average Correlation'].min()
        max_val = df_long['Average Correlation'].max()
        
        # Add a small padding, but clip to [0, 1]
        padding = 0.1
        x_min = max(0, min_val - padding)
        x_max = min(1, max_val + padding)
        
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df_long, x='Average Correlation', hue='Domain', kde=False,
                     palette=['#00376c', '#EF7C00'], alpha=0.5)
        plt.xlim(x_min, x_max)

        #plt.xlim(0, 1)
        plt.title(f'Histogram of Average Feature-Label Correlations\n(Across All Simulations) ({self.shift_type})')
        plt.tight_layout()

        # Save figure
        filename = os.path.join(self.figures_folder, f"simulation_avg_correlations_hist_{self.num_simulations}_{self._timestamp()}.pdf")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[plot_simulation_avg_correlations] Figure saved to '{filename}'")
        
        plt.show()

    def test_heteroscedasticity_vs_shift(self, shift_col="feature_shift_score", p_val=0.05):
        """
        Run Breusch-Pagan heteroscedasticity tests for all models on transfer learning R^2
        against the selected shift_col. Aggregates results in a compact table.
    
        Args:
            shift_col (str): Column name to use as the independent variable (e.g., shift intensity).
    
        Returns:
            pd.DataFrame: Summary table of test statistics and heteroscedasticity flag.
        """
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import het_breuschpagan
    
        results = []
        df = self.results_df.copy()
    
        for model_name in df["model_name"].unique():
            df_model = df[df["model_name"] == model_name]
            df_model = df_model.dropna(subset=["r2_transfer_learning", shift_col])
    
            if df_model.empty:
                continue
    
            X = sm.add_constant(df_model[shift_col])
            y = df_model["r2_transfer_learning"]
            model = sm.OLS(y, X).fit()
            residuals = model.resid
    
            lm_stat, lm_pval, f_stat, f_pval = het_breuschpagan(residuals, X)
    
            results.append({
                "model_name": model_name,
                "LM stat": round(lm_stat, 3),
                "LM p-val": round(lm_pval, 4),
                "F stat": round(f_stat, 3),
                "F p-val": round(f_pval, 4),
                "heteroscedastic": f_pval < p_val
            })
    
        heteroscedasticity_df = pd.DataFrame(results)

        # Save results
        filename = os.path.join(self.results_folder, f"heteroscedasticity_{self._timestamp()}.csv")
        heteroscedasticity_df.to_csv(filename, index=False)

        return heteroscedasticity_df


    def plot_r2_vs_shift(
        self, 
        shift_col='combined_shift_score', 
        ylim_cut=-0.2, 
        order=1,
        plot_all_scenarios=False
    ):
        """
        Plot R² vs. a chosen domain-shift measure (default: 'combined_shift_score').
        
        By default (plot_all_scenarios=False), the logic remains:
          - If a model has any non-NaN 'r2_transfer_learning', we do a scatter+regplot of R² vs. shift_col 
            (solid line), fitting a polynomial of degree 'order'.
          - If the model lacks transfer_learning but has target_only data, we draw a single dashed 
            horizontal line at that scenario's mean R², spanning the global min to global max of shift_col.
          - We don't plot SourceOnly at all unless there's no other scenario.
        
        If plot_all_scenarios=True, then for each model we attempt to plot 
          r2_source_only (dotted), 
          r2_target_only (dashed),
          r2_transfer_learning (solid),
        each if they exist, on the same axes.
        
        Args:
            shift_col (str): The name of the column in self.results_df holding the domain-shift measure
                             (e.g., 'feature_shift_score', 'label_shift_score', or 'combined_shift_score').
            ylim_cut (float): The lower limit for the y-axis.
            order (int): Polynomial order for the regression line (1=linear, 2=quadratic, etc.).
            plot_all_scenarios (bool): If True, plot all scenarios (source, target, transfer) that exist 
                                       for each model, each with distinct line styles. 
                                       If False, revert to the original logic.
        """
    
        if shift_col not in self.results_df.columns:
            print(f"[plot_r2_vs_shift] column '{shift_col}' not found in results_df.")
            return
    
        df_plot = self.results_df.copy()
        if 'model_name' not in df_plot.columns:
            print("[plot_r2_vs_shift] 'model_name' not found in results_df.")
            return
    
        # Determine the GLOBAL min and max of shift_col
        global_min_shift = df_plot[shift_col].min()
        global_max_shift = df_plot[shift_col].max()
    
        plt.figure(figsize=(10, 6))
        unique_models = df_plot['model_name'].unique()
    
        # We'll use scenario -> line style mapping:
        scenario_styles = {
            'source_only': (':', 'r2_source_only'),        # dotted
            'target_only': ('--', 'r2_target_only'),       # dashed
            'transfer_learning': ('-', 'r2_transfer_learning')  # solid
        }
    
        for model_name in unique_models:
            df_model = df_plot[df_plot['model_name'] == model_name]

            # If not plotting all scenarios => do the old "if transfer => plot transfer; else if target => plot target" logic
            if not plot_all_scenarios:

                if df_model['r2_transfer_learning'].notna().any():

                    # Transfer scenario => scatter+regplot
                    r2_col = 'r2_transfer_learning'
                    df_scenario = df_model.dropna(subset=[r2_col, shift_col])
                    if df_scenario.empty:
                        continue
    
                    avg_r2 = df_scenario[r2_col].mean()
                    label_text = f"{model_name} (Transfer, Avg {avg_r2:.2f})"

                    sns.scatterplot(
                        data=df_scenario,
                        x=shift_col,
                        y=r2_col,
                        alpha=0.3,
                        label=None
                    )
                    sns.regplot(
                        data=df_scenario,
                        x=shift_col,
                        y=r2_col,
                        scatter=False,
                        ci=None,
                        order=order,
                        label=label_text
                    )
    
                elif df_model['r2_target_only'].notna().any():
                    # If no transfer but we do have target_only
                    r2_col = 'r2_target_only'
                    df_scenario = df_model.dropna(subset=[r2_col])
                    if df_scenario.empty:
                        continue
    
                    avg_r2 = df_scenario[r2_col].mean()
                    label_text = f"{model_name} (Target, Avg {avg_r2:.2f})"
                    # Horizontal dashed line
                    plt.plot(
                        [global_min_shift, global_max_shift],
                        [avg_r2, avg_r2],
                        '--', label=label_text
                    )
    
                elif df_model['r2_source_only'].notna().any():
                    # If we have only source scenario
                    r2_col = 'r2_source_only'
                    df_scenario = df_model.dropna(subset=[r2_col])
                    if df_scenario.empty:
                        continue
    
                    avg_r2 = df_scenario[r2_col].mean()
                    label_text = f"{model_name} (SourceOnly, Avg {avg_r2:.2f})"
                    # Horizontal dotted line
                    plt.plot(
                        [global_min_shift, global_max_shift],
                        [avg_r2, avg_r2],
                        ':', label=label_text
                    )
    
                # else: skip if no scenario found
    
            else:
                # ========== plot_all_scenarios=True => We attempt to plot all that exist ==========
                # We'll keep a single color for this model, so let's pick from the default palette
                # (but for simplicity, we'll rely on the fact that each scatter/reg call picks 
                #  the next color. Instead, we can just set a color manually if we want.)
                color = next(plt.gca()._get_lines.prop_cycler)['color']
    
                for scenario_key, (ls, col_name) in scenario_styles.items():
                    if df_model[col_name].notna().any():
                        # Some valid R² values for this scenario
                        df_scenario = df_model.dropna(subset=[col_name, shift_col])
                        if df_scenario.empty:
                            continue
    
                        avg_r2 = df_scenario[col_name].mean()
                        label_text = f"{model_name} ({scenario_key}, Avg {avg_r2:.2f})"

                        if scenario_key == "transfer_learning":
                            # We'll do scatter+regplot
                            sns.scatterplot(
                                data=df_scenario,
                                x=shift_col,
                                y=col_name,
                                alpha=0.3,
                                color=color,
                                label=None
                            )
                            sns.regplot(
                                data=df_scenario,
                                x=shift_col,
                                y=col_name,
                                scatter=False,
                                ci=None,
                                order=order,
                                color=color,
                                label=label_text,
                                line_kws={"linestyle": ls}
                            )
    
                        else:
                            # For target/source, we just do horizontal lines
                            plt.plot(
                                [global_min_shift, global_max_shift],
                                [avg_r2, avg_r2],
                                ls, color=color,
                                label=label_text
                            )
    
        # Final styling
        plt.title(f'R² vs. {shift_col} ({self.shift_type})')
        plt.xlabel(shift_col)
        plt.ylabel('R²')
        plt.ylim(bottom=ylim_cut)
        plt.legend(loc='best')
        plt.tight_layout()

        # Save figure
        filename = os.path.join(self.figures_folder, f"r2_vs_shift_{shift_col}_sim{self.num_simulations}_{self._timestamp()}.pdf")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[plot_r2_vs_shift] Figure saved to '{filename}'")
    
        plt.show()


    def plot_label_feature_correlation(self, X_source, y_source, X_target, y_target, simulation_id):
        """
        Bar chart of |corr(feature, y)| for source vs target.
        Uses only shared numeric columns to avoid KeyErrors in hetero runs.
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import os
    
        shared = X_source.columns.intersection(X_target.columns)
        # numeric only
        shared = [c for c in shared if pd.api.types.is_numeric_dtype(X_source[c])]
        if not shared:
            print("[plot_label_feature_correlation] No shared numeric features; skipping.")
            return
    
        correlations = {'feature': [], 'source_corr': [], 'target_corr': []}
        for col in shared:
            # guard against degenerate variance
            if X_source[col].nunique(dropna=True) < 2 or X_target[col].nunique(dropna=True) < 2:
                continue
            sc = float(np.abs(np.corrcoef(X_source[col], y_source)[0, 1]))
            tc = float(np.abs(np.corrcoef(X_target[col], y_target)[0, 1]))
            correlations['feature'].append(col)
            correlations['source_corr'].append(sc)
            correlations['target_corr'].append(tc)
    
        if not correlations['feature']:
            print("[plot_label_feature_correlation] Nothing to plot after variance check; skipping.")
            return
    
        df_corr = pd.DataFrame(correlations)
        x = np.arange(len(df_corr))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, df_corr['source_corr'], width, label='Source', color='#00376c')
        ax.bar(x + width/2, df_corr['target_corr'], width, label='Target', color='#EF7C00')
        ax.set_ylabel('Absolute Correlation')
        ax.set_title(f'Feature Correlation with Recovery Rate ({self.shift_type})')
        ax.set_xticks(x)
        ax.set_xticklabels(df_corr['feature'], rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        filename = os.path.join(self.figures_folder, f"label_feature_correlation_{simulation_id}.pdf")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        if self.verbose > 2:
            plt.show()
        else:
            plt.close()



    def get_results(self):
        """
        Get the simulation results DataFrame.

        Returns:
            pd.DataFrame: The simulation results DataFrame.
        """
        return self.results_df