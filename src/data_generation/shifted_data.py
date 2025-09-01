import yaml
import pandas as pd
import numpy as np
from data_generation.source_data import SimulationDataGeneration

class ShiftedSimulationDataGeneration(SimulationDataGeneration):
    def __init__(self, original_config: dict, shift_config: dict):
        """
        Initialize the ShiftedSimulationDataGeneration class with dataset characteristics,
        applying shifts specified in the shift configuration.

        Args:
            original_config (dict): YAML configuration for data generation.
            shift_config (dict): YAML configuration specifying shifts.
        """

        self.original_config = original_config
        self.shift_config = shift_config
            
        # Initialize the base class
        super().__init__(self.original_config)

        # Apply shifts to the data generation parameters
        self.apply_shifts(self.shift_config)
        
        # Store the modified config as target_config
        self.target_config = {
            'n_samples': self.n_samples,
            'seed': self.seed,
            'recovery_rate': self.recovery_rate_config,
            'features': self.feature_config
        }

    def apply_shifts(self, shift_config: dict):
        """
        Apply all shifts to the data generation parameters.
        """
        # Apply shifts to feature generation parameters
        self.apply_feature_shifts(shift_config)

        # Apply shifts to recovery rate parameters if needed
        self.apply_recovery_rate_shifts(shift_config)

    def apply_feature_shifts(self, shift_config: dict):
        """
        Apply shifts to feature generation parameters.
        """
        feature_shifts = shift_config.get('feature_shifts', [])
        for shift in feature_shifts:
            feature_name = shift.get('feature_name')
            shift_params = shift.get('parameters', {})
            if feature_name in self.feature_config:
                parameters = self.feature_config[feature_name].get('parameters', {})
                # Apply shifts to beta, intercept, noise_std
                if 'beta_shift' in shift_params:
                    original_beta = parameters.get('beta', 1.0)
                    parameters['beta'] = original_beta + shift_params['beta_shift']
                if 'intercept_shift' in shift_params:
                    original_intercept = parameters.get('intercept', 0.0)
                    parameters['intercept'] = original_intercept + shift_params['intercept_shift']
                if 'noise_std_scale' in shift_params:
                    original_noise_std = parameters.get('noise_std', 1.0)
                    parameters['noise_std'] = original_noise_std * shift_params['noise_std_scale']
                if 'skewness_shift' in shift_params:
                    original_skewness = parameters.get('skewness', 0.0)
                    parameters['skewness'] = original_skewness + shift_params['skewness_shift']

                self.feature_config[feature_name]['parameters'] = parameters
            else:
                print(f"Feature '{feature_name}' not found in feature configuration.")

    def apply_recovery_rate_shifts(self, shift_config: dict):
        """
        Apply shifts to the recovery rate generation parameters.
        """
        recovery_rate_shifts = shift_config.get('recovery_rate_shifts', {}).get('parameters', {})
        if recovery_rate_shifts:
            self.recovery_rate_config['parameters'].update(recovery_rate_shifts)

    def generate_data(self) -> pd.DataFrame:
        """
        Generate the dataset with shifts applied during data generation.

        Returns:
            pd.DataFrame: Generated dataset with shifts applied.
        """
        # Generate data using the base class method
        df = super().generate_data()

        # Apply post-generation feature shifts
        df = self.apply_post_generation_feature_shifts(df)

        return df

    def apply_post_generation_feature_shifts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature shifts to the dataset after generation.

        Args:
            df (pd.DataFrame): Generated dataset.

        Returns:
            pd.DataFrame: Dataset with post-generation feature shifts applied.
        """
        feature_shifts = self.shift_config.get("post_generation_feature_shifts", [])

        for shift in feature_shifts:
            parameters = shift.get("parameters", {})
            feature_modifications = parameters.get('feature_modifications', [])

            for modification in feature_modifications:
                feature_name = modification.get('feature_name')
                shift_type = modification.get('shift_type')
                shift_value = modification.get('shift_value', 0.0)

                if feature_name not in df.columns:
                    continue  # Skip if the feature does not exist

                if shift_type == 'noise':
                    # Add Gaussian noise
                    noise_level = shift_value
                    df[feature_name] += np.random.normal(0, noise_level, size=len(df))
                elif shift_type == 'drop':
                    # Drop the feature
                    df.drop(columns=[feature_name], inplace=True)
                elif shift_type == 'scale':
                    # Scale the feature
                    df[feature_name] *= shift_value
                elif shift_type == 'offset':
                    # Add an offset to the feature
                    df[feature_name] += shift_value
                elif shift_type == 'transform':
                    # Apply a custom transformation function
                    transform_name = modification.get('transform_name')
                    transform_func = self.get_transform_function(transform_name)
                    if callable(transform_func):
                        df[feature_name] = df[feature_name].apply(transform_func)
                    else:
                        raise ValueError(f"Transform function '{transform_name}' is not callable.")
                else:
                    raise ValueError(f"Unsupported shift type: {shift_type}")

        return df

    def get_transform_function(self, name):
        """
        Retrieve a transformation function by name.

        Args:
            name (str): Name of the transformation function.

        Returns:
            Callable: The transformation function.
        """
        # Define custom transformation functions
        if name == 'log_transform':
            return lambda x: np.log(x + 1e-6)  # Add small constant to avoid log(0)
        elif name == 'sqrt_transform':
            return np.sqrt
        # Add more transformations as needed
        else:
            return None
