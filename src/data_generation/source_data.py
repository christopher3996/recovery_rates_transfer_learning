import copy
from typing import Optional, Dict, Tuple, Callable
import pandas as pd
import numpy as np
import inspect
from scipy.stats import skewnorm, norm
from scipy.optimize import fsolve
from scipy.special import softmax


class SimulationDataGeneration:
    """
    A class for generating synthetic datasets with a specified recovery rate distribution
    and features that may depend on the recovery rate and on each other.

    Attributes:
        n_samples (int): Number of samples to generate.
        seed (Optional[int]): Random seed for reproducibility.
        recovery_rate_config (dict): Configuration for the recovery rate distribution.
        feature_config (dict): Configuration for the features.
        recovery_rate (Optional[np.ndarray]): Generated recovery rate values.
        secured_feature (Optional[np.ndarray]): Indicator variable for secured/unsecured components.
        features (Dict[str, np.ndarray]): Dictionary of generated feature arrays.
        features_df (Optional[pd.DataFrame]): DataFrame containing the generated features.
    """

    def __init__(self, config: dict):
        """
        Initialize the SimulationDataGeneration class with the provided configuration.

        Args:
            config (dict): Configuration dictionary containing parameters for data generation.
        """
        self.n_samples: int = config.get('n_samples', 1000)
        self.seed: Optional[int] = config.get('seed', None)
        self.recovery_rate_config: dict = copy.deepcopy(config.get('recovery_rate', {}))
        self.feature_config: dict = copy.deepcopy(config.get('features', {}))

        # Set the random seed for reproducibility
        np.random.seed(self.seed)

        # Initialize data structures
        self.recovery_rate: Optional[np.ndarray] = None
        self.secured_feature: Optional[np.ndarray] = None
        self.features: Dict[str, np.ndarray] = {}
        self.features_df: Optional[pd.DataFrame] = None

    def mixture_cdf_inv(self, u: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inverse CDF for a mixture of two normal distributions.

        Args:
            u (np.ndarray): Array of uniform random numbers.
            params (dict): Parameters for the mixture distribution.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the recovery rate array and component indicators.
        """
        secured_p = params['secured_p']
        secured_mean = params['secured_peak']['mean']
        secured_std = params['secured_peak']['std']
        unsecured_mean = params['unsecured_peak']['mean']
        unsecured_std = params['unsecured_peak']['std']

        recovery_rate = np.zeros_like(u)
        component = np.zeros_like(u)

        for i in range(len(u)):
            if u[i] <= secured_p:
                adjusted_u = u[i] / secured_p
                recovery_rate[i] = norm.ppf(adjusted_u, loc=secured_mean, scale=secured_std)
                component[i] = 1  # Secured
            else:
                adjusted_u = (u[i] - secured_p) / (1 - secured_p)
                recovery_rate[i] = norm.ppf(adjusted_u, loc=unsecured_mean, scale=unsecured_std)
                component[i] = 0  # Unsecured

            # Clip recovery rate to [0, 1]
            recovery_rate[i] = np.clip(recovery_rate[i], 0, 1)

        return recovery_rate, component

    def generate_recovery_rate(self):
        """
        Generate the recovery rate using the specified bimodal distribution.
        """
        u = np.random.uniform(size=self.n_samples)
        params = self.recovery_rate_config.get('parameters', {})
        self.recovery_rate, self.secured_feature = self.mixture_cdf_inv(u, params)

    def skewnorm_params(
        self, target_mean: float, target_std: float, target_skewness: float
    ) -> Tuple[float, float, float]:
        """
        Compute the parameters for the skew-normal distribution to match the desired mean, std, and skewness.

        Args:
            target_mean (float): Desired mean of the distribution.
            target_std (float): Desired standard deviation of the distribution.
            target_skewness (float): Desired skewness of the distribution.

        Returns:
            Tuple[float, float, float]: Parameters (a, loc, scale) for the skew-normal distribution.

        Raises:
            ValueError: If the shape parameter 'a' cannot be found to match the desired skewness.
        """
        max_skewness = 0.9952717  # Maximum achievable skewness for skewnorm
        if abs(target_skewness) > max_skewness:
            target_skewness = np.sign(target_skewness) * max_skewness
            print(f"Adjusted target_skewness to {target_skewness} (within achievable range).")

        def skewness_function(a: float) -> float:
            delta = a / np.sqrt(1 + a**2)
            gamma1 = (
                ((4 - np.pi) / 2)
                * ((delta * np.sqrt(2 / np.pi)) ** 3)
                / ((1 - (2 * delta**2) / np.pi) ** 1.5)
            )
            return gamma1

        def objective(a: float) -> float:
            return skewness_function(a) - target_skewness

        initial_guess = np.sign(target_skewness) * 2.0  # Start with a moderate skew
        a_solution, info, ier, mesg = fsolve(objective, initial_guess, full_output=True)

        if ier != 1:
            raise ValueError(
                f"Could not find shape parameter 'a' for skewness {target_skewness}. fsolve failed: {mesg}"
            )

        a = a_solution[0]

        delta = a / np.sqrt(1 + a**2)
        mean0 = delta * np.sqrt(2 / np.pi)
        variance0 = 1 - (2 * delta**2) / np.pi
        std0 = np.sqrt(variance0)

        scale = target_std / std0
        loc = target_mean - scale * mean0

        return a, loc, scale

    # NEW =======
    def _generate_categorical_softmax(
        self,
        recovery_rate: np.ndarray,
        secured: np.ndarray,
        parameters: dict,
    ) -> np.ndarray:
        """
        Sample a categorical feature via soft-max over (possibly non-linear)
        functions of recovery_rate and secured.

        parameters expected:
            categories (list[str])
            weights    (dict[str, list[float]])  # same length for every cat
            noise_temperature (float)            # σ of added N(0,σ) noise
        """
        cats = parameters["categories"]
        wmat = np.vstack([parameters["weights"][c] for c in cats])  # shape (K,D)
        temp = parameters.get("noise_temperature", 0.0)

        # Build the design matrix Φ = [ 1, rr, rr², secured ]  -> shape (N,D)
        rr  = recovery_rate.reshape(-1, 1)
        sec = secured.reshape(-1, 1)
        Phi = np.hstack([np.ones_like(rr), rr, rr**2, sec])

        if Phi.shape[1] != wmat.shape[1]:
            raise ValueError("weights lists must all have length "
                             f"{Phi.shape[1]} (got {wmat.shape[1]})")

        # logits L = Φ·Wᵀ  → add Gaussian jitter
        logits = Phi @ wmat.T
        if temp > 0:
            logits += np.random.normal(0.0, temp, size=logits.shape)

        probs = softmax(logits, axis=1)           # (N,K)
        cum   = probs.cumsum(1)                   # for inverse-CDF sampling
        u     = np.random.rand(len(recovery_rate), 1)
        chosen = (cum > u).argmax(axis=1)         # (N,) integer indices
        return np.array([cats[i] for i in chosen])  # as strings
    #=====

    def generate_features(self):
        """
        Generate features based on the recovery rate and specified relationships,
        including dependencies among features.
        """
        # Dictionary to store features that need to be generated later (dependent features)
        deferred_features = {}

        # First pass: Generate independent features
        for feature_name, params in self.feature_config.items():
            relationship_type = params.get('relationship_type', 'linear')
            parameters = params.get('parameters', {})

            # Check if the feature depends on other features
            if relationship_type == 'interaction':
                # Defer the generation of dependent features
                deferred_features[feature_name] = params
                continue

            #NEW====
            if relationship_type == "categorical_softmax":
                self.features[feature_name] = self._generate_categorical_softmax(
                    self.recovery_rate,
                    self.secured_feature,
                    parameters,
                )
                continue
            #====

            beta = parameters.get('beta', 1.0)
            intercept = parameters.get('intercept', 0.0)
            noise_std = parameters.get('noise_std', 0.1)
            skewness = parameters.get('skewness', 0.0)

            # Independent random component
            independent_noise_std = parameters.get('independent_noise_std', 0.1)
            independent_component = np.random.normal(0, independent_noise_std, self.n_samples)

            # Generate base feature values
            feature_base = self._generate_base_feature(
                self.recovery_rate, relationship_type, parameters, independent_component
            )

            # Add noise
            feature_values = self._add_noise(feature_base, noise_std, skewness)

            self.features[feature_name] = feature_values

        # Second pass: Generate dependent features (interactions)
        for feature_name, params in deferred_features.items():
            parameters = params.get('parameters', {})
            function = parameters.get('function', 'multiply')
            involved_features = parameters.get('features', [])
            noise_std = parameters.get('noise_std', 0.1)
            skewness = parameters.get('skewness', 0.0)

            # Ensure involved features are already generated
            if not all(f in self.features for f in involved_features):
                raise ValueError(
                    f"Dependent feature '{feature_name}' relies on features that are not generated yet."
                )

            # Compute the interaction
            feature_base = self._compute_interaction(function, involved_features, parameters)

            # Add noise
            feature_values = self._add_noise(feature_base, noise_std, skewness)

            self.features[feature_name] = feature_values

        # Create the features DataFrame
        self.features_df = pd.DataFrame(self.features)
        for col in self.features_df.select_dtypes(include='object').columns:
            self.features_df[col] = self.features_df[col].astype('category')

    def _generate_base_feature(
        self,
        recovery_rate: np.ndarray,
        relationship_type: str,
        parameters: dict,
        independent_component: np.ndarray,
    ) -> np.ndarray:
        """
        Generate the base feature values based on the relationship type.

        Args:
            recovery_rate (np.ndarray): Array of recovery rate values.
            relationship_type (str): Type of relationship ('linear', 'nonlinear', etc.).
            parameters (dict): Parameters for the feature generation.
            independent_component (np.ndarray): Independent random component.

        Returns:
            np.ndarray: Base feature values before adding noise.
        """
        beta = parameters.get('beta', 1.0)
        intercept = parameters.get('intercept', 0.0)

        if relationship_type == 'linear':
            feature_base = beta * recovery_rate + intercept + independent_component
        elif relationship_type == 'nonlinear':
            function = parameters.get('function', 'sin')
            if function == 'sin':
                feature_base = beta * np.sin(recovery_rate) + intercept + independent_component
            elif function == 'log':
                feature_base = beta * np.log(recovery_rate + 1e-6) + intercept + independent_component
            elif function == 'exp':
                feature_base = beta * np.exp(recovery_rate) + intercept + independent_component
            elif function == 'polynomial':
                degree = parameters.get('degree', 2)
                feature_base = beta * np.power(recovery_rate, degree) + intercept + independent_component
            elif function == 'sigmoid':
                k = parameters.get('k', 1.0)
                x0 = parameters.get('x0', 0.5)
                feature_base = beta / (1 + np.exp(-k * (recovery_rate - x0))) + intercept + independent_component
            else:
                raise ValueError(f"Unsupported function: {function}")
        elif relationship_type == 'custom':
            custom_function = parameters.get('function')
            if callable(custom_function):
                custom_function.__globals__['np'] = np
                # Inspect the signature of the custom function
                sig = inspect.signature(custom_function)
                if len(sig.parameters) == 2:
                    # Assume second parameter is secured feature
                    feature_base = beta * custom_function(recovery_rate, self.secured_feature) + intercept + independent_component
                else:
                    feature_base = beta * custom_function(recovery_rate) + intercept + independent_component
            else:
                raise ValueError("Custom function for feature is not callable.")
        else:
            raise ValueError(f"Unsupported relationship type: {relationship_type}")

        return feature_base

    def _add_noise(self, feature_base: np.ndarray, noise_std: float, skewness: float) -> np.ndarray:
        """
        Add noise to the base feature values.

        Args:
            feature_base (np.ndarray): Base feature values.
            noise_std (float): Standard deviation of the noise.
            skewness (float): Desired skewness of the noise distribution.

        Returns:
            np.ndarray: Feature values after adding noise.
        """
        if noise_std > 0:
            if abs(skewness) > 0.0:
                a, loc, scale = self.skewnorm_params(0, noise_std, skewness)
                noise = skewnorm.rvs(a, loc=loc, scale=scale, size=self.n_samples)
            else:
                noise = np.random.normal(0, noise_std, self.n_samples)
            feature_values = feature_base + noise
        else:
            feature_values = feature_base
        return feature_values

    def _compute_interaction(
        self, function: str, involved_features: list, parameters: dict
    ) -> np.ndarray:
        """
        Compute the interaction feature based on involved features and specified function.

        Args:
            function (str): Function to compute interaction ('multiply', 'add', 'custom').
            involved_features (list): List of feature names involved in the interaction.
            parameters (dict): Parameters for the interaction feature.

        Returns:
            np.ndarray: Base values of the interaction feature.

        Raises:
            ValueError: If an unsupported function is specified or required features are missing.
        """
        if function == 'multiply':
            feature_base = np.prod([self.features[f] for f in involved_features], axis=0)
        elif function == 'add':
            feature_base = np.sum([self.features[f] for f in involved_features], axis=0)
        elif function == 'custom':
            custom_function = parameters.get('custom_function')
            if callable(custom_function):
                feature_arrays = [self.features[f] for f in involved_features]
                feature_base = custom_function(*feature_arrays)
            else:
                raise ValueError("Custom function for interaction is not callable.")
        else:
            raise ValueError(f"Unsupported interaction function: {function}")

        return feature_base

    def generate_data(self) -> pd.DataFrame:
        """
        Generate the synthetic dataset.

        Returns:
            pd.DataFrame: DataFrame containing the generated features and recovery rate.
        """
        self.generate_recovery_rate()
        self.generate_features()

        df = self.features_df.copy()
        df['recovery_rate'] = self.recovery_rate
        df['secured'] = self.secured_feature

        return df
