from pyexpat import features
from typing import Any, Union, Callable
import numpy as np
import scipy.special as special

from honeytribe.online_convex_optimization.base import OnlineAlgorithm


tol = 1e-6

class BaseExpertAlgorithm(OnlineAlgorithm):
    """
    An abstract class for expert algorithms in online convex optimization.
    This class extends the OnlineAlgorithm base class and is intended to be
    subclassed by specific expert algorithms.
    """

    def __init__(self, loss_function: Any, experts: Any):
        super().__init__(loss_function=loss_function)
        self.experts = experts

    def predict_step(self, state: Any) -> Any:
        """
        Make a prediction given the current state.

        Parameters
        ----------
        state : Any
            Current game state/input features

        Returns
        -------
        prediction : Any
            Algorithm's prediction (float, vector, or class)
        """
        out = []
        for expert in self.experts:
            if hasattr(expert, 'predict_step'):
                expert_predict = expert.predict_step(state)
                out.append(expert_predict)
            elif hasattr(expert, 'predict'):
                expert_predict = expert.predict(state)
                out.append(expert_predict)
            else:
                raise NotImplementedError("Expert must implement either 'predict_step' or 'predict' method.")

        return np.array(out)

class HedgeAlgorithm(BaseExpertAlgorithm):
    """
    An abstract class for Hedge algorithms in online convex optimization.
    This class extends the OnlineAlgorithm base class and is intended to be
    subclassed by specific Hedge algorithms.
    """

    def __init__(self, loss_function: Callable, experts: Any, prior: np.ndarray = None, learning_rate: float = 1., update_experts: bool = True, ignore_zero_loss_expert_update: bool = False):
        super().__init__(loss_function=loss_function, experts=experts)
        if prior is None:
            self._prior = np.ones(len(experts)) / len(experts)
        else:
            if len(prior) != len(experts):
                raise ValueError("Prior must have the same length as the number of experts.")
            if not np.isclose(np.sum(prior), 1.0):
                raise ValueError("Prior probabilities must sum to 1.")
            if not np.all(prior >= 0):
                raise ValueError("Prior probabilities must be non-negative.")
            self._prior = np.array(prior, dtype=float)
        self.dist = self._prior.copy()
        self.learning_rate = learning_rate
        self._update_experts = update_experts
        self._ignore_zero_loss_expert_update = ignore_zero_loss_expert_update

    def update(self, state: Any, prediction: Any, loss: Union[float, np.ndarray], y_true: Any = None):
        """
        Update algorithm parameters based on observed loss.

        Parameters
        ----------
        state : Any
            The state that was observed
        prediction : Any
            The prediction that was made
        loss : float or np.ndarray
            The loss that was incurred
        """
        self.dist = np.exp( - self.learning_rate * loss) * self.dist
        self.dist = self.dist / np.sum(self.dist)
        if self._update_experts:
            for expert, expert_prediction, expert_loss in zip(self.experts, prediction, loss):
                if self._ignore_zero_loss_expert_update and expert_loss == 0:
                    continue
                if hasattr(expert, 'update'):
                    expert.update(state, expert_prediction, expert_loss, y_true=y_true)
                elif y_true is not None and hasattr(expert, 'partial_fit'):
                    expert.partial_fit(state, y_true)
        self.update_regret(loss)

        self.t += 1

    def update_regret(self, loss: Union[float, np.ndarray]):
        """
        Update internal regret calculation.

        Parameters
        ----------
        loss : float or np.ndarray
            Loss incurred at current step
        """
        if isinstance(loss, np.ndarray):
            self.regret += np.sum(loss * self.dist)
        else:
            raise ValueError("Loss must be a numpy array for HedgeAlgorithm.")
    def reset(self):
        """
        Reset the algorithm state.
        """
        super().reset()
        self.dist = self._prior.copy()


class AggregatingAlgorithm(HedgeAlgorithm):
    """
    An abstract class for aggregating algorithms in online convex optimization.
    This class extends the OnlineAlgorithm base class and is intended to be
    subclassed by specific aggregating algorithms.
    """

    def __init__(self, loss_function: Any, experts: Any, prior: np.ndarray = None):
        super().__init__(loss_function=loss_function, experts=experts, prior=prior, learning_rate=1.0)

    def update_regret(self, loss: Union[float, np.ndarray]):
        """
        Update internal regret calculation for aggregating algorithms.

        Parameters
        ----------
        loss : float or np.ndarray
            Loss incurred at current step
        """
        if isinstance(loss, np.ndarray):
            self.regret += -np.log(np.sum(self.dist * np.exp(-loss)))
        else:
            raise ValueError("Loss must be a numpy array for AggregatingAlgorithm.")

class Exp3Algorithm(HedgeAlgorithm):
    def __init__(self, loss_function: Any, experts: Any, prior: np.ndarray = None, learning_rate: float = 1.0, random_state: int = None, prediction_strategy: Union[str, Callable] = 'random'):
        super().__init__(loss_function=loss_function, experts=experts, prior=prior, learning_rate=learning_rate, update_experts=False)
        self.prediction_strategy = prediction_strategy
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def resolve_prediction_from_strategy(self, prediction: np.ndarray):
        """Resolve the prediction based on the specified strategy.

        Parameters
        ----------
        prediction : np.ndarray
            Predictions from the experts
        Returns
        -------
        prediction : Any
            Resolved prediction based on the strategy
        """
        if self.prediction_strategy == 'random':
            expert_idx = self.rng.choice(len(self.experts), p=self.dist)
            self._last_expert_idx = expert_idx
            return prediction[expert_idx]
        elif self.prediction_strategy == 'median':
            sorted_indices = np.argsort(prediction)
            sorted_dist = self.dist[sorted_indices]
            cumulative_dist = np.cumsum(sorted_dist)
            median_index = np.searchsorted(cumulative_dist, 0.5)
            expert_idx = sorted_indices[median_index]
            self._last_expert_idx = expert_idx
            return prediction[expert_idx]
        elif self.prediction_strategy in ('max','mode'):
            max_prob = np.max(self.dist)
            spots = np.argwhere(self.dist == max_prob).flatten()
            spot_idx = self.rng.choice(len(spots))
            expert_idx = spots[spot_idx]
            self._last_expert_idx = expert_idx
            return prediction[expert_idx]
        elif callable(self.prediction_strategy):
            expert_idx = self.prediction_strategy(prediction, self.dist)
            self._last_expert_idx = expert_idx
            return prediction[expert_idx]
        else:
            raise ValueError(f"Unknown prediction strategy: {self.prediction_strategy}")

    def predict_step(self, state: Any) -> Any:
        """
        Make a prediction given the current state.

        Parameters
        ----------
        state : Any
            Current game state/input features

        Returns
        -------
        prediction : Any
            Algorithm's prediction (float, vector, or class)
        """
        expert_opinion = super().predict_step(state)
        return self.resolve_prediction_from_strategy(expert_opinion)

    def update(self, state: Any, prediction: Any, loss: Union[float, np.ndarray], y_true: Any = None):
        """
        Update algorithm parameters based on observed loss.

        Parameters
        ----------
        state : Any
            The state that was observed
        prediction : Any
            The prediction that was made
        loss : float or np.ndarray
            The loss that was incurred
        """
        if isinstance(loss, float):
            loss_vector = np.zeros(len(self.experts))
            loss_vector[self._last_expert_idx] = loss / self.dist[self._last_expert_idx]
            super().update(state, prediction, loss_vector, y_true=y_true) # This updates both the distribution and the experts, if this is undesired, override this method.
            self.dist = np.maximum(self.dist, tol)  # Ensure no zero probabilities
            self.dist = self.dist / np.sum(self.dist)  # Normalize the distribution
        else:
            raise ValueError("Loss must be a float for Exp3Algorithm.")

class FixedShareAlgorithm(HedgeAlgorithm):
    """
    An abstract class for Fixed Share algorithms in online convex optimization.
    This class extends the HedgeAlgorithm base class and is intended to be
    subclassed by specific Fixed Share algorithms.
    """

    def __init__(self, loss_function: Any, experts: Any, prior: np.ndarray = None, learning_rate: float = 1.0, switching_rate: float = 0.1):
        super().__init__(loss_function=loss_function, experts=experts, prior=prior, learning_rate=learning_rate)
        self.switching_rate = switching_rate

    def update(self, state: Any, prediction: Any, loss: Union[float, np.ndarray], y_true: Any = None):
        """
        Update algorithm parameters based on observed loss.

        Parameters
        ----------
        state : Any
            The state that was observed
        prediction : Any
            The prediction that was made
        loss : float or np.ndarray
            The loss that was incurred
        """
        if isinstance(loss, np.ndarray):
            if len(loss) != len(self.experts):
                raise ValueError("Loss must have the same length as the number of experts.")

            beta = self.switching_rate * len(self.experts) / (len(self.experts) - 1)
            super().update(state, prediction, loss, y_true=y_true)
            self.dist = (1 - beta) * self.dist + beta/len(self.experts)
            self.dist = self.dist / np.sum(self.dist)  # Normalize the distribution
        else:
            raise ValueError("Loss must be a numpy array for FixedShareAlgorithm.")


class UCBAlgorithm(BaseExpertAlgorithm):
    """
    An abstract class for UCB algorithms in online convex optimization.
    This class extends the HedgeAlgorithm base class and is intended to be
    subclassed by specific UCB algorithms.
    """

    def __init__(self, loss_function: Any, experts: Any, alpha = 1.001, _update_experts: bool = True):
        super().__init__(loss_function=loss_function, experts=experts)
        self.alpha = alpha
        self.mean_loss = np.zeros(len(experts))
        self.count = np.zeros(len(experts))
        self._update_experts = _update_experts

    def predict_step(self, state: Any) -> Any:
        """
        Make a prediction given the current state.

        Parameters
        ----------
        state : Any
            Current game state/input features

        Returns
        -------
        prediction : Any
            Algorithm's prediction (float, vector, or class)
        """
        if np.any(self.count == 0):
            # If any expert has not been selected yet, select it
            expert_idx = np.argmin(self.count)
            self._last_expert_idx = expert_idx
        else:
            expert_idx = np.argmin(self.mean_loss - np.sqrt((self.alpha * np.log(self.t) / self.count) / 2))
            self._last_expert_idx = expert_idx

        expert_opinion = super().predict_step(state)
        return expert_opinion[self._last_expert_idx]

    def update(self, state: Any, prediction: Any, loss: Union[float, np.ndarray], y_true: Any = None):
        """
        Update algorithm parameters based on observed loss.

        Parameters
        ----------
        state : Any
            The state that was observed
        prediction : Any
            The prediction that was made
        loss : float or np.ndarray
            The loss that was incurred
        y_true: Any
            The true label (if applicable, e.g., in classification tasks)
        """
        if isinstance(loss, float):
            current_count = self.count[self._last_expert_idx]
            self.mean_loss[self._last_expert_idx] = (current_count * self.mean_loss[self._last_expert_idx] + loss) / (current_count + 1)
            self.count[self._last_expert_idx] += 1
        elif isinstance(loss, np.ndarray):
            raise ValueError("Loss must be a float for UCBAlgorithm. Use a different algorithm for vectorized losses.")
        if self._update_experts:
            expert = self.experts[self._last_expert_idx]
            if hasattr(expert, 'update'):
                expert.update(state, prediction, loss, y_true=y_true)
            elif y_true is not None and hasattr(expert, 'partial_fit'):
                expert.partial_fit(state, y_true)

        self.update_regret(loss)
        self.t += 1

    def update_regret(self, loss: Union[float, np.ndarray]):
        """
        Update internal regret calculation.

        Parameters
        ----------
        loss : float or np.ndarray
            Loss incurred at current step
        """
        if isinstance(loss, float):
            self.regret += loss
        else:
            raise ValueError("Loss must be a float for UCBAlgorithm.")

    def reset(self):
        """
        Reset the algorithm state.
        """
        super().reset()
        self.mean_loss = np.zeros(len(self.experts))
        self.count = np.zeros(len(self.experts))
        self._last_expert_idx = None


class TimeseriesExponentiatedHedgeAlgorithm(BaseExpertAlgorithm):
    """
    An abstract class for time-series exponentiated hedge algorithms in online convex optimization.
    """
    def __init__(self, loss_function: Callable, experts: Any, prior: np.ndarray = None, learning_rate: float = 1.,
                 update_experts: bool = True, ignore_zero_loss_expert_update: bool = False, time_features = None):
        super().__init__(loss_function=loss_function, experts=experts)
        self.time_features = time_features if time_features is not None else lambda x: x
        n_time_features = len(self.time_features(0))
        self.n_time_features = n_time_features # Replace n_time_features with the actual number of time features. Calculate time_features(0) once to determine the shape.
        if prior is None:
            self._prior = np.zeros((len(experts), n_time_features))
        elif prior.ndim == 1:
            if len(prior) != n_time_features:
                raise ValueError("Prior must have the same length as the number of time features.")
            self._prior = np.tile(prior, (len(experts), 1))
        elif prior.ndim == 2:
            if prior.shape[0] != len(experts) or prior.shape[1] != n_time_features:
                raise ValueError("Prior must have shape (n_experts, n_time_features).")
            self._prior = np.array(prior, dtype=float)
        else:
            raise ValueError("Prior must be a 1D or 2D numpy array.")
        self.theta = self._prior.copy()
        self.learning_rate = learning_rate
        self._update_experts = update_experts
        self._ignore_zero_loss_expert_update = ignore_zero_loss_expert_update

    def compute_time_features(self):
        """
        Compute time features for the current step.

        Returns
        -------
        time_features : np.ndarray
            Time features for the current step
        """
        out = self.time_features(self.t)
        print(f"time: {self.t}, time features: {out}")
        return out

    def dist_helper(self, theta):
        features = self.compute_time_features()
        linear_combination = np.dot(theta, features)  # len(experts)
        linear_combination = linear_combination - np.min(linear_combination)
        dist = np.exp(-linear_combination)
        dist = dist / np.sum(dist)  # Normalize the distribution
        return dist

    @property
    def dist(self):
        """
        Get the current distribution of expert probabilities.

        Returns
        -------
        dist : np.ndarray
            Current distribution of expert probabilities
        """
        dist = self.dist_helper(self.theta)
        return dist

    def update(self, state: Any, prediction: Any, loss: Union[float, np.ndarray], y_true: Any = None):
        """
        Update algorithm parameters based on observed loss.

        Parameters
        ----------
        state : Any
            The state that was observed
        prediction : Any
            The prediction that was made
        loss : float or np.ndarray
            The loss that was incurred
        """
        features = self.compute_time_features()  # n_time_features
        print(f"features: {features}")
        self.theta = self.theta + self.learning_rate * loss[:, np.newaxis] * features[np.newaxis, :]  # len(experts), n_time_features
        if self._update_experts:
            for expert, expert_prediction, expert_loss in zip(self.experts, prediction, loss):
                if self._ignore_zero_loss_expert_update and expert_loss == 0:
                    continue
                if hasattr(expert, 'update'):
                    expert.update(state, expert_prediction, expert_loss, y_true=y_true)
                elif y_true is not None and hasattr(expert, 'partial_fit'):
                    expert.partial_fit(state, y_true)
        self.update_regret(loss)

        self.t += 1

    def update_regret(self, loss: Union[float, np.ndarray]):
        """
        Update internal regret calculation.

        Parameters
        ----------
        loss : float or np.ndarray
            Loss incurred at current step
        """
        if isinstance(loss, np.ndarray):
            self.regret += np.sum(loss * self.dist)
        else:
            raise ValueError("Loss must be a numpy array for HedgeAlgorithm.")

    def reset(self):
        """
        Reset the algorithm state.
        """
        super().reset()
        self.theta = self._prior.copy()

class TimeseriesExponentiatedFixedShareAlgorithm(BaseExpertAlgorithm):
    """
    An abstract class for time-series exponentiated hedge algorithms in online convex optimization.
    """
    def __init__(self, loss_function: Callable, experts: Any, prior: np.ndarray = None, learning_rate: float = 1.,
                 update_experts: bool = True, ignore_zero_loss_expert_update: bool = False, time_features = None, switching_rate: float = 0.1):
        super().__init__(loss_function=loss_function, experts=experts)
        self.time_features = time_features if time_features is not None else lambda x: x
        n_time_features = len(self.time_features(0))
        self.n_time_features = n_time_features
        if prior is None:
            self._prior = np.zeros((len(experts), n_time_features))
        elif prior.ndim == 1:
            if len(prior) != n_time_features:
                raise ValueError("Prior must have the same length as the number of time features.")
            self._prior = np.tile(prior, (len(experts), 1))
        elif prior.ndim == 2:
            if prior.shape[0] != len(experts) or prior.shape[1] != n_time_features:
                raise ValueError("Prior must have shape (n_experts, n_time_features).")
            self._prior = np.array(prior, dtype=float)
        else:
            raise ValueError("Prior must be a 1D or 2D numpy array.")
        self.theta = self._prior.copy()
        self.learning_rate = learning_rate
        self._update_experts = update_experts
        self._ignore_zero_loss_expert_update = ignore_zero_loss_expert_update
        self.switching_rate = switching_rate

    def compute_time_features(self):
        """
        Compute time features for the current step.

        Returns
        -------
        time_features : np.ndarray
            Time features for the current step
        """
        out = self.time_features(self.t)
        print(f"time: {self.t}, time features: {out}")
        return out

    def dist_helper(self, theta):
        features = self.compute_time_features()
        linear_combination = np.dot(theta, features)  # len(experts)
        linear_combination = linear_combination - np.min(linear_combination)
        dist = np.exp(-linear_combination)
        dist = dist / np.sum(dist)  # Normalize the distribution
        return dist

    @property
    def dist(self):
        """
        Get the current distribution of expert probabilities.

        Returns
        -------
        dist : np.ndarray
            Current distribution of expert probabilities
        """
        dist = self.dist_helper(self.theta)
        return dist

    def update(self, state: Any, prediction: Any, loss: Union[float, np.ndarray], y_true: Any = None):
        """
        Update algorithm parameters based on observed loss.

        Parameters
        ----------
        state : Any
            The state that was observed
        prediction : Any
            The prediction that was made
        loss : float or np.ndarray
            The loss that was incurred
        """
        features = self.compute_time_features()  # n_time_features
        print(f"features: {features}")
        print(f"features shape: {features.shape}")
        beta = self.switching_rate * len(self.experts) / (len(self.experts) - 1)
        print(f"beta: {beta}")
        linear_combination = np.dot(self.theta, features)  # len(experts)
        print(f"linear_combination: {linear_combination}")
        print(f"linear_combination shape: {linear_combination.shape}")
        fixed_share_dist = (1 - beta) * np.exp(-linear_combination - self.learning_rate * loss) + beta/len(self.experts)
        print(f"fixed_share_dist: {fixed_share_dist}")
        print(f"fixed_share_dist shape: {fixed_share_dist.shape}")
        self.theta = self.theta - (linear_combination + np.log(fixed_share_dist))[:, np.newaxis] * features[np.newaxis, :]  # len(experts), n_time_features
        print(f"theta: {self.theta}")
        print(f"theta shape: {self.theta.shape}")
        if self._update_experts:
            for expert, expert_prediction, expert_loss in zip(self.experts, prediction, loss):
                if self._ignore_zero_loss_expert_update and expert_loss == 0:
                    continue
                if hasattr(expert, 'update'):
                    expert.update(state, expert_prediction, expert_loss, y_true=y_true)
                elif y_true is not None and hasattr(expert, 'partial_fit'):
                    expert.partial_fit(state, y_true)
        self.update_regret(loss)

        self.t += 1

    def update_regret(self, loss: Union[float, np.ndarray]):
        """
        Update internal regret calculation.

        Parameters
        ----------
        loss : float or np.ndarray
            Loss incurred at current step
        """
        if isinstance(loss, np.ndarray):
            self.regret += np.sum(loss * self.dist)
        else:
            raise ValueError("Loss must be a numpy array for HedgeAlgorithm.")

    def reset(self):
        """
        Reset the algorithm state.
        """
        super().reset()
        self.theta = self._prior.copy()
