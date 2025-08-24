"""
Online Convex Optimization Framework with sklearn-compatible API
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Union, Callable, Optional, List, Tuple
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array

class OnlineGame(ABC):
    """
    Base class for online convex optimization games.

    An online game is played as follows:
    For time steps t = 0, 1, 2, ... do:
    - Environment reveals game state or input features
    - Algorithm plays a best guess
    - Environment returns a loss
    - Algorithm updates according to the loss received
    - Algorithm updates internal regret
    - Game updates history of losses
    """

    def __init__(self, n_steps: int):
        """
        Initialize the online game.

        Parameters
        ----------
        loss_function : callable, optional
            Loss function to compute loss given (y_true, y_pred).
            If None, must be implemented in subclass.
        """
        self.t = 0  # Current time step
        self.loss_history = []  # History of losses
        self.n_steps = n_steps

    @abstractmethod
    def get_state(self) -> Any:
        """
        Get the current game state/input features.

        Returns
        -------
        state : Any
            Current game state (could be features, context, etc.)
        """
        pass

    @abstractmethod
    def compute_loss(self, prediction: Any, state: Any = None) -> Union[float, np.ndarray]:
        """
        Compute loss given algorithm's prediction and current state.

        Parameters
        ----------
        prediction : Any
            Algorithm's prediction (float, vector, or class)
        state : Any, optional
            Current game state (if None, uses current state)

        Returns
        -------
        loss : float or np.ndarray
            Loss value(s)
        """
        pass

    @abstractmethod
    def get_y_true(self) -> Any:
        """
        Get the true target value for the current state. If not available, should return None.

        Returns
        -------
        y_true : Any
            True target value corresponding to the current state
        """
        return None

    def step(self, algorithm: 'OnlineAlgorithm') -> Tuple[Any, Union[float, np.ndarray]]:
        """
        Execute one step of the online game.

        Parameters
        ----------
        algorithm : OnlineAlgorithm
            The algorithm playing the game

        Returns
        -------
        prediction : Any
            Algorithm's prediction for this step
        loss : float or np.ndarray
            Loss incurred by the algorithm
        """
        # Environment reveals game state
        state = self.get_state()

        # Algorithm makes prediction
        prediction = algorithm.predict_step(state)

        # Environment returns loss
        loss = self.compute_loss(prediction, state)
        self.loss_history.append(loss)

        # Algorithm updates according to loss
        algorithm.update(state, prediction, loss, y_true=self.get_y_true())

        # Update time step
        self.t += 1

        return prediction, loss

    def steps_generator(self):
        """Generate the steps for the game."""
        return range(self.n_steps)

    def play(self, algorithm: 'OnlineAlgorithm') -> dict:
        """
        Play the game for n_steps.

        Parameters
        ----------
        algorithm : OnlineAlgorithm
            The algorithm to play against
        n_steps : int
            Number of steps to play

        Returns
        -------
        results : dict
            Dictionary containing game results and statistics
        """
        for _ in self.steps_generator():
            self.step(algorithm)

        return self.get_results()

    def get_results(self) -> dict:
        """Get game results and statistics."""
        pass

    def reset(self):
        """Reset the game to initial state."""
        self.t = 0
        self.loss_history = []


class DatasetGame(OnlineGame):
    """
    Online game created from a dataset with features X and targets y.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, loss_function: Callable,
                 n_rounds: int = 1, shuffle: bool = True, random_state: Optional[int] = None):
        """
        Initialize dataset-based game.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features
        y : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Target values
        loss_function : callable
            Loss function with signature loss_function(y_true, y_pred)
        n_rounds : int, default=1
            Number of rounds to go through the dataset
        shuffle : bool, default=True
            Whether to shuffle data each round
        random_state : int, optional
            Random state for shuffling
        """
        super().__init__(n_steps=n_rounds * len(X))

        X, y = check_X_y(X, y, multi_output=True)
        self.X = X
        self._len_X = len(self.X)
        self.y = y
        self.loss_function = loss_function
        self.n_rounds = n_rounds
        self.shuffle = shuffle
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.current_idx = 0  # Current index in dataset

    def get_y_true(self) -> np.ndarray:
        """Get the true target value for the current state."""
        if self.current_idx >= self._len_X:
            raise StopIteration("No more data points available")

        return self.y[self.current_idx]

    def get_state(self) -> np.ndarray: # REMOVE THIS
        """Get current input features."""
        if self.current_idx >= self._len_X:
            raise StopIteration("No more data points available")

        return self.X[self.current_idx]

    def compute_loss(self, prediction: Any, state: Any = None) -> Union[float, np.ndarray]:
        """Compute loss using the true target."""
        y_true = self.get_y_true()

        return self.loss_function(prediction, y_true)

    def step(self, algorithm: 'OnlineAlgorithm') -> Tuple[Any, Union[float, np.ndarray]]:
        """Execute one step and advance to next data point."""
        result = super().step(algorithm)
        return result

    def reset(self):
        """Reset game and regenerate sequence."""
        super().reset()

    def steps_generator(self):
        for _ in range(self.n_rounds):
            if self.shuffle:
                indices = self.rng.permutation(self._len_X)
            else:
                indices = range(self._len_X)

            for index in indices:
                self.current_idx = index
                yield index


class OnlineAlgorithm(BaseEstimator, ABC):
    """
    Base class for online convex optimization algorithms.
    Compatible with sklearn API.
    """

    def __init__(self, loss_function: Optional[Callable] = None):
        """Initialize the online algorithm."""
        self.regret = 0.0
        self.t = 0  # Internal time step counter
        self.is_fitted = False
        self.loss_function = loss_function

    @abstractmethod
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
        pass

    @abstractmethod
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
        y_true : Any, optional
            True target value corresponding to the current state (if available).
        """
        pass

    @abstractmethod
    def update_regret(self, loss: Union[float, np.ndarray]):
        """
        Update internal regret calculation.

        Parameters
        ----------
        loss : float or np.ndarray
            Loss incurred at current step
        """
        pass

    def fit(self, X, y) -> 'OnlineAlgorithm':
        """
        Fit the algorithm by playing the entire game.

        Parameters
        ----------
        X:

        y:


        Returns
        -------
        self : OnlineAlgorithm
            Fitted algorithm
        """
        # Reset algorithm state
        self.reset()

        self.partial_fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data (sklearn compatibility).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples

        Returns
        -------
        predictions : np.ndarray
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Algorithm must be fitted before making predictions")

        X = check_array(X)
        predictions = []

        for x in X:
            pred = self.predict_step(x)
            predictions.append(pred)

        return np.array(predictions)

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> 'OnlineAlgorithm':
        """
        Incrementally fit the algorithm with new data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples
        y : np.ndarray of shape (n_samples,)
            Target values
        loss_function : callable
            Loss function

        Returns
        -------
        self : OnlineAlgorithm
        """
        X, y = check_X_y(X, y, multi_output=True)

        game = DatasetGame(X, y, loss_function=self.loss_function, shuffle=False)
        game.play(self)

        self.is_fitted = True
        return self

    def get_regret(self) -> float:
        """Get current regret value."""
        return self.regret

    def reset(self):
        """Reset algorithm to initial state."""
        self.regret = 0.0
        self.t = 0
        self.is_fitted = False


class SklearnPartialFitWrapper(OnlineAlgorithm):
    """
    Wrapper for sklearn estimators to make them compatible with online learning.
    Uses the `partial_fit` method of the underlying estimator.
    """

    def __init__(self, estimator: BaseEstimator, loss_function: Optional[Callable] = None):
        """
        Initialize the wrapper.

        Parameters
        ----------
        estimator : BaseEstimator
            Sklearn estimator to wrap
        loss_function : callable, optional
            Loss function to compute loss given (y_true, y_pred)
        """
        super().__init__(loss_function=loss_function)
        self.estimator = estimator

    def predict_step(self, state: Any) -> Any:
        """Make a prediction using the wrapped estimator."""
        try:
            state = check_array([state])
            out = self.estimator.predict(state)[0]
        except:
            out = 0.
        return out

    def update(self, state: Any, prediction: Any, loss: Union[float, np.ndarray], y_true: Any = None):
        """Update the wrapped estimator using partial_fit."""
        if y_true is not None:
            state, y_true = check_X_y([state],[y_true])
            self.estimator.partial_fit(state, y_true)
        else:
            raise ValueError("y_true must be provided for update of SklearnPartialFitWrapper")

    def update_regret(self, loss: Union[float, np.ndarray]):
        """
        Update internal regret calculation.

        Parameters
        ----------
        loss : float or np.ndarray
            Loss incurred at current step
        """
        if isinstance(loss, np.ndarray):
            raise ValueError("Loss must be a single float value for SklearnPartialFitWrapper.")
        elif isinstance(loss, float):
            self.regret += loss
        else:
            raise ValueError("Loss must be a float for SklearnPartialFitWrapper.")


# Example loss functions
def squared_loss(y_true: Union[float, np.ndarray],
                y_pred: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Squared loss function."""
    return 0.5 * (y_true - y_pred) ** 2


def absolute_loss(y_true: Union[float, np.ndarray],
                 y_pred: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Absolute loss function."""
    return np.abs(y_true - y_pred)


def logistic_loss(y_true: Union[float, np.ndarray],
                 y_pred: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Logistic loss function."""
    return np.log(1 + np.exp(-y_true * y_pred))