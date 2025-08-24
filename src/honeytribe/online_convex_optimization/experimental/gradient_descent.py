from typing import Any, Callable, Union
from honeytribe.online_convex_optimization.gradient_descent import BaseGradientDescentAlgorithm, DifferentiableFunction, DifferentiableIdentityFunction

import numpy as np

class OnlineForgetfulNewtonStepAlgorithm(BaseGradientDescentAlgorithm):
    """
    Implements the online newton step algorithm for online convex optimization.

    Parameters
    ----------
    parameters : Any
        Initial parameters for the algorithm, can be a vector or matrix.
    objective_function : DifferentiableFunction
        The objective function to optimize, must be differentiable.
    learning_rate : float, Callable, optional
        Learning rate for the gradient descent update, default is 0.01.
    loss_function : DifferentiableFunction, optional
        The loss function to use, default is a differentiable identity function.
    projection_function : Callable, optional
        A projection function to apply to the parameters after each update, default is None.
    epsilon : float, optional
        A small value to initialize the approximate Hessian, default is 1e-8.

    Notes
    -----
    Might not be working properly yet. Implemented as described in "Online Convex Optimization" by Elad Hazan.
    Some test cases seem to provide some difficulty for this algorithm.
    """

    def __init__(self,
                 parameters: Any,
                 beta: float,
                 K: float,
                 objective_function: DifferentiableFunction,
                 learning_rate: Union[float,Callable] = 0.01,
                 loss_function: DifferentiableFunction = None,
                 projection_function: Callable = None,
                 epsilon: float = 1e-8,
                 ):
        if loss_function is None:
            loss_function = DifferentiableIdentityFunction()
        super().__init__(loss_function, parameters, objective_function)
        self.learning_rate = learning_rate
        self.projection_function = projection_function
        self._init_approx_hessian = epsilon * np.eye(len(self.parameters))
        self._approx_hessian = self._init_approx_hessian.copy()
        self._inverse_approx_hessian = np.linalg.inv(self._approx_hessian)
        self.beta = beta
        self.K = K
    def _learning_rate(self, t: int) -> float:
        """
        Compute the learning rate at time t.

        Parameters
        ----------
        t : int
            Current time step

        Returns
        -------
        float
            Learning rate at time t
        """
        if callable(self.learning_rate):
            return self.learning_rate(t)
        return self.learning_rate
    def update_inverse_approx_hessian(self, grad: np.ndarray) -> np.ndarray:
        """
        Update the inverse approximate Hessian using the Sherman-Morrison formula.

        Parameters
        ----------
        grad : np.ndarray
            The gradient used to update the inverse approximate Hessian.

        Returns
        -------
        np.ndarray
            The updated inverse approximate Hessian.
        """
        inv_A = self._inverse_approx_hessian / self.beta
        grad = self.K * (1-self.beta) * grad
        correction_matrix = inv_A @ np.outer(grad, grad) @ inv_A / (1 + grad.T @ inv_A @ grad)
        return inv_A - correction_matrix


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
            raise ValueError("Loss must be a single float value for GradientDescentAlgorithm.")
        loss_fn_gradient = self.loss_function.gradient(prediction, y_true)
        objective_fn_gradient = self.objective_function.gradient(self.parameters, state)

        end_gradient = np.dot(loss_fn_gradient, objective_fn_gradient)

        if isinstance(end_gradient, float):
            end_gradient = np.array([end_gradient])

        self._approx_hessian = self.beta * self._approx_hessian + self.K * (1 - self.beta) * np.outer(end_gradient, end_gradient)
        self._inverse_approx_hessian = self.update_inverse_approx_hessian(end_gradient)

        self.parameters -= self._learning_rate(self.t) * self._inverse_approx_hessian @ end_gradient
        if self.projection_function is not None:
            self.parameters = self.projection_function(self.parameters)

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
        elif isinstance(loss, np.ndarray):
            raise ValueError("Loss must be a single float value for GradientDescentAlgorithm.")
