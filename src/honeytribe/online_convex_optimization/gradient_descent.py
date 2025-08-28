from abc import ABC
from typing import Any, Callable, Union, Literal
from honeytribe.online_convex_optimization.base import OnlineAlgorithm

import numpy as np

class DifferentiableFunction:
    """
    A class representing a differentiable function.
    This class is used to define the loss function and its gradient.
    """

    def __init__(self, function: Callable, gradient: Callable, hessian: Callable = None):
        self.function = function
        self.gradient = gradient
        self.hessian = hessian

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def gradient(self, *args, **kwargs):
        return self.gradient(*args, **kwargs)

    def hessian(self, *args, **kwargs):
        if self.hessian is None:
            raise NotImplementedError("Hessian is not implemented for this function.")
        return self.hessian(*args, **kwargs)

class DifferentiableIdentityFunction(DifferentiableFunction):
    """
    A differentiable identity function.
    It returns the first argument as the output, and its gradient is a vector of ones.
    The Hessian is a zero matrix.
    """

    def __init__(self):
        def identity_function(x, *args, **kwargs):
            return x.copy()

        def identity_gradient(x, *args, **kwargs):
            return np.ones_like(x)

        def identity_hessian(x, *args, **kwargs):
            return np.zeros((x.shape[0], x.shape[0]))

        super().__init__(function=identity_function, gradient=identity_gradient, hessian=identity_hessian)

class DifferentiableQuadraticFunction(DifferentiableFunction):
    """
    A differentiable quadratic function of the form f(x) = 0.5 * x^T A x + b^T x + c.
    This class provides methods to compute the function value, gradient, and Hessian.
    """

    def __init__(self, A: np.ndarray, b: np.ndarray, c: float):
        self.A = A
        self.b = b
        self.c = c

        def quadratic_function(x):
            return 0.5 * np.dot(x.T, np.dot(A, x)) + np.dot(b.T, x) + c

        def quadratic_gradient(x):
            return np.dot(A, x) + b

        def quadratic_hessian(x):
            return A

        super().__init__(function=quadratic_function, gradient=quadratic_gradient, hessian=quadratic_hessian)

class DifferentiableSumFunction(DifferentiableFunction):
    """
    A differentiable sum function of the form f(x) = sum(x_i).
    This class provides methods to compute the function value, gradient, and Hessian.
    """

    def __init__(self):

        def sum_function(x, *args, **kwargs):
            return np.sum(x)

        def sum_gradient(x, *args, **kwargs):
            return np.ones_like(x)

        def sum_hessian(x, *args, **kwargs):
            return np.zeros((len(x), len(x)))

        super().__init__(function=sum_function, gradient=sum_gradient, hessian=sum_hessian)

class DifferentiableSquaredFunction(DifferentiableFunction):
    """
    A differentiable squared function of the form f(x) = sum(x_i^2).
    This class provides methods to compute the function value, gradient, and Hessian.
    """

    def __init__(self):

        def squared_function(x, *args, **kwargs):
            return np.sum(x ** 2)

        def squared_gradient(x, *args, **kwargs):
            return 2 * x

        def squared_hessian(x, *args, **kwargs):
            return 2 * np.eye(len(x))

        super().__init__(function=squared_function, gradient=squared_gradient, hessian=squared_hessian)

class DifferentiableSquaredLossFunction(DifferentiableFunction):
    """
    A differentiable squared loss function of the form f(x) = 0.5 * (x - y)^2.
    This class provides methods to compute the function value, gradient, and Hessian.
    """

    def __init__(self):

        def squared_loss_function(x, y):
            return 0.5 * np.sum((x - y) ** 2)

        def squared_loss_gradient(x, y):
            return x - y

        def squared_loss_hessian(x, y):
            return np.eye(len(x))

        super().__init__(function=squared_loss_function, gradient=squared_loss_gradient, hessian=squared_loss_hessian)

class DifferentiableAbsoluteLossFunction(DifferentiableFunction):
    """
    A differentiable absolute error loss function of the form f(x) = sum(abs(x - y)).
    This class provides methods to compute the function value, gradient, and Hessian.
    """

    def __init__(self):

        def absolute_loss_function(x, y):
            return np.sum(np.abs(x - y))

        def absolute_loss_gradient(x, y):
            return np.sign(x - y)

        def absolute_loss_hessian(x, y):
            return np.zeros((len(x), len(x)))

        super().__init__(function=absolute_loss_function, gradient=absolute_loss_gradient, hessian=absolute_loss_hessian)


class StochasticDifferentiableFunction(DifferentiableFunction):
    """
    A class representing a stochastic differentiable function.
    This class is used to define the loss function and its gradient for stochastic optimization.
    """

    def __init__(self, function: Callable, size=1000, seed: int = None, delta=0.01):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.delta = delta
        super().__init__(function=function, gradient=self.stochastic_gradient)

    def point_sampler(self, x, *args, **kwargs):
        """
        Sample a point from the function's domain.
        This method should be implemented by subclasses.

        If necessary, it can be overridden to provide a specific sampling strategy.
        Parameters
        ----------
        x : np.ndarray
            The point at which to sample the function.
        Returns
        -------
        point_cloud : np.ndarray
            A point cloud sampled from the function's domain, with shape (size, x.shape).
        weight : np.ndarray
            Weights for each point in the cloud, with shape (size,).

        Notes
        -----
        To get the creative juices flowing for implementing custom point sampling strategies,
        consider the following:
        - Sampling from a Gaussian distribution centered around the point x.
        - Sampling uniformly within a certain radius around x.
        - Sampling from a Gaussian distribution centered around the point x, and normalizing such that you get a point cloud of unit vectors around x.
        - Sampling from a discrete set of points in the vicinity of x.
        """
        if not isinstance(x, np.ndarray):
            raise ValueError("Point sampler expects a numpy array.")
        if x.ndim > 1:
            raise ValueError("Point sampler expects a 1D array.")
        elif x.ndim == 0:
            raise ValueError("Point sampler expects a non-empty array.")

        point_cloud = self.rng.normal(loc = np.zeros_like(x), size = (self.size, *x.shape))
        # point_cloud = point_cloud / np.linalg.norm(point_cloud, axis=1)[:, np.newaxis]  # Normalize and scale
        point_cloud = self.delta * point_cloud
        weight = np.ones(self.size) / self.size  # Uniform weights for each point in the cloud
        return point_cloud + x[np.newaxis, :], weight

    def stochastic_gradient(self, x, *args, **kwargs):
        """
        Compute the stochastic gradient of the function.
        This method should be implemented by subclasses.
        """
        point_cloud, weight = self.point_sampler(x, *args, **kwargs)
        N = len(x)
        Z = np.eye(N)
        base = self.function(x)
        gradient = np.zeros_like(x)
        for i in range(N):
            e = Z[i]
            overlap = np.dot(point_cloud - x[np.newaxis, :], e) # shape: size
            gradient[i] = sum([ weight[j]*(self.function(point_cloud[j])-base)/overlap[j] for j in range(self.size) ])
        return gradient

class BetaHolderReductionFunction(DifferentiableFunction):
    """
    A class representing a Beta-Hölder reduction function.
    This class is used to define the loss function and its gradient for optimization problems
    that require a Hölder condition with a beta parameter.
    """

    def __init__(self, beta: float, x_1: np.ndarray, function: Callable, gradient: Callable, hessian: Callable = None):
        if not (0 < beta <= 1):
            raise ValueError("Beta must be in the range (0, 1].")
        self.beta = beta
        self.x_1 = x_1
        self.prime_function = function
        self.prime_gradient = gradient
        self.prime_hessian = hessian
        super().__init__(function=self._function, gradient=self._gradient, hessian = hessian if hessian is not None else None)
    def alpha(self):
        """
        Compute the alpha parameter based on the beta parameter.
        The alpha parameter is defined as 1 - beta.
        """
        return 1 - self.beta
    def _function(self, x, *args, **kwargs):
        """
        Compute the value of the Beta-Hölder reduction function.
        This method should be implemented by subclasses.
        """
        return self.prime_function(x, *args, **kwargs) + self.alpha() / 2 * np.linalg.norm(x - self.x_1)**2
    def _gradient(self, x, *args, **kwargs):
        """
        Compute the gradient of the Beta-Hölder reduction function.
        This method should be implemented by subclasses.
        """
        return self.prime_gradient(x, *args, **kwargs) + self.alpha() / 2 * (x - self.x_1)
    def _hessian(self, x, *args, **kwargs):
        """
        Compute the Hessian of the Beta-Hölder reduction function.
        This method should be implemented by subclasses.
        """
        if self.prime_hessian is None:
            raise NotImplementedError("Hessian is not implemented for this function.")
        return self.prime_hessian(x, *args, **kwargs) + self.alpha() / 2 * np.eye(len(self.x_1))

class BaseGradientDescentAlgorithm(OnlineAlgorithm, ABC):
    """
    An abstract class for expert algorithms in online convex optimization.
    This class extends the OnlineAlgorithm base class and is intended to be
    subclassed by specific expert algorithms.
    """

    def __init__(self, loss_function: DifferentiableFunction, parameters: Any, objective_function: DifferentiableFunction):
        super().__init__(loss_function=loss_function)
        if parameters is None:
            raise ValueError("Parameters must be provided for the algorithm.")
        elif isinstance(parameters, int):
            prior = self.initialize_parameters(parameters)
        else:
            prior = parameters
        self._prior = prior
        self.parameters = prior # TODO: Introduce copy mechanism to avoid modifying the original parameters
        self.objective_function = objective_function

    def initialize_parameters(self, n_parameters: int):
        """
        Initialize the parameters for the algorithm based on the number of experts.

        Parameters
        ----------
        num_experts : int
            Number of experts to initialize parameters for.
        """
        return np.zeros(n_parameters)

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
        return self.objective_function(self.parameters, state)
    def reset(self):
        """Reset algorithm to initial state."""
        super().reset()
        self.parameters = self._prior  # TODO: Introduce copy mechanism to avoid modifying the original parameters

class GradientDescentAlgorithm(BaseGradientDescentAlgorithm):
    """
    A simple gradient descent algorithm for online convex optimization.
    This class implements the gradient descent update rule.

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

    Notes
    -----
    The algorithm updates the parameters based on the gradient of the loss function
    and the gradient of the objective function with respect to the parameters.
    The update rule is:
    parameters -= learning_rate * (loss_function.gradient(prediction, state, loss, y_true) @ objective_function.gradient(parameters, state))
    """

    def __init__(self,
                 parameters: Any,
                 objective_function: DifferentiableFunction,
                 learning_rate: Union[float,Callable] = 0.01,
                 loss_function: DifferentiableFunction = None,
                 projection_function: Callable = None
                 ):
        if loss_function is None:
            loss_function = DifferentiableIdentityFunction()
        super().__init__(loss_function, parameters, objective_function)
        self.learning_rate = learning_rate
        self.projection_function = projection_function
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
        self.parameters -= self._learning_rate(self.t) * np.dot(loss_fn_gradient, objective_fn_gradient)
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

class OnlineNewtonStepAlgorithm(BaseGradientDescentAlgorithm):
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
        inv_A = self._inverse_approx_hessian
        correction_matrix = inv_A @ np.outer(grad, grad) @ inv_A / (1 + grad.T @ inv_A @ grad)
        return self._inverse_approx_hessian - correction_matrix


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

        self._approx_hessian += np.outer(end_gradient, end_gradient)
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

class OnlineMirrorDescentAlgorithm(BaseGradientDescentAlgorithm):
    """
    Implements the online mirror descent algorithm for online convex optimization.

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

    Notes
    -----
    This algorithm implements the online mirror descent method as described in "Online Convex Optimization" by Elad Hazan.
    """

    def __init__(self,
                 parameters: Any,
                 objective_function: DifferentiableFunction,
                 regularization_function: DifferentiableFunction,
                 inverse_regularization_gradient_function: Callable,
                 learning_rate: Union[float,Callable] = 0.01,
                 loss_function: DifferentiableFunction = None,
                 projection_function: Callable = None,
                 ):
        if loss_function is None:
            loss_function = DifferentiableIdentityFunction()
        super().__init__(loss_function, parameters, objective_function)
        self.learning_rate = learning_rate
        self.projection_function = projection_function
        self.regularization_function = regularization_function
        self.inverse_regularization_gradient_function = inverse_regularization_gradient_function
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
    def update(self, state: Any, prediction: Any, loss: Union[float, np.ndarray], y_true: Any = None):
        """
        Update algorithm parameters based on observed loss.

        Parameters
        ----------
        """
        if isinstance(loss, np.ndarray):
            raise ValueError("Loss must be a single float value for GradientDescentAlgorithm.")
        loss_fn_gradient = self.loss_function.gradient(prediction, y_true)
        objective_fn_gradient = self.objective_function.gradient(self.parameters, state)

        end_gradient = np.dot(loss_fn_gradient, objective_fn_gradient)

        if isinstance(end_gradient, float):
            end_gradient = np.array([end_gradient])

        self.parameters = self.inverse_regularization_gradient_function(self.regularization_function.gradient(self.parameters) - self.learning_rate * end_gradient)

        if self.projection_function is not None:
            self.parameters = self.projection_function(self.parameters)

        self.update_regret(loss)

        self.t += 1
    def update_regret(self, loss: Union[float, np.ndarray]):
        """
        Update internal regret calculation.
        Parameters
        ----------
        """
        if isinstance(loss, float):
            self.regret += loss
        elif isinstance(loss, np.ndarray):
            raise ValueError("Loss must be a single float value for GradientDescentAlgorithm.")

class ExponentialGradientAlgorithm(BaseGradientDescentAlgorithm):
    """
    Implements the exponential gradient algorithm for online convex optimization.

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

    Notes
    -----
    This algorithm implements the exponential gradient method as described in "Online Convex Optimization" by Elad Hazan.
    """

    def __init__(self,
                 parameters: Any,
                 objective_function: DifferentiableFunction,
                 learning_rate: Union[float,Callable] = 0.01,
                 loss_function: DifferentiableFunction = None,
                 projection_function: Callable = None
                 ):
        if loss_function is None:
            loss_function = DifferentiableIdentityFunction()
        super().__init__(loss_function, parameters, objective_function)
        self.learning_rate = learning_rate
        self.projection_function = projection_function

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

        self.parameters *= np.exp(-self._learning_rate(self.t) * end_gradient)

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

class OnlineADAMAlgorithm(BaseGradientDescentAlgorithm):
    """
    Implements the online ADAM algorithm for online convex optimization.

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
    beta1 : float, optional
        Exponential decay rate for the first moment estimates, default is 0.9.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates, default is 0.999.
    epsilon : float, optional
        A small value to prevent division by zero, default is 1e-8.

    Notes
    -----
    This algorithm implements the ADAM optimization method as described in "Adam: A Method for Stochastic Optimization" by Diederik P. Kingma and Jimmy Ba.
    """

    def __init__(self,
                 parameters: Any,
                 objective_function: DifferentiableFunction,
                 learning_rate: Union[float,Callable] = 0.01,
                 loss_function: DifferentiableFunction = None,
                 projection_function: Callable = None,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 ):
        if loss_function is None:
            loss_function = DifferentiableIdentityFunction()
        super().__init__(loss_function, parameters, objective_function)
        self.learning_rate = learning_rate
        self.projection_function = projection_function
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros_like(self.parameters)
        self.v = np.zeros_like(self.parameters)
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

        self.m = self.beta1 * self.m + (1 - self.beta1) * end_gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (end_gradient ** 2)

        m_hat = self.m / (1 - self.beta1 ** (self.t+1))
        v_hat = self.v / (1 - self.beta2 ** (self.t+1))

        self.parameters -= self._learning_rate(self.t) * m_hat / (np.sqrt(v_hat) + self.epsilon)

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

class OnlineFKMAlgorithm(BaseGradientDescentAlgorithm):
    """
    Implements the online Flaxman-Kalai-McMahan algorithm for online convex optimization.

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
    """

    def __init__(self,
                 parameters: Any,
                 objective_function: DifferentiableFunction,
                 learning_rate: Union[float,Callable] = 0.01,
                 delta: Union[float,Callable] = 0.01,
                 loss_function: DifferentiableFunction = None,
                 projection_function: Callable = None,
                 seed: int = 0,
                 ):
        if loss_function is None:
            loss_function = DifferentiableIdentityFunction()
        super().__init__(loss_function, parameters, objective_function)
        self.learning_rate = learning_rate
        self.delta = delta
        self.projection_function = projection_function
        self.rng = np.random.default_rng(seed)
        self._perturb = True

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
    def _delta(self, t: int) -> float:
        """
        Compute the delta at time t.

        Parameters
        ----------
        t : int
            Current time step

        Returns
        -------
        float
            Delta at time t
        """
        if callable(self.delta):
            return self.delta(t)
        return self.delta

    def _random_sphere_vector(self) -> np.ndarray:
        """
        Generate a random unit vector in n-dimensional space.

        Returns
        -------
        np.ndarray
            A random unit vector of shape of parameters
        """
        vec = self.rng.normal(0, 1, size=self.parameters.shape)
        while np.isclose(vec, 0).all():
            vec = self.rng.normal(0, 1, size=self.parameters.shape)
        vec = vec / np.linalg.norm(vec)
        return vec

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
        params = self.parameters
        if not self.is_fitted:
            delta = self._delta(self.t)
            self._last_sphere_vector = self._random_sphere_vector()
            params = self.parameters + delta * self._last_sphere_vector
        return self.objective_function(params, state)

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

        gradient = self.parameters.shape[0]/self._delta(self.t) * loss * self._last_sphere_vector

        self.parameters -= self._learning_rate(self.t) * gradient

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
    def reset(self):
        """Reset algorithm to initial state."""
        super().reset()
        self._perturb = True

class BaseParameterReprocessingMixin(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reprocessed_params = None
        self._stored_algorithm_params = None
        self._live_params: Literal["unprocessed", "reprocessed"] = "unprocessed"
    def update(self, state: Any, prediction: Any, loss: Union[float, np.ndarray], y_true: Any = None):
        super().update(state, prediction, loss, y_true)
    @property
    def is_reprocessed(self):
        return self._live_params == "reprocessed"
    def set_params_to_algorithm_params(self):
        if self.is_reprocessed:
            if self._stored_algorithm_params is not None:
                stored = self._stored_algorithm_params.copy()
                self.parameters = stored
            self._live_params = "unprocessed"
    def set_params_to_reprocessed_params(self):
        if not self.is_reprocessed:
            if self._reprocessed_params is None:
                raise ValueError('Parameters not yet reprocessed. Cannot set to algorithm parameters.')
            reprocessed = self._reprocessed_params.copy()
            to_store = self.parameters.copy()
            self.parameters = reprocessed
            self._stored_algorithm_params = to_store
            self._live_params = "reprocessed"
    def partial_fit(self, X, y):
        self.set_params_to_algorithm_params()
        super().partial_fit(X, y)
        self.set_params_to_reprocessed_params()
    def reset(self):
        super().reset()
        self._reprocessed_params = None
        self._stored_algorithm_params = None
        self._live_params = "unprocessed"

class MeanParameterReprocessorMixin(BaseParameterReprocessingMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reprocessed_params = np.zeros_like(self.parameters)
    def update(self, state: Any, prediction: Any, loss: Union[float, np.ndarray], y_true: Any = None):
        super().update(state, prediction, loss, y_true)
        self._reprocessed_params = (self.t - 1)/self.t * self._reprocessed_params + 1/self.t * self.parameters
    def reset(self):
        super().reset()
        self._reprocessed_params = np.zeros_like(self.parameters)

class ExponentialMeanParameterReprocessorMixin(BaseParameterReprocessingMixin):
    def __init__(self, *args, reprocessor_beta = .9, **kwargs):
        super().__init__(*args, **kwargs)
        self._reprocessed_params = np.zeros_like(self.parameters)
        self._reprocessor_beta = reprocessor_beta
    def update(self, state: Any, prediction: Any, loss: Union[float, np.ndarray], y_true: Any = None):
        super().update(state, prediction, loss, y_true)
        self._reprocessed_params = self._reprocessor_beta * self._reprocessed_params + (1 - self._reprocessor_beta) * self.parameters.copy()
    def reset(self):
        super().reset()
        self._reprocessed_params = np.zeros_like(self.parameters)

class RootMeanParameterReprocessorMixin(BaseParameterReprocessingMixin):
    def __init__(self, *args, reprocessor_root: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self._reprocessed_params = np.zeros_like(self.parameters)
        self._reprocessor_root = reprocessor_root
    def update(self, state: Any, prediction: Any, loss: Union[float, np.ndarray], y_true: Any = None):
        super().update(state, prediction, loss, y_true)
        N = np.power(self.t, self._reprocessor_root)
        beta = (N - 1) / N
        self._reprocessed_params = beta * self._reprocessed_params + (1 - beta) * self.parameters.copy()
    def reset(self):
        super().reset()
        self._reprocessed_params = np.zeros_like(self.parameters)