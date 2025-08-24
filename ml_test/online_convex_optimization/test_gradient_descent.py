from honeytribe.online_convex_optimization.gradient_descent import GradientDescentAlgorithm, DifferentiableFunction, DifferentiableIdentityFunction, DifferentiableSquaredLossFunction, StochasticDifferentiableFunction, BetaHolderReductionFunction, DifferentiableAbsoluteLossFunction, OnlineNewtonStepAlgorithm, DifferentiableSumFunction, OnlineMirrorDescentAlgorithm, ExponentialGradientAlgorithm, DifferentiableSquaredFunction, OnlineADAMAlgorithm, OnlineFKMAlgorithm
from honeytribe.online_convex_optimization.experimental.gradient_descent import OnlineForgetfulNewtonStepAlgorithm
from ml_test.base import Algorithm, TestCase, DataGenerators, AlgorithmTester

from unittest.mock import patch
from typing import Any, List, Dict

import numpy as np


class WrapperAlgorithm(Algorithm):
    """HedgeAlgorithm wrapper for classification."""

    def tracked_updates(self, state, prediction, loss, y_true=None):
        """Track updates to the model and loss history."""
        self.model_update_fn(self.model, state, prediction, loss, y_true=y_true)
        self.loss_history.append(loss)
        self.predict_history.append(prediction)

    def __init__(self, model):
        self.model = model
        self.model_update_fn = self.model.__class__.update
        self.loss_history = []
        self.predict_history = []

        self.model.update = self.tracked_updates

    @property
    def is_fitted(self):
        """Check if the model is fitted."""
        return hasattr(self.model, 'is_fitted') and self.model.is_fitted

    def fit(self, X, y=None, **kwargs):
        """Train the decision tree."""
        self.model.fit(X, y)

    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

    def get_loss_history(self) -> List[float]:
        """Return the loss history."""
        return self.loss_history

    def get_prediction_history(self) -> List[float]:
        """Return the prediction history."""
        return self.predict_history

    def get_performance_metrics(self, X_test, y_test) -> Dict[str, float]:
        """Return performance metrics on test data."""
        if not self.is_fitted:
            print("Model is not fitted. Cannot calculate performance metrics.")
            return {}
        print("Calculating performance metrics...")
        model_regret = self.model.regret if hasattr(self.model, 'regret') else 0.0
        final_regret = self.loss_history[-1]
        average_regret = np.mean(self.loss_history) if self.loss_history else 0.0
        print(f"Model regret: {model_regret}, Final regret: {final_regret}, Average regret: {average_regret}")
        return {
            'regret':average_regret
        }


def create_decision_tree_test_cases():
    """Create test cases specifically for decision trees."""
    return [
        TestCase(
            name="constant",
            data_generator=DataGenerators.constant0,
            expected_convergence=True,
            max_iterations=100,  # Not really used for trees
            convergence_threshold=1e-3,
            performance_thresholds={'regret': 0.90},
            timeout_seconds=10.0
        ),
        TestCase(
            name="constant",
            data_generator=DataGenerators.constant1,
            expected_convergence=True,
            max_iterations=100,  # Not really used for trees
            convergence_threshold=1e-3,
            performance_thresholds={'regret': 0.90},
            timeout_seconds=10.0
        ),
        TestCase(
            name="dt_linear_separable",
            data_generator=DataGenerators.linear_separable_2d,
            expected_convergence=True,
            max_iterations=100,  # Not really used for trees
            convergence_threshold=1e-3,
            performance_thresholds={'regret': 0.90},
            timeout_seconds=10.0
        ),
        TestCase(
            name="dt_linear",
            data_generator=DataGenerators.linear_2d,
            expected_convergence=True,
            max_iterations=100,  # Not really used for trees
            convergence_threshold=1e-3,
            performance_thresholds={'regret': 0.90},
            timeout_seconds=10.0
        ),
        TestCase(
            name="dt_xor_problem",
            data_generator=DataGenerators.xor_problem,
            expected_convergence=True,  # Trees should handle XOR well
            max_iterations=100,
            convergence_threshold=1e-3,
            performance_thresholds={'regret': 0.85},
            timeout_seconds=10.0
        ),
        TestCase(
            name="dt_polynomial_regression",
            data_generator=DataGenerators.regression_polynomial,
            expected_convergence=True,
            max_iterations=100,
            convergence_threshold=1e-3,
            performance_thresholds={'regret': 0.75},
            timeout_seconds=15.0
        ),
        TestCase(
            name="dt_switching",
            data_generator=DataGenerators.switching,
            expected_convergence=True,
            max_iterations=100,
            convergence_threshold=1e-3,
            performance_thresholds={'regret': 0.75},
            timeout_seconds=15.0
        ),
        TestCase(
            name="dt_sine_wave",
            data_generator=DataGenerators.timeseries_sine,
            expected_convergence=True,
            max_iterations=100,
            convergence_threshold=1e-3,
            performance_thresholds={'regret': 0.75},
            timeout_seconds=15.0
        ),
        TestCase(
            name="dt_linear",
            data_generator=DataGenerators.timeseries_linear,
            expected_convergence=True,
            max_iterations=100,
            convergence_threshold=1e-3,
            performance_thresholds={'regret': 0.75},
            timeout_seconds=15.0
        ),
    ]


if __name__ == "__main__" and False:
    print("Testing Online Expert Algorithms with AI Algorithm Testing Framework")
    print("=" * 60)

    # Test Classification
    print("\nTesting Gradient Descent Algorithm...")

    def model_factory():
        """Factory function to create a new instance of the model."""
        learning_rate = lambda t, gamma=.1: gamma / np.sqrt(t) if t > 0 else gamma
        objective_function = DifferentiableSumFunction()
        loss_function = DifferentiableSquaredLossFunction()
        model = GradientDescentAlgorithm(2, objective_function, learning_rate=learning_rate, loss_function=loss_function)
        wrapped_model = WrapperAlgorithm(model)
        return wrapped_model

    tester = AlgorithmTester(results_dir="dt_test_results")

    # Create classification test cases (exclude regression)
    tests = [tc for tc in create_decision_tree_test_cases()]

    results = tester.run_test_suite(model_factory, tests)

    print("\n✅ All tests completed! Check the 'dt_test_results' directory for detailed results and plots.")

if __name__ == "__main__" and False:
    print("Testing Online Expert Algorithms with AI Algorithm Testing Framework")
    print("=" * 60)

    # Test Classification
    print("\nTesting Gradient Descent with Stochastised Gradient Algorithm...")

    def model_factory():
        """Factory function to create a new instance of the model."""
        learning_rate = lambda t, gamma=.1: gamma / np.sqrt(t) if t > 0 else gamma
        objective_function = StochasticDifferentiableFunction(lambda x, *args, **kwargs: np.round(x), size=100, delta=1.)
        loss_function = DifferentiableSquaredLossFunction()
        model = GradientDescentAlgorithm(1, objective_function, learning_rate=learning_rate, loss_function=loss_function)
        wrapped_model = WrapperAlgorithm(model)
        return wrapped_model

    tester = AlgorithmTester(results_dir="dt_test_results")

    # Create classification test cases (exclude regression)
    tests = [tc for tc in create_decision_tree_test_cases()]

    results = tester.run_test_suite(model_factory, tests)

    print("\n✅ All tests completed! Check the 'dt_test_results' directory for detailed results and plots.")

if __name__ == "__main__" and False:
    print("Testing Online Expert Algorithms with AI Algorithm Testing Framework")
    print("=" * 60)

    # Test Classification
    print("\nTesting Gradient Descent with Beta-Holder Gradient Algorithm...")

    def model_factory():
        """Factory function to create a new instance of the model."""
        objective_function = StochasticDifferentiableFunction(lambda x, *args, **kwargs: np.round(x), size=100, delta=1.)
        pre_holder_function = DifferentiableAbsoluteLossFunction()
        loss_function = BetaHolderReductionFunction(function = pre_holder_function.function, gradient= pre_holder_function.gradient, beta=1., x_1 = np.array(0.))
        model = GradientDescentAlgorithm(1, objective_function, learning_rate=0.1, loss_function=loss_function)
        wrapped_model = WrapperAlgorithm(model)
        return wrapped_model

    tester = AlgorithmTester(results_dir="dt_test_results")

    # Create classification test cases (exclude regression)
    tests = [tc for tc in create_decision_tree_test_cases()]

    results = tester.run_test_suite(model_factory, tests)

    print("\n✅ All tests completed! Check the 'dt_test_results' directory for detailed results and plots.")

if __name__ == "__main__" and False:
    print("Testing Online Gradient Descent Algorithms with AI Algorithm Testing Framework")
    print("=" * 60)

    # Test Classification
    print("\nTesting Online Newton Step Algorithm...")

    def model_factory():
        """Factory function to create a new instance of the model."""
        objective_function = DifferentiableSumFunction()
        loss_function = DifferentiableSquaredLossFunction()
        model = OnlineNewtonStepAlgorithm(2, objective_function, learning_rate=.1, loss_function=loss_function)
        wrapped_model = WrapperAlgorithm(model)
        return wrapped_model

    tester = AlgorithmTester(results_dir="dt_test_results")

    # Create classification test cases (exclude regression)
    tests = [tc for tc in create_decision_tree_test_cases()]

    results = tester.run_test_suite(model_factory, tests)

    print("\n✅ All tests completed! Check the 'dt_test_results' directory for detailed results and plots.")


if __name__ == "__main__" and False:
    print("Testing Online Gradient Descent Algorithms with AI Algorithm Testing Framework")
    print("=" * 60)

    # Test Classification
    print("\nTesting Online Mirror Descent Algorithm...")

    def model_factory():
        """Factory function to create a new instance of the model."""
        objective_function = DifferentiableSumFunction()
        loss_function = DifferentiableSquaredLossFunction()
        regularization_function = DifferentiableSquaredFunction()
        inverse_regularization_gradient_function = lambda x: x / 2.
        model = OnlineMirrorDescentAlgorithm(
            parameters = 2,
            objective_function = objective_function,
            regularization_function = regularization_function,
            inverse_regularization_gradient_function = inverse_regularization_gradient_function,
            learning_rate = .01,
            loss_function = loss_function,
        )
        wrapped_model = WrapperAlgorithm(model)
        return wrapped_model

    tester = AlgorithmTester(results_dir="dt_test_results")

    # Create classification test cases (exclude regression)
    tests = [tc for tc in create_decision_tree_test_cases()]

    results = tester.run_test_suite(model_factory, tests)

    print("\n✅ All tests completed! Check the 'dt_test_results' directory for detailed results and plots.")


if __name__ == "__main__" and False:
    print("Testing Online Gradient Descent Algorithms with AI Algorithm Testing Framework")
    print("=" * 60)

    # Test Classification
    print("\nTesting Exponential Gradient Algorithm...") # NOT TESTED!!! NEEDS CUSTOM TEST CASES

    def model_factory():
        """Factory function to create a new instance of the model."""
        objective_function = DifferentiableSumFunction()
        loss_function = DifferentiableSquaredLossFunction()
        projection_function = lambda x: np.maximum(0, x) / np.sum(np.maximum(0, x)) if np.sum(np.maximum(0, x)) > 0 else np.ones_like(x) / len(x)
        model = ExponentialGradientAlgorithm(
            parameters = 2,
            objective_function = objective_function,
            learning_rate = 0.1,
            loss_function = loss_function,
            projection_function = projection_function,
        )
        wrapped_model = WrapperAlgorithm(model)
        return wrapped_model

    tester = AlgorithmTester(results_dir="dt_test_results")

    # Create classification test cases (exclude regression)
    tests = [tc for tc in create_decision_tree_test_cases()]

    results = tester.run_test_suite(model_factory, tests)

    print("\n✅ All tests completed! Check the 'dt_test_results' directory for detailed results and plots.")

if __name__ == "__main__" and False:
    print("Testing Online Gradient Descent Algorithms with AI Algorithm Testing Framework")
    print("=" * 60)

    # Test Classification
    print("\nTesting Adam Gradient Algorithm...")

    def model_factory():
        """Factory function to create a new instance of the model."""
        # learning_rate = lambda t, gamma=.01: gamma / np.sqrt(t) if t > 0 else gamma
        learning_rate = 0.01
        objective_function = DifferentiableSumFunction()
        loss_function = DifferentiableSquaredLossFunction()
        model = OnlineADAMAlgorithm(
            parameters = 2,
            objective_function = objective_function,
            learning_rate = learning_rate,
            loss_function = loss_function,
            beta1=0.9,
            beta2=0.999,
        )
        wrapped_model = WrapperAlgorithm(model)
        return wrapped_model

    tester = AlgorithmTester(results_dir="dt_test_results")

    # Create classification test cases (exclude regression)
    tests = [tc for tc in create_decision_tree_test_cases()]

    results = tester.run_test_suite(model_factory, tests)

    print("\n✅ All tests completed! Check the 'dt_test_results' directory for detailed results and plots.")


if __name__ == "__main__" and False:
    print("Testing Online Gradient Descent Algorithms with AI Algorithm Testing Framework")
    print("=" * 60)

    # Test Classification
    print("\nTesting Exponential Gradient Algorithm...") # NOT TESTED!!! NEEDS CUSTOM TEST CASES

    def model_factory():
        """Factory function to create a new instance of the model."""
        learning_rate = lambda t, gamma=.1: gamma / np.sqrt(t) if t > 0 else gamma
        delta = lambda t, gamma=.01, scale=2.: scale * np.sqrt(learning_rate(t, gamma))
        objective_function = DifferentiableSumFunction()
        loss_function = DifferentiableSquaredLossFunction()
        model = OnlineFKMAlgorithm(
            parameters = 2,
            objective_function = objective_function,
            learning_rate = learning_rate,
            delta= delta,
            loss_function = loss_function,
            seed=0
        )
        wrapped_model = WrapperAlgorithm(model)
        return wrapped_model

    tester = AlgorithmTester(results_dir="dt_test_results")

    # Create classification test cases (exclude regression)
    tests = [tc for tc in create_decision_tree_test_cases()]

    results = tester.run_test_suite(model_factory, tests)

    print("\n✅ All tests completed! Check the 'dt_test_results' directory for detailed results and plots.")


if __name__ == "__main__" and True:
    print("Testing Online Gradient Descent Algorithms with AI Algorithm Testing Framework")
    print("=" * 60)

    # NOTE: With most hyperparameters FKM has terrible performance. There must be a better way to do this.

    # Test Classification
    print("\nTesting Exponential Gradient Algorithm...") # NOT TESTED!!! NEEDS CUSTOM TEST CASES

    def model_factory():
        """Factory function to create a new instance of the model."""
        objective_function = DifferentiableIdentityFunction()
        loss_function = DifferentiableSquaredLossFunction()
        model = OnlineForgetfulNewtonStepAlgorithm(
            parameters = 1,
            beta = .9,
            K = 20,
            objective_function = objective_function,
            learning_rate = .5,
            loss_function = loss_function,
            # epsilon=1.
        )
        wrapped_model = WrapperAlgorithm(model)
        return wrapped_model

    tester = AlgorithmTester(results_dir="dt_test_results")

    # Create classification test cases (exclude regression)
    tests = [tc for tc in create_decision_tree_test_cases()]

    results = tester.run_test_suite(model_factory, tests)

    print("\n✅ All tests completed! Check the 'dt_test_results' directory for detailed results and plots.")

