from honeytribe.online_convex_optimization.expert import HedgeAlgorithm, Exp3Algorithm, FixedShareAlgorithm, \
    UCBAlgorithm, TimeseriesExponentiatedHedgeAlgorithm, TimeseriesExponentiatedFixedShareAlgorithm
from honeytribe.online_convex_optimization.base import SklearnPartialFitWrapper
from ml_test.base import Algorithm, TestCase, DataGenerators, AlgorithmTester
from sklearn.linear_model import SGDRegressor

from unittest.mock import patch
from typing import Any, List, Dict

import numpy as np


class WrapperAlgorithm(Algorithm):
    """HedgeAlgorithm wrapper for classification."""

    def tracked_updates(self, state, prediction, loss, y_true=None):
        """Track updates to the model and loss history."""
        self.model_update_fn(self.model, state, prediction, loss, y_true=y_true)
        reduced_loss = self.loss_reduction_fn(loss)
        if not isinstance(reduced_loss, float):
            raise ValueError("Loss reduction function must return a float")
        self.loss_history.append(reduced_loss)
        self.full_loss_history.append(loss)
        self.full_predict_history.append(prediction)
        reduced_prediction = self.prediction_reduction_fn(prediction)
        if not isinstance(reduced_prediction, float):
            raise ValueError("Prediction reduction function must return a float")
        self.predict_history.append(reduced_prediction)

    def __init__(self, model, loss_reduction_fn = lambda x: x, prediction_reduction_fn=lambda x: x):
        self.model = model
        self.model_update_fn = self.model.__class__.update
        self.loss_reduction_fn = loss_reduction_fn
        self.loss_history = []
        self.full_loss_history = []
        self.prediction_reduction_fn = prediction_reduction_fn
        self.predict_history = []
        self.full_predict_history = []

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
        best_expert_regret = np.array(self.full_loss_history).sum(axis=0).min() if self.full_loss_history else 0.0
        final_regret = model_regret - best_expert_regret
        average_regret = final_regret / len(self.full_loss_history) if self.full_loss_history else 0.0
        print(f"Model regret: {model_regret}, Best expert regret: {best_expert_regret}, Final regret: {final_regret}, Average regret: {average_regret}")
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


class ConstantExpert:
    """A constant expert that always predicts a fixed value."""

    def __init__(self, value: float):
        self.value = value

    def predict(self, state: Any) -> float:
        """Predict a constant value."""
        return self.value


if __name__ == "__main__" and False:
    print("Testing Online Expert Algorithms with AI Algorithm Testing Framework")
    print("=" * 60)

    # Test Classification
    print("\nTesting Hedge Algorithm...")

    def model_factory():
        """Factory function to create a new instance of the model."""
        experts = [ConstantExpert(value=i) for i in range(11)]
        experts += [SklearnPartialFitWrapper(SGDRegressor())]
        model = HedgeAlgorithm(loss_function=lambda x, y: (x - y) ** 2, experts=experts,
                               prior=np.ones(len(experts)) / len(experts), learning_rate=1.)
        wrapped_model = WrapperAlgorithm(model, loss_reduction_fn=lambda loss, model=model: np.sum(model.dist * loss), prediction_reduction_fn=lambda x, model=model: np.sum(model.dist * x))
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
    print("\nTesting Exp3 Algorithm...")

    def model_factory():
        """Factory function to create a new instance of the model."""
        experts = [ConstantExpert(value=i) for i in range(11)]
        experts += [SklearnPartialFitWrapper(SGDRegressor())]
        model = Exp3Algorithm(loss_function=lambda x, y: (x - y) ** 2, experts=experts,
                               prior=np.ones(len(experts)) / len(experts), learning_rate=1.)
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
    print("\nTesting Fixed Share Algorithm...")

    def model_factory():
        """Factory function to create a new instance of the model."""
        experts = [ConstantExpert(value=i) for i in range(11)]
        experts += [SklearnPartialFitWrapper(SGDRegressor())]
        model = FixedShareAlgorithm(loss_function=lambda x, y: (x - y) ** 2, experts=experts,
                               prior=np.ones(len(experts)) / len(experts), learning_rate=1., switching_rate=.1)
        wrapped_model = WrapperAlgorithm(model, loss_reduction_fn=lambda loss, model=model: np.sum(model.dist * loss), prediction_reduction_fn=lambda x, model=model: np.sum(model.dist * x))
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
    print("\nTesting UCB Algorithm...")

    def model_factory():
        """Factory function to create a new instance of the model."""
        experts = [ConstantExpert(value=i) for i in range(11)]
        experts += [SklearnPartialFitWrapper(SGDRegressor())]
        model = UCBAlgorithm(loss_function=lambda x, y: (x - y) ** 2, experts=experts, alpha=1.1)
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
    print("\nTesting TimeSeries Hedge Algorithm...")

    def time_features_fn(count):
        """Generate time features for the TimeSeriesHedgeAlgorithm."""
        out = [1]
        out += [count]
        freq = [10, 100, 200]
        out += [np.sin(count/num * 2 * np.pi) for num in freq]
        out += [np.cos(count/num * 2 * np.pi) for num in freq]
        return np.array(out)

    def model_factory():
        """Factory function to create a new instance of the model."""
        experts = [ConstantExpert(value=i) for i in range(11)]
        experts += [SklearnPartialFitWrapper(SGDRegressor())]
        model = TimeseriesExponentiatedHedgeAlgorithm(
            loss_function=lambda x, y: (x - y) ** 2,
            experts=experts,
            time_features=time_features_fn,
        )
        wrapped_model = WrapperAlgorithm(model, loss_reduction_fn=lambda loss, model=model: np.sum(model.dist * loss), prediction_reduction_fn=lambda x, model=model: np.sum(model.dist * x))
        return wrapped_model

    tester = AlgorithmTester(results_dir="dt_test_results")

    # Create classification test cases (exclude regression)
    tests = [tc for tc in create_decision_tree_test_cases()]

    results = tester.run_test_suite(model_factory, tests)

    print("\n✅ All tests completed! Check the 'dt_test_results' directory for detailed results and plots.")

if __name__ == "__main__" and True:
    print("Testing Online Expert Algorithms with AI Algorithm Testing Framework")
    print("=" * 60)

    # Test Classification
    print("\nTesting TimeSeries Hedge Algorithm...")

    def time_features_fn(count):
        """Generate time features for the TimeSeriesHedgeAlgorithm."""
        out = [1]
        out += [count]
        freq = [10, 100, 200]
        out += [np.sin(count/num * 2 * np.pi) for num in freq]
        out += [np.cos(count/num * 2 * np.pi) for num in freq]
        return np.array(out)

    ### NOTE!!! Dataset reduced to 10 points for faster testing. Reset to 1000 for full tests.

    def model_factory():
        """Factory function to create a new instance of the model."""
        experts = [ConstantExpert(value=i) for i in range(11)]
        experts += [SklearnPartialFitWrapper(SGDRegressor())]
        model = TimeseriesExponentiatedFixedShareAlgorithm(
            loss_function=lambda x, y: (x - y) ** 2,
            experts=experts,
            time_features=time_features_fn,
            switching_rate=.1,
        )
        wrapped_model = WrapperAlgorithm(model, loss_reduction_fn=lambda loss, model=model: loss[np.argmax(model.dist)], prediction_reduction_fn=lambda x, model=model: x[np.argmax(model.dist)])
        return wrapped_model

    tester = AlgorithmTester(results_dir="dt_test_results")

    # Create classification test cases (exclude regression)
    tests = [tc for tc in create_decision_tree_test_cases()]

    results = tester.run_test_suite(model_factory, tests)

    print("\n✅ All tests completed! Check the 'dt_test_results' directory for detailed results and plots.")
