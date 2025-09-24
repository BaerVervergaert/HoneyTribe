import torch.nn
from torch.nn import MSELoss
from torch.optim import Adam

from honeytribe.fireflies.utils import TorchAdapter
from honeytribe.fireflies.expert_layers import ExpertSignLinear

from ml_test.base import Algorithm, TestCase, DataGenerators, AlgorithmTester

from typing import Any, List, Dict

import numpy as np


class WrapperAlgorithm(Algorithm):
    """Torch wrapper for ml task."""

    def tracked_batch_step(self, batch_X, batch_y, optimizer):
        """Track updates to the model and loss history."""
        outputs, loss = self.model_batch_step_fn(self.model, batch_X, batch_y, optimizer, )
        self.loss_history.append(loss.detach().item())
        self.predict_history.extend(outputs.detach().numpy())

    def __init__(self, model):
        self.model = model
        self.model_batch_step_fn = self.model.__class__.batch_step
        self.loss_history = []
        self.predict_history = []

        self.model.batch_step = self.tracked_batch_step

    @property
    def is_fitted(self):
        """Check if the model is fitted."""
        return hasattr(self.model, 'is_fitted') and self.model.is_fitted

    def fit(self, X, y=None, **kwargs):
        """Train the decision tree."""
        self.model.fit(X, y, epochs = 100)

    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

    def get_loss_history(self) -> List[float]:
        """Return the loss history."""
        return self.loss_history

    # def get_prediction_history(self) -> List[float]:
    #     """Return the prediction history."""
    #     return self.predict_history

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


def create_2_variable_test_cases():
    """Create test cases specifically for decision trees."""
    return [
        TestCase(
            name="constant",
            data_generator=DataGenerators.constant0,
            expected_convergence=True,
            max_iterations=10_000,  # Not really used for trees
            convergence_threshold=1e-3,
            performance_thresholds={'regret': 0.90},
            timeout_seconds=10.0
        ),
        TestCase(
            name="constant",
            data_generator=DataGenerators.constant1,
            expected_convergence=True,
            max_iterations=10_000,  # Not really used for trees
            convergence_threshold=1e-3,
            performance_thresholds={'mse': 0.90},
            timeout_seconds=10.0
        ),
        TestCase(
            name="dt_linear_separable",
            data_generator=DataGenerators.linear_separable_2d,
            expected_convergence=True,
            max_iterations=10_000,  # Not really used for trees
            convergence_threshold=1e-3,
            performance_thresholds={'mse': 0.90},
            timeout_seconds=10.0
        ),
        TestCase(
            name="dt_linear",
            data_generator=DataGenerators.linear_2d,
            expected_convergence=True,
            max_iterations=10_000,  # Not really used for trees
            convergence_threshold=1e-3,
            performance_thresholds={'mse': 0.90},
            timeout_seconds=10.0
        ),
        TestCase(
            name="dt_xor_problem",
            data_generator=DataGenerators.xor_problem,
            expected_convergence=True,  # Trees should handle XOR well
            max_iterations=10_000,
            convergence_threshold=1e-3,
            performance_thresholds={'mse': 0.85},
            timeout_seconds=10.0
        ),
        TestCase(
            name="dt_switching",
            data_generator=DataGenerators.switching,
            expected_convergence=True,
            max_iterations=10_000,
            convergence_threshold=1e-3,
            performance_thresholds={'mse': 0.75},
            timeout_seconds=15.0
        ),
    ]


def create_1_variable_test_cases():
    """Create test cases specifically for decision trees."""
    return [
        TestCase(
            name="dt_polynomial_regression",
            data_generator=DataGenerators.regression_polynomial,
            expected_convergence=True,
            max_iterations=10_000,
            convergence_threshold=1e-3,
            performance_thresholds={'mse': 0.75},
            timeout_seconds=15.0
        ),
        TestCase(
            name="dt_sine_wave",
            data_generator=DataGenerators.timeseries_sine,
            expected_convergence=True,
            max_iterations=10_000,
            convergence_threshold=1e-3,
            performance_thresholds={'mse': 0.75},
            timeout_seconds=15.0
        ),
        TestCase(
            name="dt_linear",
            data_generator=DataGenerators.timeseries_linear,
            expected_convergence=True,
            max_iterations=10_000,
            convergence_threshold=1e-3,
            performance_thresholds={'mse': 0.75},
            timeout_seconds=15.0
        ),
    ]


if __name__ == "__main__" and False:
    print("Testing NN Algorithms with AI Algorithm Testing Framework")
    print("=" * 60)

    # Test Classification
    print("\nTesting Linear Layer...")

    def model_factory():
        """Factory function to create a new instance of the model."""
        torch_model = torch.nn.Linear(
            1, 1, bias=True
        )
        optimizer = lambda params: Adam(params, lr=0.01)
        model = TorchAdapter(
            model = torch_model,
            optim = optimizer,
            criterion = MSELoss(),
            device = 'cpu',
        )
        wrapped_model = WrapperAlgorithm(model)
        return wrapped_model

    tester = AlgorithmTester(results_dir="dt_test_results")

    # Create classification test cases (exclude regression)
    tests = [tc for tc in create_1_variable_test_cases()]

    results = tester.run_test_suite(model_factory, tests)

    def model_factory():
        """Factory function to create a new instance of the model."""
        torch_model = torch.nn.Linear(
            2, 1, bias=True
        )
        optimizer = lambda params: Adam(params, lr=0.01)
        model = TorchAdapter(
            model = torch_model,
            optim = optimizer,
            criterion = MSELoss(),
            device = 'cpu',
        )
        wrapped_model = WrapperAlgorithm(model)
        return wrapped_model

    tester = AlgorithmTester(results_dir="dt_test_results")

    # Create classification test cases (exclude regression)
    tests = [tc for tc in create_2_variable_test_cases()]

    results = tester.run_test_suite(model_factory, tests)

    print("\n✅ All tests completed! Check the 'dt_test_results' directory for detailed results and plots.")

if __name__ == "__main__" and False:
    print("Testing NN Algorithms with AI Algorithm Testing Framework")
    print("=" * 60)

    # Test Classification
    print("\nTesting Expert Sign Layer Algorithm...")

    def model_factory():
        """Factory function to create a new instance of the model."""
        torch_model = ExpertSignLinear(
            1, 1, include_zero=True, bias=True, learning_rate=0.1, alpha=0.01
        )
        optimizer = lambda params: Adam(params, lr=0.01)
        model = TorchAdapter(
            model = torch_model,
            optim = optimizer,
            criterion = MSELoss(),
            device = 'cpu',
        )
        wrapped_model = WrapperAlgorithm(model)
        return wrapped_model

    tester = AlgorithmTester(results_dir="dt_test_results")

    # Create classification test cases (exclude regression)
    tests = [tc for tc in create_1_variable_test_cases()]

    results = tester.run_test_suite(model_factory, tests)

    def model_factory():
        """Factory function to create a new instance of the model."""
        torch_model = ExpertSignLinear(
            2, 1, include_zero=True, bias=True, learning_rate=0.1, alpha=0.01,
        )
        optimizer = lambda params: Adam(params, lr=0.01)
        model = TorchAdapter(
            model = torch_model,
            optim = optimizer,
            criterion = MSELoss(),
            device = 'cpu',
        )
        wrapped_model = WrapperAlgorithm(model)
        return wrapped_model

    tester = AlgorithmTester(results_dir="dt_test_results")

    # Create classification test cases (exclude regression)
    tests = [tc for tc in create_2_variable_test_cases()]

    results = tester.run_test_suite(model_factory, tests)

    print("\n✅ All tests completed! Check the 'dt_test_results' directory for detailed results and plots.")

if __name__ == "__main__" and True:
    print("Testing NN Algorithms with AI Algorithm Testing Framework")
    print("=" * 60)

    # Test Classification
    print("\nTesting Single Layer...")

    def model_factory():
        """Factory function to create a new instance of the model."""
        class SingleLayer(torch.nn.Module):
            def __init__(self, input_dim, output_dim, neurons):
                super().__init__()
                self.first_layer = torch.nn.Linear(input_dim, neurons)
                self.second_layer = torch.nn.Linear(neurons, output_dim)
                self.activation_function = torch.nn.ReLU()

            def forward(self, x):
                x = self.first_layer(x)
                x = self.activation_function(x)
                x = self.second_layer(x)
                return x

        torch_model = SingleLayer(
            1, 1, neurons=100
        )
        optimizer = lambda params: Adam(params, lr=0.01)
        model = TorchAdapter(
            model = torch_model,
            optim = optimizer,
            criterion = MSELoss(),
            device = 'cpu',
        )
        wrapped_model = WrapperAlgorithm(model)
        return wrapped_model

    tester = AlgorithmTester(results_dir="dt_test_results")

    # Create classification test cases (exclude regression)
    tests = [tc for tc in create_1_variable_test_cases()]

    results = tester.run_test_suite(model_factory, tests)

    def model_factory():
        """Factory function to create a new instance of the model."""
        class SingleLayer(torch.nn.Module):
            def __init__(self, input_dim, output_dim, neurons):
                super().__init__()
                self.first_layer = torch.nn.Linear(input_dim, neurons)
                self.second_layer = torch.nn.Linear(neurons, output_dim)
                self.activation_function = torch.nn.ReLU()

            def forward(self, x):
                x = self.first_layer(x)
                x = self.activation_function(x)
                x = self.second_layer(x)
                return x

        torch_model = SingleLayer(
            2, 1, neurons=100
        )
        optimizer = lambda params: Adam(params, lr=0.01)
        model = TorchAdapter(
            model = torch_model,
            optim = optimizer,
            criterion = MSELoss(),
            device = 'cpu',
        )
        wrapped_model = WrapperAlgorithm(model)
        return wrapped_model

    tester = AlgorithmTester(results_dir="dt_test_results")

    # Create classification test cases (exclude regression)
    tests = [tc for tc in create_2_variable_test_cases()]

    results = tester.run_test_suite(model_factory, tests)

    print("\n✅ All tests completed! Check the 'dt_test_results' directory for detailed results and plots.")

if __name__ == "__main__" and True:
    print("Testing NN Algorithms with AI Algorithm Testing Framework")
    print("=" * 60)

    # Test Classification
    print("\nTesting Expert Sign Layer Algorithm...")

    def model_factory():
        """Factory function to create a new instance of the model."""
        class SingleLayer(torch.nn.Module):
            def __init__(self, input_dim, output_dim, neurons):
                super().__init__()
                self.first_layer = ExpertSignLinear(input_dim, neurons, learning_rate=.1, alpha=.0)
                self.second_layer = ExpertSignLinear(neurons, output_dim, learning_rate=.1, alpha=.0)
                self.activation_function = torch.nn.ReLU()

            def forward(self, x):
                x = self.first_layer(x)
                x = self.activation_function(x)
                x = self.second_layer(x)
                return x

            def update(self, loss):
                self.first_layer.update(loss)
                self.second_layer.update(loss)

        torch_model = SingleLayer(
            1, 1, neurons=100
        )
        optimizer = lambda params: Adam(params, lr=0.01)
        model = TorchAdapter(
            model = torch_model,
            optim = optimizer,
            criterion = MSELoss(),
            device = 'cpu',
        )
        wrapped_model = WrapperAlgorithm(model)
        return wrapped_model

    tester = AlgorithmTester(results_dir="dt_test_results")

    # Create classification test cases (exclude regression)
    tests = [tc for tc in create_1_variable_test_cases()]

    results = tester.run_test_suite(model_factory, tests)

    def model_factory():
        """Factory function to create a new instance of the model."""

        class SingleLayer(torch.nn.Module):
            def __init__(self, input_dim, output_dim, neurons):
                super().__init__()
                self.first_layer = ExpertSignLinear(input_dim, neurons, learning_rate=.1, alpha=.0)
                self.second_layer = ExpertSignLinear(neurons, output_dim, learning_rate=.1, alpha=.0)
                self.activation_function = torch.nn.ReLU()

            def forward(self, x):
                x = self.first_layer(x)
                x = self.activation_function(x)
                x = self.second_layer(x)
                return x

            def update(self, loss):
                self.first_layer.update(loss)
                self.second_layer.update(loss)

        torch_model = SingleLayer(
            2, 1, neurons=100
        )
        optimizer = lambda params: Adam(params, lr=0.01)
        model = TorchAdapter(
            model = torch_model,
            optim = optimizer,
            criterion = MSELoss(),
            device = 'cpu',
        )
        wrapped_model = WrapperAlgorithm(model)
        return wrapped_model

    tester = AlgorithmTester(results_dir="dt_test_results")

    # Create classification test cases (exclude regression)
    tests = [tc for tc in create_2_variable_test_cases()]

    results = tester.run_test_suite(model_factory, tests)

    print("\n✅ All tests completed! Check the 'dt_test_results' directory for detailed results and plots.")
