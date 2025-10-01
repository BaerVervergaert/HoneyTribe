from honeytribe.metrics.correlation.standard import *
import scipy as sp

class CorrelationSuite:
    def __init__(self, metrics=None):
        if metrics is None:
            metrics = [
                pearsonr,
                kendalltau,
                spearmanr,
                chatterjeexi,
                stepanovr,
            ]
        self.metrics = metrics
        self._saved_results = None

    def results_columns(self, a, b):
        self._saved_results = [ metric(a, b) for metric in self.metrics ]
        return tuple(self._saved_results)

    def results_matrix(self, A):
        self._saved_results = [ correlation_matrix(A, metric=metric) for metric in self.metrics ]
        return tuple(self._saved_results)

    def results(self, a, b=None):
        if b is None:
            return self.results_matrix(a)
        else:
            return self.results_columns(a, b)

    def print(self):
        if self._saved_results is None:
            raise ValueError('Compute results first by calling the `.results` method.')
        str_outputs = []
        for result in self._saved_results:
            str_outputs.append(self.print_single_result(result).lstrip('\n').rstrip('\n'))
        sep = "\n" + "="*50 + "\n"
        print("", *str_outputs, sep=sep, end=sep)

    def print_single_result(self, result):
        if isinstance(result, CorrelationOutput):
            return self.format_column_result(result)
        elif isinstance(result, CorrelationMatrixOutput):
            return self.format_matrix_result(result)
        else:
            raise ValueError(f'Unknown results output {result} of type {type(result)}.')

    def format_column_result(self, result: CorrelationOutput):
        base_info = Template("""
    Correlation metric: $name
    Correlation value: $value
        """.lstrip('\n').rstrip('\n'))
        hypothesis_info = Template("""
    Hypothesis test: $hypothesis_test
    Hypothesis description: $hypothesis_test_description
    P-value: $p_value $hypothesis_test_notes
        """.lstrip('\n').rstrip('\n'))
        output = []
        base_info = base_info.substitute(
            name=result.name,
            value=result.value,
        )
        output.append(base_info)
        if result.p_value is not None:
            hypothesis_info = hypothesis_info.substitute(
                hypothesis_test=result.hypothesis_test,
                hypothesis_test_description=result.hypothesis_test_description,
                p_value=result.p_value,
                hypothesis_test_notes="" if result.hypothesis_test_notes is None else f"Notes: {result.hypothesis_test_notes}"
            )
            output.append(hypothesis_info)
        return "\n\n".join(output)

    def format_matrix_result(self, result: CorrelationMatrixOutput):
        base_info = Template("""
            Correlation metric: $name
            Correlation value: $value
        """)
        hypothesis_info = Template("""
            Hypothesis test: $hypothesis_test
            Hypothesis description: $hypothesis_test_description
            P-value: $p_value
            $hypothesis_test_notes
        """)
        output = []
        base_info = base_info.substitute(
            name=result.name,
            value=result.value,
        )
        output.append(base_info)
        if result.p_value is not None:
            hypothesis_info = hypothesis_info.substitute(
                hypothesis_test=result.hypothesis_test,
                hypothesis_test_description=result.hypothesis_test_description,
                p_value=result.p_value,
                hypothesis_test_notes="" if result.hypothesis_test_notes is None else f"\n\tNotes: {result.hypothesis_test_notes}"
            )
            output.append(hypothesis_info)
        return "\n\n".join(output).lstrip().rstrip()