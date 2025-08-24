from typing import Self
from ngboost import NGBRegressor
import numpy as np
from sklearn import BaseEstimator
from honeytribe.online_convex_optimization.expert import BaseExpertAlgorithm


class InitialNGBoost(BaseEstimator):
    def __init__(self, params: np.ndarray, norm: float):
        self.params = params
        self.norm = norm
    def predict(self, X) -> np.ndarray:
        m, n = X.shape
        params = np.ones((m, self.params.shape[0])) * self.params * self.norm
        return params

class SubmodelNGBoost(BaseEstimator):
    def __init__(self, models, col_idx, lr, scale, norm):
        self.models = models
        self.col_idx = col_idx
        self.lr = lr
        self.scale = scale
        self.norm = norm
    def predict(self, X) -> np.ndarray:
        resids = np.array([model.predict(X[:, self.col_idx]) for model in self.models]).T
        return - self.lr * resids * self.scale * self.norm

def get_submodels_from_ngboost(ngboost_fitted: NGBRegressor) -> list:
    N = len(ngboost_fitted.base_models) + 1
    out = []
    init_submodel = InitialNGBoost(ngboost_fitted.init_params, N)
    out.append(init_submodel)
    for i, (models, s, col_idx) in enumerate(
        zip(ngboost_fitted.base_models, ngboost_fitted.scalings, ngboost_fitted.col_idxs)
    ):
        submodel = SubmodelNGBoost(models, col_idx, ngboost_fitted.learning_rate, s, N)
        out.append(submodel)
    return out

class ReprocessorExpertAlgorithm:
    def __init__(self, ngboost_fitted: NGBRegressor, expert_algorithm: BaseExpertAlgorithm):
        self.fitted_model = ngboost_fitted
        self.expert_algorithm = expert_algorithm

    def fit(self, X, y) -> Self:
        self.expert_algorithm.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        weights = self.expert_algorithm.predict(X)
        m, n = X.shape
        out = []
        for i in range(m):
            w = weights[i]
            res = 0
            for p, expert in zip(w, self.expert_algorithm.experts):
                res += p * expert.predict(X[i:i+1])
            out.append(res)
        return np.concatenate(out, axis=0)

    @staticmethod
    def example() -> None:
        msg = """
        ngboost_model = NGBRegressor()
        ngboost_model.fit(X,y)
        
        def loss_function(y_pred, y_true):
            return -sp.stats.normal(y_pred).logpdf(y_true)
        
        expert_algorithm = HedgeAlgorithm(
            loss_function = loss_function,
            experts = get_submodels_from_ngboost(ngboost_model), 
            learning_rate = 1., 
            update_experts = False, 
        )
        
        reprocessed_model = ReprocessorExpertAlgorithm(ngboost_model, expert_algorithm)
        reprocessed_model.fit(X, y)
        
        guess = reprocessed_model.predict(X)
        guess        
        """
        print(msg)