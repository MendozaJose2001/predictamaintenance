# ./src/models/negative_binomial

import numpy as np
import statsmodels.api as sm
import warnings
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array

class NegativeBinomialPiecewise(BaseEstimator, RegressorMixin):
    def __init__(self, 
                 alpha: float = 1.0, 
                 clipping_threshold: int = 125,
                 alpha_reg: float = 0.0,
                 l1_ratio: float = 0.5,
                 link_type: str = 'log'): # 'log', 'identity', 'sqrt'
        self.alpha = alpha
        self.clipping_threshold = clipping_threshold
        self.alpha_reg = alpha_reg
        self.l1_ratio = l1_ratio
        self.link_type = link_type
        self.is_fitted_ = False

    def _get_link(self):
        links = {
            'log': sm.families.links.Log(),
            'identity': sm.families.links.Identity(),
            'sqrt': sm.families.links.Sqrt()
        }
        return links.get(self.link_type, sm.families.links.Log())

    def predict(self, X):
        if not getattr(self, 'is_fitted_', False) or self.model_stats_ is None:
            return np.full(X.shape[0], np.nan)
        X = check_array(X)
        X_with_const = sm.add_constant(X, has_constant='add')
        return self.model_stats_.predict(X_with_const)
    
    def fit(self, X, y):
            X, y = check_X_y(X, y, accept_sparse=True)
            y_piecewise = np.minimum(y, self.clipping_threshold)
            X_with_const = sm.add_constant(X, has_constant='add')
            
            try:
                family_nb = sm.families.NegativeBinomial(
                    alpha=self.alpha, 
                    link=self._get_link()
                )
                model = sm.GLM(y_piecewise, X_with_const, family=family_nb)

                if self.alpha_reg > 0:
                    self.model_stats_ = model.fit_regularized(
                        method='elastic_net',
                        alpha=self.alpha_reg,
                        L1_wt=self.l1_ratio,
                        maxiter=500
                    )
                else:
                    self.model_stats_ = model.fit()
                
                self.is_fitted_ = True
                self.fit_success_ = 1  # Marcamos éxito
                
            except Exception as e:
                self.is_fitted_ = False
                self.model_stats_ = None
                self.fit_success_ = 0 
                warnings.warn(f"Iteración fallida: {self.link_type} con alpha_reg {self.alpha_reg}")
                
            return self
