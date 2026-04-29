"""
Generic ensemble utilities for sklearn-compatible pipelines.
Designed to be reusable across ML projects.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


class BlendingEnsemble:
    """
    Holdout blending ensemble for binary classification.

    Strategy:
      1. X_train is split into blend_train (1 - blend_ratio) + holdout (blend_ratio).
      2. Base models are trained on blend_train only.
      3. Holdout predict_proba outputs form the meta-feature matrix.
      4. The meta-model is trained on that matrix.
      5. At inference: base models score X → meta-model combines probabilities.

    Unlike CV-stacking, there is no cross-validation loop — leakage is prevented
    by using a dedicated holdout set that the base models never see.

    Parameters
    ----------
    base_models  : list of (name, estimator) tuples
    meta_model   : sklearn-compatible classifier for the second layer
    blend_ratio  : fraction of training data reserved for the holdout (default 0.2)
    random_state : reproducibility seed

    Examples
    --------
    blending = BlendingEnsemble(
        base_models=[("lgbm", lgbm_clf), ("rf", rf_clf)],
        meta_model=LogisticRegression(C=1.0, max_iter=1000),
    )
    blending.fit(X_train, y_train)
    proba = blending.predict_proba(X_test)[:, 1]
    """

    def __init__(
        self,
        base_models: list,
        meta_model,
        blend_ratio: float = 0.2,
        random_state: int = 42,
    ):
        self.base_models  = base_models
        self.meta_model   = meta_model
        self.blend_ratio  = blend_ratio
        self.random_state = random_state

    def fit(self, X, y):
        X_bt, X_hold, y_bt, y_hold = train_test_split(
            X, y,
            test_size=self.blend_ratio,
            stratify=y,
            random_state=self.random_state,
        )
        for _, m in self.base_models:
            m.fit(X_bt, y_bt)

        meta_train = np.column_stack([
            m.predict_proba(X_hold)[:, 1] for _, m in self.base_models
        ])
        self.meta_model.fit(meta_train, y_hold)
        return self

    def predict_proba(self, X):
        meta = np.column_stack([
            m.predict_proba(X)[:, 1] for _, m in self.base_models
        ])
        prob1 = self.meta_model.predict_proba(meta)[:, 1]
        return np.column_stack([1 - prob1, prob1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
