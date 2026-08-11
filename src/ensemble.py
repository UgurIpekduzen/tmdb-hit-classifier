"""
sklearn-uyumlu pipeline'lar için genel amaçlı ensemble yardımcıları.
Farklı ML projelerinde yeniden kullanılabilir olacak şekilde tasarlanmıştır.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


class BlendingEnsemble:
    """
    Binary sınıflandırma için holdout blending ensemble.

    Strateji:
      1. X_train, blend_train (1 - blend_ratio) + holdout (blend_ratio) olarak bölünür.
      2. Base modeller sadece blend_train üzerinde eğitilir.
      3. Holdout predict_proba çıktıları meta-feature matrisini oluşturur.
      4. Meta-model bu matris üzerinde eğitilir.
      5. Inference'da: base modeller X'i skorlar → meta-model olasılıkları birleştirir.

    CV-stacking'den farklı olarak cross-validation döngüsü yoktur — leakage,
    base modellerin hiç görmediği ayrı bir holdout set kullanılarak önlenir.

    Parameters
    ----------
    base_models  : (name, estimator) tuple'larından oluşan liste
    meta_model   : ikinci katman için sklearn-uyumlu sınıflandırıcı
    blend_ratio  : holdout için ayrılan eğitim verisi oranı (varsayılan 0.2)
    random_state : tekrarlanabilirlik seed'i

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
