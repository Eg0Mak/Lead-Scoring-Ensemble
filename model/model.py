import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

class EnsembleModel:
    def __init__(self, cb_params=None, lgb_params=None, xgb_params=None, cat_features=None, seeds=(42,)):
        self.cb_params = cb_params
        self.lgb_params = lgb_params
        self.xgb_params = xgb_params
        self.cat_features = cat_features or []
        self.seeds = seeds
        self.cb_models = []
        self.lgb_models = []
        self.xgb_models = []

    def fit(self, X, y):
        self.cb_models = []
        self.lgb_models = []
        self.xgb_models = []

        for seed in self.seeds:
            if self.cb_params is not None:
                cb = CatBoostClassifier(**self.cb_params, cat_features=self.cat_features, random_state=seed, verbose=False)
                cb.fit(X, y, verbose=False)
                self.cb_models.append(cb)

            if self.lgb_params is not None:
                lgb = LGBMClassifier(**self.lgb_params, random_state=seed, verbose=-1)
                lgb.fit(X, y, categorical_feature=self.cat_features)
                self.lgb_models.append(lgb)

            if self.xgb_params is not None:
                xgb = XGBClassifier(**self.xgb_params, random_state=seed)
                xgb.fit(X, y)
                self.xgb_models.append(xgb)

        return self

    def predict_proba(self, X):
        preds = [m.predict_proba(X)[:, 1] for m in self.cb_models]
        preds += [m.predict_proba(X)[:, 1] for m in self.lgb_models]
        preds += [m.predict_proba(X)[:, 1] for m in self.xgb_models]
        return np.mean(preds, axis=0)