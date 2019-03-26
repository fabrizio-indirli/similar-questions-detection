import lightgbm as lgb
from sklearn.model_selection import train_test_split

import numpy as np

class LGBMHelper():
    
    def __init__(self, params=None):
        self.params=params
    
    def fit(self, X, y):
        X_train_, X_valid, y_train_, y_valid = train_test_split(X, y, test_size=0.2, random_state=4242)

        d_train = lgb.Dataset(X_train_, label=y_train_)
        d_valid = lgb.Dataset(X_valid, label=y_valid)
        
        if self.params is None:
            self.params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_leaves': 31,
                'learning_rate': 0.01,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0,
            }

        self.bst = lgb.train(self.params,
                    d_train,
                    num_boost_round=30000,
                    verbose_eval=0,
                    valid_sets=[d_train, d_valid],
                    early_stopping_rounds=100)

    def predict_proba(self, X):
        return self.bst.predict(X, num_iteration=self.bst.best_iteration)