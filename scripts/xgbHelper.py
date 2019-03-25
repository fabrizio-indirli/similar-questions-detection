import xgboost as xgb 
from sklearn.model_selection import train_test_split

import numpy as np

class XGBHelper():
    def __init__(self, params=None):
        self.params=params
    
    def fit(self, X, y, params=None):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=4242)

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_valid = xgb.DMatrix(X_valid, label=y_valid)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        
        if self.params is None:
            # Set our parameters for xgboost
            self.params = {}
            self.params['objective'] = 'binary:logistic'
            self.params['eval_metric'] = ['logloss', 'error']
            self.params['eta'] = 0.01
            self.params['max_depth'] = 6
            self.params["silent"] = True
    
        self.bst = xgb.train(self.params, d_train, 30000, watchlist, early_stopping_rounds=200, verbose_eval=0)

    def predict_proba(self, X):
        d_test = xgb.DMatrix(X)
        return self.bst.predict(d_test)