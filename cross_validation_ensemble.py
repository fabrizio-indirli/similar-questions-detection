import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
import xgboost as xgb 
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from scripts.lgbmHelper import LGBMHelper
from scripts.xgbHelper import XGBHelper
from scripts.postprocess_submission import postprocess

# Load data
train_all = pd.read_csv("./data/train.csv", names=['row_ID', 'text_a_ID', 'text_b_ID', 'text_a_text', 'text_b_text', 'have_same_meaning'], index_col=0)
train_nlp_features = pd.read_csv("./data/nlp_features_train.csv")
train_graph_features = pd.read_csv("./data/graph_features_train.csv")
train_distance_features = pd.read_csv("./data/distance_features_train.csv")
train = pd.concat((train_nlp_features, train_graph_features, train_distance_features), axis=1)

test_all = pd.read_csv("./data/test.csv", names=['row_ID', 'text_a_ID', 'text_b_ID', 'text_a_text', 'text_b_text', 'have_same_meaning'], index_col=0)
test_nlp_features = pd.read_csv("./data/nlp_features_test.csv")
test_graph_features = pd.read_csv("./data/graph_features_test.csv")
test_distance_features = pd.read_csv("./data/distance_features_test.csv")
test = pd.concat((test_nlp_features, test_graph_features, test_distance_features), axis=1)

# Remove NaN values
train = train.dropna(axis=1)
train = train.replace([np.inf, -np.inf], -1)
test = test.dropna(axis=1)
test= test.replace([np.inf, -np.inf], -1)

train_ensemble = train.copy()
test_ensemble = test.copy()

# Folds
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
kfold_splits = list(kfold.split(train, train_all.have_same_meaning))

# Run cross validation and do out of folds predictions that will be later used by 2nd level model

# Train Models
models = [(LGBMHelper(), False),
          (LogisticRegression(solver='lbfgs', max_iter=1000), True), 
          (RandomForestClassifier(), False),
          (GradientBoostingClassifier(), False),
          (XGBHelper(), False),
          (KNeighborsClassifier(n_neighbors=3), True)]

results = {}

for i, (model,scale) in enumerate(models):
    print("-----------------------------------------------------------")
    print("Train model {}".format(model.__class__.__name__))
    print("-----------------------------------------------------------")
    log_loss_fold = []
    print("Split ", end="")
    for j, (train_indices, test_indices) in enumerate(kfold_splits):
        print(str(j+1) + ",", end=" ")
        X_train = train.iloc[train_indices].astype(np.float64)
        y_train = train_all.have_same_meaning.iloc[train_indices].astype(np.float64)
        
        X_test = train.iloc[test_indices].astype(np.float64)
        y_test = train_all.have_same_meaning.iloc[test_indices].astype(np.float64)
        
        if scale: 
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
        model.fit(X_train, y_train)
        
        train_pred = model.predict_proba(train.iloc[test_indices,:])
        test_pred = model.predict_proba(test)

        if len(train_pred.shape) > 1 and train_pred.shape[1] == 2:
            train_pred = train_pred[:,1]
            test_pred = test_pred[:,1]

        train_ensemble.loc[test_indices,"predictions_" + model.__class__.__name__] = train_pred
        test_ensemble["predictions" + model.__class__.__name__] = test_pred
        
        log_loss_fold.append(log_loss(y_test, train_pred))
                
    loss = np.mean(log_loss_fold)
    results[model.__class__.__name__] = loss

    print("")
    print("Result model {}: {}".format(model.__class__.__name__,loss))


if os.path.exists("./predictions/test_ensemble_lstm.csv") and os.path.exists("./predictions/train_ensemble_lstm.csv"):
    train_pred_lstm = pd.read_csv("./predictions/test_ensemble_lstm.csv")
    test_pred_lstm = pd.read_csv("./predictions/train_ensemble_lstm.csv")
    
    train_ensemble["predictions_lstm"] = train_pred_lstm.pred_lstm
    test_ensemble["predictions_lstm"] = test_pred_lstm.pred_lstm

    
print("Train LGBM with train and validation set...")
X_train, X_valid, y_train, y_valid = train_test_split(train_ensemble, train_all.have_same_meaning, test_size=0.2, random_state=4242)

d_train = lgb.Dataset(X_train, label=y_train)
d_valid = lgb.Dataset(X_valid, label=y_valid)
d_test = lgb.Dataset(test_ensemble)

# Set our parameters for xgboost
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 5,
    'learning_rate': 0.01,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.5,
    'bagging_freq': 5,
    'verbose': 0,
    'max_depth':5
}
bst = lgb.train(params,
                d_train,
                num_boost_round=10000,
                verbose_eval=300,
                valid_sets=[d_train, d_valid],
                early_stopping_rounds=300)

ax = lgb.plot_importance(bst, max_num_features=30)
plt.savefig("feature_importance.png")
plt.show()

print("Train LGBM with full data set...")
n_iter = bst.best_iteration

d_train = lgb.Dataset(train_ensemble, label=train_all.have_same_meaning)
d_test = lgb.Dataset(test_ensemble)

bst = lgb.train(params,
                d_train,
                num_boost_round=n_iter,
                valid_sets=[d_train],
                verbose_eval=300)


# Store features for weighted graph
y_test_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
y_train_pred = bst.predict(train, num_iteration=bst.best_iteration)

df_test_res = pd.DataFrame(y_test_pred, columns=["weight"])
df_test_res.to_csv("./predictions/predictions_ensemble_test.csv")

df_train_res = pd.DataFrame(y_train_pred, columns=["weight"])
df_train_res.to_csv("./predictions/predictions_ensemble_train.csv")

# Submit results
y_pred = bst.predict(test_ensemble, num_iteration=bst.best_iteration)
submission = pd.DataFrame(test.index.values, columns=["Id"])
submission["Score"] = y_pred
submission.to_csv("submission.csv", index=None)

# Post-process
print(" ")
print("POST PROCESSING")
postprocess()
