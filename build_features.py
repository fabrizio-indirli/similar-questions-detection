import sys
import os

sys.path.insert(0, './scripts/')
from distance_features import compute_distance_features
from graph_features import compute_graph_features
from nlp_features import compute_nlp_features

# Features files paths
train_graph_feats_paths = "./data/graph_features_train.csv"
test_graph_feats_paths = "./data/graph_features_test.csv"

train_distance_feats_path = "./data/distance_features_train.csv"
test_distance_feats_path = "./data/distance_features_test.csv"

train_nlp_feats_path = "./data/nlp_features_train.csv"
test_nlp_feats_path = "./data/nlp_features_test.csv"


# Compute graph features
if not os.path.isfile(train_graph_feats_paths) or not os.path.isfile(test_graph_feats_paths):
    print("COMPUTING GRAPH FEATURES")
    compute_graph_features()
else:
    print("Graph features already present in files for both train and test set.")
print(" ")


# Compute distance features
if not os.path.isfile(train_distance_feats_path) or not os.path.isfile(test_distance_feats_path):
    print("COMPUTING DISTANCE FEATURES")
    compute_distance_features()
else:
    print("Distance features already present in files for both train and test set.")
print(" ")


# Compute NLP features
if not os.path.isfile(train_nlp_feats_path) or not os.path.isfile(test_nlp_feats_path):
    print("COMPUTING NLP FEATURES")
    compute_nlp_features()
else:
    print("NLP features already present in files for both train and test set.")
print(" ")
